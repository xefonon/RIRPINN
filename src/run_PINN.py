import torch
from torch.utils.data import DataLoader
from aux_functions import (FCN, PINNDataset, PINNDataset_w_IC, construct_input_vec,
                           plot_results, scan_checkpoint, save_checkpoint,
                           load_checkpoint, get_measurement_vectors,
                           standardize_rirs, normalize_rirs, config_from_yaml, construct_rir_input_vec,
                           get_jabra_data, unit_norm_normalization, maxabs_normalize_rirs,plot_rir_training,
                           get_019_measurement_vectors, get_simulated_measurement_vectors,
                           get_avg_edc,get_odeon_data)
import numpy as np
import click
import os
import matplotlib.pyplot as plt

# Set default dtype to float32
torch.set_default_dtype(torch.float)
# PyTorch random number generator
torch.manual_seed(1234)
# Random number generators in other libraries
np.random.seed(1234)
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
if device == 'cuda':
    print(torch.cuda.get_device_name())
def training_examples(hparams, x_ref, y_ref, t, grid_measured, grid_initial, t_indx_plt, refdata=None):
    if hparams.sparse_data:
        xyt_plt, p_plt = construct_input_vec( x_ref, y_ref, t, rirdata= None, t_ind=t_indx_plt)
        grids = dict(grid_measured = grid_measured, grid_initial = grid_initial)
    else:
        rand_indx = np.random.randint(0, len(x_ref))
        t_plt = [.011, .013, .019, .021, .03]
        t_indx_plt = np.array([np.argmin(t < t_plt[indx]) for indx in range(len(t_plt))])
        xyt_plt, p_plt = construct_input_vec( x_ref, y_ref, t, rirdata= refdata, t_ind=t_indx_plt)
        xyt_rir  = construct_rir_input_vec(x_ref[rand_indx], y_ref[rand_indx], t)
        p_rir = refdata[rand_indx]
        grids = None
    return xyt_plt, p_plt, xyt_rir, p_rir, grids
# %%
"""Command line interface for training the PINN"""
@click.command()
@click.option(
    "--data_dir", default='../Data', type=str, help="Directory of training data"
)
@click.option(
    "--config_file", default='./config.yml', type=str, help="Directory of training data"
)
@click.option('--use_wandb', is_flag=True,
              help='Use weights and biases to monitor training')
def train_PINN(data_dir, config_file, use_wandb):
    if use_wandb:
        import wandb
        print("Using Weights and Biases to track training!")
        wandb.login()
        run = wandb.init(project='PINN_sound_field',
                         config=config_file)
        hparams = wandb.config

    else:
        hparams = config_from_yaml(config_file)

    checkpoint_dir = hparams.checkpoint_dir
    if hparams.sparse_data:
        filename = os.path.join(data_dir, 'Jabra_measurements.h5')
        (measureddata, initial_pressure_data, fs, grid_measured, grid_initial, grid,
         loudspeaker_position, c) = get_jabra_data(filename)
        refdata = measureddata
    elif hparams.classroom_data:
        filename = os.path.join(data_dir, '019_data.h5')

        refdata, fs, grid, measureddata, grid_measured, c = get_019_measurement_vectors(filename, subsample_points=hparams.n_mics_per_dimension)
        grid_initial = None
    elif hparams.simulated_data:
        filename = os.path.join(data_dir, 'odeon_rirs.h5')

        refdata, fs, grid, measureddata, grid_measured, c = get_odeon_data(filename,subsample_points=hparams.n_mics_per_dimension)  # per dimension
        grid_initial = None
    else:
        filename = os.path.join(data_dir, 'SoundFieldControlPlanarDataset.h5')
        refdata, fs, grid, measureddata, grid_measured, c = get_measurement_vectors(filename,
                                                                                    subsample_points=hparams.n_mics_per_dimension)  # per dimension
        grid_initial = None
    # %%
    # set t_weighting_factor from EDCs of measurement data set
    if hparams.t_weighting_factor:
        t_weighting_factor = get_avg_edc(measureddata)
    else:
        t_weighting_factor = None
    """Training Data"""
    if hparams.simulated_data:
        data = measureddata[:, int(0.085 * fs):int((hparams.rir_time + 0.085) * fs)]  # truncate
        refdata = refdata[:, int(0.085 * fs):int((hparams.rir_time + 0.085) * fs)] # truncate
        temp_data = refdata
    elif hparams.sparse_data:
        data = measureddata[:, :int(hparams.rir_time * fs)]
        refdata = refdata[:, :int(hparams.rir_time * fs)]
        initial_pressure_data = initial_pressure_data[:, :int(hparams.rir_time * fs)]
        temp_data = np.vstack([data, initial_pressure_data])
    else:
        data = measureddata[:, int(0.003 * fs):int(hparams.rir_time * fs)]  # truncate
        refdata = refdata[:, int(0.003 * fs):int(hparams.rir_time * fs)]  # truncate
        temp_data = np.vstack([data, refdata])
    if hparams.standardize_data:
        scaler = standardize_rirs(temp_data, device=device)
    else:
        if hparams.map_input:
            l_inf_norm = 0.1
        else:
            l_inf_norm = 0.9
        scaler = maxabs_normalize_rirs(temp_data, device=device, l_inf_norm=l_inf_norm)

    t_ind = np.arange(0, refdata.shape[-1])
    t = np.linspace(0., refdata.shape[-1] / fs, refdata.shape[-1])
    x_m = grid_measured[0]
    y_m = grid_measured[1]
    x_ref = grid[0]
    y_ref = grid[1]
    if hparams.sparse_data:
        x_ic = grid_initial[0]
        y_ic = grid_initial[1]
    # %%

    bounds = {
        'x': (1.1 * x_ref.min(), 1.1 * x_ref.max()),
        'y': (1.1 * y_ref.min(), 1.1 * y_ref.max()),
        't': (0, hparams.rir_time),
        'xdata' : (x_ref.min(), x_ref.max()),
        'ydata' : (y_ref.min(), y_ref.max())
    }

    if hparams.adaptive_loss_weights:
        sigmas = []
        if hparams.lambda_data != 0.:
            sigmas.append(torch.tensor(2/hparams.lambda_data, dtype=torch.float32, device=device, requires_grad=True))
        if hparams.lambda_pde != 0.:
            sigmas.append(torch.tensor(2/hparams.lambda_pde, dtype=torch.float32, device=device, requires_grad=True))
        if hparams.lambda_bc != 0.:
            sigmas.append(torch.tensor(2/hparams.lambda_bc, dtype=torch.float32, device=device, requires_grad=True))
        if hparams.lambda_ic != 0.:
            sigmas.append(torch.tensor(2/hparams.lambda_ic, dtype=torch.float32, device=device, requires_grad=True))
        lambda_optimizer = torch.optim.Adam(sigmas, hparams.sigma_lr, betas=(0.9, 0.9),
                                            weight_decay=1e-5)
        # scheduler with cosine annealing
        lambda_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(lambda_optimizer, T_max=hparams.train_epochs,
                                                                        eta_min=1e-6)
    else:
        sigmas = [None]

    PINN = FCN(n_hidden_layers=hparams.n_hidden_layers, bounds=bounds, device=device, siren=hparams.siren,
               lambda_data=hparams.lambda_data, lambda_pde=hparams.lambda_pde, lambda_bc=hparams.lambda_bc,
               c=c, lambda_ic=hparams.lambda_ic, loss_fn=hparams.loss_fn, output_scaler=scaler,
               fs=fs, map_input=hparams.map_input,
               activation=hparams.activation_fn, n_hidden_features=hparams.n_hidden_features,
               sigmas = sigmas, network_type = hparams.network_architecture)


    net_params = list(PINN.dnn.parameters())
    if use_wandb:
        if hparams.activation_fn != 'gabor':
            wandb.watch(PINN.dnn, PINN.loss_function, log='all')

    '''Optimization'''
    gamma = 0.9  # final learning rate will be gamma * initial_lr
    optimizer = torch.optim.Adam(net_params, hparams.lr, betas=(0.9, 0.999), eps=1e-08, amsgrad=False,
                                 weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hparams.train_epochs, eta_min=1e-6)

    'Neural Network Summary'
    print(PINN.dnn)
    # %%
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if os.path.isdir(checkpoint_dir):
        cp_pinn = scan_checkpoint(checkpoint_dir, "PINN_")

    steps = 0
    if cp_pinn is None:
        last_epoch = -1
    else:
        state_dict_pinn = load_checkpoint(cp_pinn, device)
        PINN.dnn.load_state_dict(state_dict_pinn["net"])
        steps = state_dict_pinn["steps"] + 1
        last_epoch = state_dict_pinn["epoch"]
        optimizer.load_state_dict(state_dict_pinn["optim"])
        if hparams.adaptive_loss_weights:
            lambda_optimizer.load_state_dict(state_dict_pinn["lambda_optim"])
    # Dataset
    if hparams.sparse_data:
        dataset = PINNDataset_w_IC(measured_data=refdata, ic_data=initial_pressure_data,
                                   x_ic=x_ic, y_ic=y_ic,
                                   x_m=x_m, y_m=y_m, t=t, t_ind=t_ind,
                                   n_pde_samples=hparams.n_pde_samples, counter=steps + 1,
                                   maxcounter=hparams.max_t_counter,
                                   curriculum_training=hparams.curriculum_training,
                                   batch_size=hparams.batch_size,
                                   t_weighting_factor=t_weighting_factor,
                                   bounds= bounds)
    else:
        dataset = PINNDataset(refdata=refdata, measured_data=data, x_ref=x_ref, y_ref=y_ref,
                              x_m=x_m, y_m=y_m, t=t, t_ind=t_ind,
                              n_pde_samples=hparams.n_pde_samples, counter=steps + 1,
                              maxcounter=hparams.max_t_counter,
                              curriculum_training=hparams.curriculum_training,
                              batch_size=hparams.batch_size,
                              t_weighting_factor=t_weighting_factor)
    train_dataloader = DataLoader(dataset, batch_size=hparams.batch_size,
                                  shuffle=True,
                                  pin_memory=True, num_workers=0)

    PINN.dnn.train()
    t_plt = np.linspace(.01, hparams.rir_time - .01, 5)
    t_indx_plt = np.array([np.argmin(t < t_plt[indx]) for indx in range(len(t_plt))])
    if not hparams.adaptive_loss_weights:
        # lambda_pde_decay = (1 - .98) ** np.linspace(0, 1, hparams.max_t_counter)
        lambda_pde_decay = np.ones(hparams.train_epochs)

    xyt_plt, p_plt, xyt_rir, p_rir, grids = training_examples(hparams, x_ref, y_ref, t, grid_measured,
                                                              grid_initial, t_indx_plt, refdata=refdata)

    loss_dict = {'loss_total': [], 'loss_data': [], 'loss_pde': [], 'loss_bc': [], 'loss_ic': []}
    for epoch in range(max(0, last_epoch), hparams.train_epochs):
        for i, batch in enumerate(train_dataloader):
            data_input = batch['collocation_train']
            pde_input = batch['collocation_pde']
            ic_input = batch['collocation_ic']
            bc_input = batch['collocation_bc']
            p_data = batch['pressure_batch']
            if hparams.sparse_data:
                p_ic = batch['pressure_ic_batch']
            else:
                p_ic = torch.zeros_like(p_data)
            t_lims = batch['t_lims']
            data_loss_weights = batch['data_loss_weights']
            # p_test = batch['pressure_all']
            optimizer.zero_grad()
            if hparams.adaptive_loss_weights:
                lambda_optimizer.zero_grad()

            loss_total, loss_data, loss_pde, loss_bc, loss_ic, norm_ratio, std_ratio, maxabs_ratio = PINN.SGD_step(
                data_input.to(device),
                pde_input.to(device),
                bc_input.to(device),
                ic_input.to(device),
                p_data.to(device),
                p_ic.to(device),
                data_loss_weights.to(device),
                iter = epoch
            )
            loss_dict['loss_total'].append(loss_total.item())
            loss_dict['loss_data'].append(loss_data.item())
            loss_dict['loss_pde'].append(loss_pde.item())
            loss_dict['loss_bc'].append(loss_bc.item())
            loss_dict['loss_ic'].append(loss_ic.item())
            optimizer.step()
            if hparams.adaptive_loss_weights:
                lambda_optimizer.step()

            if hparams.scheduler_step > 0:
                scheduler.step()
                if hparams.adaptive_loss_weights:
                    lambda_scheduler.step()
            if steps % 100 == 0:
                if use_wandb:
                    wandb.log({
                        "total_loss": np.mean(loss_dict['loss_total']),
                        "data_loss": np.mean(loss_dict['loss_data']),
                        "PDE_loss": np.mean(loss_dict['loss_pde']),
                        "BCs_loss": np.mean(loss_dict['loss_bc']),
                        "ICs_loss": np.mean(loss_dict['loss_ic']),
                        "norm_ratio": norm_ratio,
                        "std_ratio": std_ratio,
                        "maxabs_ratio": maxabs_ratio,
                        "lr": optimizer.param_groups[0]['lr']})
            # reset loss dict
            loss_dict = {'loss_total': [], 'loss_data': [], 'loss_pde': [], 'loss_bc': [], 'loss_ic': []}
            if steps % int(1000) == 0:
                # plot adaptive pde weight and log with wandb
                if hparams.adaptive_loss_weights:
                    # plot sigma_1, sigma_2 and sigma_3
                    if use_wandb:
                        for o, sig in enumerate(PINN.sigmas):
                            wandb.log({f"adaptive_loss_weights_{o}": 1/(2 * sig.pow(2))})
                    plt.close('all')
                xyt_plt, p_plt, xyt_rir, p_rir, grids = training_examples(hparams, x_ref, y_ref, t, grid_measured,
                                                                          grid_initial, t_indx_plt, refdata=refdata)
                fig, errors, avg_snapshot_error = plot_results(xyt_plt, PINN, p_plt, bounds, grids)
                fig2 = plot_rir_training(xyt_rir, p_rir, PINN, grid_measured)
                if use_wandb:
                    wandb.log({"Sound_Fields": wandb.Image(fig)})
                    wandb.log({"RIRs": wandb.Image(fig2)})
                plt.close('all')
                if use_wandb:
                    if not hparams.sparse_data:
                        for ii, error in enumerate(errors):
                            wandb.log({
                                f"MSE - t: {xyt_plt[ii, 2, 0]:.3f} s": error,
                                "steps": steps})
                        wandb.log({
                            "Average snapshot square error": avg_snapshot_error,
                            "steps": steps})
                    wandb.log({
                        "Training (time) window lower bound": t_lims[0].item(),
                        "Training (time) window upper bound": t_lims[1].item(),
                        "steps": steps})
            if steps % int(1000) == 0:
                checkpoint_path = "{}/PINN_{:08d}".format(checkpoint_dir, steps)
                state_dict_ = {
                    "net": PINN.dnn.state_dict(),
                    "optim": optimizer.state_dict(),
                    "steps": steps,
                    "epoch": epoch,
                }
                if hparams.adaptive_loss_weights:
                    state_dict_["lambda_optim"] = lambda_optimizer.state_dict()
                save_checkpoint(checkpoint_dir,
                                checkpoint_path,
                                state_dict_,
                                remove_below_step=steps // 2,
                                current_step=steps)

            if not hparams.adaptive_loss_weights:
                PINN.lambda_pde = hparams.lambda_pde * lambda_pde_decay[steps]
            steps += 1
            if hparams.adaptive_loss_weights:
                sigma_print = []
                for o, sig in enumerate(PINN.sigmas):
                    sigma_print.append(sig.mean().data)
            else:
                sigma_print = [hparams.lambda_data ,hparams.lambda_pde ,hparams.lambda_bc,hparams.lambda_ic]
            print(
                f'\repochs: {epoch + 1} total steps: {steps}, loss: {loss_total:.3}, t limits: ({t_lims[0].item():.3}, '
                f'{t_lims[1].item():.3}) sec, sigma_1: {sigma_print[0]:.4}, sigma_2: {sigma_print[1]:.4}',
                end='',
                flush=True)


if __name__ == "__main__":
    train_PINN()
# train_PINN()
