# RIRPINN
***
This repository contains the code for the paper [Room impulse response reconstruction with physics-informed deep learning
](https://arxiv.org/abs/2401.01206). The paper is currently under review at the Journal of the Acoustical Society of America.

## Abstract
***
A method is presented for estimating and reconstructing the sound field within a room using physics-informed neural networks. By incorporating a limited set of experimental room impulse responses as training data, this approach combines neural network processing capabilities with the underlying physics of sound propagation, as articulated by the wave equation. The network's ability to estimate particle velocity and intensity, in addition to sound pressure, demonstrates its capacity to represent the flow of acoustic energy and completely characterise the sound field with only a few measurements. Additionally, an investigation into the potential of this network as a tool for improving acoustic simulations is conducted. This is due to its profficiency in offering grid-free sound field mappings with minimal inference time. Furthermore, a study is carried out which encompasses comparative analyses against current approaches for sound field reconstruction. Specifically, the proposed approach is evaluated against both data-driven techniques and elementary wave-based regression methods. The results demonstrate that the physics-informed neural network stands out when reconstructing the early part of the room impulse response, while simultaneously allowing for complete sound field characterisation in the time domain.

## Usage
***

To create a conda environment with all the required dependencies, execute the following command in your terminal:

```bash 
conda env create -f environment.yml
```

To activate the environment, execute the following command in your terminal:

```bash
conda activate rir_pinn
```
Before running the code, 
you need to download the
[RIR dataset](https://data.dtu.dk/articles/dataset/Planar_Room_Impulse_Response_Dataset_-_ACT_DTU_Electro_b_355_r_008_/21740453) 
and place it in the `data` folder. This is done automatically by running the `DL_data.py` script found in the `data` folder.

```bash
python ./data/DL_data.py
```

To train the network, execute the following command in your terminal:

```bash
python ./src/run_PINN.py
```

An example of how to use the trained network to reconstruct the sound field is provided in the `./src/inference_nb.ipynb` notebook.

## Citation
***
If you use this code in your research, please cite the following paper:

```
@article{karakonstantis2024room,
      title={Room impulse response reconstruction with physics-informed deep learning}, 
      author={Xenofon Karakonstantis and Diego Caviedes-Nozal and Antoine Richard and Efren Fernandez-Grande},
      year={2024},
      eprint={2401.01206},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```
![](sf.gif)