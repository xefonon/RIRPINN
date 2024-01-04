import urllib.request
import os
# Download dataset
dir_ = os.path.dirname(os.path.abspath(__file__))
datapath = os.path.join(dir_, 'SoundFieldControlPlanarDataset.h5')
url = 'https://github.com/xefonon/RIRPINN/releases/download/dataset/SoundFieldControlPlanarDataset.h5'
if not os.path.exists(datapath):
    urllib.request.urlretrieve(url, datapath)