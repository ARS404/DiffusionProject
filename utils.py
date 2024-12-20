import pickle

import dnnlib
import torch


def init_edm():
    network_pkl = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-ve.pkl'
    device = torch.device('cuda')

    with dnnlib.util.open_url(network_pkl) as f:
        net = pickle.load(f)['ema'].to(device)
    
    return net

