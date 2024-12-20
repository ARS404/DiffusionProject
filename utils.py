import pickle

import torch

from models.edm import dnnlib


def init_edm():
    network_pkl = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-ve.pkl'
    device = torch.device('cuda')

    with dnnlib.util.open_url(network_pkl) as f:
        net = pickle.load(f)['ema'].to(device)
    
    return net

