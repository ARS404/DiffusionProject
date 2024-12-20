import pickle

import click
import numpy as np
import scipy
import torch

from abc import abstractmethod

from tqdm import tqdm

from .base import BaseMetric
from models.edm import dnnlib
from models.edm.training import dataset

class FID(BaseMetric):
    def __call__(self, generated_path, ref_path, num_samples, batch):
        with dnnlib.util.open_url(ref_path) as f:
            ref = dict(np.load(f))

        mu, sigma = self.calculate_inception_stats(
            image_path=generated_path, 
            num_expected=num_samples, 
            max_batch_size=batch
        )
        fid = self.calculate_fid_from_inception_stats(mu, sigma, ref['mu'], ref['sigma'])
        return fid
    
    def calculate_inception_stats(
        self, image_path, num_expected=None, seed=0, max_batch_size=64,
        num_workers=3, prefetch_factor=2, device=torch.device('cuda')
    ):
        print('Loading Inception-v3 model...')
        detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
        detector_kwargs = dict(return_features=True)
        feature_dim = 2048
        with dnnlib.util.open_url(detector_url, verbose=True) as f:
            detector_net = pickle.load(f).to(device)

        # List images.
        print(f'Loading images from "{image_path}"...')
        dataset_obj = dataset.ImageFolderDataset(path=image_path, max_size=num_expected, random_seed=seed)
        if num_expected is not None and len(dataset_obj) < num_expected:
            raise click.ClickException(f'Found {len(dataset_obj)} images, but expected at least {num_expected}')
        if len(dataset_obj) < 2:
            raise click.ClickException(f'Found {len(dataset_obj)} images, but need at least 2 to compute statistics')

        # Divide images into batches.
        num_batches = ((len(dataset_obj) - 1) // max_batch_size + 1)
        all_batches = torch.arange(len(dataset_obj)).tensor_split(num_batches)
        rank_batches = all_batches
        data_loader = torch.utils.data.DataLoader(dataset_obj, batch_sampler=rank_batches, num_workers=num_workers, prefetch_factor=prefetch_factor)

        # Accumulate statistics.
        print(f'Calculating statistics for {len(dataset_obj)} images...')
        mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
        sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)
        for images, _labels in tqdm(data_loader, unit='batch', disable=False):
            if images.shape[0] == 0:
                continue
            if images.shape[1] == 1:
                images = images.repeat([1, 3, 1, 1])
            features = detector_net(images.to(device), **detector_kwargs).to(torch.float64)
            mu += features.sum(0)
            sigma += features.T @ features

        # Calculate grand totals.
        mu /= len(dataset_obj)
        sigma -= mu.ger(mu) * len(dataset_obj)
        sigma /= len(dataset_obj) - 1
        return mu.cpu().numpy(), sigma.cpu().numpy()

    def calculate_fid_from_inception_stats(self, mu, sigma, mu_ref, sigma_ref):
        m = np.square(mu - mu_ref).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
        fid = m + np.trace(sigma + sigma_ref - s * 2)
        return float(np.real(fid))