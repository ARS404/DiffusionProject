import os
import shutil

import torch

from hydra.utils import instantiate
from PIL import Image
from tqdm import tqdm

from metrics.fid import FID


class Tester(object):
    def __init__(self, config):
        self.config = config
        self.out_root = config.data_config.out_root
        self.dataset_name = config.data_config.dataset

        self.num_samples = config.num_samples
        self.batch_size = 128

        self.model = None
        self.solver = None
        self.fid = FID()


    def set_solver(self, solver_config):
        self.solver_config = solver_config
        self.solver = instantiate(solver_config)

    def run_test(self):
        result_path = self.__get_result_path()
        if os.path.exists(result_path):
            shutil.rmtree(result_path)
        os.makedirs(result_path, exist_ok=True)
        
        self.save_samples()
        if self.dataset_name == "CIFAR10":
            return self.fid(
                result_path, 
                'https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz', 
                self.num_samples,
                self.batch_size
            )
        raise Exception(f"Unexpected dataset {self.dataset_name}")

    def save_samples(self) -> None:
        result_path = self.__get_result_path()
        count = 0
        assert self.num_samples % 10 == 0

        with tqdm(total= self.num_samples) as pbar:
            while count < self.num_samples:
                cur_batch_size = min(self.num_samples - count, self.batch_size)
                noise = torch.randn(cur_batch_size, 3, 32, 32, device='cuda')
                out, trajectory = self.solver(self.model, noise, self.solver_config)
                out = (out * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
                for i in range(out.shape[0]):
                    img = Image.fromarray(out[i])
                    n_digits = len(str(count))
                    img_name = (6 - n_digits) * '0' + str(count) + '.png'
                    img.save(os.path.join(result_path, img_name))
                    count += 1
                    pbar.update(1)
                    pbar.set_description('%d images saved' % (count,))

    def __get_result_path(self):
        assert self.solver is not None, "Set solver!"
        return os.path.join(self.out_root, self.solver.get_name(), self.dataset_name)
