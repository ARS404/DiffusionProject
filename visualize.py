import os

import hydra
import matplotlib.pyplot as plt
import torch

from hydra.utils import instantiate
from omegaconf import DictConfig
from torchvision.utils import make_grid

from utils import *



@hydra.main(version_base=None, config_path="./configs", config_name="visualize")
def visualize(config: DictConfig) -> None:
    solver = instantiate(config["solvers"])
    config = config["vis"]
    nrow = config["nrow"]
    ncol = config["ncol"]
    num_steps = config["num_steps"]
    out_root = config["out_root"]

    model = init_edm()
    visualize_model_samples(
        model,
        solver=solver,
        num_steps=num_steps,
        title=f"Samples for {solver.get_name()} (num_steps={num_steps})",
        nrow=nrow,
        ncol=ncol,
        out_root=out_root
    )


def visualize_model_samples(model, solver, num_steps, 
                            title, nrow, ncol, out_root):
    noise = torch.randn(nrows * ncol, 3, 32, 32).cuda()
    out = solver(model, noise, num_steps)
    out = out * 0.5 + 0.5
    visualize_batch(
        out.detach().cpu(), 
        title=title,
        nrow=nrow,
        ncol=ncol,
        out_root=out_root
    )


def visualize_batch(img_vis, title, nrow, ncol, out_root):
    img_grid = make_grid(img_vis, nrow=nrow)
    fig, ax = plt.subplots(1, figsize=(nrow, ncol))
    remove_ticks(ax)
    ax.set_title(title, fontsize=14)
    ax.imshow(img_grid.permute(1, 2, 0))
    plt.show()
    plt.savefig(f"{os.path.join(out_root, title.replace(' ', '_'))}.pdf")


def remove_ticks(ax):
    ax.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False,
        left=False,
        labelleft=False
    )


def remove_xticks(ax):
    ax.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False,
        left=True,
        labelleft=True
    )


if __name__ == "__main__":
    visualize()