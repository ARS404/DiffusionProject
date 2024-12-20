import hydra

from omegaconf import OmegaConf, DictConfig

from validation import *
from utils import *

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.resolve()))
sys.path.append(str(os.path.join(Path(__file__).parent.resolve(), "models", "emd")))


@hydra.main(version_base=None, config_path="./configs", config_name="main_config")
def main(config: DictConfig) -> None:
    tester = Tester(config.exp_config)
    tester.set_solver(config.solvers)
    fid = tester.run_test()
    print(f"Solver: {tester.solver.get_name()} \t FID: {fid}")


if __name__ == "__main__":
    main()
