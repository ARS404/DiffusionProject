import hydra

from omegaconf import OmegaConf, DictConfig

from validation import *
from utils import *


@hydra.main(version_base=None, config_path="./configs", config_name="main_config")
def main(config: DictConfig) -> None:
    tester = Tester(config.exp_config)
    # for solver_config in config.solvers:
    print(config.solvers)
    tester.set_solver(config.solvers)
    fid = tester.run_test()
    print(f"Solver: {tester.solver.get_name()} \t FID: {fid}")


if __name__ == "__main__":
    main()
