import hydra

from omegaconf import OmegaConf

from validation import *
from utils import *


hydra.main(config_path="configs", config_name="main_config")
def main(config: OmegaConf) -> None:
    tester = Tester(config.exp_config)
    for solver_config in config.solvers:
        tester.set_solver(solver_config)
        tester.run_test()


if __name__ == "__main__":
    main()