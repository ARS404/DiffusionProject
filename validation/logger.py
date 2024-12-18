from typing import Dict, Iterable


class Logger(object):
    def __init__(self, configs):
        self.configs = configs
        self.wandb_dict = {}

    def log_message(self, *messages: Iterable[str]) -> None:
        pass

    def add_to_wandb_log(self, wandb_dict: Dict) -> None:
        self.wandb_dict.update(wandb_dict)

    def flush_wandb_log(self) -> None:
        pass
