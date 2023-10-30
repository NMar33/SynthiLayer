import os
import logging
import logging.config
import yaml

from typing import List

def setup_logging(logging_config_path: str) -> None:
    if logging_config_path != "default":        
        with open(logging_config_path, "r") as f:
            log_config = yaml.safe_load(f)
            logging.config.dictConfig(log_config)
    else:
        logger = logging.getLogger("data_gen")
        handler = logging.StreamHandler()
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

def check_make_dirs(dirs_paths: List[str]) -> None:
    for dir_path in dirs_paths:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    return None