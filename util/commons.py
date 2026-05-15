import os
import sys
import yaml
import random
import logging
import logging.config
import datetime
from zoneinfo import ZoneInfo
import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig

DATASET_ROOT = os.environ["DATASET_ROOT"]
VITON_HD_ROOT = os.path.join(DATASET_ROOT, "vitonhd")
DRESSCODE_ROOT = os.path.join(DATASET_ROOT, "DressCode")
SYNTH_VTON_ROOT = os.path.join(DATASET_ROOT, "synth_vton")
VTONQBENCH_ROOT = os.path.join(DATASET_ROOT, "vtonqbench")

VTON_MODEL_IDS = [
    "fitdit", "any2any", "catvton_flux", "catvton", "idm", "oot", "vitonhd", 
    "hrviton", "sdviton", "qwen_edit", "nanobanana", "openai", "catdm", "ladi"
                ]

DATASET_TYPES = ["synth_vton", "dc", "vitonhd", "vitonhd_by_dc", "vton_evals_dc", "vton_evals_vitonhd"]


DEFAULT_LOGGER_YAML = """
version: 1
disable_existing_loggers: false
formatters:
  default:
    format: "[%(asctime)s][%(levelname)s] %(message)s"
handlers:
  file:
    class: logging.FileHandler
    level: INFO
    formatter: default
    filename: default.log
    encoding: utf-8
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: default
    stream: ext://sys.stdout
loggers:
  __main__:
    level: INFO
    handlers: [file, console]
    propagate: no
root:
  level: INFO
  handlers: []
"""


def create_logger(name: str, file_name: str, template_path: str = None) -> logging.Logger:

    if template_path is not None and os.path.exists(template_path):
        with open(template_path, "r") as f:
            logger_config = yaml.safe_load(f)
    else:
        logger_config = yaml.safe_load(DEFAULT_LOGGER_YAML)

    logger_config["handlers"]["file"]["filename"] = file_name

    logging.config.dictConfig(logger_config)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    return logger

def set_random_seed(seed: int, *, deterministic: bool = True, set_cuda_env: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)         
    torch.cuda.manual_seed_all(seed)

    if set_cuda_env:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
        os.environ.setdefault("PYTHONHASHSEED", str(seed))

    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.backends.cudnn.benchmark = True


def create_config() -> DictConfig:
    
    cli_conf = OmegaConf.from_cli()

    if "config" not in cli_conf:
        raise ValueError("Missing argument: please specify config=<path/to/config.yaml>")

    base_path = cli_conf.config
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Config file not found: {base_path}")

    base_conf = OmegaConf.load(base_path)

    merged_conf = OmegaConf.merge(base_conf, cli_conf)

    OmegaConf.resolve(merged_conf)

    return merged_conf

def get_now() -> str:
    
    now = datetime.datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%dT%H:%M:%S")
    
    return now

def is_ampera_gpu_available() -> bool:
    cuda_version = torch.version.cuda
    device = torch.cuda.current_device()
    flg = cuda_version is not None and int(cuda_version.split(".")[0]) >= 11 and torch.cuda.get_device_properties(device).major >= 8
    return flg

def get_exec_cmd_as_str() -> str:
    return " ".join(["python"] + [v for v in sys.argv])
