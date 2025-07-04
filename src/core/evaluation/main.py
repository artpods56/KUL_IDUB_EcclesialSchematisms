import json
import os
from datetime import datetime
from typing import cast
import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf


import wandb

from lmv3.utils.config import config_to_dict

from shared import CONFIGS_DIR

from lmv3.utils.utils import get_device



load_dotenv()



@hydra.main(config_path=str(CONFIGS_DIR / "lmv3"), config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:

    device = get_device(cfg)
    print(f"Using device: {device}")

    run = None
    if cfg.wandb.enable:
        run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=f"{cfg.wandb.name}-{datetime.now().isoformat()}",
            tags=cfg.wandb.tags,
            config=config_to_dict(cfg),
        )


    if run is not None:
        run.finish()



if __name__ == "__main__":
    main()
