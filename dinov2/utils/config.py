# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
from pathlib import Path

import math
import logging
import os
from omegaconf import OmegaConf

import dinov2.distributed as distributed
from dinov2.logging import setup_logging
from dinov2.utils import utils
from dinov2.configs import dinov2_default_config
from dinov2.utils.tracking import ExperimentTracker

logger = logging.getLogger("dinov2")
project_root = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent.parent


def apply_scaling_rules_to_cfg(cfg):  # to fix
    if cfg.optim.scaling_rule == "sqrt_wrt_1024":
        base_lr = cfg.optim.base_lr
        cfg.optim.lr = base_lr
        cfg.optim.lr *= math.sqrt(cfg.train.batch_size_per_gpu * distributed.get_global_size() / 1024.0)
        logger.info(f"sqrt scaling learning rate; base: {base_lr}, new: {cfg.optim.lr}")
    else:
        raise NotImplementedError
    return cfg


def write_config(cfg, output_dir, name="config.yaml"):
    logger.info(OmegaConf.to_yaml(cfg))
    saved_cfg_path = os.path.join(output_dir, name)
    with open(saved_cfg_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    return saved_cfg_path


def get_cfg_from_args(args, is_eval):
    if not is_eval:
        args.config_file = (
                project_root / "configs" / "dinov2" / f"{args.config_file}.yaml"
        )
    args.output_dir = os.path.abspath(args.output_dir)
    args.opts += [f"train.output_dir={args.output_dir}"]
    default_cfg = OmegaConf.create(dinov2_default_config)
    cfg = OmegaConf.load(args.config_file)
    cfg = OmegaConf.merge(default_cfg, cfg, OmegaConf.from_cli(args.opts))
    return cfg


def default_setup(args):
    distributed.enable(overwrite=True)
    seed = getattr(args, "seed", 0)
    rank = distributed.get_global_rank()

    global logger
    setup_logging(output=args.output_dir, level=logging.INFO)
    logger = logging.getLogger("dinov2")

    utils.fix_random_seeds(seed + rank)
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))


def setup(args, is_eval=False):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg_from_args(args, is_eval=is_eval)
    tracker = None
    if not is_eval:
        if args.input != '':
            cfg.train.dataset_path = str(project_root / args.input)
        else:
            cfg.train.dataset_path = str(project_root / cfg.train.dataset_path)
        tracker = ExperimentTracker(str(project_root), OmegaConf.to_container(cfg, resolve=True))
        cfg.train.run_id = tracker.run_id
        cfg.train.output_dir = str(project_root / tracker.run_dir)
        args.output_dir = cfg.train.output_dir
    os.makedirs(cfg.train.output_dir, exist_ok=True)
    default_setup(args)
    apply_scaling_rules_to_cfg(cfg)
    write_config(cfg, args.output_dir)
    return cfg, tracker
