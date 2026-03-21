import os
import sys

import argparse
from omegaconf import OmegaConf
import datetime
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, set_seed

import util.misc as utils
from util.scheduler import create_scheduler_pytorch
from util.config_manager import load_config
from datasets import build_dataset
from engine import train_one_epoch, evaluate_swig, evaluate_hico
from models import build_model

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

import logging

import deepspeed
deepspeed.logger.setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


torch.autograd.set_detect_anomaly(True)


def load_model_only_weights(accelerator, model, pretrained_path: Path, logger):
    """
    Load model-only weights (no optimizer / scheduler / ZeRO partitioned state).

    Supported cases:
      1) pretrained_path is a single file (.pt / .bin / .safetensors)
      2) pretrained_path is a directory containing a merged model weight file:
         - consolidated_fp32.pt
         - model_final.pt
         - model.pt
         - pytorch_model.bin
         - any *.safetensors

    If the directory only has ZeRO shards (e.g. 'pytorch_model/' with mp_rank_* files)
    and no merged file is found, raise an error instructing the user to run zero_to_fp32.py first.
    """
    if not pretrained_path.exists():
        raise FileNotFoundError(f"PRETRAINED path does not exist: {pretrained_path}")

    weight_file = pretrained_path

    logger.info(f"Loading model-only weights from: {weight_file}")

    # Load state dict
    if weight_file.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as e:
            raise ImportError("Please install safetensors: pip install safetensors") from e
        state_dict = load_file(str(weight_file))
    else:
        ckpt = torch.load(weight_file, map_location="cpu", weights_only=True)
        if isinstance(ckpt, dict) and any(k in ckpt for k in ["model", "module"]):
            state_dict = ckpt.get("model", ckpt.get("module"))
        else:
            state_dict = ckpt

    unwrapped = accelerator.unwrap_model(model)
    missing, unexpected = unwrapped.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"[load_model_only_weights] Missing keys: {missing}")
    if unexpected:
        logger.warning(f"[load_model_only_weights] Unexpected keys: {unexpected}")

    logger.info("Model-only weights loaded successfully.")


def main(args, unknown_cli_opts):
    
    kwargs = InitProcessGroupKwargs(timeout=datetime.timedelta(minutes=30))
    accelerator = Accelerator(step_scheduler_with_optimizer=False, kwargs_handlers=[kwargs])

    log_level = logging.INFO if accelerator.is_main_process else logging.ERROR
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    logger.setLevel(log_level)
    logging.getLogger("dinov3").setLevel(logging.WARNING)
    logger.info(f"Accelerator state: {accelerator.state}")

    cfg = load_config(
        accelerator=accelerator,
        config_path=args.config,
        default_config_path=args.default_config,
        cli_config_overrides=unknown_cli_opts
    )

    cfg_output_dir = cfg.RUNTIME.get("OUTPUT_DIR")
    if cfg_output_dir is None or cfg_output_dir == '':
        if accelerator.is_main_process:
            logger.error("Configuration error: RUNTIME.OUTPUT_DIR is not set or is empty in your config file.")
        accelerator.wait_for_everyone() 
        sys.exit(1)

    logger.info("Setting up seeds")
    set_seed(cfg.SOLVER.SEED + accelerator.process_index)

    is_eval = cfg.RUNTIME.get("EVAL", False)
    if is_eval:
        pretrained_path = Path(cfg.RUNTIME.PRETRAINED)
        ckpt_name = pretrained_path.name
        system_log_name = f"eval_system_{ckpt_name}.log"
        metrics_log_name = f"eval_metrics_{ckpt_name}.jsonl"
    else:
        system_log_name = "train_system.log"
        metrics_log_name = "train_metrics.jsonl"

    output_dir = Path(cfg_output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory set to: {output_dir}")
        try:
            output_file_name = cfg.RUNTIME.get("OUTPUT_FILE", "final_config.yaml")
            OmegaConf.save(config=cfg, f=output_dir / output_file_name)
            logger.info(f"Final configuration saved to {output_dir / output_file_name}")
        except Exception as e:
            logger.warning(f"Could not save the final configuration file: {e}")

        fh = logging.FileHandler(output_dir / system_log_name)
        _LOG_FMT = "%(asctime)s - %(levelname)s - %(name)s -   %(message)s"
        _DATE_FMT = "%m/%d/%Y %H:%M:%S"
        formatter = logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.info(f"File logger configured. System log: {system_log_name}, Metrics log: {metrics_log_name}")

    accelerator.wait_for_everyone()

    resume_from_checkpoint = cfg.RUNTIME.get("RESUME", None)
    is_resume = resume_from_checkpoint is not None and os.path.isdir(resume_from_checkpoint)
    is_fresh_train = not is_eval and not is_resume

    model, criterion, postprocessors = build_model(cfg, is_fresh_train=is_fresh_train)
    logger.info("Model has been built")

    param_dicts = [{"params": [p for n, p in model.named_parameters() if p.requires_grad]}]

    optimizer = torch.optim.AdamW(param_dicts, lr=cfg.SOLVER.LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    lr_scheduler = create_scheduler_pytorch(cfg, optimizer)

    dataset_train = build_dataset(image_set="train", cfg=cfg)
    dataset_val = build_dataset(image_set='val', cfg=cfg)

    data_loader_train = DataLoader(
        dataset_train, shuffle=True, collate_fn=utils.collate_fn,
        batch_size=cfg.SOLVER.BATCH_SIZE, num_workers=cfg.RUNTIME.NUM_WORKERS
    )

    data_loader_val = DataLoader(
        dataset_val, shuffle=False, collate_fn=utils.collate_fn, 
        batch_size=cfg.SOLVER.BATCH_SIZE, num_workers=cfg.RUNTIME.NUM_WORKERS
    )

    model, optimizer, lr_scheduler, data_loader_train, data_loader_val = accelerator.prepare(
        model, optimizer, lr_scheduler, data_loader_train, data_loader_val)
    accelerator.register_for_checkpointing(lr_scheduler)

    if accelerator.is_main_process:
        n_parameters = sum(p.numel() for p in model.parameters())
        logger.info(f"Number of params: {n_parameters:,}")
        n_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Number of trainable params: {n_trainable_parameters:,}")

    start_epoch = 0
    resume_from_checkpoint = cfg.RUNTIME.get("RESUME", None)

    if resume_from_checkpoint is not None and os.path.isdir(resume_from_checkpoint):
        logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
        accelerator.load_state(resume_from_checkpoint)
        try:
            start_epoch = int(os.path.basename(resume_from_checkpoint).split("-")[-1]) + 1
        except:
            logger.warning("Could not parse epoch from checkpoint directory. Resuming from epoch 0.")
    elif cfg.RUNTIME.PRETRAINED and os.path.isfile(cfg.RUNTIME.PRETRAINED) and not cfg.RUNTIME.EVAL:
        logger.info(f"Loading pretrained model weights from: {cfg.RUNTIME.PRETRAINED}")
        checkpoint = torch.load(cfg.RUNTIME.PRETRAINED, map_location='cpu', weights_only=True)
        unwrapped_model = accelerator.unwrap_model(model)
        state_dict = checkpoint.get('model', checkpoint.get('model_state_dict', checkpoint))
        unwrapped_model.load_state_dict(state_dict, strict=False)



    if cfg.RUNTIME.EVAL:
        pretrained = cfg.RUNTIME.get("PRETRAINED")
        if not pretrained:
            raise ValueError("Eval mode requires RUNTIME.PRETRAINED to point to a model file or a directory containing a merged model file.")

        pretrained_path = Path(pretrained)
        logger.info(f"[EVAL] Using PRETRAINED={pretrained_path}")

        if pretrained_path.is_file():
        # Load only the model weights (no optimizer/scheduler state)
            load_model_only_weights(accelerator, model, pretrained_path, logger)
        elif pretrained_path.is_dir():
            accelerator.load_state(pretrained_path)
            logger.info("Checkpoint loaded successfully.")
        else:
            raise FileNotFoundError(f"PRETRAINED path does not exist: {pretrained_path}")
        
        model.eval()

        checkpoint_name = pretrained_path.name
        log_file_path = output_dir / metrics_log_name

        logger.info(f"Evaluating on validation split for {checkpoint_name}")
        with torch.no_grad():
            if cfg.INPUT.DATASET_FILE == "swig":
                test_stats, evaluator = evaluate_swig(model, postprocessors, criterion, data_loader_val, accelerator, cfg)
            elif cfg.INPUT.DATASET_FILE == "hico":
                test_stats = evaluate_hico(model, postprocessors, data_loader_val, accelerator, cfg)

        if output_dir and accelerator.is_main_process:
            eval_record = {
                "phase": "eval",
                "checkpoint": checkpoint_name,
                "metrics": test_stats
            }
            
            with log_file_path.open("a") as f:
                f.write(json.dumps(eval_record) + "\n")
            
            global_dir_str = cfg.RUNTIME.get("GLOBAL_OUTPUT_DIR", None)
            if global_dir_str:
                global_dir = Path(global_dir_str)
                global_dir.mkdir(parents=True, exist_ok=True)
                global_log_path = global_dir / "all_eval_metrics.jsonl"
                
                with global_log_path.open("a") as f:
                    f.write(json.dumps(eval_record) + "\n")
                    
            logger.info(f"[EVAL RESULT] Checkpoint: {checkpoint_name} | Stats: {json.dumps(test_stats)}")
            
            if cfg.INPUT.DATASET_FILE == "swig":
                eval_save_dir = output_dir / f"eval_{checkpoint_name}"
                eval_save_dir.mkdir(parents=True, exist_ok=True)
                evaluator.save(eval_save_dir)
        return

    if resume_from_checkpoint:
        logger.info("Resuming training, best_performance will be re-evaluated.")

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(start_epoch, cfg.SOLVER.EPOCHS):
        logger.info(f"--- Starting Epoch {epoch} ---")

        train_stats = train_one_epoch(model, criterion, data_loader_train, 
                                      optimizer, accelerator, epoch, cfg)
        lr_scheduler.step()

        save_start_epoch = cfg.SOLVER.get("SAVE_START_EPOCH", cfg.SOLVER.LR_DROP_EPOCHS[0])
        save_every_epochs = cfg.SOLVER.get("SAVE_EVERY_EPOCHS", 5)
        is_last_epoch = (epoch == cfg.SOLVER.EPOCHS - 1)

        if epoch > save_start_epoch and (epoch + 1) % save_every_epochs == 0 and not is_last_epoch:
            logger.info(f"Saving checkpoint for epoch {epoch}.")
            accelerator.wait_for_everyone()
            checkpoint_dir = output_dir / f"checkpoint-epoch-{epoch}"
            accelerator.save_state(output_dir=checkpoint_dir)
            logger.info(f"Checkpoint saved to {checkpoint_dir}")

        if is_last_epoch:
            logger.info(f"Saving final model checkpoint for epoch {epoch}.")
            accelerator.wait_for_everyone()
            final_checkpoint_dir = output_dir / f"checkpoint-epoch-{epoch}"
            accelerator.save_state(output_dir=final_checkpoint_dir)
            logger.info(f"Final epoch reached.")
            accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            log_stats = {
                'phase': 'train',
                'epoch': epoch,
                **{f'train_{k}': v for k, v in train_stats.items()}
            }

            metrics_file_path = output_dir / metrics_log_name
            with metrics_file_path.open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        accelerator.wait_for_everyone()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))
    logger.info('Training complete.')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Load configuration for the application.")
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="configs/hico.yaml", # Default main config file name
        help="Path to the main configuration file (YAML)."
    )
    parser.add_argument(
        "--default-config",
        type=str,
        default="configs/base.yaml", # Default base config file name
        help="Path to the default/base configuration file (YAML)."
    )

    args, unknown_cli_opts = parser.parse_known_args()

    main(args, unknown_cli_opts)
