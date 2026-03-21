"""
Train and eval functions used in main.py
"""
import copy
import datetime
import itertools
import math
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from accelerate import Accelerator

import util.misc as utils
from datasets.swig_evaluator import SWiGEvaluator
from datasets.hico_eval_triplet import HICOEvaluator

import logging

logger = logging.getLogger(__name__)


def train_swig_one_epoch(
        model: torch.nn.Module, criterion: torch.nn.Module,
        data_loader: Iterable, optimizer: torch.optim.Optimizer,
        accelerator: Accelerator, epoch: int, cfg=None
    ):
    model.train()
    criterion.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)

    print_freq = 80
    
    classifier_weights = torch.load(cfg.ZERO_SHOT.CLASSIFIER.EVAL, map_location="cpu", weights_only=False)
    hoi_classifier = classifier_weights['hoi_embeddings']
    # object classifier will not used in open-vocabulary HOI detection, only hoi classifier is used.
    # obj_classifier = classifier_weights['object_embeddings'] --- IGNORE ---

    iterable_data_loader = data_loader
    if accelerator.is_main_process:
        iterable_data_loader = metric_logger.log_every(data_loader, print_freq, header)

    for samples, targets in iterable_data_loader:

        samples, targets, text_ids = prepare_swig_inputs(samples, targets, data_loader, accelerator.device)
        # text_ids is actually a ordereddict dict mapping from hoi_id to text, so a more formal way is below
        # text_ids = list(unique_hois.keys()) --- IGNORE ---
        if not text_ids:
            continue
        text_embeddings = torch.cat([hoi_classifier[text_id].unsqueeze(0) for text_id in text_ids], dim=0).to(accelerator.device)

        if accelerator.mixed_precision == "fp16" or accelerator.mixed_precision == "bf16":
            correct_dtype = next(model.parameters()).dtype
            samples.tensors = samples.tensors.to(correct_dtype)
            text_embeddings = text_embeddings.to(correct_dtype)

        with accelerator.autocast():
            outputs = model(samples, text_embeddings)
            loss_dict, indices = criterion(outputs, targets)

        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            accelerator.print("Loss is {}, stopping training".format(loss_value))
            accelerator.print(loss_dict_reduced)
            sys.exit(1)

        accelerator.backward(losses)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), cfg.SOLVER.CLIP_MAX_NORM)
        optimizer.step()
        optimizer.zero_grad()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    accelerator.print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_hico_one_epoch(
        model: torch.nn.Module, criterion: torch.nn.Module,
        data_loader: Iterable, optimizer: torch.optim.Optimizer,
        accelerator: Accelerator, epoch: int, cfg=None
    ):
    model.train()
    criterion.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    if "loss_hoi_labels" in criterion.weight_dict:
        metric_logger.add_meter("hoi_class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))
    if "loss_obj_ce" in criterion.weight_dict:
        metric_logger.add_meter("obj_class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 800

    iterable_data_loader = data_loader
    if accelerator.is_main_process:
        iterable_data_loader = metric_logger.log_every(data_loader, print_freq, header)

    for samples, targets in iterable_data_loader:
        samples = samples.to(accelerator.device)
        file_names = [{"filename": i["filename"]} for i in targets]
        targets = [{k: v.to(accelerator.device) for k, v in t.items() if k not in ["filename"]} for t in targets]

        if accelerator.mixed_precision == "fp16" or accelerator.mixed_precision == "bf16":
            correct_dtype = next(model.parameters()).dtype
            samples.tensors = samples.tensors.to(correct_dtype)
            # targets = [{k: v.to(correct_dtype) for k, v in t.items()} for t in targets]
        
        for t, f in zip(targets, file_names):
            t.update(f)

        with accelerator.autocast():
            outputs = model(samples)
            loss_dict = criterion(outputs, targets)

        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            accelerator.print("Loss is {}, stopping training".format(loss_value))
            accelerator.print(loss_dict_reduced)
            sys.exit(1)

        accelerator.backward(losses)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), cfg.SOLVER.CLIP_MAX_NORM)
        optimizer.step()
        optimizer.zero_grad()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(hoi_class_error=loss_dict_reduced['hoi_class_error'])
        metric_logger.update(obj_class_error=loss_dict_reduced['obj_class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    accelerator.print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch(model, criterion, data_loader, optimizer, accelerator, epoch, cfg=None):
    if cfg.INPUT.DATASET_FILE == "hico":
        return train_hico_one_epoch(model, criterion, data_loader, optimizer, accelerator, epoch, cfg)
    elif cfg.INPUT.DATASET_FILE == "swig":
        return train_swig_one_epoch(model, criterion, data_loader, optimizer, accelerator, epoch, cfg)
    else:
        raise ValueError(f"Unsupported dataset: {cfg.INPUT.DATASET_FILE}")


@torch.inference_mode()
def evaluate_swig(model, postprocessors, criterion, data_loader, accelerator, cfg):
    model.eval()
    criterion.eval()

    device = accelerator.device

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # Build evaluator
    output_dir = Path(cfg.RUNTIME.OUTPUT_DIR)
    swig_val_anno = cfg.INPUT.PATH + "/annotations/swig_test_1000.json"
    evaluator = SWiGEvaluator(swig_val_anno, output_dir, accelerator)

    # Convert all interaction categories into embeddings
    classifier_weights = torch.load(cfg.ZERO_SHOT.CLASSIFIER.EVAL, map_location="cpu", weights_only=False)
    hoi_classifier = classifier_weights['hoi_embeddings']
    
    text_ids = list(data_loader.dataset.text_mapper.values())
    text_embeddings = torch.cat([hoi_classifier[text_id].unsqueeze(0) for text_id in text_ids], dim=0).to(device)

    # Inference
    print_freq = 20

    iterable_data_loader = data_loader
    if accelerator.is_main_process:
        iterable_data_loader = metric_logger.log_every(data_loader, print_freq, header)

    for samples, targets in iterable_data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) if k != "hois" else v for k, v in t.items()} for t in targets]

        if accelerator.mixed_precision == "fp16" or accelerator.mixed_precision == "bf16":
            correct_dtype = next(model.parameters()).dtype
            samples.tensors = samples.tensors.to(correct_dtype)
            text_embeddings = text_embeddings.to(correct_dtype)

        with accelerator.autocast():
            outputs = model(samples, text_embeddings)

        logits_per_hoi = outputs['logits_per_hoi']
        pred_boxes = outputs['pred_boxes']
        box_scores = outputs['box_scores']

        results = {int(targets[i]['image_id']): postprocessors['hoi'](
            {'pred_logits': logits_per_hoi[i], 'pred_boxes': pred_boxes[i], 'box_scores': box_scores[i]},
            targets[i]['orig_size'],
            data_loader.dataset.text_mapper
        ) for i in range(len(samples.tensors))}

        evaluator.update(results)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # accelerator.print("Averaged stats:", metric_logger)

    evaluator.save_preds()
    # accumulate predictions from all images
    evaluator.accumulate()
    stats = evaluator.summarize()
    return stats, evaluator


def prepare_swig_inputs(samples, targets, data_loader, device):
    """Prepare model inputs."""
    # image inputs
    samples = samples.to(device)
    targets = [{k: v.to(device) if k != "hois" else v for k, v in t.items()} for t in targets]

    # text inputs
    text_inputs = []
    unique_hois = {}

    for t in targets:
        for hoi in t["hois"]:
            # Ensure all texts are unique (no duplicates).
            hoi_id = hoi["hoi_id"]
            if hoi_id not in unique_hois:
                action_text, object_text = hoi["text"]
                text = action_text + " " + object_text
                unique_hois[hoi_id] = text
                text_inputs.append(text)
    
    return samples, targets, unique_hois


@torch.inference_mode()
def evaluate_hico(model, postprocessors, data_loader, accelerator, cfg):
    model.eval()
    device = accelerator.device

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    preds, gts = [], []
    print_freq = 300
    iterable_data_loader = data_loader
    if accelerator.is_main_process:
        iterable_data_loader = metric_logger.log_every(data_loader, print_freq, header)

    for samples, targets in iterable_data_loader:
        samples = samples.to(device)

        if accelerator.mixed_precision == "fp16" or accelerator.mixed_precision == "bf16":
            correct_dtype = next(model.parameters()).dtype
            samples.tensors = samples.tensors.to(correct_dtype)

        with accelerator.autocast():
            outputs = model(samples)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['hoi'](outputs, orig_target_sizes)

        # accelerator gather supports only tensors on GPU
        # preds.extend(accelerator.gather_for_metrics(results))
        # all_gts_list.extend(accelerator.gather_for_metrics(targets))

        preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results))))
        # For avoiding a runtime error, the copy is used
        gts.extend(list(itertools.chain.from_iterable(utils.all_gather(copy.deepcopy(targets)))))

        # counter += 1

        # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    stats = {}
    if accelerator.is_main_process:
        img_ids = [img_gts['id'] for img_gts in gts]
        _, indices = np.unique(img_ids, return_index=True)
        preds = [img_preds for i, img_preds in enumerate(preds) if i in indices]
        gts = [img_gts for i, img_gts in enumerate(gts) if i in indices]

        evaluator = HICOEvaluator(
            preds, gts, data_loader.dataset.rare_triplets, 
            data_loader.dataset.non_rare_triplets, cfg=cfg
        )

        start_time = time.time()
        stats = evaluator.evaluate()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info("Total time computing mAP: {}".format(total_time_str))

    accelerator.wait_for_everyone()
    return stats
