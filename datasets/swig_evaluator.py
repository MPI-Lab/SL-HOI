import os
import pickle
import collections
import json
import numpy as np
from typing import Optional

from .swig_v1_categories import SWIG_INTERACTIONS

try:
    from accelerate import Accelerator
    _ACCELERATE_AVAILABLE = True
except ImportError:
    _ACCELERATE_AVAILABLE = False


class SWiGEvaluator(object):
    ''' Evaluator for SWIG-HOI dataset, adapted for distributed evaluation. '''
    def __init__(self, anno_file: str, output_dir: str, accelerator: Optional['Accelerator'] = None):
        if accelerator is not None and not _ACCELERATE_AVAILABLE:
            raise ImportError("`accelerate` is not installed, but an Accelerator instance was provided.")

        self.accelerator = accelerator
        self.output_dir = output_dir

        eval_hois = [x["id"] for x in SWIG_INTERACTIONS if x["evaluation"] == 1]
        size = max(eval_hois) + 1
        self.size = size
        self.eval_hois = eval_hois

        if self.is_main_process:
            self.gts = self.load_anno(anno_file)
        else:
            self.gts = None
        
        self._local_predictions = []

        self.scores = {i: [] for i in range(size)}
        self.boxes = {i: [] for i in range(size)}
        self.keys = {i: [] for i in range(size)}
        self.swig_ap  = np.zeros(size)
        self.swig_rec = np.zeros(size)
        
    @property
    def is_main_process(self) -> bool:
        """Returns True if in a single-process run or if this is the main process in a distributed run."""
        return self.accelerator is None or self.accelerator.is_main_process

    def update(self, predictions):
        # update predictions
        for img_id, preds in predictions.items():
            for pred in preds:
                # pred is [hoi_id, score, box...]
                self._local_predictions.append(
                    {
                        "image_id": img_id,
                        "hoi_id": pred[0],
                        "score": pred[1],
                        "boxes": pred[2:],
                    }
                )

    def accumulate(self):
        # gather the results from all processes
        if self.accelerator and self.accelerator.num_processes > 1:
            # Gathers the list of dicts from all processes.
            gathered_preds = self.accelerator.gather_for_metrics(self._local_predictions)
        else:
            # In a single-process run, the local predictions are all predictions.
            gathered_preds = self._local_predictions

        if self.is_main_process:
            all_predictions = []
            if gathered_preds and isinstance(gathered_preds[0], list):
                 # Handle list of lists case (newer accelerate versions)
                 for sublist in gathered_preds:
                    all_predictions.extend(sublist)
            else:
                 # Handle flattened list case (older accelerate versions)
                 all_predictions = gathered_preds

            for pred in all_predictions:
                hoi_id = pred["hoi_id"]
                if hoi_id in self.scores:
                    self.scores[hoi_id].append(pred["score"])
                    self.boxes[hoi_id].append(pred["boxes"])
                    self.keys[hoi_id].append(pred["image_id"])

            for hoi_id in self.eval_hois:
                gts_per_hoi = self.gts[hoi_id]
                ap, rec = calc_ap(self.scores[hoi_id], self.boxes[hoi_id], self.keys[hoi_id], gts_per_hoi)
                self.swig_ap[hoi_id], self.swig_rec[hoi_id] = ap, rec
    
    def summarize(self):
        # summarize the results
        if not self.is_main_process:
            return {}
        
        eval_hois = np.asarray([x["id"] for x in SWIG_INTERACTIONS if x["evaluation"] == 1])
        zero_hois = np.asarray([x["id"] for x in SWIG_INTERACTIONS if x["frequency"] == 0 and x["evaluation"] == 1])
        rare_hois = np.asarray([x["id"] for x in SWIG_INTERACTIONS if x["frequency"] == 1 and x["evaluation"] == 1])
        nonrare_hois = np.asarray([x["id"] for x in SWIG_INTERACTIONS if x["frequency"] == 2 and x["evaluation"] == 1])

        full_mAP = np.mean(self.swig_ap[eval_hois])
        zero_mAP = np.mean(self.swig_ap[zero_hois])
        rare_mAP = np.mean(self.swig_ap[rare_hois])
        nonrare_mAP = np.mean(self.swig_ap[nonrare_hois])
        
        print("mAP (full/zero/rare/nonrare): {:.2f} / {:.2f} / {:.2f} / {:.2f}".format(
            full_mAP * 100., zero_mAP * 100., rare_mAP * 100., nonrare_mAP * 100.))

        return {"full_mAP": full_mAP * 100, "zero_mAP": zero_mAP * 100, 
                "rare_mAP": rare_mAP * 100, "nonrare_mAP": nonrare_mAP * 100}

    def save_preds(self):
         if self.is_main_process:
            if not self.scores:
                 self.accumulate()
            with open(os.path.join(self.output_dir, "preds.pkl"), "wb") as f:
                pickle.dump({"scores": self.scores, "boxes": self.boxes, "keys": self.keys}, f)

    def save(self, output_dir=None):
        if self.is_main_process:
            if output_dir is None:
                output_dir = self.output_dir
            if not self.scores:
                 self.accumulate()
            with open(os.path.join(output_dir, "dets.pkl"), "wb") as f:
                pickle.dump({"gts": self.gts, "scores": self.scores, "boxes": self.boxes, "keys": self.keys}, f)

    def load_anno(self, anno_file):
        with open(anno_file, "r") as f:
            dataset_dicts = json.load(f)

        hoi_mapper = {(x["action_id"], x["object_id"]): x["id"] for x in SWIG_INTERACTIONS}

        size = max(self.eval_hois) + 1
        gts = {i: collections.defaultdict(list) for i in range(size)}
        for anno_dict in dataset_dicts:
            image_id = anno_dict["img_id"]
            box_annos = anno_dict.get("box_annotations", [])
            hoi_annos = anno_dict.get("hoi_annotations", [])
            for hoi in hoi_annos:
                person_box = box_annos[hoi["subject_id"]]["bbox"]
                object_box = box_annos[hoi["object_id"]]["bbox"]
                action_id = hoi["action_id"]
                object_id = box_annos[hoi["object_id"]]["category_id"]
                hoi_id = hoi_mapper[(action_id, object_id)]
                gts[hoi_id][image_id].append(person_box + object_box)

        for hoi_id in gts:
            for img_id in gts[hoi_id]:
                gts[hoi_id][img_id] = np.array(gts[hoi_id][img_id])

        return gts


def calc_ap(scores, boxes, keys, gt_boxes):
    if len(keys) == 0:
        return 0, 0

    if isinstance(boxes, list):
        scores, boxes, key = np.array(scores), np.array(boxes), np.array(keys)

    hit = []
    idx = np.argsort(scores)[::-1]
    npos = 0
    used = {}

    for key in gt_boxes.keys():
        npos += gt_boxes[key].shape[0]
        used[key] = set()

    for i in range(min(len(idx), 19999)):
        pair_id = idx[i]
        box = boxes[pair_id, :]
        key = keys[pair_id]
        if key in gt_boxes:
            maxi = 0.0
            k = -1
            for i in range(gt_boxes[key].shape[0]):
                tmp = calc_hit(box, gt_boxes[key][i, :])
                if maxi < tmp:
                    maxi = tmp
                    k = i
            if k in used[key] or maxi < 0.5:
                hit.append(0)
            else:
                hit.append(1)
                used[key].add(k)
        else:
            hit.append(0)
    bottom = np.array(range(len(hit))) + 1
    hit = np.cumsum(hit)
    rec = hit / npos
    prec = hit / bottom
    ap = 0.0
    for i in range(11):
        mask = rec >= (i / 10.0)
        if np.sum(mask) > 0:
            ap += np.max(prec[mask]) / 11.0

    return ap, np.max(rec)


def calc_hit(det, gtbox):
    gtbox = gtbox.astype(np.float64)
    hiou = iou(det[:4], gtbox[:4])
    oiou = iou(det[4:], gtbox[4:])
    return min(hiou, oiou)


def iou(bb1, bb2):
    x1 = bb1[2] - bb1[0]; y1 = bb1[3] - bb1[1]
    if x1 < 0: x1 = 0
    if y1 < 0: y1 = 0

    x2 = bb2[2] - bb2[0]; y2 = bb2[3] - bb2[1]
    if x2 < 0: x2 = 0
    if y2 < 0: y2 = 0

    xiou = min(bb1[2], bb2[2]) - max(bb1[0], bb2[0])
    yiou = min(bb1[3], bb2[3]) - max(bb1[1], bb2[1])
    if xiou < 0: xiou = 0
    if yiou < 0: yiou = 0

    return 0 if xiou * yiou <= 0 else xiou * yiou / (x1 * y1 + x2 * y2 - xiou * yiou)