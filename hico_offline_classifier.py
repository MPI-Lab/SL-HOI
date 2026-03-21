import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple

from models.dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l
from datasets.hico_text_label import hico_text_label, hico_obj_text_label, hico_unseen_index


''' Usage:
python hico_offline_classifier.py \
    --dinotxt_weights <path_to_dinov3_text_head_and_vision_head_weights> \
    --backbone_weights <path_to_dinov3_backbone_weights> \
    --bpe_path_or_url <path_or_url_to_bpe_vocab> \
    --save_dir params/hico

'''


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for DINOv3 classifier embeddings saving.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Save DINOv3 classifier embeddings for HOI detection (train/eval splits)."
    )
    parser.add_argument(
        "--dinotxt_weights", type=str,
        default="dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth",
        help="Path to DINOv3 text head + vision head weights."
    )
    parser.add_argument(
        "--backbone_weights", type=str,
        default="dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        help="Path to DINOv3 backbone weights."
    )
    parser.add_argument(
        "--bpe_path_or_url", type=str,
        default="bpe_simple_vocab_16e6.txt.gz",
        help="Path or URL to BPE vocabulary for DINOv3 tokenizer."
    )
    parser.add_argument(
        "--save_dir", type=str,
        default="classifier_weights",
        help="Directory to save classifier embedding files."
    )
    return parser.parse_args()

def init_classifier_with_dino(
    del_unseen: bool,
    zero_shot_type: str,
    hoi_text_label: Dict[Any, str],
    obj_text_label: list,
    unseen_index: Dict[str, list],
    model: torch.nn.Module,
    tokenizer: Any,
    device: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[Any, str], torch.Tensor]:
    """
    Initialize classifier text embeddings for HOI and object labels using DINOv3 text encoder.

    Args:
        del_unseen (bool): If True, filter out unseen HOI classes for training embeddings.
        zero_shot_type (str): Key to select unseen indices.
        hoi_text_label (Dict[Any, str]): Mapping from HOI label indices to label text.
        obj_text_label (list): List of tuples (idx, text) for object labels.
        unseen_index (Dict[str, list]): Mapping from zero-shot type to unseen indices.
        model (torch.nn.Module): DINOv3 model with text encoder.
        tokenizer (Any): DINOv3 tokenizer object.
        device (str): Device string ('cuda' or 'cpu').

    Returns:
        Tuple containing:
            - hoi_embedding_train (torch.Tensor): HOI text embeddings for training. Shape [K_train, D].
            - hoi_embedding_eval (torch.Tensor): HOI text embeddings for evaluation (full set). Shape [K_eval, D].
            - obj_text_embedding_eval (torch.Tensor): Object text embeddings for evaluation. Shape [K_obj, D].
            - hoi_text_label_train (Dict[Any, str]): Filtered HOI label dict for training.
            - obj_text_inputs (torch.Tensor): Tokenized object label texts. Shape [K_obj, L].
    """
    # Tokenize all HOI label texts (for evaluation)
    text_inputs = torch.cat([tokenizer.tokenize(hoi_text_label[id]) for id in hoi_text_label.keys()])

    # Filter HOI labels for training if del_unseen is True
    if del_unseen and unseen_index is not None:
        unseen_index_list = unseen_index.get(zero_shot_type, [])
        hoi_text_label_train = {
            k: hoi_text_label[k]
            for idx, k in enumerate(hoi_text_label.keys())
            if idx not in unseen_index_list
        }
    else:
        hoi_text_label_train = hoi_text_label.copy()

    # Tokenize HOI label texts for training
    text_inputs_train = torch.cat([tokenizer.tokenize(hoi_text_label_train[id]) for id in hoi_text_label_train.keys()])

    # Tokenize object label texts
    obj_text_inputs = torch.cat([tokenizer.tokenize(obj_text[1]) for obj_text in obj_text_label])

    # Encode text embeddings with DINOv3
    with torch.no_grad():
        # [K_eval, D]
        hoi_embedding_eval = model.encode_text(text_inputs.to(device)).float()
        # [K_train, D]
        hoi_embedding_train = model.encode_text(text_inputs_train.to(device)).float()
        # [K_obj, D]
        obj_text_embedding_eval = model.encode_text(obj_text_inputs.to(device)).float()

    # L2 normalize embeddings along the last dimension (feature dim), as required by theory
    hoi_embedding_train = F.normalize(hoi_embedding_train, p=2, dim=-1)
    hoi_embedding_eval = F.normalize(hoi_embedding_eval, p=2, dim=-1)
    obj_text_embedding_eval = F.normalize(obj_text_embedding_eval, p=2, dim=-1)

    return hoi_embedding_train, hoi_embedding_eval, obj_text_embedding_eval, hoi_text_label_train, obj_text_inputs

def save_classifier_eval(
    hoi_embedding_eval: torch.Tensor,
    obj_text_embedding_eval: torch.Tensor,
    save_path: Path
) -> None:
    """
    Save classifier eval stage embeddings (HOI + object) to a single file.

    Args:
        hoi_embedding_eval (torch.Tensor): HOI text embeddings [K_eval, D].
        obj_text_embedding_eval (torch.Tensor): Object text embeddings [K_obj, D].
        save_path (Path): Output .pt file path.
    """
    torch.save(
        {
            "hoi_embedding_eval": hoi_embedding_eval.cpu(),
            "obj_text_embedding_eval": obj_text_embedding_eval.cpu()
        },
        str(save_path)
    )

def save_classifier_train(
    hoi_embedding_train: torch.Tensor,
    save_path: Path
) -> None:
    """
    Save classifier train stage embeddings (HOI, filtered by zero-shot split) to a single file.

    Args:
        hoi_embedding_train (torch.Tensor): HOI text embeddings for training [K_train, D].
        save_path (Path): Output .pt file path.
    """
    torch.save(
        {
            "hoi_embedding_train": hoi_embedding_train.cpu()
        },
        str(save_path)
    )

def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load DINOv3 model and tokenizer
    model, tokenizer = dinov3_vitl16_dinotxt_tet1280d20h24l(
        dinotxt_weights=args.dinotxt_weights,
        backbone_weights=args.backbone_weights,
        bpe_path_or_url=args.bpe_path_or_url
    )
    model = model.to(device)
    model.eval()

    # Load HOI and object labels, unseen index
    hoi_text_label = hico_text_label
    obj_text_label = hico_obj_text_label
    unseen_index = hico_unseen_index

    # Compute eval embeddings (these do not depend on del_unseen or zero_shot_type)
    _, hoi_embedding_eval, obj_text_embedding_eval, _, _ = init_classifier_with_dino(
        del_unseen=False,
        zero_shot_type="default",
        hoi_text_label=hoi_text_label,
        obj_text_label=obj_text_label,
        unseen_index=unseen_index,
        model=model,
        tokenizer=tokenizer,
        device=device
    )

    # Save eval embeddings (HOI + object) in a single file
    eval_path = save_dir / "classifier_eval.pt"
    save_classifier_eval(hoi_embedding_eval, obj_text_embedding_eval, eval_path)
    print(f"Saved eval classifier embeddings to {eval_path}")

    # Define zero-shot splits; you can add or modify splits as needed
    zero_shot_splits = [
        ("default", False),
        ("rare_first", True),
        ("non_rare_first", True),
        ("unseen_object", True),
        ("unseen_verb", True),
    ]

    # For each split, save its classifier train embedding
    for zero_shot_type, del_unseen in zero_shot_splits:
        print(f"Processing train split: zero_shot_type={zero_shot_type}, del_unseen={del_unseen}")
        hoi_embedding_train, _, _, hoi_text_label_train, _ = init_classifier_with_dino(
            del_unseen=del_unseen,
            zero_shot_type=zero_shot_type,
            hoi_text_label=hoi_text_label,
            obj_text_label=obj_text_label,
            unseen_index=unseen_index,
            model=model,
            tokenizer=tokenizer,
            device=device
        )
        print(f"  hoi_embedding_train shape: {hoi_embedding_train.shape}")
        print(f"  hoi_text_label_train count: {len(hoi_text_label_train)}")

        train_path = save_dir / f"classifier_{zero_shot_type}.pt"
        save_classifier_train(hoi_embedding_train, train_path)
        print(f"Saved train classifier embeddings to {train_path}")

if __name__ == "__main__":
    main()