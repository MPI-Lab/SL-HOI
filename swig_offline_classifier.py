import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, List

from tqdm import tqdm

from models.dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l
from datasets.swig_v1_categories import SWIG_INTERACTIONS, SWIG_CATEGORIES


''' Usage:
python swig_offline_classifier.py \
    --dinotxt_weights <path_to_dinov3_text_head_and_vision_head_weights> \
    --backbone_weights <path_to_dinov3_backbone_weights> \
    --bpe_path_or_url <path_or_url_to_bpe_vocab> \
    --save_dir params/swig
'''


def prepare_swig_labels(
    interactions: List[Dict[str, Any]],
    categories: List[Dict[str, Any]]
) -> Tuple[Dict[int, str], Dict[int, str]]:
    """
    Prepare HOI and object text labels from SWIG_INTERACTIONS and SWIG_CATEGORIES lists.

    Args:
        interactions (List[Dict[str, Any]]): SWiG interaction dictionaries list.
        categories (List[Dict[str, Any]]): SWiG object category dictionaries list.

    Returns:
        A tuple containing:
        - hoi_text_label (Dict[int, str]): Mapping from interaction ID to formatted text.
        - object_text_label (Dict[int, str]): Mapping from object ID to object name.
    """
    hoi_text_label = {}
    object_text_label = {}

    print("Preparing text labels from SWiG lists...")
    # 1. Prepare HOI labels
    for item in interactions:
        # The 'name' field in SWiG is already in "action object" format
        hoi_text_label[item['id']] = item['name']
    
    # 2. Prepare object labels
    for item in categories:
        object_text_label[item['id']] = item['name']

    print(f"Generated {len(hoi_text_label)} HOI labels.")
    print(f"Generated {len(object_text_label)} object labels.")
    return hoi_text_label, object_text_label


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for saving DINOv3 classifier embeddings.
    """
    parser = argparse.ArgumentParser(
        description="Save DINOv3 classifier embeddings for SWiG dataset."
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
        default="classifier_weights_swig",
        help="Directory to save the classifier embedding file."
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for text embedding generation to prevent OOM.")

    return parser.parse_args()


def generate_text_embedding_dict(
    text_labels: Dict[int, str],
    model: torch.nn.Module,
    tokenizer: Any,
    device: str,
    batch_size: int = 64  # Added batch_size parameter
) -> Dict[int, torch.Tensor]:
    """
    Generate text embeddings for given labels using DINOv3 text encoder.
    Args:
        text_labels (Dict[int, str]): Mapping from IDs to text labels.
        model (torch.nn.Module): DINOv3 model with text encoder.
        tokenizer (Any): DINOv3 tokenizer.
        device (str): Device to run the model on ('cuda' or 'cpu').
        batch_size (int): Batch size for processing text labels.
    """
    print(f"Generating embeddings with batch size {batch_size}...")
    
    sorted_ids = sorted(text_labels.keys())
    texts = [text_labels[id] for id in sorted_ids]
    
    embedding_dict = {}
    
    # Use tqdm to create a progress bar
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding text"):
        # 1. Get the current batch data
        batch_ids = sorted_ids[i : i + batch_size]
        batch_texts = texts[i : i + batch_size]
        
        # 2. Tokenize the current batch
        text_inputs = torch.cat([tokenizer.tokenize(text) for text in batch_texts])

        # 3. Use inference_mode for inference
        with torch.inference_mode():
            batch_embeddings = model.encode_text(text_inputs.to(device)).float()

        # 4. L2 normalize the embeddings
        batch_embeddings = F.normalize(batch_embeddings, p=2, dim=-1)

        # 5. Store the current batch results in the dictionary
        for j, id_val in enumerate(batch_ids):
            embedding_dict[id_val] = batch_embeddings[j]
    
    return embedding_dict

def save_embedding_dicts(
    hoi_embeddings: Dict[int, torch.Tensor],
    object_embeddings: Dict[int, torch.Tensor],
    save_path: Path
) -> None:
    hoi_cpu = {k: v.cpu() for k, v in hoi_embeddings.items()}
    object_cpu = {k: v.cpu() for k, v in object_embeddings.items()}
    torch.save(
        {"hoi_embeddings": hoi_cpu, "object_embeddings": object_cpu},
        str(save_path)
    )

def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load model and tokenizer
    print("Loading DINOv3 model and tokenizer...")
    model, tokenizer = dinov3_vitl16_dinotxt_tet1280d20h24l(
        dinotxt_weights=args.dinotxt_weights,
        backbone_weights=args.backbone_weights,
        bpe_path_or_url=args.bpe_path_or_url
    )
    model = model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # 2. Prepare labels
    hoi_text_label, object_text_label = prepare_swig_labels(SWIG_INTERACTIONS, SWIG_CATEGORIES)

    # 3. Generate embedding dictionaries, passing batch_size
    print("\nGenerating SWiG HOI classifier embeddings (as dict)...")
    hoi_embedding_dict = generate_text_embedding_dict(
        text_labels=hoi_text_label,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=args.batch_size
    )
    print(f"Generated {len(hoi_embedding_dict)} HOI embeddings.")

    print("\nGenerating SWiG object classifier embeddings (as dict)...")
    object_embedding_dict = generate_text_embedding_dict(
        text_labels=object_text_label,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=args.batch_size
    )
    print(f"Generated {len(object_embedding_dict)} object embeddings.")
    
    # 4. Save dictionaries
    save_path = save_dir / "classifier_swig_dict.pt"
    save_embedding_dicts(hoi_embedding_dict, object_embedding_dict, save_path)
    print(f"\nSaved SWiG classifier embedding dictionaries to {save_path}")


if __name__ == "__main__":
    main()