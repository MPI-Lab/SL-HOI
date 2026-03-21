from pathlib import Path


def build_dataset(image_set, cfg):
    if cfg.INPUT.DATASET_FILE == 'swig':
        from .swig import build as build_swig
        return build_swig(image_set, cfg)
    if cfg.INPUT.DATASET_FILE == 'hico':
        from .hico import build as build_hico
        return build_hico(image_set, cfg)
    raise ValueError(f'dataset {cfg.INPUT.DATASET_FILE} not supported')
