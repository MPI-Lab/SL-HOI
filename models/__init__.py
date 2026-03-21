import logging

logger = logging.getLogger(__name__)

def build_model(cfg, is_fresh_train: bool = True):
    logger.info("Building model: %s", cfg.MODEL.NAME)

    if cfg.INPUT.DATASET_FILE == "swig":
        from .detection.swig.slhoi import build_detector as build_hoi
    elif cfg.INPUT.DATASET_FILE == "hico":
        from .detection.hico.slhoi import build_detector as build_hoi
    else:
        raise ValueError(f"Unsupported dataset file: {cfg.INPUT.DATASET_FILE}")

    return build_hoi(cfg, is_fresh_train=is_fresh_train)