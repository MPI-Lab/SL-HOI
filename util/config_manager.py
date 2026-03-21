# util/config_manager.py

import os
import sys
from typing import Optional, List
from omegaconf import OmegaConf, DictConfig, MissingMandatoryValue, errors

from accelerate import Accelerator
import logging


logger = logging.getLogger(__name__)


def load_config(
    accelerator: Accelerator,
    config_path: Optional[str] = None,
    default_config_path: Optional[str] = None,
    cli_config_overrides: Optional[List[str]] = None,
) -> DictConfig:
    """
    Loads and merges configuration files and command-line overrides using OmegaConf.
    This function is aware of the distributed environment via the Accelerator object.
    """
    
    is_main = True if accelerator is None else accelerator.is_main_process
    
    configs_to_merge: List[DictConfig] = []

    # Load default/base configuration
    if default_config_path and os.path.exists(default_config_path):
        if is_main:
            logger.info(f"Loading default config from: {default_config_path}")
        try:
            configs_to_merge.append(OmegaConf.load(default_config_path))
        except Exception as e:
            if is_main:
                logger.warning(f"Could not load default config file {default_config_path}: {e}")
    elif default_config_path and is_main:
        logger.warning(f"Default config file not found: {default_config_path}")

    # Load main configuration file
    if config_path:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Main configuration file not found: {config_path}")
        if is_main:
            logger.info(f"Loading main config from: {config_path}")
        try:
            configs_to_merge.append(OmegaConf.load(config_path))
        except Exception as e:
            raise RuntimeError(f"Could not load main config file {config_path}: {e}")
    elif not default_config_path and is_main:
        logger.warning("No configuration file specified via args or defaults. Starting with empty config.")

    # Apply CLI dotlist overrides
    if cli_config_overrides:
        try:
            cli_conf = OmegaConf.from_dotlist(cli_config_overrides)
            if cli_conf:
                if is_main:
                    logger.info("Applying command-line OmegaConf overrides...")
                    logger.info("--- CLI Provided OmegaConf Configuration ---")
                    for line in OmegaConf.to_yaml(cli_conf, resolve=False).strip().split('\n'):
                        logger.info(line)
                    logger.info("------------------------------------------")
                
                configs_to_merge.append(cli_conf)
        except Exception as e:
            if is_main:
                logger.warning(f"Failed to parse CLI OmegaConf overrides: {e}. Format should be 'key=value'.")

    # Merge configuration sources
    if not configs_to_merge:
        final_config: DictConfig = OmegaConf.create()
        if is_main:
            logger.info("Initialized with an empty configuration.")
    else:
        try:
            final_config = OmegaConf.merge(*configs_to_merge)
        except Exception as e:
            conf_types = ", ".join([str(type(cfg)) for cfg in configs_to_merge])
            raise RuntimeError(f"Failed to merge configuration sources ({conf_types}): {e}")

    # Resolve interpolations
    try:
        OmegaConf.resolve(final_config)
    except Exception as e:
        if is_main:
            logger.error("--- Error resolving configuration interpolations ---")
            logger.error(f"Details: {e}")
            logger.error("--- Current Unresolved Configuration ---")
            for line in OmegaConf.to_yaml(final_config, resolve=False).strip().split('\n'):
                logger.error(line)
            logger.error("------------------------------------")
        
        if isinstance(e, MissingMandatoryValue):
             raise MissingMandatoryValue(
                f"Configuration error: Missing value required for interpolation. Details: {e}\n"
                f"Please ensure all `${{...}}` variables are defined or have defaults."
             )
        else:
            raise RuntimeError(f"Error resolving configuration interpolations: {e}")

    if is_main:
        logger.info("Configuration loaded and resolved successfully.")

    return final_config