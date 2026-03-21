import torch
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, LinearLR, SequentialLR


def create_scheduler_pytorch(cfg, optimizer: torch.optim.Optimizer):
    """
    Creates a learning rate scheduler using native PyTorch components.
    """
    num_epochs = cfg.SOLVER.EPOCHS
    warmup_epochs = cfg.SOLVER.get("WARMUP_EPOCHS", 0)

    if cfg.SOLVER.LR_SCHEDULER == 'multisteplr':
        main_milestones = [m - warmup_epochs for m in cfg.SOLVER.LR_DROP_EPOCHS if m > warmup_epochs]
        main_scheduler = MultiStepLR(
            optimizer,
            milestones=main_milestones,
            gamma=cfg.SOLVER.LR_DROP_GAMMA
        )
    elif cfg.SOLVER.LR_SCHEDULER == 'cosineannealinglr':
        main_scheduler_epochs = num_epochs - warmup_epochs
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=main_scheduler_epochs,
            eta_min=cfg.SOLVER.get("MIN_LR", 0.)
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {cfg.SOLVER.LR_SCHEDULER}")

    if warmup_epochs == 0:
        return main_scheduler

    initial_lr = optimizer.param_groups[0]['lr']
    warmup_lr_init = cfg.SOLVER.get("WARMUP_LR", 1e-6)
    start_factor = warmup_lr_init / initial_lr
    
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=start_factor,
        total_iters=warmup_epochs
    )
    
    combined_scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs]
    )
    
    return combined_scheduler