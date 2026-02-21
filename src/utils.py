"""
Utility functions for ProVLA training and visualization.
"""

import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to config.yaml
        
    Returns:
        Dictionary with configuration
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_directories(config: Dict[str, Any]) -> None:
    """
    Create necessary directories from config.
    
    Args:
        config: Configuration dictionary
    """
    dirs_to_create = [
        config["data"]["data_dir"],
        config["checkpointing"]["checkpoint_dir"],
    ]
    
    if config["validation"].get("save_plots"):
        dirs_to_create.append(config["validation"]["plot_save_dir"])
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)


def set_seed(seed: int = 42) -> torch.Generator:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
        
    Returns:
        PyTorch Generator with seed
    """
    import random
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["HF_SEED"] = str(seed)
    
    return torch.Generator().manual_seed(seed)


def get_device(use_cuda: bool = True) -> str:
    """
    Get device string (cuda or cpu).
    
    Args:
        use_cuda: Whether to use CUDA if available
        
    Returns:
        Device string
    """
    if use_cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def visualize_batch(batch: Dict[str, torch.Tensor], dataset_stats) -> None:
    """
    Visualize a batch: trajectory plots and image.
    
    Args:
        batch: Batch dictionary from dataloader
        dataset_stats: Dataset with action statistics (mean, std, tokenizer)
    """
    # Un-normalize actions from sample 2 in batch
    pred_chunk = batch["actions"][2]
    mean = dataset_stats.action_mean
    std = dataset_stats.action_std
    trajectory_radians = (pred_chunk * std) + mean

    # Plot arm joints
    plt.figure(figsize=(10, 6))

    # Left arm (first 6 dimensions)
    plt.subplot(2, 1, 1)
    plt.plot(trajectory_radians[:, :6].numpy())
    plt.title("Left Arm Trajectory (Next 16 Steps)")
    plt.ylabel("Joint Angle (Rad)")
    plt.grid(True, alpha=0.3)

    # Right arm (dimensions 7-13)
    plt.subplot(2, 1, 2)
    plt.plot(trajectory_radians[:, 7:13].numpy())
    plt.title("Right Arm Trajectory (Next 16 Steps)")
    plt.xlabel("Time (Chunks)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Visualize image
    img_tensor = batch["pixel_values"][2]
    img_vis = (img_tensor * 0.5) + 0.5  # Undo SigLIP normalization
    img_vis = img_vis.clamp(0, 1).permute(1, 2, 0).numpy()

    decoded_instruction = dataset_stats.tokenizer.decode(
        batch["input_ids"][2], skip_special_tokens=True
    )

    plt.figure(figsize=(8, 6))
    plt.imshow(img_vis)
    plt.title(f"Robot Vision (Instruction: {decoded_instruction})")
    plt.axis("off")
    plt.show()


def plot_trajectories(
    gt_actions: np.ndarray,
    pred_actions: np.ndarray,
    epoch: int,
    save_dir: str = "./plots",
    joint_idx: int = 0,
) -> str:
    """
    Plot ground truth vs predicted trajectories.
    
    Args:
        gt_actions: Ground truth actions [B, horizon, action_dim]
        pred_actions: Predicted actions [B, horizon, action_dim]
        epoch: Epoch number (for filename)
        save_dir: Directory to save plots
        joint_idx: Which joint to visualize
        
    Returns:
        Path to saved plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    B = gt_actions.shape[0]
    fig, axes = plt.subplots(1, B, figsize=(4 * B, 4))
    if B == 1:
        axes = [axes]

    for i in range(B):
        axes[i].plot(
            gt_actions[i, :, joint_idx],
            "b-",
            label="Ground Truth",
            linewidth=2,
            alpha=0.8,
        )
        axes[i].plot(
            pred_actions[i, :, joint_idx],
            "r--",
            label="Predicted",
            linewidth=2,
        )
        axes[i].set_title(f"Sample {i} | Joint {joint_idx}")
        axes[i].set_xlabel("Time Step (0-15)")
        axes[i].grid(True, alpha=0.3)
        if i == 0:
            axes[i].legend()

    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"trajectories_ep{epoch}.png")
    plt.savefig(plot_path)
    plt.close(fig)
    
    return plot_path


def compute_trajectory_metrics(
    gt_actions: np.ndarray, pred_actions: np.ndarray
) -> Dict[str, float]:
    """
    Compute trajectory quality metrics.
    
    Args:
        gt_actions: Ground truth actions [B, horizon, action_dim]
        pred_actions: Predicted actions [B, horizon, action_dim]
        
    Returns:
        Dictionary with metrics
    """
    # Endpoint error (arms only - indices 0-5, 7-12)
    arm_indices = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
    last_step_gt_arms = gt_actions[:, -1, arm_indices]
    last_step_pred_arms = pred_actions[:, -1, arm_indices]
    epe_arms = np.linalg.norm(
        last_step_gt_arms - last_step_pred_arms, axis=1
    ).mean()

    # Gripper endpoint error (indices 6, 13)
    gripper_indices = [6, 13]
    gripper_gt = gt_actions[:, -1, gripper_indices]
    gripper_pred = pred_actions[:, -1, gripper_indices]
    epe_gripper = np.abs(gripper_gt - gripper_pred).mean()

    # First step error
    first_step_gt = gt_actions[:, 0, :]
    first_step_pred = pred_actions[:, 0, :]
    fse = np.mean(np.linalg.norm(first_step_gt - first_step_pred, axis=-1))

    # Trajectory MSE
    traj_mse = np.mean((gt_actions - pred_actions) ** 2)

    # Smoothness (second-order derivative)
    diffs = np.diff(pred_actions, n=2, axis=1)
    smoothness = np.mean(np.abs(diffs))

    return {
        "endpoint_error_arms": float(epe_arms),
        "endpoint_error_gripper": float(epe_gripper),
        "first_step_error": float(fse),
        "trajectory_mse": float(traj_mse),
        "smoothness": float(smoothness),
    }


def save_checkpoint(
    model,
    optimizer,
    scaler,
    epoch: int,
    loss: float,
    val_loss: float,
    patience: int,
    checkpoint_path: str,
) -> None:
    """
    Save training checkpoint.
    
    Args:
        model: ProVLA model
        optimizer: Training optimizer
        scaler: GradScaler for mixed precision
        epoch: Current epoch
        loss: Training loss
        val_loss: Validation loss
        patience: Early stopping patience counter
        checkpoint_path: Path to save checkpoint
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": {
            k: v for k, v in model.named_parameters() if v.requires_grad
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "loss": loss,
        "best_val_loss": val_loss,
        "patience": patience,
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    checkpoint_path: str, device: str
) -> Dict[str, Any]:
    """
    Load training checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
        device: Device to load onto
        
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint
