import os
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from pathlib import Path
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
from dotenv import load_dotenv
from huggingface_hub import login
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import wandb

from src.dataset import AlohaVLADataset
from src.model import ProVLA
from src.utils import (
    load_config,
    create_directories,
    set_seed,
    episodic_split,
    plot_trajectories,
    compute_trajectory_metrics,
    save_checkpoint,
)

def build_model(config: dict, device: str):
    """
    Build ProVLA model with LoRA adapters.
    
    Args:
        config: Configuration dictionary
        device: Device to load model onto
        
    Returns:
        Configured ProVLA model
    """
    model = ProVLA(
        vision_model_id=config["model"]["vision_model_id"],
        text_model_id=config["model"]["text_model_id"],
        cache_dir=config["model"]["cache_dir"],
        action_dim=config["model"]["action_dim"],
        unet_dim=config["model"]["unet_dim"],
        num_train_timesteps=config["model"]["diffusion"]["num_train_timesteps"],
        inference_steps=config["model"]["diffusion"]["inference_steps"],
        beta_schedule=config["model"]["diffusion"]["beta_schedule"],
        quantization_config=config["model"]["quantization"],
    )
    
    # Only move to device if not using quantization (quantization uses device_map="auto")
    if not (config["model"].get("quantization", {}).get("enabled", False)):
        model = model.to(device)
    else:
        # Prepare quantized model for LoRA 
        model.vision_encoder = prepare_model_for_kbit_training(model.vision_encoder)
        model.text_encoder = prepare_model_for_kbit_training(model.text_encoder)
        
        # Explicitly move the custom PyTorch modules to the GPU 
        model.fusion = model.fusion.to(device)
        model.state_encoder = model.state_encoder.to(device)
        model.final_proj = model.final_proj.to(device)
        model.unet = model.unet.to(device)

    # Apply LoRA to text encoder
    text_peft_config = LoraConfig(
        r=config["model"]["lora"]["r"],
        lora_alpha=config["model"]["lora"]["lora_alpha"],
        target_modules=config["model"]["lora"]["target_modules"],
        lora_dropout=config["model"]["lora"]["lora_dropout"],
        bias=config["model"]["lora"]["bias"],
    )
    model.text_encoder = get_peft_model(model.text_encoder, text_peft_config)

    # Apply LoRA to vision encoder
    vision_peft_config = LoraConfig(
        r=config["model"]["lora"]["r"],
        lora_alpha=config["model"]["lora"]["lora_alpha"],
        target_modules=config["model"]["lora"]["target_modules"],
        lora_dropout=config["model"]["lora"]["lora_dropout"],
        bias=config["model"]["lora"]["bias"],
    )
    model.vision_encoder = get_peft_model(model.vision_encoder, vision_peft_config)

    # Print trainable parameters to verify LoRA is working
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(f"Trainable params: {trainable_params} || All params: {all_param} || %: {100 * trainable_params / all_param:.2f}")
    return model


def validate_and_visualize(
    model,
    val_loader,
    fixed_samples: dict,
    epoch: int,
    config: dict,
    device: str,
):
    """
    Run validation and generate trajectory visualizations.
    
    Args:
        model: ProVLA model
        val_loader: Validation dataloader
        fixed_samples: Fixed samples for visualization
        epoch: Current epoch
        config: Configuration dictionary
        device: Device
        
    Returns:
        Validation loss
    """
    print(f"\nRunning Validation for Epoch {epoch}...")
    model.eval()

    # Validation loss on full validation set
    total_noise_loss = 0
    for batch in val_loader:
        pixels = batch["pixel_values"].to(device)
        ids = batch["input_ids"].to(device)
        state = batch["state"].to(device)
        actions = batch["actions"].to(device)

        noise = torch.randn_like(actions)
        t = torch.randint(
            0, config["model"]["diffusion"]["num_train_timesteps"],
            (actions.shape[0],),
            device=device,
        ).long()
        noisy_action = model.train_scheduler.add_noise(actions, noise, t)

        with autocast(device_type=device):
            pred = model(pixels, ids, state, noisy_action.transpose(1, 2), t)
            loss = F.mse_loss(pred.transpose(1, 2), noise)
        total_noise_loss += loss.item()

    avg_val_loss = total_noise_loss / len(val_loader)

    # Generate trajectories on fixed samples
    pixels = fixed_samples["pixel_values"].to(device)
    ids = fixed_samples["input_ids"].to(device)
    state = fixed_samples["state"].to(device)
    gt_actions = fixed_samples["actions"].numpy()

    with autocast(device_type=device):
        pred_actions = model.generate(pixels, ids, state).cpu().numpy()

    # Compute metrics
    metrics = compute_trajectory_metrics(gt_actions, pred_actions)

    # Save visualization
    plot_path = plot_trajectories(
        gt_actions,
        pred_actions,
        epoch,
        save_dir=config["validation"]["plot_save_dir"],
        joint_idx=0,
    )

    # Log metrics
    metrics_log = {
        "val/noise_loss": avg_val_loss,
        "val/endpoint_error_arms": metrics["endpoint_error_arms"],
        "val/endpoint_error_gripper": metrics["endpoint_error_gripper"],
        "val/first_step_error": metrics["first_step_error"],
        "val/trajectory_mse": metrics["trajectory_mse"],
        "val/smoothness": metrics["smoothness"],
    }
    
    if wandb.run is not None:
        metrics_log["val/trajectories"] = wandb.Image(plot_path)
        wandb.log(metrics_log)

    print(
        f"FSE: {metrics['first_step_error']:.4f} | "
        f"Traj MSE: {metrics['trajectory_mse']:.4f} | "
        f"Smoothness: {metrics['smoothness']:.4f} | "
        f"EPE Arms: {metrics['endpoint_error_arms']:.4f} | "
        f"Val Loss: {avg_val_loss:.4f}"
    )

    model.train()
    return avg_val_loss


def main():
    """Main training loop."""
    load_dotenv()
    
    # HuggingFace login for gated models
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("Logged in to HuggingFace Hub")
    
    parser = argparse.ArgumentParser(description="Train ProVLA model")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--wandb-key",
        type=str,
        default=None,
        help="WandB API key (or set WANDB_API_KEY env var or in .env file)",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    print("Configuration loaded successfully")

    # Create directories
    create_directories(config)
    print("Directories created")

    # Set seed
    set_seed(config["data"]["seed"])
    print(f"Seed set to {config['data']['seed']}")

    # Device
    use_cuda = config["device"]["use_cuda"]
    if use_cuda and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Training on: {device}")

    # Data Loading
    print("\nLoading dataset...")

    raw_dataset = LeRobotDataset(
        config["data"]["dataset_name"],
        root=config["data"]["data_dir"],
    )
    print(f"Raw dataset size: {len(raw_dataset)}")

    vla_dataset = AlohaVLADataset(
        raw_dataset,
        vision_model_id=config["model"]["vision_model_id"],
        text_model_id=config["model"]["text_model_id"],
        horizon=config["dataset_processing"]["horizon"],
        cache_dir=config["model"]["cache_dir"],
    )
    print(f"VLA dataset size: {len(vla_dataset)}")

    # Episodic train/val split (respects episode boundaries)
    train_indices, val_indices = episodic_split(
        raw_dataset,
        val_split=config["data"]["val_split"],
        seed=config["data"]["seed"],
        num_episodes_subset=config["data"].get("num_episodes_subset"),
    )

    train_dataset = Subset(vla_dataset, train_indices)
    val_dataset = Subset(vla_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["dataset_processing"]["batch_size"],
        shuffle=True,
        num_workers=config["dataset_processing"]["num_workers"],
        pin_memory=config["dataset_processing"]["pin_memory"],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["dataset_processing"]["batch_size"],
        shuffle=False,
        num_workers=config["dataset_processing"]["num_workers"],
        pin_memory=config["dataset_processing"]["pin_memory"],
    )

    # Model Setup
    print("\nBuilding model...")
    model = build_model(config, device)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(config["training"]["learning_rate"]),
    )
    scaler = GradScaler(enabled=config["device"]["mixed_precision"])

    # Resume from Checkpoint (if enabled)
    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0

    checkpoint_dir = config["checkpointing"]["checkpoint_dir"]
    latest_ckpt_path = os.path.join(
        checkpoint_dir, config["checkpointing"]["latest_checkpoint"]
    )
    best_val_path = os.path.join(
        checkpoint_dir, config["checkpointing"]["best_val_checkpoint"]
    )

    # Resume from Checkpoint & WandB Setup
    
    # Get API key once
    api_key = args.wandb_key or os.getenv("WANDB_API_KEY")
    if config["wandb"]["enabled"] and not api_key:
        print("Warning: WandB enabled but no API key provided")
        config["wandb"]["enabled"] = False
    
    # Initialize checkpoint variables
    checkpoint_loaded = False
    
    if config["wandb"]["resume"]:
        # Try local checkpoint first
        if os.path.exists(latest_ckpt_path):
            print(f"\nLoading local checkpoint: {latest_ckpt_path}")
            try:
                checkpoint = torch.load(latest_ckpt_path, map_location=device)
                checkpoint_loaded = True
            except Exception as e:
                print(f"Could not load local checkpoint: {e}")
        
        # If local not found, try WandB artifact
        if not checkpoint_loaded and config["wandb"]["enabled"]:
            print(f"Local checkpoint not found. Downloading from WandB...")
            try:
                wandb.login(key=api_key)
                artifact = wandb.use_artifact(
                    f"{config['wandb']['run_id']}_latest:latest"
                )
                artifact_dir = artifact.download()
                artifact_checkpoint_path = os.path.join(
                    artifact_dir, config["checkpointing"]["latest_checkpoint"]
                )
                checkpoint = torch.load(artifact_checkpoint_path, map_location=device)
                checkpoint_loaded = True
                print("Downloaded checkpoint from WandB artifact")
            except Exception as e:
                print(f"Could not download from WandB: {e}")
        
        # Apply loaded checkpoint
        if checkpoint_loaded:
            try:
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                scaler.load_state_dict(checkpoint["scaler"])
                
                start_epoch = checkpoint["epoch"] + 1
                best_val_loss = checkpoint["best_val_loss"]
                patience_counter = checkpoint["patience"]
                
                print(f"Resumed from epoch {start_epoch-1}")
                print(f"Best val loss: {best_val_loss:.4f}")
                print(f"Patience counter: {patience_counter}")
            except Exception as e:
                print(f"Error loading checkpoint data: {e}")
                print("Training from scratch...")
        else:
            print("Warning: Resume enabled but no checkpoint found (local or WandB)")
            print("Training from scratch...")

    # Learning Rate Scheduler (created after checkpoint load)
    total_steps = (len(train_loader) // config["training"]["accum_steps"]) * config["training"]["epochs"]
    remaining_steps = total_steps - (start_epoch * (len(train_loader) // config["training"]["accum_steps"]))
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["training"]["warmup_steps"],
        num_training_steps=remaining_steps,
        last_epoch=(start_epoch * (len(train_loader) // config["training"]["accum_steps"])) - 1,
    )

    # WandB Setup (initialized after checkpoint handling)
    if config["wandb"]["enabled"]:
        if not api_key:  # Double check (shouldn't happen due to earlier check)
            print("Warning: WandB API key required but not found")
            config["wandb"]["enabled"] = False
        else:
            wandb.login(key=api_key)
            
            if config["wandb"]["resume"]:
                # Resume existing run
                print(f"\nResuming WandB run: {config['wandb']['run_id']}")
                wandb.init(
                    project=config["wandb"]["project_name"],
                    id=config["wandb"]["run_id"],
                    resume="must",
                )
            else:
                # Start new run
                print(f"\nStarting new WandB run: {config['wandb']['run_id']}")
                wandb.init(
                    project=config["wandb"]["project_name"],
                    id=config["wandb"]["run_id"],
                    config=config,
                )
            
            wandb.define_metric("epochs")
            wandb.define_metric("train/*", step_metric="epochs")
            wandb.define_metric("val/*", step_metric="epochs")
            print("WandB initialized")


    # Prepare Fixed Samples for Visualization
    fixed_indices = config["validation"]["fixed_sample_indices"]
    fixed_samples_list = [val_dataset[i] for i in fixed_indices]
    fixed_samples = {
        "pixel_values": torch.stack([x["pixel_values"] for x in fixed_samples_list]),
        "input_ids": torch.stack([x["input_ids"] for x in fixed_samples_list]),
        "state": torch.stack([x["state"] for x in fixed_samples_list]),
        "actions": torch.stack([x["actions"] for x in fixed_samples_list]),
    }
    print(f"Locked in {len(fixed_indices)} fixed samples for visualization")

    # Training Loop
    print("\n" + "=" * 70)
    print("Starting Training Loop")
    print("=" * 70 + "\n")

    model.train()

    for epoch in range(start_epoch, config["training"]["epochs"]):
        if config["wandb"]["enabled"] and wandb.run is not None:
            wandb.log({"epochs": epoch})

        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        optimizer.zero_grad()

        for step, batch in enumerate(progress_bar):
            pixels = batch["pixel_values"].to(device)
            text = batch["input_ids"].to(device)
            state = batch["state"].to(device)
            clean_actions = batch["actions"].to(device)

            noise = torch.randn_like(clean_actions)
            B = clean_actions.shape[0]
            timesteps = torch.randint(
                0,
                config["model"]["diffusion"]["num_train_timesteps"],
                (B,),
                device=device,
            ).long()
            noisy_actions = model.train_scheduler.add_noise(clean_actions, noise, timesteps)

            with autocast(device_type=device, enabled=config["device"]["mixed_precision"]):
                noise_pred = model(
                    pixels,
                    text,
                    state,
                    noisy_actions.transpose(1, 2),
                    timesteps,
                )
                loss = F.mse_loss(noise_pred.transpose(1, 2), noise)
                loss = loss / config["training"]["accum_steps"]

            scaler.scale(loss).backward()

            if (step + 1) % config["training"]["accum_steps"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config["training"]["gradient_clip_norm"],
                )
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()

                current_loss = loss.item() * config["training"]["accum_steps"]
                epoch_loss += current_loss

                if config["wandb"]["enabled"] and wandb.run is not None:
                    wandb.log({
                        "train_batchaccumstep/loss": current_loss,
                        "train_batchaccumstep/lr": lr_scheduler.get_last_lr()[0],
                    })
                progress_bar.set_postfix({"loss": f"{current_loss:.4f}"})

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

        # Validation
        if (epoch % config["training"]["validation_frequency"] == 0 or
            epoch == config["training"]["epochs"] - 1):
            
            current_val_loss = validate_and_visualize(
                model=model,
                val_loader=val_loader,
                fixed_samples=fixed_samples,
                epoch=epoch,
                config=config,
                device=device,
            )

            # Save best checkpoint
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                patience_counter = 0

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    loss=avg_loss,
                    val_loss=best_val_loss,
                    patience=patience_counter,
                    checkpoint_path=best_val_path,
                )
                print("Best validation checkpoint saved!")

            else:
                patience_counter += 1
                print(
                    f"Validation loss did not improve. "
                    f"Patience: {patience_counter}/{config['training']['patience_limit']}"
                )

                if patience_counter >= config["training"]["patience_limit"]:
                    print("\n" + "!" * 70)
                    print("Early stopping triggered - model is overfitting")
                    print("!" * 70)
                    break

        # Save latest checkpoint
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            loss=avg_loss,
            val_loss=best_val_loss,
            patience=patience_counter,
            checkpoint_path=latest_ckpt_path,
        )
        
        # Log latest checkpoint as WandB artifact
        if config["wandb"]["enabled"] and wandb.run is not None:
            try:
                artifact = wandb.Artifact(
                    name=f"{wandb.run.id}_latest",
                    type="model",
                    description=f"Latest checkpoint at epoch {epoch+1}",
                )
                artifact.add_file(latest_ckpt_path)
                wandb.log_artifact(artifact)
            except Exception as e:
                print(f"Failed to upload latest checkpoint artifact: {e}")

        if config["wandb"]["enabled"] and wandb.run is not None:
            wandb.log({"train/epoch_loss": avg_loss})

    # Cleanup
    if config["wandb"]["enabled"]:
        wandb.finish()

    print("\n" + "=" * 70)
    print("Training Complete!")
    print(f"Best checkpoint saved to: {best_val_path}")
    print(f"Latest checkpoint saved to: {latest_ckpt_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
