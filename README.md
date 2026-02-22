# ProVLA: Vision-Language-Action Model for Robot Control

A modular PyTorch implementation of a Vision-Language-Action (VLA) model for robotic manipulation. The model combines vision, language instructions, and robot state to predict multi-step action trajectories using diffusion-based action generation.

## Project Overview

This project implements a trainable VLA system that learns to map from observations (images + task descriptions + joint states) to action sequences. The architecture fuses multi-modal inputs through a cross-attention module and predicts actions using a conditioned diffusion model (DDPM for training, DDIM for inference).

## Key Features

### Model Architecture
- **Vision Encoder**: SigLIP for efficient image understanding
- **Text Encoder**: Lightweight LLM (Qwen 2.5 0.5B / TinyLlama) for instruction encoding
- **Fusion Module**: Cross-attention with learnable latent vectors for efficient multi-modal fusion
- **Action Predictor**: 1D U-Net with DDPM/DDIM diffusion schedulers for action trajectory generation
- **State Encoder**: MLP for encoding robot joint positions as conditioning

### Training Pipeline
- **LoRA Fine-tuning**: Parameter-efficient adaptation of vision and text encoders 
- **Optional Quantization**: 4-bit or 8-bit model quantization with BitsAndBytes 
- **Mixed Precision Training**: PyTorch AMP (Automatic Mixed Precision) with GradScaler
- **Gradient Accumulation**: Simulate larger batch sizes without increased GPU memory
- **Learning Rate Scheduling**: Cosine schedule with warmup for stable convergence
- **Early Stopping**: Validation-based stopping with patience counter

### Data & Reproducibility
- **Episode-Aware Splitting**: Train/val split respects episode boundaries (no temporal leakage)
- **Deterministic Training**: Seed control across numpy, torch, and random

### Experiment Tracking
- **Weights & Biases Integration**: Full logging of metrics, trajectories, and model checkpoints
- **Checkpoint Management**: Local file + WandB artifact backup for resume across runtimes
- **Trajectory Visualization**: Plots of predicted vs ground-truth action sequences
- **Metrics Tracking**: First-step error, endpoint error, trajectory MSE, smoothness

### Code Organization
- **Config-Driven**: All hyperparameters in `config.yaml` 
- **Modular Structure**: Clean separation of dataset, model, training, and utilities
- **Type Hints**: Full type annotations for maintainability
- **Docstrings**: Comprehensive documentation for all classes and functions

## Project Structure

```
ProVLA/
├── config.yaml                    # Model, training, and data configuration
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── train.py                       # Main training script 
├── analyze_dataset.py             # Dataset analysis and episode statistics
├── .env                           # Credentials (credentials, HF_TOKEN, etc)
├── .gitignore                     # Git ignore rules
├── src/
│   ├── __init__.py               # Package init
│   ├── dataset.py                # AlohaVLADataset class 
│   ├── model.py                  # ProVLA and PerceiverFusion 
│   └── utils.py                  # Helpers: config, metrics, checkpointing 
```

## Installation

### Setup Steps

```bash
# 1. Clone repository
git clone https://github.com/PrudhviGudla/ProVLA.git
cd ProVLA

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create .env and add API Keys
```

## Configuration

All settings are managed through `config.yaml`. Key sections:

### Model Configuration
```yaml
model:
  vision_model_id: "google/siglip-base-patch16-224"  # Vision encoder
  text_model_id: "Qwen/Qwen2.5-0.5B"                 # Language encoder
  action_dim: 14                                     # ALOHA action dimension
  unet_dim: 256                                      # Diffusion model hidden dim
  
  lora:
    r: 16                                            # LoRA rank
    lora_alpha: 32                                   # LoRA scaling
    target_modules: ["q_proj", "v_proj"]             # Which modules to adapt
    lora_dropout: 0.05
  
  quantization:
    enabled: false                                   # Set to true for 4/8-bit quantization
    bits: 4
```

### Training Configuration
```yaml
training:
  epochs: 10
  learning_rate: 2e-4
  batch_size: 1                    # Effective batch size with accumulation
  accum_steps: 10                  # Gradient accumulation steps
  warmup_steps: 100
  patience_limit: 5                # Early stopping patience
  validation_frequency: 3          # Validate every N epochs
```

### Data Configuration
```yaml
data:
  dataset_name: "lerobot/aloha_mobile_cabinet"
  val_split: 0.1                   # 10% validation split
  seed: 42
  num_episodes_subset: 2           # Limit to N episodes (null = all 85)
```

## Usage

### Train from Scratch

```bash
python train.py --config config.yaml
```

Optional arguments:
```bash
python train.py \
  --config config.yaml \
  --wandb-key YOUR_WANDB_API_KEY
```

### Resume Training

Set `resume: true` in `config.yaml` and the script will automatically load the latest checkpoint:

```yaml
wandb:
  resume: true
```

Then run:
```bash
python train.py --config config.yaml
```

The script will:
1. Load latest checkpoint from `checkpoints/vla_latest.pth`
2. Resume from saved epoch and LR schedule
3. Restore optimizer and gradient scaler states
4. Continue validation metrics tracking

### Analyze Dataset

Inspect episodes and train/val split:

```bash
python analyze_dataset.py --config config.yaml
```


## Training Details

### Training Loop
1. **Data Loading**: Batch images, instructions, states, and action trajectories
2. **Noise Injection**: Add random Gaussian noise to clean actions
3. **Denoising Prediction**: Forward pass predicts noise given noisy actions + conditioning
4. **Loss Computation**: MSE between predicted and actual noise
5. **Backpropagation**: Backprop with gradient accumulation over accum_steps
6. **Optimization**: AdamW + cosine LR schedule with warmup
7. **Validation**: Every N epochs, evaluate on held-out episode validation set

### Optimization Strategy
- **Optimizer**: AdamW with weight decay (prevents overfitting)
- **Learning Rate**: Cosine annealing from 2e-4 → 0 with warmup
- **Gradient Clipping**: Norm-based clipping to prevent explosion
- **Mixed Precision**: FP32 for stability-critical ops, FP16 elsewhere
- **Gradient Accumulation**: Accumulate over 10 steps to increase effective batch size

### Evaluation Metrics
- **Noise Loss**: Training objective (diffusion MSE)
- **First Step Error**: L2 distance of predicted vs actual first action [0, 1]
- **Endpoint Error**: Position difference at trajectory end
- **Trajectory MSE**: Overall trajectory prediction quality
- **Smoothness**: Second-order action derivative (lower = smoother predictions)

## Results

**Status**: Model training and evaluation are still in progress. The ALOHA dataset is large (~127K samples across 85 episodes), and full convergence testing is ongoing.


**Expected Future Updates**:
- Quantitative metrics on validation trajectories
- Comparison with baseline architectures
- Ablation studies on fusion module design
- Guidance on hyperparameter tuning 
- Multi-GPU / multi-node via PyTorch DistributedDataParallel or Accelerate
- Image augmentation (rotation, brightness), action smoothing
- History of past actions as additional conditioning


## Known Limitations

- **Single Dataset**: Currently supports LeRobot ALOHA dataset; generalization untested
- **Small Models**: Vision/text encoders are compact; larger models may improve performance
- **No Real Robot Testing**: Evaluation is offline on held-out trajectories
- **Limited Baselines**: No quantitative comparison with other VLA methods
- **Deterministic Inference**: DDIM is deterministic; stochasticity might help in uncertainty scenarios

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in config
- Increase `accum_steps` to compensate for smaller batch
- Disable mixed precision (`mixed_precision: false`)


## References

- LeRobot: https://github.com/huggingface/lerobot
- Mobile ALOHA: https://mobile-aloha.github.io/
- Diffusion Model Review: https://arxiv.org/abs/2308.16384
- LoRA: https://arxiv.org/abs/2106.09685


**For questions or issues**, please open a GitHub issue 