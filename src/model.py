"""
ProVLA Model Architecture.
Combines Vision, Language, and Diffusion for action prediction.
"""

import os
import torch
import torch.nn as nn
from transformers import SiglipVisionModel, AutoModelForCausalLM
from diffusers import UNet1DModel, DDPMScheduler, DDIMScheduler


class PerceiverFusion(nn.Module):
    """
    Perceiver-based fusion module for compressing Vision+Text features.
    Uses learnable latent parameters and cross-attention to compress
    high-dimensional visual and textual features into a compact representation.
    """

    def __init__(self, vision_dim: int, text_dim: int, hidden_dim: int = 768, num_latents: int = 32):
        """
        Initialize Perceiver fusion module.
        
        Args:
            vision_dim: Dimension of vision encoder output
            text_dim: Dimension of text encoder output
            hidden_dim: Hidden dimension for projection and fusion
            num_latents: Number of learnable latent vectors
        """
        super().__init__()
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Learnable latent parameters
        self.latents = nn.Parameter(torch.randn(1, num_latents, hidden_dim) * 0.02)

        # Transformer decoder for cross-attention fusion
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)

    def forward(self, vision_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse vision and text features using cross-attention.
        
        Args:
            vision_features: [B, num_vision_tokens, vision_dim]
            text_features: [B, num_text_tokens, text_dim]
            
        Returns:
            Fused embedding [B, hidden_dim]
        """
        B = vision_features.shape[0]
        v_emb = self.vision_proj(vision_features)
        t_emb = self.text_proj(text_features)
        context = torch.cat([v_emb, t_emb], dim=1)
        
        latents = self.latents.repeat(B, 1, 1)  # [B, num_latents, hidden_dim]
        fused = self.decoder(tgt=latents, memory=context)
        
        return fused.mean(dim=1)  # Global pooling [B, hidden_dim]


class ProVLA(nn.Module):
    """
    Professional Vision-Language-Action model.
    
    Architecture:
    - Vision encoder (SigLIP): processes images
    - Text encoder (TinyLlama): processes instructions
    - Perceiver fusion: combines vision+text features
    - Diffusion head: predicts action trajectories
    
    Training uses DDPM scheduler, inference uses DDIM for speed.
    """

    def __init__(
        self,
        vision_model_id: str = "google/siglip-so400m-patch14-384",
        text_model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        cache_dir: str = None,
        action_dim: int = 14,
        unet_dim: int = 256,
        num_train_timesteps: int = 100,
        inference_steps: int = 50,
        beta_schedule: str = "squaredcos_cap_v2",
    ):
        """
        Initialize ProVLA model.
        
        Args:
            vision_model_id: HuggingFace model ID for vision encoder
            text_model_id: HuggingFace model ID for text encoder
            cache_dir: Directory to cache model weights
            action_dim: Dimension of action space
            unet_dim: Hidden dimension for U-Net and fusion
            num_train_timesteps: Number of diffusion timesteps
            inference_steps: Number of DDIM steps for inference
            beta_schedule: Beta schedule for diffusion
        """
        super().__init__()

        if cache_dir and os.path.exists(cache_dir):
            print(f"Loading Weights from: {cache_dir}")

        # Vision and text encoders (frozen by default, LoRA applied later)
        self.vision_encoder = SiglipVisionModel.from_pretrained(
            vision_model_id, cache_dir=cache_dir
        )
        self.text_encoder = AutoModelForCausalLM.from_pretrained(
            text_model_id, cache_dir=cache_dir
        )

        vision_dim = self.vision_encoder.config.hidden_size
        text_dim = self.text_encoder.config.hidden_size

        # Fusion module
        self.fusion = PerceiverFusion(vision_dim, text_dim, hidden_dim=unet_dim)

        # State encoder (robot's joint positions)
        self.state_encoder = nn.Sequential(
            nn.Linear(14, 128),
            nn.Mish(),
            nn.Linear(128, unet_dim),
            nn.LayerNorm(unet_dim),
        )

        # Final projection combining fused features and state
        self.final_proj = nn.Sequential(
            nn.Linear(unet_dim + unet_dim, unet_dim),
            nn.Mish(),
            nn.Linear(unet_dim, unet_dim),
        )

        # Diffusion U-Net for action prediction
        self.unet = UNet1DModel(
            in_channels=action_dim + unet_dim,
            out_channels=action_dim,
            down_block_types=("DownBlock1D",),
            up_block_types=("UpBlock1D",),
            block_out_channels=(512,),
        )

        # Training scheduler (DDPM)
        self.train_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            clip_sample=True,
            prediction_type="epsilon",
        )

        # Inference scheduler (DDIM - faster)
        self.inference_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            clip_sample=True,
            prediction_type="epsilon",
        )
        self.inference_steps = inference_steps

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        state: torch.Tensor,
        noisy_actions_input: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for training (noise prediction).
        
        Args:
            pixel_values: [B, 3, H, W] - Camera images
            input_ids: [B, seq_len] - Tokenized instructions
            state: [B, 14] - Current robot joint states
            noisy_actions_input: [B, action_dim, horizon] - Noisy action trajectories
            timesteps: [B] - Diffusion timestep for each sample
            
        Returns:
            Noise predictions [B, action_dim, horizon]
        """
        # Encode vision and text
        vision_feats = self.vision_encoder(pixel_values).last_hidden_state
        text_feats = self.text_encoder(
            input_ids, output_hidden_states=True
        ).hidden_states[-1]

        # Fuse features
        fused_emb = self.fusion(vision_feats, text_feats)
        state_emb = self.state_encoder(state)

        combined = torch.cat([fused_emb, state_emb], dim=1)
        global_cond = self.final_proj(combined)

        # Condition diffusion model
        horizon = noisy_actions_input.shape[2]
        global_cond_repeated = global_cond.unsqueeze(-1).expand(-1, -1, horizon)
        unet_input = torch.cat([noisy_actions_input, global_cond_repeated], dim=1)

        # Predict noise
        return self.unet(unet_input, timesteps).sample

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        state: torch.Tensor,
        horizon: int = 16,
    ) -> torch.Tensor:
        """
        Generate clean action trajectory using DDIM denoising.
        
        Args:
            pixel_values: [B, 3, H, W] - Camera images
            input_ids: [B, seq_len] - Tokenized instructions
            state: [B, 14] - Current robot joint states
            horizon: Number of future action steps to generate
            
        Returns:
            Clean action trajectory [B, horizon, action_dim]
        """
        B = pixel_values.shape[0]
        device = pixel_values.device

        # Encode vision and text
        vision_feats = self.vision_encoder(pixel_values).last_hidden_state
        text_feats = self.text_encoder(
            input_ids, output_hidden_states=True
        ).hidden_states[-1]

        # Fuse features
        fused_emb = self.fusion(vision_feats, text_feats)
        state_emb = self.state_encoder(state)

        combined = torch.cat([fused_emb, state_emb], dim=1)
        global_cond = self.final_proj(combined)

        global_cond_repeated = global_cond.unsqueeze(-1).expand(-1, -1, horizon)

        # Start with pure Gaussian noise [B, 14, 16]
        latents = torch.randn(B, 14, horizon, device=device)

        self.inference_scheduler.set_timesteps(self.inference_steps)

        # Denoising loop
        for t in self.inference_scheduler.timesteps:
            unet_input = torch.cat([latents, global_cond_repeated], dim=1)

            # Predict noise (t must be [B])
            t_batch = t.repeat(B).to(device)
            noise_pred = self.unet(unet_input, t_batch).sample

            # Remove noise
            latents = self.inference_scheduler.step(
                noise_pred, t, latents
            ).prev_sample

        # Return [B, horizon, 14]
        return latents.transpose(1, 2)
