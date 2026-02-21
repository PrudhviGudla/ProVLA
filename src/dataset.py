"""
Dataset module for ProVLA training.
Handles data loading, preprocessing, and action chunking.
"""

import torch
import random
from torch.utils.data import Dataset
from transformers import AutoImageProcessor, AutoTokenizer


class AlohaVLADataset(Dataset):
    """
    PyTorch Dataset for ALOHA robot vision-language-action data.
    
    Handles:
    - Vision preprocessing via SigLIP processor
    - Text tokenization via LLM tokenizer
    - Action normalization using dataset statistics
    - Action chunking for multi-step prediction
    """

    def __init__(
        self,
        lerobot_dataset,
        vision_model_id: str,
        text_model_id: str,
        horizon: int = 16,
        cache_dir: str = None,
    ):
        """
        Initialize ALOHA VLA Dataset.
        
        Args:
            lerobot_dataset: LeRobotDataset instance
            vision_model_id: HuggingFace model ID for vision processor
            text_model_id: HuggingFace model ID for text tokenizer
            horizon: Number of future action steps to predict
            cache_dir: Directory to cache model weights
        """
        if cache_dir:
            print(f"Loading Tokenizer and Processors from: {cache_dir}")

        self.dataset = lerobot_dataset
        self.horizon = horizon
        self.image_processor = AutoImageProcessor.from_pretrained(
            vision_model_id, cache_dir=cache_dir
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            text_model_id, cache_dir=cache_dir
        )

        # Load action statistics for normalization
        stats = self.dataset.meta.stats["action"]
        self.action_mean = torch.tensor(stats["mean"], dtype=torch.float32)
        self.action_std = torch.tensor(stats["std"], dtype=torch.float32)
        self.action_std[self.action_std < 1e-5] = 1.0

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        """
        Get item at index.
        
        Args:
            idx: Dataset index
            
        Returns:
            Dictionary with keys: pixel_values, input_ids, state, actions, timestamp
        """
        # Fetch primary frame
        item = self.dataset[idx]

        # Get action chunk (multiple future steps)
        actions_chunk = self._get_action_chunk(idx)

        # Normalize actions using dataset statistics
        actions_norm = (actions_chunk - self.action_mean) / self.action_std

        # Process image
        image = item["observation.images.cam_high"]
        if image.dtype == torch.float32 or image.dtype == torch.float16:
            image = (image * 255).type(torch.uint8)

        vision_outputs = self.image_processor(images=image, return_tensors="pt")
        pixel_values = vision_outputs.pixel_values.squeeze(0)

        # Process text instruction
        text_outputs = self.tokenizer(
            item["task"],
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": text_outputs.input_ids.squeeze(0),
            "state": item["observation.state"],
            "actions": actions_norm,
            "timestamp": item["timestamp"],
        }

    def _get_action_chunk(self, start_idx: int) -> torch.Tensor:
        """
        Get chunk of future actions starting from start_idx.
        Respects episode boundaries and pads with last action if needed.
        
        Args:
            start_idx: Starting index
            
        Returns:
            Tensor of shape [horizon, action_dim]
        """
        # Access raw HuggingFace dataset for O(1) column access
        hf_ds = self.dataset.hf_dataset.select_columns(
            ["action", "episode_index"]
        )

        # Get current episode ID for boundary checking
        start_ep = hf_ds[int(start_idx)]["episode_index"]

        valid_actions = []

        # Collect 'horizon' steps
        for i in range(self.horizon):
            target_idx = start_idx + i

            # Boundary check: dataset end
            if target_idx >= len(self.dataset):
                break

            row = hf_ds[int(target_idx)]

            # Boundary check: episode end
            if row["episode_index"] != start_ep:
                break

            valid_actions.append(torch.tensor(row["action"]))

        # Handle empty case
        if len(valid_actions) == 0:
            valid_actions = [torch.tensor(hf_ds[int(start_idx)]["action"])]

        # Stack actions
        chunk_tensor = torch.stack(valid_actions)

        # Pad with last action if less than horizon
        remaining = self.horizon - chunk_tensor.shape[0]
        if remaining > 0:
            padding = chunk_tensor[-1].unsqueeze(0).repeat(remaining, 1)
            chunk_tensor = torch.cat([chunk_tensor, padding], dim=0)

        return chunk_tensor
