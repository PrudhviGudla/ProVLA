import argparse
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from src.utils import load_config, episodic_split


def analyze_dataset(config_path: str, val_split: float = None, num_episodes_subset: int = None):
    """
    Show dataset stats: episodes, samples per episode, and split results.
    
    Args:
        config_path: Path to config.yaml
        val_split: Override config val_split (optional)
        num_episodes_subset: Override config num_episodes_subset (optional)
    """
    config = load_config(config_path)
    
    if val_split is None:
        val_split = config["data"]["val_split"]
    if num_episodes_subset is None:
        num_episodes_subset = config["data"].get("num_episodes_subset")
    
    print("DATASET ANALYSIS")
    
    # Load dataset
    print(f"\nLoading {config['data']['dataset_name']}...")
    raw_dataset = LeRobotDataset(
        config["data"]["dataset_name"],
        root=config["data"]["data_dir"],
    )
    print(f"Total samples: {len(raw_dataset)}")
    
    # Get episode boundaries from HuggingFace dataset
    episode_index_array = np.array(raw_dataset.hf_dataset["episode_index"])
    
    # print("Episode index array loaded from HuggingFace dataset.")
    # print(episode_index_array[:10])  # Show first 10 for sanity check

    # Find unique episodes
    unique_episodes = sorted(np.unique(episode_index_array).tolist())
    
    print(f"Total episodes: {len(unique_episodes)}")
    
    print(f"\nEpisodes and samples per episode:")
    for ep_id in unique_episodes:
        count = np.sum(episode_index_array == ep_id)
        print(f"  Episode {ep_id}: {count} samples")
    
    # Perform episodic split
    train_indices, val_indices = episodic_split(
        raw_dataset,
        val_split=val_split,
        seed=config["data"]["seed"],
        num_episodes_subset=num_episodes_subset,
    )
    print(f"\nTrain samples: {len(train_indices)}")
    print(f"Val samples: {len(val_indices)}")       


def main():
    parser = argparse.ArgumentParser(description="Analyze dataset structure and splits")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--val-split", type=float, default=None, help="Override val_split")
    parser.add_argument("--num-episodes", type=int, default=None, help="Limit to N episodes")
    args = parser.parse_args()
    
    analyze_dataset(args.config, args.val_split, args.num_episodes)


if __name__ == "__main__":
    main()

