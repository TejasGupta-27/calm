"""
Example: Creating a Custom Dataset for CALM

This example shows how to create a custom video-text dataset
compatible with CALM.
"""

import sys
from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader
import clip

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from calm.data.video_text_dataset import VideoTextDataset, collate_fn


class CustomVideoTextDataset(VideoTextDataset):
    """
    Example custom video-text dataset.

    Expected annotation file format (JSON):
    {
        "videos": [
            {
                "video_id": "video_001",
                "video_file": "video_001.mp4",
                "captions": [
                    "A person is walking",
                    "Someone walks down the street"
                ]
            },
            ...
        ]
    }
    """

    def load_annotations(self, annotation_file):
        """Load custom annotations from JSON file."""
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        return data

    def build_index(self):
        """Build dataset index from annotations."""
        samples = []

        videos = self.annotations.get('videos', [])

        for video_data in videos:
            video_id = video_data['video_id']
            video_file = video_data['video_file']
            captions = video_data.get('captions', [])

            if self.split == 'train':
                # Use all captions during training
                for caption in captions:
                    samples.append({
                        'video_id': video_id,
                        'video_path': video_file,
                        'text': caption
                    })
            else:
                # Use first caption for evaluation
                if captions:
                    samples.append({
                        'video_id': video_id,
                        'video_path': video_file,
                        'text': captions[0],
                        'all_captions': captions
                    })

        return samples


def create_sample_annotation():
    """Create a sample annotation file for demonstration."""
    sample_data = {
        "videos": [
            {
                "video_id": "video_001",
                "video_file": "video_001.mp4",
                "captions": [
                    "A person is walking down the street",
                    "Someone walks on the sidewalk"
                ]
            },
            {
                "video_id": "video_002",
                "video_file": "video_002.mp4",
                "captions": [
                    "A cat is playing with a toy",
                    "A kitten plays with a ball"
                ]
            },
            {
                "video_id": "video_003",
                "video_file": "video_003.mp4",
                "captions": [
                    "A chef is cooking in the kitchen",
                    "Someone prepares food"
                ]
            }
        ]
    }

    # Save sample annotation
    annotation_path = Path('./sample_annotations.json')
    with open(annotation_path, 'w') as f:
        json.dump(sample_data, f, indent=4)

    print(f"Sample annotation file created: {annotation_path}")
    return str(annotation_path)


def main():
    """Main example function."""
    print("="*60)
    print("Custom Dataset Example for CALM")
    print("="*60)

    # Create sample annotation file
    print("\n1. Creating sample annotation file...")
    annotation_file = create_sample_annotation()

    # Create dataset
    print("\n2. Creating custom dataset...")

    # Note: This will fail if videos don't exist, but shows the structure
    try:
        dataset = CustomVideoTextDataset(
            data_root='./videos',  # Directory containing your videos
            annotation_file=annotation_file,
            num_frames=12,
            frame_size=(224, 224),
            split='train'
        )

        print(f"   ✓ Dataset created")
        print(f"   → Number of samples: {len(dataset)}")

        # Show sample structure
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\n3. Sample structure:")
            print(f"   - video_id: {sample['video_id']}")
            print(f"   - text: {sample['text']}")
            print(f"   - video_frames shape: {sample['video_frames'].shape}")

    except FileNotFoundError as e:
        print(f"   ⚠ Videos not found (expected for demo)")
        print(f"   → To use this dataset, place videos in './videos/' directory")

    # Show how to create dataloader
    print("\n4. Creating DataLoader:")
    print("""
    from torch.utils.data import DataLoader
    from calm.data.video_text_dataset import collate_fn

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Iterate over batches
    for batch in dataloader:
        video_frames = batch['video_frames']  # [B, T, C, H, W]
        texts = batch['texts']                # List of strings
        video_ids = batch['video_ids']        # List of video IDs

        # Process batch...
    """)

    # Example annotation formats for different use cases
    print("\n5. Annotation Format Examples:")
    print("-" * 60)

    print("\nSimple format (single caption per video):")
    simple_format = {
        "videos": [
            {
                "video_id": "vid_001",
                "video_file": "vid_001.mp4",
                "caption": "A person walking"
            }
        ]
    }
    print(json.dumps(simple_format, indent=2))

    print("\nMultiple captions format:")
    multi_format = {
        "videos": [
            {
                "video_id": "vid_001",
                "video_file": "vid_001.mp4",
                "captions": ["Caption 1", "Caption 2", "Caption 3"]
            }
        ]
    }
    print(json.dumps(multi_format, indent=2))

    print("\nWith metadata:")
    metadata_format = {
        "videos": [
            {
                "video_id": "vid_001",
                "video_file": "vid_001.mp4",
                "captions": ["A person walking"],
                "duration": 10.5,
                "fps": 30,
                "category": "outdoor"
            }
        ]
    }
    print(json.dumps(metadata_format, indent=2))

    print("\n" + "="*60)
    print("Example completed!")
    print("="*60)

    print("\nTo use your own dataset:")
    print("1. Prepare videos in a directory")
    print("2. Create annotation file in JSON format")
    print("3. Extend VideoTextDataset class if needed")
    print("4. Use with CALM training script")

    print("\nTraining command example:")
    print("""
    python scripts/train.py \\
        --dataset custom \\
        --data_root ./your_videos/ \\
        --annotation_train ./your_annotations.json \\
        --batch_size 32 \\
        --num_epochs 5
    """)


if __name__ == '__main__':
    main()
