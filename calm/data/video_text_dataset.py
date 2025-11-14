"""
Video-Text Dataset Loaders for CALM

Implements data loading for benchmark datasets:
- MSR-VTT
- DiDeMo
- MSVD
- LSMDC
"""

import os
import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from PIL import Image


class VideoTextDataset(Dataset):
    """
    Base class for video-text datasets.

    Args:
        data_root (str): Root directory containing videos
        annotation_file (str): Path to annotation file
        num_frames (int): Number of frames to sample
        frame_size (tuple): Size to resize frames to
        split (str): Dataset split ('train', 'val', 'test')
        transform: Image transformations
    """

    def __init__(
        self,
        data_root,
        annotation_file,
        num_frames=12,
        frame_size=(224, 224),
        split='train',
        transform=None,
        max_samples=None
    ):
        self.data_root = Path(data_root)
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.split = split
        self.transform = transform
        self.max_samples = max_samples

        # Load annotations
        self.annotations = self.load_annotations(annotation_file)

        # Build dataset index
        self.samples = self.build_index()

    def load_annotations(self, annotation_file):
        """Load annotations from file."""
        raise NotImplementedError

    def build_index(self):
        """Build dataset index."""
        raise NotImplementedError

    def load_video(self, video_path):
        """
        Load and sample frames from video.

        Args:
            video_path (str): Path to video file

        Returns:
            torch.Tensor: Video frames [num_frames, C, H, W]
        """
        cap = cv2.VideoCapture(str(video_path))

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            # Return black frames if video can't be loaded
            frames = torch.zeros(self.num_frames, 3, *self.frame_size)
            return frames

        # Sample frame indices uniformly
        if total_frames < self.num_frames:
            # Repeat frames if video is too short
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if not ret:
                # Use previous frame if read fails
                if len(frames) > 0:
                    frame = frames[-1]
                else:
                    frame = np.zeros((self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8)
            else:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Resize frame
                frame = cv2.resize(frame, self.frame_size)

            frames.append(frame)

        cap.release()

        # Convert to tensor
        frames = np.stack(frames)  # [T, H, W, C]

        if self.transform:
            # Apply transforms to each frame
            transformed_frames = []
            for frame in frames:
                frame_pil = Image.fromarray(frame)
                frame_transformed = self.transform(frame_pil)
                transformed_frames.append(frame_transformed)
            frames = torch.stack(transformed_frames)
        else:
            # Convert to tensor [T, C, H, W]
            frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0

        return frames

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Returns:
            dict: Sample containing video frames, text, and metadata
        """
        sample = self.samples[idx]

        # Load video
        video_path = self.data_root / sample['video_path']
        video_frames = self.load_video(video_path)

        return {
            'video_id': sample['video_id'],
            'video_frames': video_frames,
            'text': sample['text'],
            'video_path': str(video_path)
        }


class MSRVTTDataset(VideoTextDataset):
    """
    MSR-VTT dataset loader.

    MSR-VTT contains 10,000 video clips with 200,000 captions.
    """

    def load_annotations(self, annotation_file):
        """Load MSR-VTT annotations."""
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        return data

    def build_index(self):
        """Build dataset index for MSR-VTT."""
        samples = []

        # Handle two possible annotation formats
        if isinstance(self.annotations, list):
            # Format: List of dicts with video_id and caption list
            for item in self.annotations:
                video_id = item['video_id']
                captions = item.get('caption', [])

                # Verify video file exists
                video_path = self.data_root / f"{video_id}.mp4"
                if not video_path.exists():
                    continue

                # For training, use all captions; for testing, use one per video
                if self.split == 'train':
                    for caption in captions:
                        samples.append({
                            'video_id': video_id,
                            'video_path': f"{video_id}.mp4",
                            'text': caption
                        })
                else:
                    # For eval, create one sample per video with all captions
                    if captions:
                        samples.append({
                            'video_id': video_id,
                            'video_path': f"{video_id}.mp4",
                            'text': captions[0],  # Use first caption as representative
                            'all_captions': captions
                        })

        elif 'videos' in self.annotations:
            # Format: Dict with 'videos' and 'sentences' keys
            videos = self.annotations['videos']
            sentences = self.annotations.get('sentences', [])

            # Create video_id to sentences mapping
            video_to_sents = {}
            for sent in sentences:
                video_id = sent['video_id']
                if video_id not in video_to_sents:
                    video_to_sents[video_id] = []
                video_to_sents[video_id].append(sent['caption'])

            # Build samples
            for video in videos:
                video_id = video['video_id']

                if video_id in video_to_sents:
                    captions = video_to_sents[video_id]

                    # For training, use all captions; for testing, use one per video
                    if self.split == 'train':
                        for caption in captions:
                            samples.append({
                                'video_id': video_id,
                                'video_path': f"{video_id}.mp4",
                                'text': caption
                            })
                    else:
                        # For eval, create one sample per video with all captions
                        samples.append({
                            'video_id': video_id,
                            'video_path': f"{video_id}.mp4",
                            'text': captions[0],  # Use first caption as representative
                            'all_captions': captions
                        })

        # Limit samples if max_samples is set
        if self.max_samples is not None and len(samples) > self.max_samples:
            samples = samples[:self.max_samples]

        return samples


class DiDeMoDataset(VideoTextDataset):
    """
    DiDeMo dataset loader.

    DiDeMo contains 10,642 personal videos with temporally localized descriptions.
    """

    def load_annotations(self, annotation_file):
        """Load DiDeMo annotations."""
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        return data

    def build_index(self):
        """Build dataset index for DiDeMo."""
        samples = []

        for video_id, video_data in self.annotations.items():
            # DiDeMo has multiple descriptions per video
            descriptions = video_data.get('description', [])

            video_path = f"{video_id}.mp4"

            if self.split == 'train':
                # Use all descriptions during training
                for desc in descriptions:
                    samples.append({
                        'video_id': video_id,
                        'video_path': video_path,
                        'text': desc
                    })
            else:
                # Use first description for evaluation
                if descriptions:
                    samples.append({
                        'video_id': video_id,
                        'video_path': video_path,
                        'text': descriptions[0],
                        'all_captions': descriptions
                    })

        return samples


class MSVDDataset(VideoTextDataset):
    """
    MSVD dataset loader.

    MSVD contains 1,970 videos with diverse annotations (~40 per video).
    """

    def load_annotations(self, annotation_file):
        """Load MSVD annotations."""
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        return data

    def build_index(self):
        """Build dataset index for MSVD."""
        samples = []

        for item in self.annotations:
            video_id = item['video_id']
            captions = item.get('captions', [])

            video_path = f"{video_id}.avi"

            if self.split == 'train':
                # Use all captions during training
                for caption in captions:
                    samples.append({
                        'video_id': video_id,
                        'video_path': video_path,
                        'text': caption
                    })
            else:
                # Use first caption for evaluation
                if captions:
                    samples.append({
                        'video_id': video_id,
                        'video_path': video_path,
                        'text': captions[0],
                        'all_captions': captions
                    })

        return samples


class LSMDCDataset(VideoTextDataset):
    """
    LSMDC dataset loader.

    LSMDC contains 118,081 clips from 200 movies with script descriptions.
    """

    def load_annotations(self, annotation_file):
        """Load LSMDC annotations."""
        annotations = []

        with open(annotation_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    video_id = parts[0]
                    caption = parts[1]
                    annotations.append({
                        'video_id': video_id,
                        'caption': caption
                    })

        return annotations

    def build_index(self):
        """Build dataset index for LSMDC."""
        samples = []

        for item in self.annotations:
            video_id = item['video_id']
            caption = item['caption']

            video_path = f"{video_id}.avi"

            samples.append({
                'video_id': video_id,
                'video_path': video_path,
                'text': caption
            })

        return samples


def collate_fn(batch):
    """
    Custom collate function for batching video-text data.

    Args:
        batch (list): List of samples

    Returns:
        dict: Batched data
    """
    video_frames = torch.stack([item['video_frames'] for item in batch])
    texts = [item['text'] for item in batch]
    video_ids = [item['video_id'] for item in batch]

    batched = {
        'video_frames': video_frames,
        'texts': texts,
        'video_ids': video_ids
    }

    # Include all captions if available (for evaluation)
    if 'all_captions' in batch[0]:
        batched['all_captions'] = [item['all_captions'] for item in batch]

    return batched


def create_dataloader(
    dataset_name,
    data_root,
    annotation_file,
    num_frames=12,
    frame_size=(224, 224),
    split='train',
    batch_size=32,
    num_workers=4,
    transform=None,
    max_samples=None
):
    """
    Create a dataloader for specified dataset.

    Args:
        dataset_name (str): Name of dataset ('msrvtt', 'didemo', 'msvd', 'lsmdc')
        data_root (str): Root directory containing videos
        annotation_file (str): Path to annotation file
        num_frames (int): Number of frames to sample
        frame_size (tuple): Size to resize frames to
        split (str): Dataset split
        batch_size (int): Batch size
        num_workers (int): Number of workers for data loading
        transform: Image transformations
        max_samples (int): Maximum number of samples to use (None for all)

    Returns:
        DataLoader: PyTorch DataLoader
    """
    dataset_name = dataset_name.lower()

    if dataset_name == 'msrvtt':
        dataset = MSRVTTDataset(
            data_root, annotation_file, num_frames, frame_size, split, transform, max_samples
        )
    elif dataset_name == 'didemo':
        dataset = DiDeMoDataset(
            data_root, annotation_file, num_frames, frame_size, split, transform
        )
    elif dataset_name == 'msvd':
        dataset = MSVDDataset(
            data_root, annotation_file, num_frames, frame_size, split, transform
        )
    elif dataset_name == 'lsmdc':
        dataset = LSMDCDataset(
            data_root, annotation_file, num_frames, frame_size, split, transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Disable pin_memory on CPU to save memory
    pin_memory = torch.cuda.is_available()
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory
    )

    return dataloader
