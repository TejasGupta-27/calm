"""
Class Anchor Extraction Module for CALM
Implements class-based prompts as semantic anchors for multi-modal alignment.
"""

import torch
import torch.nn as nn


class ClassAnchorExtractor(nn.Module):
    """
    Extracts class anchors from class labels and encodes them as prompts.

    Args:
        class_labels (list): List of class label strings
        text_encoder: Pre-trained CLIP text encoder
        prompt_template (str): Template for creating prompts from labels
        embed_dim (int): Embedding dimension
    """

    def __init__(self, class_labels, text_encoder, prompt_template="The content of {}", embed_dim=512):
        super().__init__()
        self.class_labels = class_labels
        self.num_classes = len(class_labels)
        self.text_encoder = text_encoder
        self.prompt_template = prompt_template
        self.embed_dim = embed_dim

        # Learnable positional embeddings for each class anchor
        self.positional_embeddings = nn.Parameter(
            torch.randn(self.num_classes, embed_dim)
        )

        # Initialize positional embeddings with small values
        nn.init.normal_(self.positional_embeddings, mean=0.0, std=0.02)

        # Pre-compute and store class anchor embeddings
        self.register_buffer('class_anchors', torch.zeros(self.num_classes, embed_dim))

    def create_prompts(self):
        """
        Create prompts from class labels using the template.

        Returns:
            list: List of prompt strings
        """
        prompts = [self.prompt_template.format(label) for label in self.class_labels]
        return prompts

    def encode_anchors(self, tokenizer, device='cuda'):
        """
        Encode class labels as prompts using CLIP text encoder.

        Args:
            tokenizer: CLIP tokenizer
            device: Device to use for encoding

        Returns:
            torch.Tensor: Encoded class anchors [K, embed_dim]
        """
        prompts = self.create_prompts()

        # Tokenize prompts
        tokens = tokenizer(prompts).to(device)

        # Encode using CLIP text encoder
        with torch.no_grad():
            prompt_features = self.text_encoder(tokens)

        # Add learnable positional embeddings (Equation 5 in paper)
        class_anchors = prompt_features + self.positional_embeddings

        # Store for future use
        self.class_anchors = class_anchors.detach()

        return class_anchors

    def get_anchors(self):
        """
        Get the stored class anchor embeddings.

        Returns:
            torch.Tensor: Class anchors [K, embed_dim]
        """
        return self.class_anchors + self.positional_embeddings

    def forward(self):
        """
        Forward pass returns class anchors with positional embeddings.

        Returns:
            torch.Tensor: Class anchors [K, embed_dim]
        """
        return self.get_anchors()


def load_class_labels(dataset_name='charades'):
    """
    Load class labels from different datasets.

    Args:
        dataset_name (str): Name of dataset to load labels from

    Returns:
        list: List of class label strings
    """
    if dataset_name.lower() == 'charades':
        # Charades action classes (157 classes)
        # For demonstration, here's a subset. Full list should be loaded from file.
        charades_labels = [
            "Holding some clothes",
            "Putting clothes somewhere",
            "Taking some clothes from somewhere",
            "Throwing clothes somewhere",
            "Tidying some clothes",
            "Washing some clothes",
            "Closing a door",
            "Fixing a door",
            "Opening a door",
            "Putting something on a table",
            "Sitting on a table",
            "Sitting at a table",
            "Tidying up a table",
            "Washing a table",
            "Working at a table",
            "Holding a phone/camera",
            "Playing with a phone/camera",
            "Putting a phone/camera somewhere",
            "Taking a phone/camera from somewhere",
            "Talking on a phone/camera",
            "Holding a bag",
            "Opening a bag",
            "Putting a bag somewhere",
            "Taking a bag from somewhere",
            "Throwing a bag somewhere",
            "Closing a book",
            "Holding a book",
            "Opening a book",
            "Putting a book somewhere",
            "Smiling at a book",
            "Taking a book from somewhere",
            "Throwing a book somewhere",
            "Watching/Reading/Looking at a book",
            "Holding a towel/s",
            "Putting a towel/s somewhere",
            "Taking a towel/s from somewhere",
            "Throwing a towel/s somewhere",
            "Tidying up a towel/s",
            "Washing something with a towel",
            "Closing a box",
            "Holding a box",
            "Opening a box",
            "Putting a box somewhere",
            "Taking a box from somewhere",
            "Taking something from a box",
            "Throwing a box somewhere",
            "Holding a laptop",
            "Putting a laptop somewhere",
            "Taking a laptop from somewhere",
            "Watching a laptop or something on a laptop",
            "Working on a laptop",
            # Add remaining Charades labels here...
            # This is a subset for demonstration
        ]
        return charades_labels

    elif dataset_name.lower() == 'coco':
        # COCO object categories (91 classes)
        coco_labels = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
            "truck", "boat", "traffic light", "fire hydrant", "stop sign",
            "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
            "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
            "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
            "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
            "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
            "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "couch", "potted plant", "bed", "dining table", "toilet", "tv",
            "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
            "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
            "scissors", "teddy bear", "hair drier", "toothbrush"
        ]
        return coco_labels

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_full_charades_labels():
    """
    Returns the complete list of 157 Charades action classes.
    In practice, this should be loaded from a file.
    """
    # This is a placeholder - in actual implementation, load from Charades dataset
    # For now, returning a representative subset
    labels = load_class_labels('charades')

    # TODO: Load complete 157 labels from Charades dataset file
    # e.g., from charades_v1_classes.txt

    return labels
