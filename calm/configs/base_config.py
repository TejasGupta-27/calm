"""
Base Configuration for CALM

Contains default hyperparameters and settings.
"""


class BaseConfig:
    """Base configuration class for CALM."""

    # Model parameters
    MODEL = {
        'clip_model_name': 'ViT-B/32',  # CLIP model variant
        'num_frames': 12,  # Number of frames to sample from video
        'latent_dim': 256,  # VAE latent dimension
        'temperature': 0.07,  # Temperature for probability distributions
        'fusion_method': 'mean',  # Temporal fusion method: 'mean', 'max', 'attention'
        'vae_hidden_dims': [512, 384],  # Hidden dimensions for VAE
        'dropout': 0.1,  # Dropout rate
        'embed_dim': 512,  # CLIP embedding dimension
    }

    # Class anchors
    CLASS_ANCHORS = {
        'dataset': 'charades',  # Dataset to load class labels from
        'num_classes': 157,  # Number of class anchors (Charades has 157)
        'prompt_template': 'The content of {}',  # Template for creating prompts
    }

    # Training parameters
    TRAIN = {
        'batch_size': 128,
        'num_epochs': 5,  # 5 for retrieval, 20 for captioning
        'learning_rate': 1e-5,
        'weight_decay': 0.01,
        'warmup_epochs': 1,
        'alpha': 0.1,  # Weight for KL divergence loss
        'gradient_clip': 1.0,  # Gradient clipping
        'accumulation_steps': 1,  # Gradient accumulation
    }

    # Optimizer
    OPTIMIZER = {
        'name': 'AdamW',
        'betas': (0.9, 0.999),
        'eps': 1e-8,
    }

    # Learning rate scheduler
    SCHEDULER = {
        'name': 'cosine',  # 'cosine', 'step', 'linear'
        'warmup_steps': 1000,
        'min_lr': 1e-7,
    }

    # Data parameters
    DATA = {
        'num_workers': 4,
        'pin_memory': True,
        'frame_size': (224, 224),
        'prefetch_factor': 2,
    }

    # Dataset paths (to be overridden for specific datasets)
    DATASET = {
        'name': 'msrvtt',
        'data_root': '/path/to/videos',
        'annotation_train': '/path/to/train_annotations.json',
        'annotation_val': '/path/to/val_annotations.json',
        'annotation_test': '/path/to/test_annotations.json',
    }

    # Evaluation parameters
    EVAL = {
        'batch_size': 32,
        'top_k': [1, 5, 10],  # For retrieval metrics
        'eval_frequency': 1,  # Evaluate every N epochs
    }

    # Logging and checkpointing
    LOGGING = {
        'log_interval': 50,  # Log every N batches
        'save_interval': 1,  # Save checkpoint every N epochs
        'output_dir': './outputs',
        'experiment_name': 'calm_default',
    }

    # Device
    DEVICE = {
        'cuda': True,
        'gpu_ids': [0],
        'distributed': False,
    }

    # Random seed
    SEED = 42

    def __init__(self):
        """Initialize configuration."""
        pass

    def to_dict(self):
        """Convert config to dictionary."""
        config_dict = {}
        for key in dir(self):
            if key.isupper():
                config_dict[key] = getattr(self, key)
        return config_dict

    def update(self, config_dict):
        """
        Update configuration from dictionary.

        Args:
            config_dict (dict): Dictionary of config updates
        """
        for key, value in config_dict.items():
            if hasattr(self, key):
                if isinstance(getattr(self, key), dict) and isinstance(value, dict):
                    # Update nested dictionary
                    getattr(self, key).update(value)
                else:
                    setattr(self, key, value)
            else:
                setattr(self, key, value)

    def __repr__(self):
        """String representation of config."""
        lines = ['Configuration:']
        for key in dir(self):
            if key.isupper():
                value = getattr(self, key)
                lines.append(f"  {key}: {value}")
        return '\n'.join(lines)
