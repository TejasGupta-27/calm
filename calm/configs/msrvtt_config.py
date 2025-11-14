"""
MSR-VTT Dataset Configuration
"""

from .base_config import BaseConfig


class MSRVTTConfig(BaseConfig):
    """Configuration for MSR-VTT dataset."""

    def __init__(self):
        super().__init__()

        # Update dataset-specific parameters
        self.DATASET.update({
            'name': 'msrvtt',
            'data_root': './datasets/msrvtt/videos',
            'annotation_train': './datasets/msrvtt/annotations/train.json',
            'annotation_val': './datasets/msrvtt/annotations/val.json',
            'annotation_test': './datasets/msrvtt/annotations/test.json',
        })

        # MSR-VTT specific training parameters
        self.TRAIN.update({
            'num_epochs': 5,
            'batch_size': 128,
        })

        # Experiment name
        self.LOGGING.update({
            'experiment_name': 'calm_msrvtt',
        })
