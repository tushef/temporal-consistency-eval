import os
from typing import List, Tuple, Any

import torch
import random
import torchvision

from torch.utils.data import Subset
from torchvision.datasets import UCF101
from tqdm import tqdm


class VideoDatasetWrapper:
    """
    A wrapper class for video datasets to provide additional functionality and customization
    for training and evaluation in deep learning workflows.

    Attributes:
        name (str):
            The name of the dataset or its identifier
            Used as the unique identifier for the dataset during data gathering.
        dataset (torchvision.datasets):
            The underlying video dataset instance from the torchvision library.
        frames_per_clip (int):
            The number of consecutive frames to be included in each video clip.
            Default is 16.
        step_between_clips (int):
            The number of frames to skip between consecutive clips.
            Keep it equal to frame_per_clip such that generated clips do not overlap
            Default is 16.
        train (bool):
            Indicates whether the dataset is being used for training.
            Default is False.
        transform (torchvision.transforms, optional):
            Transformations to be applied to the video data (e.g., resizing, normalization).
            Default is None.
        collate_callback (callable, optional):
            A custom callback function for collating batches of data.
            Default is the removal of audio in batches of videos. Feel free to remove the default callback.
    """
    def __init__(self, name: str, dataset: torchvision.datasets, frames_per_clip: int = 16, step_between_clips: int = 16,
                 batch_size: int = 1, num_clips_per_label: int = 10, limit_dataset: bool = True,
                 train: bool = False, transform: torchvision.transforms = None, collate_callback: Any = None):
        self.name = name
        self.dataset = dataset
        self.frames_per_clip = frames_per_clip
        self.step_between_clips = step_between_clips
        self.batch_size = batch_size
        self.num_clips_per_label = num_clips_per_label
        self.limit_dataset = limit_dataset
        self.train = train
        self.transform = transform
        if collate_callback is None:
            self.collate_callback = self.remove_audio_collate
        else:
            self.collate_callback = collate_callback

        self.test_loader = self.get_ucf101_test_loader()

    def get_ucf101_test_loader(self,
                               ucf_data_dir: str = "datasets/UCF101/UCF-101",
                               ucf_label_dir: str = "datasets/UCF101/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist"):
        """
        Returns the test dataloader for UCF101.
        The files of UCF101 must be found locally and the directory must be customized accordingly

        :return:
        """

        test_dataset = UCF101(ucf_data_dir, ucf_label_dir,
                              frames_per_clip=self.frames_per_clip,
                              step_between_clips=self.step_between_clips,
                              train=self.train,
                              transform=self.transform)

        # Limited dataset
        if self.limit_dataset:
            dataset = self.limit_ucf101_dataset(test_dataset, self.num_clips_per_label)
        else:
            dataset = test_dataset

        test_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=self.batch_size,
                                                  shuffle=False,
                                                  drop_last=True,
                                                  collate_fn=self.collate_callback)

        return test_loader

    """
        Helper functions
    """

    def parse_class_indices(self, class_ind_path):
        """
        Parse the classInd.txt file to create a mapping of class names to labels.

        Args:
            class_ind_path (str): Path to the classInd.txt file.
        Returns:
            dict: A dictionary mapping class names to numeric labels.
        """
        class_mapping = {}
        with open(class_ind_path, 'r') as f:
            for line in f:
                label, class_name = line.strip().split()
                class_mapping[label] = class_name
        return class_mapping

    def get_label_from_path(self, video_path, class_mapping):
        """
        Extract the label from the video path based on the folder name.

        Args:
            video_path (str): Path to the video file.
            class_mapping (dict): Mapping of class names to labels.
        Returns:
            int: The numeric label for the video.
        """
        # Extract folder name (class name) from video path
        folder_name = os.path.basename(os.path.dirname(video_path))
        return class_mapping.get(folder_name, -1)  # Return -1 if class not found


    def limit_ucf101_dataset(self, ucf101_dataset, num_clips_per_label=2):
        """
        Helper function for limiting the UCF101 dataset.

        :param ucf101_dataset: The UCF101 dataset wrapper from trochvision.datasets
        :param num_clips_per_label: The number of Clips per label
        :return: torch.utils.data.Subset
        """

        # Mapping from label to indices of clips
        label_to_indices = {}
        for idx in tqdm(range(len(ucf101_dataset)), desc="Processing dataset"):
            video, audio, label = ucf101_dataset[idx]
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(idx)

        # Randomly sample a fixed number of clips per label
        selected_indices = []
        for label, indices in label_to_indices.items():
            if len(indices) > num_clips_per_label:
                selected_indices.extend(random.sample(indices, num_clips_per_label))
            else:
                selected_indices.extend(indices)  # If fewer clips are available, take all

        return Subset(ucf101_dataset, selected_indices)

    def remove_audio_collate(self, batch) -> torch.Tensor:
        """
            A custom collate function for processing batches of data.

            Parameters:
                batch (List[Tuple[Any, Any, Any]]): A batch of data, where each element is a tuple
                                                    containing video, the audio (ignored), and a label.

            Returns:
                torch.Tensor: A tensor containing the collated data (videos and labels).
            """
        filtered_batch = []
        for video, _, label in batch:
            filtered_batch.append((video, label))
        return torch.utils.data.dataloader.default_collate(filtered_batch)