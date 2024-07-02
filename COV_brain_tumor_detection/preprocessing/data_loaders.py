from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from numpy import ndarray
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from COV_brain_tumor_detection.config import IMG_SIZE, SEED


class BrainTumorDataset(Dataset):
    """
    PyTorch Dataset for loading brain tumor images and labels.

    Parameters:
        images (List[np.ndarray]): List of image data (numpy arrays or paths).
        labels (List[int]): List of corresponding labels.
        transform (Optional[Callable]): Optional transform applied to images.

    Returns:
        None: Initializes the dataset object.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx: int) -> Tuple[torch.Tensor, torch.Tensor]: Fetches the idx-th sample from the dataset.
    """

    def __init__(
        self,
        images: List[np.ndarray],
        labels: List[int],
        transform: Optional[Callable] = None,
    ) -> None:
        self.images = images
        self.labels = np.array(labels).astype(np.float32)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[Union[ndarray, Any], int]:
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = Image.fromarray(image)
            image = self.transform(image)

        return image, label


def create_data_loaders(
    images: List[np.ndarray],
    labels: List[int],
    IMG_SIZE: Tuple[int, int] = IMG_SIZE,
    SEED: int = SEED,
) -> Tuple[DataLoader, DataLoader]:
    """
    Splits the dataset into training and validation sets, applies transformations, and creates DataLoader instances.

    Parameters:
        images (List[np.ndarray]): List of image data (numpy arrays).
        labels (List[int]): List of corresponding labels.
        IMG_SIZE (Tuple[int, int]): Size of input images (default: IMG_SIZE).
        SEED (int): Random seed for data splitting (default: SEED).

    Returns:
        Tuple[DataLoader, DataLoader]: train_loader, val_loader.
    """
    transform_train = transforms.Compose(
        [
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(
                brightness=(0.5, 1.5), contrast=0.5, saturation=0.5, hue=0.5
            ),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    transform_val = transforms.Compose(
        [
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, random_state=SEED
    )

    train_dataset = BrainTumorDataset(X_train, y_train, transform=transform_train)
    val_dataset = BrainTumorDataset(X_val, y_val, transform=transform_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=30,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=30,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    )

    return train_loader, val_loader
