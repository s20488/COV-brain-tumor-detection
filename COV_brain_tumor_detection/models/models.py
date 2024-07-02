from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule
from torchvision import models


class BaseModel(LightningModule):
    history: Dict[str, List[float]]

    def __init__(self, base_model: torch.nn.Module, num_features: int):
        super().__init__()
        """
        Initializes a PyTorch Lightning module for binary classification using a pretrained base model.

        Parameters:
            base_model (torch.nn.Module): Instance of the pretrained base model (e.g., VGG16, EfficientNet, MNASNet).
            num_features (int): Number of features in the classifier layer of the base model.

        Returns:
            None: Initializes the model object.
        """
        self.base_model = base_model
        self.base_model.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.5), nn.Linear(num_features, 1)
        )
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "accuracy": [],
            "val_accuracy": [],
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the model.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Predicted probabilities after sigmoid activation.
        """
        x = self.base_model(x)
        return self.sigmoid(x)

    def configure_optimizers(self) -> optim.Optimizer:
        """
        Configures the optimizer for training.

        Returns:
            torch.optim.Optimizer: Optimizer object.
        """
        return optim.RMSprop(self.parameters(), lr=0.001)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Performs a training step.

        Parameters:
            batch (tuple): Batch of input data and labels.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Loss tensor for the current batch.
        """
        x, y = batch
        y_hat = self(x).squeeze(1)
        loss = self.loss(y_hat, y)
        acc = (y_hat.round() == y).float().mean()
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.history["train_loss"].append(loss.item())
        self.history["accuracy"].append(acc.item())
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Performs a validation step.

        Parameters:
            batch (tuple): Batch of input data and labels.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Loss tensor for the current batch.
        """
        x, y = batch
        y_hat = self(x).squeeze(1)
        loss = self.loss(y_hat, y)
        acc = (y_hat.round() == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.history["val_loss"].append(loss.item())
        self.history["val_accuracy"].append(acc.item())
        return loss


class BrainTumorModelVGG16(BaseModel):
    def __init__(self) -> None:
        """
        Initializes a binary classification model using the VGG16 architecture.

        Returns:
            None: Initializes the model object.

        Inspiration:
        https://www.kaggle.com/code/ruslankl/brain-tumor-detection-v1-0-cnn-vgg-16
        """
        super().__init__(models.vgg16(pretrained=True), 25088)


class BrainTumorModelEfficientNet(BaseModel):
    def __init__(self) -> None:
        """
        Initializes a binary classification model using the EfficientNet architecture.

        Returns:
            None: Initializes the model object.
        """
        super().__init__(models.efficientnet_b0(pretrained=True), 1280)


class BrainTumorModelMNASNet(BaseModel):
    def __init__(self) -> None:
        """
        Initializes a binary classification model using the MNASNet architecture.

        Returns:
            None: Initializes the model object.
        """
        super().__init__(models.mnasnet1_3(pretrained=True), 1280)
