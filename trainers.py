import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import os, shutil

def visualize_prediction(images, ground_truths, predictions, filepath="evaluation.png"):
    """
    Visualizes and saves a comparison of images, ground truth masks, and predicted masks.

    Args:
        images (torch.Tensor): Batch of input images (shape: B x C x H x W, where C=1 or C=3).
        ground_truths (torch.Tensor): Batch of ground truth masks (shape: B x 1 x H x W).
        predictions (torch.Tensor): Batch of predicted masks (shape: B x 1 x H x W).
        filepath (str): Path to save the visualization.
    """
    # Ensure batches have at least 5 samples
    num_samples = min(len(images), 5)
    
    # Prepare the figure
    fig, axs = plt.subplots(3, num_samples, figsize=(15, 9))
    fig.suptitle("Evaluation: Images, Ground Truth, Predictions", fontsize=16)

    for i in range(num_samples):
        # Original image
        img = images[i].detach().cpu().numpy()
        if img.shape[0] == 3:  # RGB Image
            img = img.transpose(1, 2, 0)
        else:  # Grayscale Image
            img = img.squeeze(0)
        axs[0, i].imshow(img, cmap="gray" if img.ndim == 2 else None)
        axs[0, i].axis("off")
        axs[0, i].set_title("Image")

        # Ground truth mask
        gt = ground_truths[i].detach().cpu().squeeze().numpy()
        axs[1, i].imshow(gt, cmap="gray")
        axs[1, i].axis("off")
        axs[1, i].set_title("Ground Truth")

        # Predicted mask
        pred = predictions[i].detach().cpu().squeeze().numpy()
        axs[2, i].imshow(pred, cmap="gray")
        axs[2, i].axis("off")
        axs[2, i].set_title("Prediction")

    # Adjust layout and save
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    if os.path.dirname(filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)  # Ensure directory exists
    plt.savefig(filepath, bbox_inches="tight")
    plt.close(fig)




class Trainer:
    def __init__(
        self,
        model,
        
        train_loader,
        valid_loader,
        test_loader = None,
        
        checkpoint_path = None,

        metrics_path = "training/metrics",
        metrics_rate = 2, # save metrics every epoch

        snapshots_path = "training/snapshots",
        only_last_snapshot = True,
        snapshot_rate = 0,
        snapshot_best = True,

        learing_rate = 1e-4,
        num_epochs = 10,
        loss_fn = nn.BCEWithLogitsLoss(),
        optimizer_cls = optim.Adam,
        scaler = torch.amp.GradScaler(),
        scheduler_cls = torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_params = {"mode": "min", "factor": 0.5, "patience": 2},
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initializes the trainer with the given model, data loaders, and training parameters.

        Args:
            model (torch.nn.Module): The neural network model to be trained.
            train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
            valid_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
            test_loader (torch.utils.data.DataLoader, optional): DataLoader for the test dataset. Defaults to None.
            checkpoint_path (str, optional): Path to save model checkpoints. Defaults to None.
            metrics_path (str, optional): Path to save metrics. Defaults to "training/metrics".
            metrics_rate (int, optional): Save metrics every N epochs. Defaults to 2.
            snapshots_path (str, optional): Path to save model snapshots. Defaults to "training/snapshots".
            only_last_snapshot (bool, optional): Save only the last snapshot. Defaults to False.
            snapshot_rate (int, optional): Save model snapshots every N epochs. Defaults to 0.
            snapshot_best (bool, optional): Save the best model snapshot. Defaults to True.
            learing_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-4.
            num_epochs (int, optional): Number of epochs to train the model. Defaults to 10.
            loss_fn (torch.nn.Module, optional): Loss function for training. Defaults to nn.BCEWithLogitsLoss().
            optimizer_cls (torch.optim.Optimizer, optional): Optimizer class. Defaults to optim.Adam.
            scaler (torch.cuda.amp.GradScaler, optional): Gradient scaler for mixed precision training. Defaults to torch.cuda.amp.GradScaler().
            scheduler_cls (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler class. Defaults to torch.optim.lr_scheduler.ReduceLROnPlateau.
            scheduler_params (dict, optional): Parameters for the learning rate scheduler. Defaults to {"mode": "min", "factor": 0.5, "patience": 2}.
            device (str, optional): Device to use for training. Defaults to "cuda" if available, else "cpu".
        """
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.checkpoint_path = checkpoint_path

        self.metrics_path = metrics_path
        self.metrics_rate = metrics_rate

        self.snapshots_path = snapshots_path
        self.only_last_snapshot = only_last_snapshot
        self.snapshot_rate = snapshot_rate
        self.snapshot_best = snapshot_best

        self.num_epochs = num_epochs
        self.loss_fn = loss_fn
        self.optimizer_cls = optimizer_cls
        self.scaler = scaler
        self.scheduler_cls = scheduler_cls
        self.device = device

        self.model = self.model.to(self.device)
        self.loss_fn = self.loss_fn.to(self.device)
        self.optimizer = self.optimizer_cls(self.model.parameters(), lr=learing_rate)
        self.scheduler = self.scheduler_cls(self.optimizer, **scheduler_params) if self.scheduler_cls else None


        self.metrics = {
            "train": {"loss": [], "accuracy": [], "dice_score": []},
            "validation": {"loss": [], "accuracy": [], "dice_score": []},
            "learning_rate": []
        }

        if self.checkpoint_path:
            self.__load_checkpoint()

    
    def __load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path)
        self.model.load_state_dict(checkpoint["model"], strict=False)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        self.metrics = checkpoint["metrics"]



    def epoch_train(self):
        self.model.train()
        loop = tqdm(self.train_loader, desc=f"Training", leave=False)

        epoch_loss = 0
        num_pixel_correct = 0
        num_pixel = 0
        dice_score = 0
        iou_score = 0

        for data, target in loop:
            data, target = data.to(self.device), target.float().permute(0, 3, 1, 2).to(self.device)

            # Reset gradients
            self.optimizer.zero_grad()

            # Forward pass
            with torch.amp.autocast(self.device):
                outputs = self.model(data)
                loss = self.loss_fn(outputs, target)
                epoch_loss += loss.item()

            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Calculate metrics
            outputs = torch.sigmoid(outputs)
            preds = (outputs > 0.5).float()

            num_pixel_correct += (preds == target).sum().item()
            num_pixel += torch.numel(preds)
            
            dice_score += (2 * (preds * target).sum().item()) / ((preds + target).sum().item() + 1e-8)

             # IoU
            intersection = (preds * target).sum()
            union = preds.sum() + target.sum() - intersection
            iou_score += (intersection + 1e-8) / (union + 1e-8)

            loop.set_postfix(loss=loss.item())

        # Calculate metrics
        epoch_accuracy = (iou_score / len(self.train_loader)).item()
        epoch_dice_score = dice_score / len(self.train_loader)

        # Save metrics
        self.metrics["train"]["loss"].append(epoch_loss / len(self.train_loader))
        self.metrics["train"]["accuracy"].append(epoch_accuracy)
        self.metrics["train"]["dice_score"].append(epoch_dice_score)

    def evaluate(self, loader, phase="validation"):
        self.model.eval()
        loop = tqdm(loader, desc=f"Evaluating {phase.capitalize()}", leave=False)

        epoch_loss = 0
        num_pixel_correct = 0
        num_pixel = 0
        dice_score = 0
        iou_score = 0
        with torch.no_grad():
            for data, target in loop:
                data, target = data.to(self.device), target.float().permute(0, 3, 1, 2).to(self.device)
                outputs = self.model(data)
                loss = self.loss_fn(outputs, target)
                epoch_loss += loss.item()

                # Calculate metrics
                outputs = torch.sigmoid(outputs)
                preds = (outputs > 0.5).float()
                num_pixel_correct += (preds == target).sum().item()
                num_pixel += torch.numel(preds)
                dice_score += (2 * (preds * target).sum().item()) / ((preds + target).sum().item() + 1e-8)

                 # IoU
                intersection = (preds * target).sum()
                union = preds.sum() + target.sum() - intersection
                iou_score += (intersection + 1e-8) / (union + 1e-8)


                loop.set_postfix(loss=loss.item())

        # Calculate metrics
        accuracy = (iou_score / len(loader)).item()
        avg_dice_score = dice_score / len(loader)
        avg_loss = epoch_loss / len(loader)

        # Save metrics
        if phase in self.metrics:
            self.metrics[phase]["loss"].append(avg_loss)
            self.metrics[phase]["accuracy"].append(accuracy)
            self.metrics[phase]["dice_score"].append(avg_dice_score)

        return avg_loss, accuracy, avg_dice_score

    def fit(self):
        for epoch in tqdm(range(self.num_epochs), desc="Training", leave=True):
            self.epoch_train()
            val_loss, accuracy, avg_dice_score = self.evaluate(self.valid_loader, phase="validation")
            if self.scheduler:
                self.scheduler.step(val_loss)
                self.metrics["learning_rate"].append(self.optimizer.param_groups[0]["lr"])
            
            if self.snapshot_best and self.determine_best(val_loss, accuracy, avg_dice_score):
                self.save_best()

            elif self.snapshot_rate > 0 and epoch % self.snapshot_rate == 0:
                if self.only_last_snapshot:
                    self.delete_old_snapshots()
                self.save_snapshot()
            elif epoch == self.num_epochs - 1:
                self.save_snapshot() # save the last snapshot
            if self.metrics_rate > 0 and (epoch % self.metrics_rate == 0 or epoch == self.num_epochs - 1):
                self.save_metrics()

        if self.test_loader:
            _, test_accuracy, test_dice_score = self.evaluate(self.test_loader, phase="test")
            self.metrics["test"]["accuracy"] = test_accuracy
            self.metrics["test"]["dice_score"] = test_dice_score


    def get_checkpoint(self):
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "metrics": self.metrics,
        }

    def save_metrics(self):
        os.makedirs(self.metrics_path, exist_ok=True)
        torch.save(self.metrics, os.path.join(self.metrics_path, "metrics.pth"))
        self.plot_metrics(os.path.join(self.metrics_path, "metrics.png"))
        self.get_sample(num_samples=5, fname=os.path.join(self.metrics_path, "evaluation.png"))

    def delete_old_snapshots(self):
        # Delete all snapshots except the best one
        if not os.path.exists(self.snapshots_path):
            return
        for folder in os.listdir(self.snapshots_path):
            if folder != "best":
                shutil.rmtree(os.path.join(self.snapshots_path, folder))

    def save_snapshot(self):
        epoch = len(self.metrics["train"]["loss"])
        snapshot_folder = os.path.join(self.snapshots_path, f"epoch_{epoch}")
        os.makedirs(snapshot_folder, exist_ok=True)
        torch.save(self.get_checkpoint(), os.path.join(snapshot_folder, "checkpoint.pth"))
        self.get_sample(num_samples=5, fname=os.path.join(snapshot_folder, "evaluation.png"))
        self.plot_metrics(os.path.join(snapshot_folder, "metrics.png"))

        

    def determine_best(self, val_loss, accuracy, avg_dice_score):
        best_loss = min(self.metrics["validation"]["loss"])
        best_accuracy = max(self.metrics["validation"]["accuracy"])
        best_dice_score = max(self.metrics["validation"]["dice_score"])
        return val_loss <= best_loss and accuracy >= best_accuracy and avg_dice_score >= best_dice_score


    def save_best(self):
        best_folder = os.path.join(self.snapshots_path, "best")
        os.makedirs(best_folder, exist_ok=True)
        torch.save(self.get_checkpoint(), os.path.join(best_folder, "checkpoint.pth"))
        self.get_sample(num_samples=5, fname=os.path.join(best_folder, "evaluation.png"))
        self.plot_metrics(os.path.join(best_folder, "metrics.png"))


    def plot_metrics(self, path='metrics.png'):
        # Plot the metrics for both training and validation, and include the learning rate
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))

        # Loss plot
        axes[0, 0].plot(self.metrics["train"]["loss"], label="Train Loss", color="blue")
        axes[0, 0].plot(self.metrics["validation"]["loss"], label="Validation Loss", color="orange")
        axes[0, 0].set_title("Loss")
        axes[0, 0].set_xlabel("Epochs")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid()

        # Accuracy plot
        axes[0, 1].plot(self.metrics["train"]["accuracy"], label="Train Accuracy", color="blue")
        axes[0, 1].plot(self.metrics["validation"]["accuracy"], label="Validation Accuracy", color="orange")
        axes[0, 1].set_title("Accuracy")
        axes[0, 1].set_xlabel("Epochs")
        axes[0, 1].set_ylabel("Accuracy")
        axes[0, 1].legend()
        axes[0, 1].grid()

        # Dice Score plot
        axes[1, 0].plot(self.metrics["train"]["dice_score"], label="Train Dice Score", color="blue")
        axes[1, 0].plot(self.metrics["validation"]["dice_score"], label="Validation Dice Score", color="orange")
        axes[1, 0].set_title("Dice Score")
        axes[1, 0].set_xlabel("Epochs")
        axes[1, 0].set_ylabel("Dice Score")
        axes[1, 0].legend()
        axes[1, 0].grid()

        # Learning Rate plot
        axes[1, 1].plot(self.metrics["learning_rate"], label="Learning Rate", color="purple")
        axes[1, 1].set_title("Learning Rate")
        axes[1, 1].set_xlabel("Epochs")
        axes[1, 1].set_ylabel("Learning Rate")
        axes[1, 1].legend()
        axes[1, 1].grid()

        # Save the figure
        plt.tight_layout()
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        plt.close()
        
    def get_sample(self, num_samples=5, fname="evaluation.png"):
        # Get predictions for the validation set
        self.model.eval()
        
        with torch.no_grad():
            for i, (data, target) in enumerate(self.valid_loader):
                if i == num_samples:
                    break
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                outputs = torch.sigmoid(outputs)
                preds = (outputs > 0.5).float()
                visualize_prediction(data, target, preds, fname)
                break
