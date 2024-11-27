from dataloaders import get_dataloaders
from trainers import Trainer
from network import UNET
from albumentations.pytorch import ToTensorV2
import albumentations as A


IMAGE_HEIGHT = 320
IMAGE_WIDTH = 480



def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    train_loader, val_loader, _ = get_dataloaders(["datasets/segmentacao.v2i.yolov11", 
                                                    # "datasets/balls.v2i.yolov11",
                                                    # "datasets/pills.v10i.yolov11",
                                                    "datasets/foram-species-genus-groups_sma.v4i.yolov11"
                                                    ],
                                                    train_transform=train_transform,
                                                    batch_size=16,
                                                    num_workers=4,
                                                    val_transform=val_transforms)

    unet_trainer = Trainer(UNET(),
                           train_loader=train_loader, valid_loader=val_loader,
                           metrics_rate=1, snapshot_rate=1, snapshot_best=True)

    unet_trainer.fit()
   

if __name__ == "__main__":
    main()