import cv2
import os, pathlib, shutil, uuid
import tqdm
import numpy as np

target_datasets = [] # list of paths to datasets

target_datasets = [pathlib.Path(dataset) for dataset in target_datasets]


def merge_subsets(dataset):
    """
    datasets usually come in subsets
    merge the subsets into a single dataset
    """
    subsets = [dataset / subset for subset in ["train", "test", "valid"]]
    os.makedirs(dataset / "images", exist_ok=True)
    os.makedirs(dataset / "labels", exist_ok=True)
    for subset in subsets:
        if not subset.exists():
            continue  # test and valid are optional
        images = list(subset.glob("images/*"))
        for image in images:
            new_id = str(uuid.uuid4().hex)
            label = subset / "labels" / f"{image.stem}.txt"
            shutil.copy(image, dataset / "images" / f"{new_id}{image.suffix}")
            if not label.exists():
                print(f"Label not found: {label}")
                continue
            shutil.copy(label, dataset / "labels" / f"{new_id}.txt")

def label_to_mask(image_path, label_path, output_path):
    img = cv2.imread(str(image_path))
    height, width, _ = img.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    with open(label_path, "r") as f:
        labels = f.readlines()
    for label in labels:
        parts = label.strip().split()
        if len(parts) < 5:
            print(f"Invalid label format in file: {label_path}")
            continue
        class_id = int(parts[0])
        polygon_points = list(map(float, parts[1:]))
        polygon_points = np.array(polygon_points).reshape(-1, 2)
        polygon_points[:, 0] *= width
        polygon_points[:, 1] *= height
        polygon_points = polygon_points.astype(np.int32)
        cv2.fillPoly(mask, [polygon_points], 255)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, mask)


def dataset_make_masks(dataset):
    images = list((dataset / "images").glob("*"))
    for image in tqdm.tqdm(images):
        label = dataset / "labels" / f"{image.stem}.txt"
        output = dataset / "masks" / f"{image.stem}.png"
        label_to_mask(image, label, output)




def normalize_images(dataset):
    # convert all images to png
    images = list((dataset / "images").glob("*"))
    for image in tqdm.tqdm(images):
        img = cv2.imread(str(image))
        cv2.imwrite(str(image), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])

for dataset in target_datasets:
    # merge_subsets(dataset)
    # dataset_make_masks(dataset)
    # normalize_images(dataset)
    pass
