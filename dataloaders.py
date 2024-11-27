from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
import pathlib
import numpy as np
import PIL
from typing import List, Tuple, Union, Optional, Callable

### need a wrapper class that acts as interface for the dataset folder in system
# take dataset name, train and test ratio, seed for random split

class DatasetFolder:
    """
    A class to represent a dataset folder containing images and masks.

    Attributes:
    -----------
    datasets_path : list of pathlib.Path
        List of paths to the dataset folders.
    test_ratio : float
        Ratio of the dataset to be used for testing.
    valid_ratio : float
        Ratio of the dataset to be used for validation.
    seed : int
        Seed for random number generator.
    local_rng : numpy.random.Generator
        Local random number generator.
    dataset : numpy.ndarray
        Array of tuples containing image and mask paths.
    test_set : numpy.ndarray
        Array of tuples for the test set.
    valid_set : numpy.ndarray
        Array of tuples for the validation set.
    train_set : numpy.ndarray
        Array of tuples for the training set.
    """

    IMAGES_FOLDER = "images"
    MASKS_FOLDER = "masks"
    # LABELS_FOLDER = "labels"

    def _load_dataset(self, dataset_path: pathlib.Path) -> np.ndarray:
        """
        Load dataset from the given path.

        Parameters:
        -----------
        dataset_path : pathlib.Path
            Path to the dataset folder.

        Returns:
        --------
        numpy.ndarray
            Array of tuples containing image and mask paths.
        """
        # load dataset in arrays of images and masks tuples
        images_path = dataset_path / self.IMAGES_FOLDER
        masks_path = dataset_path / self.MASKS_FOLDER
        images = sorted(images_path.glob("*"))
        masks = sorted(masks_path.glob("*"))
        return np.array(list(zip(images, masks)))
    
    def load_datasets(self) -> None:
        """
        Load all datasets from the datasets_path and shuffle them.
        """
        # load all datasets in the datasets_path
        self.dataset = np.empty((0, 2), dtype=object)
        for dataset_path in self.datasets_path:
            self.dataset = np.concatenate((self.dataset, self._load_dataset(dataset_path)))
        self.local_rng.shuffle(self.dataset)
    
    def split_subsets(self) -> None:
        """
        Split the dataset into training, validation, and test subsets.
        """
        # split dataset into train, test and validation subsets
        n_samples = len(self.dataset)
        n_test = int(n_samples * self.test_ratio)
        n_valid = int(n_samples * self.valid_ratio)
        self.test_set = self.dataset[:n_test]
        self.valid_set = self.dataset[n_test:n_test+n_valid]
        self.train_set = self.dataset[n_test+n_valid:]

    def __init__(self, datasets_path: Union[str, List[str]], test_ratio: float = 0.2, valid_ratio: float = 0.2, seed: int = 42):
        """
        Initialize the DatasetFolder with the given parameters.

        Parameters:
        -----------
        datasets_path : str or list of str
            Path(s) to the dataset folder(s).
        test_ratio : float, optional
            Ratio of the dataset to be used for testing (default is 0.2).
        valid_ratio : float, optional
            Ratio of the dataset to be used for validation (default is 0.2).
        seed : int, optional
            Seed for random number generator (default is 42).
        """
        datasets_path = [datasets_path] if isinstance(datasets_path, str) else datasets_path
        self.datasets_path = map(pathlib.Path, datasets_path)
        self.test_ratio = test_ratio
        self.valid_ratio = valid_ratio
        self.seed = seed
        self.local_rng = np.random.default_rng(seed)
        self.load_datasets()
        self.split_subsets()

class DataIterator(TorchDataset):
    """
    A class to iterate over the dataset.

    Attributes:
    -----------
    dataset : numpy.ndarray
        Array of tuples containing image and mask paths.
    transform : callable, optional
        A function/transform to apply to the images and masks.
    """
    
    def __init__(self, dataset: np.ndarray, transform: Optional[Callable] = None):
        """
        Initialize the DataIterator with the given dataset and transform.

        Parameters:
        -----------
        dataset : numpy.ndarray
            Array of tuples containing image and mask paths.
        transform : callable, optional
            A function/transform to apply to the images and masks.
        """
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
        --------
        int
            Number of samples in the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the image and mask at the given index.

        Parameters:
        -----------
        idx : int
            Index of the sample to retrieve.

        Returns:
        --------
        tuple
            Tuple containing the image and mask.
        """
        image_path, mask_path = self.dataset[idx]
        image = np.array(PIL.Image.open(str(image_path)).convert("RGB"))
        mask = np.array(PIL.Image.open(str(mask_path)).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0
        mask = np.expand_dims(mask, axis=2)
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        return image, mask


def get_dataloaders(dataset: Union[str, List[str]], valid_ratio: float = 0.2, test_ratio: float = 0, seed: int = 42,
                    train_transform: Optional[Callable] = None, val_transform: Optional[Callable] = None, test_transform: Optional[Callable] = None,
                    batch_size: int = 8, num_workers: int = 4, pin_memory: bool = True) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    Get the data loaders for the dataset.

    Parameters:
    -----------
    dataset : str or list of str
        Path(s) to the dataset folder(s).
    valid_ratio : float, optional
        Ratio of the dataset to be used for validation (default is 0.2).
    test_ratio : float, optional
        Ratio of the dataset to be used for testing (default is 0).
    seed : int, optional
        Seed for random number generator (default is 42).
    train_transform : callable, optional
        A function/transform to apply to the training images and masks.
    val_transform : callable, optional
        A function/transform to apply to the validation images and masks.
    test_transform : callable, optional
        A function/transform to apply to the test images and masks.
    batch_size : int, optional
        Number of samples per batch (default is 8).
    num_workers : int, optional
        Number of subprocesses to use for data loading (default is 4).
    pin_memory : bool, optional
        If True, the data loader will copy tensors into CUDA pinned memory before returning them (default is True).

    Returns:
    --------
    tuple
        Tuple containing the training, validation, and test data loaders.
    """
    subsets = DatasetFolder(dataset, test_ratio, valid_ratio, seed)
    train_ds = DataIterator(subsets.train_set, train_transform)
    val_ds = DataIterator(subsets.valid_set, val_transform)
    test_ds = DataIterator(subsets.test_set, test_transform)

    train_loader = DataLoader(
                    train_ds,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    shuffle=True,
                    )

    val_loader = DataLoader(
                    val_ds,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    shuffle=True,
                    ) if val_ds else None
    
    test_loader = DataLoader(
                    test_ds,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    shuffle=False,
                    ) if test_ds else None
    
    return train_loader, val_loader, test_loader