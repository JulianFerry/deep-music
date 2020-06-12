import numpy as np
import pickle
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from . import gsutil


def listdir(path):
    files = os.listdir(path)
    if '.DS_Store' in files:
        files.remove('.DS_Store')
    return files


class SpectrogramDataset(Dataset):
    """CQT spectrograms dataset"""

    def __init__(self, root, instruments, spec_transform=None, label_transform=None):
        """
        Parameters
        ----------
        root: string
            Directory which contains all the spectrogram arrays
        instruments: list
            List of instruments to keep
        spec_transform: PyTorch transform object
            Transform applied to the training data (spectrograms)
        label_transform: PyTorch transform object
            Transform applied to the training labels (instrument names)

        """
        root = Path(root)
        nested_files = [[root/instr/instr_id/instr_file
                         for instr_id in listdir(root/instr)
                         for instr_file in listdir(root/instr/instr_id)]
                        for instr in instruments]
        self.files = [f for instr_files in nested_files
                        for f in instr_files]  # noqa
        self.class_counts = [len(instr_files) for instr_files in nested_files]
        self.instruments = instruments
        self.spec_transform = spec_transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        Fetch a spectrogram and label at the specified index

        Parameters
        ----------
        idx: int
            Index of the data (slices do not work)

        Returns
        -------
        sample: (spectrogram, label)
            If spec_transform does not modify type, spectrogram is an audiolib.Spectrogram
            If label_transform does not modify type, label is an int

        """
        # Load pickled Spectrogram object
        with open(self.files[idx], 'rb') as f:
            spec = pickle.load(f)
        # Extract spectrogram
        spec = np.abs(spec).T
        if self.spec_transform:
            spec = self.spec_transform(spec)
        # Extract label
        instr = '_'.join(self.files[idx].name.split('_')[:-1])
        label = self.instruments.index(instr)
        if self.label_transform:
            label = self.label_transform(label)
        return (spec, label)


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.

    """
    def __init__(self, float32=False):
        self.float32 = float32

    def __call__(self, item):
        tensor = torch.from_numpy(np.asarray(item))
        return tensor.float() if self.float32 else tensor


class Normalise(object):
    """
    Normalise 1-channel image Tensor to mean=0 std=1

    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, item):
        return (item - self.mean) / self.std


def stratified_split(dataset, split=0.8, seed=42):
    """
    Train/test split, stratified for each label

    """
    train_indices = []
    test_indices = []
    class_delim = np.cumsum([0] + dataset.class_counts)
    # Get indices
    for i in range(1, len(class_delim)):
        indices = np.arange(class_delim[i-1], class_delim[i])
        idx_split = int(np.floor(split * indices.shape[0]))
        np.random.seed(seed)
        np.random.shuffle(indices)
        train_indices.extend(list(indices[:idx_split]))
        test_indices.extend(list(indices[idx_split:]))
    # PyTorch sampler
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    return train_sampler, test_sampler


def load_data(data_dir, intstruments):
    """
    Create PyTorch data loader from data in data_dir

    """
    # Copy data from cloud storage bucket
    if data_dir.startswith('gs://'):
        gs_path = data_dir
        data_dir = '/root/data'
        for instrument in instruments:
            gsutil.download(gs_path/instrument, data_dir)
    # Load dataset
    spec_dataset = SpectrogramDataset(root_dir, instruments)
    # Calculate mean and std
    specs = np.hstack([spec_dataset[i][0] for i in range(len(spec_dataset))])
    mean = specs.mean()
    std = specs.std()
    # Prepare dataset for training
    spec_transform = transforms.Compose([
        ToTensor(float32=True),
        Normalise(mean, std)
    ])
    label_transform = ToTensor()
    spec_dataset = SpectrogramDataset(
        root_dir, instruments, spec_transform, label_transform)
    # PyTorch data loaders (train/test split)
    train_sampler, test_sampler = stratified_split(spec_dataset)
    train_loader = DataLoader(spec_dataset, batch_size=32, sampler=train_sampler)
    test_loader = DataLoader(spec_dataset, batch_size=32, sampler=test_sampler)
    return train_loader, test_loader