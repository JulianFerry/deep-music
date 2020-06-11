import numpy as np
import pickle
from pathlib import Path
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms


def listdir(path):
    files = os.listdir(path)
    if '.DS_Store' in files:
        files.remove('.DS_Store')
    return files


class SpectrogramDataset(torch.utils.data.Dataset):
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


def stratified_split(dataset, split=0.8, seed=42):
    """
    Stratified train test split

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


def download_data(args):
    """
    """
    # Authenticate with local GCP credentials if they have been mounted
    if os.path.isdir('/root/credentials'):
        subprocess.run(['gcloud', 'auth', 'activate-service-account',
                        '--key-file', '/root/credentials/gs-access-key.json'])
    # Copy data from cloud storage bucket
    for instrument in args['instruments']:
        subprocess.run(['gsutil', '-m', 'cp', '-r',
                        os.path.join(args['data_dir'], instrument), '/root/data'])

def load_data(args):
    """
    """

    if args.get('data_dir', '').startswith('gs://'):
        download_data(args)
        args['data_dir'] = '/root/data'

    # Load dataset
    root_dir = args['data_dir']
    spec_dataset = SpectrogramDataset(root_dir, args['instruments'])

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
        root_dir, args['instruments'], spec_transform, label_transform)
    train_sampler, test_sampler = stratified_split(spec_dataset)
    train_loader = torch.utils.data.DataLoader(spec_dataset, batch_size=32,
                                               sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(spec_dataset, batch_size=32,
                                              sampler=test_sampler)
    return train_loader, test_loader


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.pool = nn.MaxPool1d(2, 2)
        self.conv1 = nn.Conv1d(1, 16, 5)
        self.conv2 = nn.Conv1d(16, 32, 5)
        self.fc1 = nn.Linear(116 * 32, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 1x476 in, 16x236 out
        x = self.pool(F.relu(self.conv2(x)))  # 16x236 in, 32x116 out
        x = x.view(-1, 32*116)                # flatten each mini-batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x))
        return x


def validation(model, test_loader):
    test_loss = 0
    accuracy = 0
    for n, (inputs, labels) in enumerate(test_loader):
        # Loss
        if torch.cuda.is_available():
            inputs = inputs.to('cuda')
        output = model.forward(inputs)
        test_loss += F.nll_loss(output, labels).item()
        # Acc
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    test_loss /= n+1
    accuracy /= n+1
    return test_loss, accuracy


def save_model(model, args):
    """
    """
    # Define save path
    if args.get('job_dir', '').startswith('gs://'):
        job_dir = '/root/train-output'
    else:
        job_dir = args['job_dir']
    # Save model
    os.makedirs(job_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(job_dir, 'model.pt'))
    # Upload to cloud
    if args.get('job_dir', '').startswith('gs://'):
        job_dir = '/root/train-output'
        subprocess.run(['gsutil', '-m', 'cp', '-r',
                        '/root/train-output', args['job_dir']])


def train_and_evaluate(args):
    """
    Train and evaluate model

    Parameters
    ----------
    args: dict
        Arguments parsed by task.py when the package is called with command-line

    """
    train_loader, test_loader = load_data(args)

    model = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(args['epochs']):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):

            # Forward pass and backprop
            inputs, labels = data
            if torch.cuda.is_available():
                inputs = inputs.to('cuda')
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Training log
            running_loss += loss.item()
            log_rate = 100
            if i % log_rate == log_rate-1:
                print('Epoch %d, batch %d - loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / log_rate))
                running_loss = 0.0

    print('Finished Training')
    print('train_loss: %.4f - train_accuracy: %.4f' % validation(model, train_loader))
    print('val_loss: %.4f - val_accuracy: %.4f' % validation(model, test_loader))

    save_model(model, args)