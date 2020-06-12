from tensorboardX import SummaryWriter
from . import gsutil

class Callback:

    def __init__(self):
        pass
    def on_train_start(self):
        pass
    def on_epoch_start(self, epoch):
        pass
    def on_batch_end(self, batch, logs):
        pass
    def on_epoch_end(self, epoch, logs):
        pass
    def on_train_end(self):
        pass


class _PrintCallback(Callback):

    def __init__(self, num_epochs, num_batches):
        self.num_epochs = num_epochs
        self.num_batches = num_batches

    def on_epoch_start(self, epoch):
        print(f'Epoch {epoch}/{self.num_epochs}')

    def on_batch_end(self, batch, batch_total, logs):
        self._progress_bar(batch)
        self._print_logs(logs, 'batch', 'train', end='\r')

    def on_epoch_end(self, epoch, logs):
        self._progress_bar(self.num_batches)
        self._print_logs(logs, 'epoch', 'train', end=' - ')
        self._print_logs(logs, 'epoch', 'val', end='\n')
    
    def on_train_end(self):
        print('Training finished.')

    def _progress_bar(self, batch):
        size = 19
        progress_size = int(size * batch/self.num_batches)
        dot_size = size - progress_size
        progress = '[%s>%s]' % (progress_size * '=', dot_size * '.')
        print(f'    {batch}/{self.num_batches} {progress} - ', end='')

    def _print_logs(self, logs, stage, dataset, end):
        # Loss
        key = f'{stage}_loss/{dataset}'
        if logs.get(key):
            print(f'{dataset}_loss: {logs[key]:.3f}', end='')
        # Accuracy
        key = f'{stage}_acc/{dataset}'
        if logs.get(key):
            print(f' - {dataset}_accuracy: {logs[key]:.3f}', end='')
        print('', end=end)


class SummaryWriterCallback(SummaryWriter, Callback):

    def __init__(self, path):
        if str(path).startswith('gs://'):
            gsutil.gcloud_auth()
        super().__init__(path)

    def on_batch_end(self, batch, batch_total, logs):
        for key in logs.keys():
            self.add_scalar(key, logs[key], batch_total)

    def on_epoch_end(self, epoch, logs):
        for key in logs.keys():
            self.add_scalar(key, logs[key], epoch)

    def on_train_end(self):
        self.close()

