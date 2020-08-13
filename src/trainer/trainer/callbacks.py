import tensorboardX as tbx
import re
import os
import json
from . import gsutil


class Callback:
    """
    Base callback class.

    Each callback method is called at the appropriate time during training.

    """
    def __init__(self):
        pass
    def on_train_start(self):
        pass
    def on_epoch_start(self, epoch=None):
        pass
    def on_batch_start(self, batch=None):
        pass
    def on_batch_end(self, batch, batch_total, logs):
        pass
    def on_epoch_end(self, epoch, logs):
        pass
    def on_train_end(self):
        pass


class PrintCallback(Callback):
    """
    Callback which prints training progress and performance during training.

    """
    def __init__(self, num_epochs, num_batches, verbosity=2):
        self.num_epochs = num_epochs
        self.num_batches = num_batches
        self.verbosity = verbosity

    def on_epoch_start(self, epoch):
        if self.verbosity >= 1:
            print(f'Epoch {epoch}/{self.num_epochs}')

    def on_batch_end(self, batch, batch_total, logs):
        if self.verbosity >= 2:
            self._progress_bar(batch)
            self._print_logs(logs, 'batch', 'train', end='\r')

    def on_epoch_end(self, epoch, logs):
        if self.verbosity >= 1:
            self._progress_bar(self.num_batches)
            self._print_logs(logs, 'epoch', 'train', end=' - ')
            self._print_logs(logs, 'epoch', 'val', end='\n')
    
    def on_train_end(self):
        if self.verbosity >= 1:
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


class SummaryWriterCallback(tbx.SummaryWriter, Callback):
    """
    Callback which saves training summary logs: config, hparams and performance.

    Parameters
    ----------
    path: str
        The path used to save the logs
    train_config: dict
        The data configuration used for the training job
    update_freq: str
        How often to write logs. Format "N epoch(s)" or "N batche(s)"
    hparmas: dict
        Hyperparameters used for the training job

    """
    def __init__(self, path, train_config, update_freq='epoch', hparams=None):
        # Parse params
        self.path = path
        self.log_stage = self._parse_stage(update_freq)
        self.log_freq = self._parse_freq(update_freq)
        self.train_config = train_config
        self.hparams = hparams
        # Initialise summary writer
        if path.startswith('gs://'):
            gsutil.gcloud_auth()
        log_path = os.path.join(path, 'logs')
        super().__init__(log_path)

    def on_train_start(self):
        # Save data config
        local_path = self.path.replace('gs://', '')
        os.makedirs(local_path, exist_ok=True)
        config_path = os.path.join(local_path, 'train_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.train_config, f)
        if self.path.startswith('gs://'):
            gsutil.upload(config_path, os.path.join(self.path, 'train_config.json'))
        # Save hparams
    #     if self.hparams is not None:
    #         self.add_hparams(hparam_dict=self.hparams,
    #                              metric_dict={'val_accuracy': logs['epoch_acc/val']},
    #                              name='hparams',
    #                              global_step=epoch)

    def on_batch_end(self, batch, batch_total, logs):
        if (self.log_stage == 'batches') and (batch % self.log_freq == 0):
            for key in logs.keys():
                self.add_scalar(key, logs[key], batch_total)

    def on_epoch_end(self, epoch, logs):
        if (self.log_stage == 'batches') or \
           ((self.log_stage == 'epochs') and (epoch % self.log_freq == 0)):
            for key in logs.keys():
                self.add_scalar(key, logs[key], epoch)

    def on_train_end(self):
        self.close()

    def _parse_stage(self, update_freq):
        """
        Parse logging "stage" (epochs or batches)
        """
        log_stage = None
        stage_error = ValueError(
            'Please enter an N integer number of epochs, '
            'or an string of the format "N epoch(s)" or "N batche(s)".')
        # Parse
        if isinstance(update_freq, int):
            log_stage = 'epochs'
        elif isinstance(update_freq, str):
            re_stage = {}
            re_stage['epochs'] = re.findall(r'\b(ep?|epochs?)\b', update_freq)
            re_stage['batches'] = re.findall(r'\b(ba?|batch(?:es)?)\b', update_freq)
            update_freq_alphas = re.sub("[^A-Za-z]+", '', update_freq)
            # If a re_stage is the only word in update_freq, assign it as the log stage
            for stage in ['epochs', 'batches']:
                if len(re_stage[stage]) == 1:
                    if re_stage[stage][0] == update_freq_alphas:
                        log_stage = stage
                    else:
                        raise(stage_error)
        if log_stage is None:
            raise(stage_error)
        return log_stage

    def _parse_freq(self, update_freq):
        """
        Parse logging frequency
        """
        log_freq = None
        numeric_error = ValueError(
            'Please enter an N integer number of epochs, '
            'or an string of the format "N epoch(s)" or "N batche(s)".')
        # Parse
        if isinstance(update_freq, int):
            log_freq = update_freq
        elif isinstance(update_freq, str):
            update_freq_numeric = re.sub("[^0-9 ]+", '', update_freq).strip(' ')
            if update_freq_numeric == '':
                log_freq = 1
            elif ' ' not in update_freq_numeric:
                log_freq = int(update_freq_numeric)
            else:
                raise numeric_error
        if log_freq is None:
            raise(numeric_error)
        return log_freq
