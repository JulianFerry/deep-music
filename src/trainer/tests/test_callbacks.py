from trainer.callbacks import PrintCallback, SummaryWriterCallback

import os
import json
import io
from contextlib import redirect_stdout


def capture(func, *args, **kwargs):
    with io.StringIO() as buf, redirect_stdout(buf):
        func(*args, **kwargs)
        output = buf.getvalue()
        return output


class TestPrintCallback:

    cb = PrintCallback(num_epochs=5, num_batches=2)
    logs = {
        'batch_loss/train': 4.58665,
        'batch_acc/train': 0.70768,
        'epoch_loss/train': 3.50868,
        'epoch_acc/train': 0.80871,
        'epoch_loss/val': 4.5421,
        'epoch_acc/val': 0.6087
    }
    progress_bar1 = '    1/2 [=========>..........] - '
    progress_bar2 = '    2/2 [===================>] - '
    batch_train = 'train_loss: 4.587 - train_accuracy: 0.708\r'
    epoch_train = 'train_loss: 3.509 - train_accuracy: 0.809 - '
    epoch_val = 'val_loss: 4.542 - val_accuracy: 0.609\n'

    def test_on_epoch_start(self):
        for epoch in range(1, 6):
            output = capture(self.cb.on_epoch_start, epoch)
            assert output == f'Epoch {epoch}/5\n'

    def test_progress_bar(self):
        output = capture(self.cb._progress_bar, batch=1)
        assert output == self.progress_bar1
        output = capture(self.cb._progress_bar, batch=2)
        assert output == self.progress_bar2

    def test_print_logs(self):
        output = capture(
            self.cb._print_logs, self.logs, 'batch', 'train', end='\r')
        assert output == self.batch_train
        output = capture(
            self.cb._print_logs, self.logs, 'epoch', 'train', end=' - ')
        assert output == self.epoch_train
        output = capture(
            self.cb._print_logs, self.logs, 'epoch', 'val', end='\n')
        assert output == self.epoch_val
        output = capture(
            self.cb._print_logs, self.logs, 'epoch', 'test', end='')
        assert output == ''

    def test_on_batch_end(self):
        output = capture(self.cb.on_batch_end, 1, '', self.logs)
        assert output == self.progress_bar1 + self.batch_train

    def test_on_epoch_end(self):
        output = capture(self.cb.on_epoch_end, 3, self.logs)
        assert output == self.progress_bar2 + self.epoch_train + self.epoch_val

    def test_on_train_end(self):
        output = capture(self.cb.on_train_end)
        assert output == 'Training finished.\n'


class TestSummaryWriterCallback:

    def create_cb(self, tmpdir):
        return SummaryWriterCallback(
            tmpdir.strpath,
            train_config={'lr': 0.01},
            hparams={'lr': 0.01}
        )

    def test_parse_stage(self, tmpdir):
        cb = self.create_cb(tmpdir)
        log_stage = cb._parse_stage(3)
        assert log_stage == 'epochs'
        log_stage = cb._parse_stage('1 epoch')
        assert log_stage == 'epochs'
        log_stage = cb._parse_stage('5 epochs')
        assert log_stage == 'epochs'
        log_stage = cb._parse_stage('1 batch')
        assert log_stage == 'batches'
        log_stage = cb._parse_stage('5 batches')
        assert log_stage == 'batches'

    def test_parse_freq(self, tmpdir):
        cb = self.create_cb(tmpdir)
        log_stage = cb._parse_freq(3)
        assert log_stage == 3
        log_stage = cb._parse_freq('1 epoch')
        assert log_stage == 1
        log_stage = cb._parse_freq('5 epochs')
        assert log_stage == 5
        log_stage = cb._parse_freq('1 batch')
        assert log_stage == 1
        log_stage = cb._parse_freq('5 batches')
        assert log_stage == 5

    def test_on_train_start(self, tmpdir):
        """Test that the train config is saved"""
        cb = self.create_cb(tmpdir)
        cb.on_train_start()
        config_path = os.path.join(tmpdir, 'train_config.json')
        assert os.path.exists(config_path)
        with open(config_path, 'r') as f:
            config = json.load(f)
        assert config == {'lr': 0.01}

    def test_on_epoch_end(self, tmpdir):
        """Test that the scalar and hparams are logged"""
        cb = self.create_cb(tmpdir)
        cb.on_epoch_end(epoch=1, logs={'epoch_acc/val': 0.873})
        assert os.listdir(tmpdir) == ['logs']
        assert len(os.listdir(os.path.join(tmpdir, 'logs'))) > 0
