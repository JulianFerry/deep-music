import os
from google.cloud import storage
from tempfile import TemporaryFile
import json
import numpy as np
import torch

from preprocessing.preprocess import audio_to_spectrogram
from trainer.dataset import ToTensor, Normalise
from trainer.model import MusicNet


class Model:

    def __init__(self):
        # gcloud settings - this should become pre-set environment variables or JSON files
        key_path = '../../credentials/gs-access-key.json'
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = key_path
        BUCKET = 'deep-musik-data'
        client = storage.Client()
        self.bucket = client.bucket(BUCKET)
        self.JOB = 'trainer/config0/200623_124035'
        self._load_train_config()
        self._load_preprocessing_config()
        self._load_norm_params()
        self._load_model()

    def preprocess(self, audio):
        """
        Convert audio to spectrogram, using the same parameters as in preprocessing
        Then normalise the spectrogram, using the same parameters as in training

        """
        spec = audio_to_spectrogram(audio, self.pp_config)
        spec_norm = self.tensor(self.norm(np.abs(spec).reshape(1, 1, -1)))
        return spec_norm

    def score(self, spec):
        """
        Forward pass and return the score as an instrument label
  
        """
        with torch.no_grad():
            label_proba = torch.exp(self.model.forward(spec))
            label_numeric = torch.argmax(label_proba)
            label = self.train_config['instruments'][label_numeric]
            confidence = 100*torch.max(label_proba).numpy()
            return label, confidence

    def _load_train_config(self):
        """
        Load training config used for the training job

        """
        config_blob = self.bucket.blob(f'output/{self.JOB}/data_config.json')
        with TemporaryFile() as tmp:
            config_blob.download_to_file(tmp)
            tmp.seek(0)
            self.train_config = json.load(tmp)

    def _load_preprocessing_config(self):
        """
        Load preprocessing config associated with the training config

        """
        # This should be stored remotely / merged into data_config.json
        with open('../preprocessing/shell/configs.json') as f:
            pp_configs = json.load(f)
        self.pp_config = pp_configs[self.train_config['data_id']]['config']

    def _load_norm_params(self):
        """
        Load the normalisation params calculated from the subset of data used in training

        """
        norm_blob = self.bucket.blob(f'output/{self.JOB}/norm_params.json')
        with TemporaryFile() as tmp:
            norm_blob.download_to_file(tmp)
            tmp.seek(0)
            self.norm_params = json.load(tmp)
        self.norm = Normalise(self.norm_params['mean'], self.norm_params['std'])
        self.tensor = ToTensor(float32=True)

    def _load_model(self):
        """
        Load model state dict (weights) and create model

        """
        model_blob = self.bucket.blob(f'output/{self.JOB}/model.pt')
        with TemporaryFile() as tmp:
            model_blob.download_to_file(tmp)
            tmp.seek(0)
            state_dict = torch.load(tmp)

        self.model = MusicNet()
        self.model.load_state_dict(state_dict)
