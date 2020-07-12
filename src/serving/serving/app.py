import os
from flask import Flask, request
from flask_restful import Api, Resource
from scipy.io import wavfile
import numpy as np
import torch

from audiolib import Audio
from serving.model import Model


# Start RESTful app
app = Flask(__name__)
api = Api(app)
app.config['model'] = Model()


class Predict(Resource):
    """
    Returns predictions using the trained model (data will be processed first)
    The POST request needs to contain a 'data' argument, sent as a file or in JSON format.

    Example use:
        curl -X POST -F data=@path/to/data/piano.wav http://127.0.0.1:8080/predict

    """
    def post(self):
        """
        Handles post request

        """
        # Load model if not already loaded
        if not app.config.get('model'):
            app.config['model'] = Model()

        # Load file from POST request
        if request.files and 'data' in request.files.keys():
            print('File received. Scoring data...')
            sampling_rate, audio = wavfile.read(request.files['data'])
        else:
            response = {
                'status': 'error',
                'message': 'Send file in a POST request under the `data` header'
            }
            return response, 400

        # Preprocess and score
        audio = Audio(audio, sampling_rate)
        spec = app.config['model'].preprocess(audio)
        label, confidence = app.config['model'].score(spec)

        print('Successfully scored data using model.')
        response = {
            'status': 'success',
            'data': {
                'label': label,
                'confidence': confidence
        }}
        return response, 200


# Add endpoints to RESTful api
api.add_resource(Predict, '/predict')


# For local debugging only
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
