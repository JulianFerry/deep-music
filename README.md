# Deep Music

## Objective

Music recognition and generation using deep learning.

## Project structure / design:

The project currently has three components, which exist as [standalone packages](https://github.com/JulianFerry/deep-music/tree/master/src):
1. **Preprocessing** - convert [NSynth](https://magenta.tensorflow.org/datasets/nsynth) audio .wav files to spectrograms with [audiolib](https://github.com/JulianFerry/audiolib).
2. **Training** - train a Convolutional Neural Network to classify instrument spectrograms with `PyTorch`.
3. **Serving** - serve the classifier as a RESTful API with `flask` and `gunicorn`.

Each package:

- Is callable from the command-line and has configurable parameters. For example, preprocessing is called with:
```
python -m preprocessing.task
   --data_dir path/to/import/raw/data \
   --job_dir path/to/export/processed/data \
   --filters_dir path/to/import/instrument/filters \
   --config $config \
   --instruments '["keyboard_acoustic", "guitar_acoustic"]'
```

- Contains a JSON file of run configurations for reproducibility. For example, [this preprocessing config file](https://github.com/JulianFerry/deep-music/blob/master/src/preprocessing/shell/configs.json):
   - Gets parsed as `$config` in the above preprocessing example.
   - Gets exported by the `training` stage, so that the data used for training can be reproduced.

- Contains shell scripts to run the package locally, with docker, and to deploy the docker image to cloud with a specific configuration ID.
  [Example training scripts](https://github.com/JulianFerry/deep-music/tree/master/src/trainer/shell)


## Roadmap

1. Instrument recognition (current):
   - Instrument classification from single note audio
   - Instrument detection from multiple note audio (songs)

2. Genre recognition:
   - Genre classification from songs

3. Music generation:
   - Instrument note generation
   - Musical piece generation
   - Song generation
