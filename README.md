# Deep Music

## Objective

Music recognition and generation using deep learning.

## Project structure / design:

The project currently has three components:
1. Preprocessing
2. Training
3. Serving

Each component exists as a standalone package. Each package:

- Is callable from the command-line and has configurable arguments. E.g.
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
   - Gets exported by the `training` stage, so that the state of the data used for training can be reproduced.

- Contains shell scripts to test the package locally, with docker and to deploy the docker image to google cloud.
  [Example: training scripts](https://github.com/JulianFerry/deep-music/tree/master/src/trainer/shell)


## Roadmap

1. Instrument recognition:
   - Instrument classification from single note audio
   - Instrument detection from multiple note audio (songs)

2. Genre recognition:
   - Genre classification from songs

3. Music generation:
   - Instrument note generation
   - Musical piece generation
   - Song generation
