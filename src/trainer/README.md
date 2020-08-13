## Setup

`poetry install`

## Local run (subset of data)

`. shell/1.\ package-local.zsh`

## Local docker run (subset of data)

`. shell/2.\ docker-local.zsh -r` (`-r` rebuilds the docker image)

## Local docker run (GCP storage)

`. shell/3.\ docker-local-gs.zsh -r` (`-r` rebuilds the docker image)

## GCP AI Platform run

`. shell/3.\ package-dataflow.zsh -r -p` (`-r` rebuilds the docker image, `-p` pushes the image)