#!/bin/bash

docker run \
    -it \
    --rm \
    -v $(pwd):/code \
    --gpus '"device=0"' \
    --ipc="host" \
    --name clip_for_mushrooms \
    pmorris2012/clip_for_mushrooms:latest \
    python3 /code/train.py --images-dir=/code/data/mushroom_images/ --tsv-path=/code/data/mushrooms.tsv.gz --checkpoint-dir=/code/checkpoints/
