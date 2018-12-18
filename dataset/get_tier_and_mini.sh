#!/usr/bin/env bash
# make sure you are using Python 3
# under the `few_shot_learning/` folder, execute this file

DATA_ROOT=dataset

echo "Downloading tier-imagenet data (12.9G) ... gonna take a while"

mkdir -p $DATA_ROOT/tier_imagenet
python tools/download_gdrive.py 1hqVbS2nhHXa51R9_aB6QDXeC0P2LQG_u $DATA_ROOT/tier_imagenet/tier-imagenet.tar

echo "Done! Decompressing ..."
(cd $DATA_ROOT/tier_imagenet ; tar -xvf tier-imagenet.tar)

echo "tiered-Imagenet Done!"

# get mini-imagenet
mkdir -p $DATA_ROOT/mini_imagenet
python tools/download_gdrive.py 1HkgrkAwukzEZA0TpO7010PkAOREb2Nuk $DATA_ROOT/mini_imagenet/mini-imagenet.zip
(cd $DATA_ROOT/mini_imagenet ; unzip mini-imagenet.zip)
echo "mini-Imagenet Done!"
