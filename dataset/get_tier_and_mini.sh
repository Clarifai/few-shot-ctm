#!/usr/bin/env bash
# make sure you are using Python 3

DATA_ROOT=dataset

## get tier-ImageNet
#echo "Downloading tier-imagenet data (12.9G) ... gonna take a while"
#mkdir -p $DATA_ROOT/tier_imagenet
#python tools/download_gdrive.py 1hqVbS2nhHXa51R9_aB6QDXeC0P2LQG_u $DATA_ROOT/tier_imagenet/tier-imagenet.tar
#echo "Done! Decompressing ..."
#(cd $DATA_ROOT/tier_imagenet ; tar -xvf tier-imagenet.tar)
#echo "tiered-Imagenet Done!"

# get mini-imagenet
echo "Downloading mini-imagenet (~3G) ..."
mkdir -p $DATA_ROOT/miniImagenet
python tools/download_gdrive.py 1HkgrkAwukzEZA0TpO7010PkAOREb2Nuk $DATA_ROOT/miniImagenet/mini-imagenet.zip
(cd $DATA_ROOT/miniImagenet ; unzip mini-imagenet.zip)
echo "mini-Imagenet Done!"
