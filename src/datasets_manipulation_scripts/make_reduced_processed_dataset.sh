#!/bin/bash

mkdir data/processed/splits

split -C 11650k --numeric-suffixes data/processed/train.txt data/processed/splits/train_
cp data/processed/train.txt data/processed/train_full.txt
cp data/processed/splits/train_00 data/processed/train.txt

split -C 2m --numeric-suffixes data/processed/test.txt data/processed/splits/test_
cp data/processed/test.txt data/processed/test_full.txt
cp data/processed/splits/test_00 data/processed/test.txt

split -C 2m --numeric-suffixes data/processed/valid.txt data/processed/splits/valid_
cp data/processed/valid.txt data/processed/valid_full.txt
cp data/processed/splits/valid_00 data/processed/valid.txt

rm -rf data/processed/splits
