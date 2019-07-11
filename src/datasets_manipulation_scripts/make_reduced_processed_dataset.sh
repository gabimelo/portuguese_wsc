#!/bin/bash

mkdir data/processed/splits

cp data/processed/train.txt data/processed/train_bkp.txt
split -C 11m --numeric-suffixes data/processed/train.txt data/processed/splits/train_
cp data/processed/splits/train_00 data/processed/train.txt

cp data/processed/train.txt data/processed/test_bkp.txt
split -C 2m --numeric-suffixes data/processed/test.txt data/processed/splits/test_
cp data/processed/splits/test_00 data/processed/test.txt

cp data/processed/train.txt data/processed/valid_bkp.txt
split -C 2m --numeric-suffixes data/processed/valid.txt data/processed/splits/valid_
cp data/processed/splits/valid_00 data/processed/valid.txt

rm -rf data/processed/splits
