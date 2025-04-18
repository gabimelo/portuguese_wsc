#!/bin/bash

mkdir data/processed/splits

split -C 51690k --numeric-suffixes data/interim/wiki_pt_splits/wiki_pt00.txt data/processed/splits/main_
split -C 1200k --numeric-suffixes data/processed/splits/main_01 data/processed/splits/test_val_

cp data/processed/splits/main_00 data/processed/train.txt
cp data/processed/splits/test_val_00 data/processed/test.txt
cp data/processed/splits/test_val_01 data/processed/valid.txt

rm -rf data/processed/splits
