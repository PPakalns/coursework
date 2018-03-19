#!/bin/bash

if [[ ! -e ../CAT_DATASET_01.zip ]]; then
    echo "Downloading CAT_DATASET_01.zip"
    aria2c -o ../CAT_DATASET_01.zip https://archive.org/download/CAT_DATASET/CAT_DATASET_01.zip
fi

if [[ ! -e ../CAT_DATASET_02.zip ]]; then
    echo "downloading CAT_DATASET_02.zip"
    aria2c -o ../CAT_DATASET_01.zip https://archive.org/download/CAT_DATASET/CAT_DATASET_02.zip
fi

mkdir -p ../cats

echo "Extracting cats"

unzip -n ../CAT_DATASET_01.zip -d ../cats -n
unzip -n ../CAT_DATASET_02.zip -d ../cats -n

cat_dirs=("CAT_00" "CAT_01" "CAT_02" "CAT_03" "CAT_04")

rm -rf ../CAT
mkdir -p ../CAT

for i in "${cat_dirs[@]}"
do
    echo "Copying $i to ../CAT/"
    cp ../cats/$i/* ../CAT/
done

rm ../CAT/Thumbs.db

echo "Great success!"
# CAT_05, CAT_06 paliks testēšanai
