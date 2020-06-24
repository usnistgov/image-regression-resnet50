#!/bin/bash

modelPath="$3 $4"
imagesFolder="$1 $2"
outputFile="$5 $6/predictions.csv"

python3 inference.py ${modelPath} ${imagesFolder} ${outputFile}
