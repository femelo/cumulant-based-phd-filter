#!/bin/bash

datasets=('WAMI')

for dataset in "${datasets[@]}"; do
    if [[ "$dataset" = "WAMI" ]]; then
        videos=('wami01' 'wami02');
    fi;
    echo "Processing dataset $dataset";
    for video in "${videos[@]}"; do
        echo "Processing video $video";
	python3 track.py -d -w -ds $dataset -iv $video
    done;
done
