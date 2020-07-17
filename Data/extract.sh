#!/bin/bash

INPUT_FILE=$1
TIFF_INFO=tiffinfo
TIFF_CROP=tiffcrop
PRINTF=printf
CONVERT=convert
N=$($TIFF_INFO $INPUT_FILE | grep "TIFF Directory" | wc -l)

if [ ! -d "./tmp" ]; then
	/bin/mkdir ./tmp
else
	/bin/rm -f ./tmp/*
fi

if [ ! -d "./images" ]; then
	/bin/mkdir ./images
else
	/bin/rm -f ./images/*
fi

for i in $(seq 1 $N); do
	echo -e "Extracting image $i" 
	$PRINTF -v j "%05d" $(($i-1))
	$TIFF_CROP -N $i $INPUT_FILE ./tmp/image_$j.tiff
	$CONVERT ./tmp/image_$j.tiff ./images/image_$j.jpg
done

/bin/rm -rf ./tmp
