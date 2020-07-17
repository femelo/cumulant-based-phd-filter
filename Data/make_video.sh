#!/bin/bash

/usr/bin/ffmpeg -framerate 15 -i images/image_%05d.jpg -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p wami.mp4
