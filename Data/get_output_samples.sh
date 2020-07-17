#!/bin/bash
/usr/bin/ffmpeg -i WAMI/output_videos/wami01.mp4 -r 1 -f image2 WAMI/output_images/wami01/wami01-%3d.png
/usr/bin/ffmpeg -i WAMI/output_videos/wami02.mp4 -r 1 -f image2 WAMI/output_images/wami02/wami02-%3d.png
