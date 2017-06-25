#!/bin/bash

image_name=image.ppm
width=8192
height=8192
block_size_w=32
block_size_h=32

./main $image_name $width $height $block_size_w $block_size_h r
 
