#!/bin/bash

image_name=image.ppm
width=1024
height=1024
block_size_w=32
block_size_h=32

./main $image_name $width $height $block_size_w $block_size_h r
 
