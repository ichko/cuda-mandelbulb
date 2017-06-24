#!/bin/bash

convert image.ppm image.png
curl --upload-file ./image.png https://transfer.sh/image.png
echo
rm image.png

