

# Segmentation İnferences Tool for U-Net and YOLOv7

This project was conducted to compare the two most widely used segmentation architectures in real time.

The system features:

    OS : Ubuntu 20.04 LTS 64-bit 
    CPU : Intel(R) Core(TM) i5-10200H CPU @ 2.40GHz
    GPU : Nvidia GTX 1650ti 4GB
    RAM : Samsung M471A1K43DB1-CWE 16GB

## Inferences on Photo and Video

U-Net

![into gif](https://github.com/oguzaybilir/Lane-Segmentation/blob/main/gifs/unet.gif)

YOLOv7

![into gif](https://github.com/oguzaybilir/Lane-Segmentation/blob/main/gifs/yolov7.gif)


## Cloning the Repository

Install this repository with git.

```bash
  git clone https://github.com/oguzaybilir/Lane-Segmentation.git
  cd Lane-Segmentation
```

## Installing Required Libraries

There is a requirements.txt file to install packages you need. This file contains almost all libraries and modules used in the project.

To install this libraries and packages:

```bash
    pip3 install -r requirements.txt
```

## Run 
The tool is so easy to use.
```bash
  python3 main.py --weights "path to your YOLOv7 weights" --source "path to your photo or video"
  python3 main.py --weights "path to your U-Net weights" --source "path to your photo or video"
```
This repository only accepts source files as .mp4, .jpg, .png
and only accepts weight files as .pt, .h5 and .hdf5 .
But you can adjust the extensions in the main.py and segment/predict.py

## Authors

- [@oguzaybilir](https://github.com/oguzaybilir)

## Special Thanks and Regards

I manipulated Rizwan Munawar's https://github.com/RizwanMunawar/yolov7-segmentation repository a little bit for my usage. So I owe a thank you to him.

I also owe a debt of gratitude to my mentor Mehmet Okuyar. He guided me in my journey in the fields of Artificial Intelligence and Image Processing.

## Acknowledgements

 - [Mehmet Okuyar](https://github.com/MehmetOKUYAR)
 - [RizwanMunawar](https://github.com/RizwanMunawar/yolov7-segmentation)
