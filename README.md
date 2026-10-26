# 🧠Object Detection System

## Overview
A Python desktop application that detects objects in images using the YOLOv3 model. It provides a graphical user interface (GUI) for selecting images, viewing detection results, zooming in on output images, saving results, and resetting the application.

## Features
- Select an image for object detection.
- Detect objects using YOLOv3.
- Display detected objects with bounding boxes and labels.
- Zoom in and out of the output image.
- Save the detection result.
- Reset the application for a new image.

## Requirements
- Python 3
- OpenCV
- NumPy
- Tkinter


## Required Files
Keep these files in the project directory:

- `main.py` — application source code
- `yolov3.cfg` — model configuration
- `yolov3.weights` — pretrained model weights
- `coco.names` — object class names


4. Select an image and click **Detect Objects**.
5. Use the zoom controls to inspect the output.
6. Click **Save Output** to save the result or **Reset** to start again.

## Technology Used
- Python
- OpenCV
- NumPy
- Tkinter
- YOLOv3

## Note
Detection accuracy and speed depend on the input image and computer hardware.
