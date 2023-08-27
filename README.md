# NFX Tasks

This repository contains code for detecting shoplifting activities using Python along with the OpenCV library, MediaPipe for pose estimation, and YOLOv3 for object detection. The combined approach aims to identify potential instances of shoplifting behavior in video footage.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Customization](#customization)
- [Results](#results)
- [Credits](#credits)
- [License](#license)

## Introduction

Shoplifting is a common issue faced by retail businesses. This project leverages computer vision techniques to automatically detect and flag suspicious activities in surveillance videos, specifically focusing on potential shoplifting behavior. The approach combines pose estimation to identify body posture and gestures, and object detection to recognize items being taken from shelves.

## Prerequisites

Before running the code, ensure you have the following prerequisites installed:

- Python (3.6 or higher)
- OpenCV (cv2)
- MediaPipe (mediapipe)
- YOLOv3 weights and configuration files

You can install the required libraries using the following command:

```bash
pip install opencv-python mediapipe
```

Download YOLOv3 weights (`yolov3.weights`) and configuration files (`yolov3.cfg`) from the official YOLO website or repository.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/shoplifting-detection.git
```

2. Navigate to the repository directory:

```bash
cd shoplifting-detection
```

3. Place YOLOv3 weights and configuration files in the repository directory.

## Usage

1. Run the `shoplifting_detection.py` script to start detecting shoplifting activities:

```bash
python shoplifting_detection.py
```

The script will process a video file (specified in the script) and analyze it for shoplifting behaviors.

## Customization

You can customize the following aspects:

- Adjust YOLOv3 configuration and weights paths in the `shoplifting_detection.py` script.
- Tune the confidence thresholds for object detection and pose estimation.
- Modify the video file path and other parameters as needed.

## Results

Provide visual and descriptive results of the shoplifting detection process. You can include screenshots or GIFs showing suspicious activities being highlighted.

## Credits

- Pose estimation functionality utilizes the MediaPipe library by Google.
- Object detection implementation is based on the YOLO algorithm.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

Feel free to customize the repository details, instructions, and other information to match your project. Include proper licenses, credits, and any additional information required for your project.
