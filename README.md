# FrameDuplix: Duplicate Frames Detection from Videos

## Overview
##### This project implements a Python-based solution for detecting duplicate frames in video files. It offers multiple methods for detection, including SSIM (Structural Similarity Index), deep learning-based feature comparison, and a combined approach that leverages optical flow alongside deep learning.

## Features
* **SSIM Detection:** Utilizes Structural Similarity Index for finding duplicates. 
* **Deep Learning Detection:** Employs a pretrained ResNet50 model to compare frame features.
* **Combined Detection:** Integrates SSIM, optical flow, and deep learning for enhanced accuracy.
* **Frame Extraction and Saving:** Extracts frames from videos and saves duplicates for analysis.
* **Video Saving Without Duplicates:** Offers functionality to save a new video file excluding the detected duplicate frames.

## Installation
To install the dependencies run the below command:
```bash
$ conda env create -f requirements.yaml
```
## Usage
To use the frame detection, provide the path to your video file and select the detection method:
1. **Set the video path:** Replace **'input/Clip3.mp4'** with the path to your video file.
2. **Select the detection method:** Choose between 'ssim', 'deep_learning', or 'advanced' (combined approach).
3. **Run the script:** Execute the script to identify and print the duplicate frames.

## Customization
You can adjust the thresholds and parameters in the class initializations to suit your specific requirements.

## Output
* Duplicate frames are saved in the outputs directory under respective method-named folders.
* A new video file, excluding the detected duplicates, is saved in the input directory.

