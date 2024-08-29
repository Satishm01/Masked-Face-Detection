# Mask Detection Model

This repository contains scripts for training and using a mask detection model. The model can be trained, tested on individual images, and used for real-time detection via video.

## Table of Contents
- [Training the Model](#training-the-model)
- [Validating Individual Test Data](#validating-individual-test-data)
- [Real-Time Mask Detection](#real-time-mask-detection)
- [Requirements](#requirements)

## Training the Model

To train the mask detection model, use the following command:

```bash
python train_mask_detector.py --dataset <path-to-dataset>

python detect_mask_image.py --image <path-to-image>

python detect_mask_image.py --image images/pic1.jpeg

python detect_mask_image.py --image images/pic3.jpg

python detect_mask_video.py

for installing requirements
pip install -r requirements.txt

