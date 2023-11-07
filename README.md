# Table Detector

Welcome to the Table Detector GitHub repository!

Table Detector is a Python program designed for a specific use case. It is hosted as a local Tornado web application that can take an image of data tables and localize the table, header, and columns within it.

## Overview

- The 'yolov8n' model is trained from scratch using custom data.
- The model is further fine-tuned through data augmentation techniques and hyperparameter tuning.
- Check out the differences in localizations between the base model and the fine-tuned model in the `src/inference/predictions_base` and `predictions_finetuned` directories.

## Key Packages

- Ultralytics' YOLO: YOLO (You Only Look Once) is a real-time object detection system.
- Other modules from Ultralytics
- OpenCV (cv2): A popular computer vision library
- Tornado: A Python web framework and asynchronous networking library
- scikit-learn: A machine learning library

## How it Works

Here's an overview of how the Table Detector works:

1. The `notebook.ipynb` explores and visualizes the data.
2. The `src/data_ingestion.py` script takes the images and YOLO format labels folder path, splits the data into train, validation, and test sets as per YOLO requirements, and stores it in a new folder named `training_data`.
3. The `src/model_trainer.py` script calls the data ingestion component, loads and trains the model. A powerful GPU is utilized for training, typically done using a Colab notebook.
4. The model is validated and fine-tuned in the same notebook.
5. The fine-tuned model performs significantly better than the base model.
6. The `src/inference/predict.py` defines two functions:
   - To compare the predictions of the base and fine-tuned models.
   - To make predictions on a new image received from the Tornado web application.
7. A Tornado web application is created and hosted to facilitate an interactive user interface for localizing a new image.

Thank you for exploring Table Detector! Feel free to contribute or use it for your own data table localization needs.
