import sys
sys.path.append('/home/ankit/Data_Science/CV_Projects/OrderStack')  # fix import errors

from ultralytics import YOLO
from src.data_ingestion import Data_Ingestion
import os
import shutil

class Model_Trainer:

    def __init__(self, image_dir_path, label_dir_path):
        """
        Initialize the Model Trainer.

        Args:
            image_dir_path (str): The path to the directory containing images of tables.
            label_dir_path (str): The path to the directory containing YOLO format labels for the images.
        """

        self.image_dir= image_dir_path
        self.label_dir = label_dir_path

    def initiate_model_training(self):
        """
        Initiate the model training process.
        - Data ingestion: Organizes data into train, validation, and test sets.
        - Model loading: Loads the YOLOv8 model.
        - Training: Trains the model for table detection.
        - Best Model Saving: Saves the best model as 'trained.pt'.
        """
        try:

            # call the data_ingestion module
            obj = Data_Ingestion(image_dir=self.image_dir, labels_dir=self.label_dir)
            obj.initiate_data_ingestion()

            # load the model
            model = YOLO("yolov8n.yaml")
            print("Loaded the YOLOv8 nano model")

            # initiate training
            model.train(data='./yolo_config.yaml', epochs=1) # a dummy training 
            # model.train(data='./yolo_config.yaml', epochs=500) #uncoment if you want to train loacally
            
            print("Training completed")

            # create a copy of the best model in models dir
            dir_name= "models"
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
                print(f"'{dir_name}' directory created.")
            else:
                print(f" '{dir_name}' already exists.")

            # copy the best model to the dir
            best_model_path = './runs/detect/train/weights/best.pt'
            trained_model_path = './models/trained.pt'
            shutil.copy(best_model_path, trained_model_path)
            print(f"Best model saved as '{trained_model_path}'")

        except Exception as e:
            print('An error occurred while trying to train the model: ', str(e))

if __name__ == "__main__":
    image_dir = "/home/ankit/Data_Science/CV_Projects/OrderStack/data/images"
    label_dir = "/home/ankit/Data_Science/CV_Projects/OrderStack/data/labels"
    obj = Model_Trainer(image_dir_path=image_dir, label_dir_path=label_dir)
    obj.initiate_model_training()