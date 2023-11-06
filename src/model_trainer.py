import sys
sys.path.append('/home/ankit/Data_Science/CV_Projects/OrderStack')  # fix import errors

from ultralytics import YOLO
from src.data_ingestion import Data_Ingestion

class Model_Trainer:

    def __init__(self, image_dir_path, label_dir_path):
        self.image_dir= image_dir_path
        self.label_dir = label_dir_path

    def initiate_model_training(self):
        try:

            # call the data_ingestion module
            obj = Data_Ingestion(image_dir=self.image_dir, labels_dir=self.label_dir)
            obj.initiate_data_ingestion()

            # load the model
            model = YOLO("yolov8n.yaml")
            print("Loaded the YOLOv8 nano model")

            # initiate training
            model.train(data='./training_data/yolo_config.yaml', epochs=500)
            print("Training completed")

        except Exception as e:
            print('An error occurred while trying to train the model: ', str(e))

if __name__ == "__main__":
    image_dir = "/home/ankit/Data_Science/CV_Projects/OrderStack/data/images"
    label_dir = "/home/ankit/Data_Science/CV_Projects/OrderStack/data/labels"
    obj = Model_Trainer(image_dir_path=image_dir, label_dir_path=label_dir)
    obj.initiate_model_training()