from ultralytics import YOLO
import os

class Predict:
    def __init__(self):
        self.trained_model  = "./trained_model/best.pt"
        self.test_image_dir = "./training_data/images/test"

        # get list of test image paths
        self.test_image_paths = sorted([os.path.join(self.test_image_dir, filename) 
                                   for filename in os.listdir(self.test_image_dir) if filename.endswith('.jpg')])

    def make_predictions(self):
        try:
            # load the trained model
            trained_model = YOLO(self.trained_model)

            # make predictions and save to inference/predictions
            trained_model.predict(self.test_image_paths, project = './src/inference', name = 'predictions', imgsz = 640, conf = 0.6, save=True)
            

        except Exception as e:
            print('An error occurred while predicting: ', str(e))

if __name__ == "__main__":
    Predict.make_predictions()