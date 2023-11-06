from ultralytics import YOLO
import os
import numpy as np

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

            # make predictions on test set and save to inference/predictions
            trained_model.predict(self.test_image_paths, project = './src/inference', name = 'predictions', imgsz = 640, conf = 0.6, save=True)
            

        except Exception as e:
            print('An error occurred while predicting: ', str(e))

    def new_image_predict(self, image):
        try:
            trained_model = YOLO(self.trained_model)

            # make prediction for the new image
            prediction_result = trained_model.predict(image, conf = 0.6, show_conf = False)

            # Extract the image from the prediction result
            predicted_image = prediction_result[0].orig_img
            print(predicted_image)

            # Ensure that the image is a NumPy array
            if isinstance(predicted_image, np.ndarray):
                return predicted_image
            else:
                raise ValueError("Predicted image is not a valid NumPy array.")


        except Exception as e:
            print('An error occurred while making prediction for a new image: ', str(e))


# if __name__ == "__main__":
#     obj = Predict()
#     obj.make_predictions