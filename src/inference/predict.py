from ultralytics import YOLO
import os
import numpy as np
from ultralytics.utils.plotting import Annotator

class Predict:
    def __init__(self):
        """
        Initialize the Predict module.
        """

        self.trained_model  = "./models/trained.pt"
        self.finetuned_model = "./models/finetuned.pt"
        self.test_image_dir = "./training_data/images/test"

        # get list of test image paths
        self.test_image_paths = sorted([os.path.join(self.test_image_dir, filename) 
                                   for filename in os.listdir(self.test_image_dir) if filename.endswith('.jpg')])

    def make_predictions(self):
        """
        Make predictions on test images using both trained and finetuned YOLO models.
        Save the predictions in separate directories.
        """
        try:
            # load the trained model
            trained_model = YOLO(self.trained_model)

            # load the finetuned model
            finetuned_model = YOLO(self.finetuned_model)

            # make predictions using trained model on test set and save to inference/predictions_base
            trained_model.predict(self.test_image_paths, project = './src/inference', name = 'predictions_base', imgsz = 640, conf = 0.6, save=True)

            # make predictions using trained model on test set and save to inference/predictions_finetuned
            finetuned_model.predict(self.test_image_paths, project = './src/inference', name = 'predictions_finetuned', imgsz = 640, conf = 0.6, save=True)
            

        except Exception as e:
            print('An error occurred while predicting: ', str(e))

    def new_image_predict(self, image):
        """
        Function that takes an image path, makes predictions with the finetuned model,
        annotates the detected objects, and returns the annotated image as a NumPy array.

        Args:
            image: The image to make predictions on.

        Returns:
            np.ndarray: Annotated image as a NumPy array.
        """
        try:
            model = YOLO(self.finetuned_model)

            # make prediction for the new image
            results = model.predict(image, conf = 0.6, show_conf = False)

            for r in results:
        
                annotator = Annotator(image)
                
                boxes = r.boxes
                for box in boxes:
                    
                    b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
                    c = box.cls
                    annotator.box_label(b, model.names[int(c)])
                
            predicted_image = annotator.result()

            # Ensure that the image is a NumPy array
            if isinstance(predicted_image, np.ndarray):
                return predicted_image
            else:
                raise ValueError("Predicted image is not a valid NumPy array.")


        except Exception as e:
            print('An error occurred while making prediction for a new image: ', str(e))


# if __name__ == "__main__":
#     obj = Predict()
#     obj.make_predictions()