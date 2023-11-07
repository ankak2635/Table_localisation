import sys
sys.path.append('/home/ankit/Data_Science/CV_Projects/OrderStack')  # fix utils import error

import cv2
import os
from sklearn.model_selection import train_test_split
import utils



class Data_Ingestion:

    """
    Initializes the Data_Ingestion object with the paths to image and label directories.

    Args:
        image_dir (str): Path to the directory containing image files (in .jpg format).
        labels_dir (str): Path to the directory containing label files (in .txt format).

    Returns:
        None
    """

    def __init__(self, image_dir, labels_dir):
        self.image_dir = image_dir
        self.labels_dir = labels_dir

    def initiate_data_ingestion(self):

        """
        Initiates data ingestion and prepares the data for YOLO model training.

        This function loads image and label file paths, performs data transformations, splits
        the data into training, validation, and testing sets, and organizes it into YOLO-compatible
        directory structures.

        Args:
            None

        Returns:
            None

        Note:
            - Image size argument is directly passed to the model trainer.
            - Data augmentation methods are specified during model fine tuning.
            - The function assumes the 'ultralytics' library and other required modules are available.
        """
        try:
            # get list of image paths
            image_paths = sorted([os.path.join(self.image_dir, filename) for filename in os.listdir(self.image_dir) if filename.endswith('.jpg')])
            print(f"Loaded {len(image_paths)} image paths")

            # get list of labels paths
            label_paths = sorted([os.path.join(self.labels_dir, filename) for filename in os.listdir(self.labels_dir) if filename.endswith('.txt')])
            print(f" Loaded {len(label_paths)} label paths")
            print("Data paths reading completed")

            # prepare data as per YOLO requirements
            # split images and labels into train, val and test set

            # splitting into train test split
            train_image_paths, val_image_paths, train_label_paths, val_label_paths = train_test_split(
                image_paths, label_paths, test_size=0.2, random_state=7)
            
            # splitting the val data into val and test set
            val_image_paths, test_image_paths, val_label_paths, test_label_paths = train_test_split(
                val_image_paths, val_label_paths, test_size=0.5, random_state=7)
            
            # create required directories and move the data to respective directories
            utils.create_dirs()

            # move the train, test and val set to folders
            utils.move_to_folder(train_image_paths, './training_data/images/train')
            utils.move_to_folder(train_label_paths, './training_data/labels/train')

            utils.move_to_folder(val_image_paths, './training_data/images/val')
            utils.move_to_folder(val_label_paths, './training_data/labels/val')

            utils.move_to_folder(test_image_paths, './training_data/images/test')
            utils.move_to_folder(test_label_paths, './training_data/labels/test')
             
            print("Data ready for training.")

        except Exception as e:
            print(f"An error occurred: {e}")

 