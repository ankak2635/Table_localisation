import os
import torch
import cv2

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from src.utils import parse_yolo_annotations

class CustomDataset(Dataset):

    def __init__(self, image_dir, annotation_dir, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform

        # create image file path list
        self.image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.endswith('.jpg')]

        # create annotations file path list
        self.annotation_paths = [os.path.join(annotation_dir, filename.replace('.jpg', '.txt')) for filename in os.listdir(image_dir) if filename.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)
    
    def __getitems__(self, idx):
        image_path = self.image_paths[idx]
        annotation_path = self.annotation_paths[idx]

        # Load image
        image = cv2.imread(image_path)
        cv2.imshow(image)

        # Parse and process YOLO annotations
        with open(annotation_path, 'r') as file:
            lines = file.readlines()
        
        annotations = [line.strip().split() for line in lines]
        annotations = [[float(a) for a in annotation] for annotation in annotations]

        return {
            'image': image,
            'annotations': torch.tensor(annotations)
        }



if __name__ == "__main__":
    obj = CustomDataset()
    



