import os
import cv2
import shutil

def parse_yolo_annotations(directory):
    """
    Parses YOLO format annotations from text files in the specified directory.

    Args:
        directory (str): The directory containing YOLO format annotation text files.

    Returns:
        list: A list of annotations for each image.
    """
    all_annotations = []
    
    # List all text files in the directory
    annotation_files = [file for file in os.listdir(directory) if file.endswith(".txt")]
    
    for annotation_file in annotation_files:
        with open(os.path.join(directory, annotation_file), 'r') as file:
            lines = file.readlines()

        annotations = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue  # Skip lines with insufficient data

            class_id = int(parts[0])  # Class ID (integer)
            center_x = float(parts[1])  # X-coordinate of the bounding box center (float)
            center_y = float(parts[2])  # Y-coordinate of the bounding box center (float)
            width = float(parts[3])  # Width of the bounding box (float)
            height = float(parts[4])  # Height of the bounding box (float)

            annotation = [class_id, center_x, center_y, width, height]
            annotations.append(annotation)

        all_annotations.append(annotations)

    return all_annotations

def draw_bounding_boxes(image, annotations):
    """
    Draws bounding boxes on an image based on the provided annotations.

    Args:
        image (numpy.ndarray): The image on which to draw bounding boxes.
        annotations (list): A list of annotations for each bounding box.

    Returns:
        numpy.ndarray: The image with bounding boxes drawn.
    """
    for annotation in annotations:
        class_id = annotation[0]
        center_x = annotation[1]
        center_y = annotation[2]
        width = annotation[3]
        height = annotation[4]
        
        # Calculate bounding box coordinates
        x = int((center_x - width / 2) * image.shape[1])
        y = int((center_y - height / 2) * image.shape[0])
        w = int(width * image.shape[1])
        h = int(height * image.shape[0])
        
        # Draw the bounding box
        color = (255, 0, 0)  # Red
        thickness = 2
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
    
    return image

# define a function to create and move the data to directories
def create_dirs():
    """
    Creates the necessary directory structure for organizing data for YOLO training.
    """

    # create directories
    main_folder = "training_data"  #parent folder

    # Check if the main folder already exists
    if not os.path.exists(main_folder):
        # Create the main folder in the current working directory
        os.makedirs(main_folder)
        print(f"Main folder '{main_folder}' created successfully.")
    else:
        print(f"Main folder '{main_folder}' already exists.")


    # create subfolders images and labels within main folder
    image_folder = os.path.join(main_folder, 'images')
    labels_folder = os.path.join(main_folder, 'labels')

    # create images folder
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
        print(f"Subfolder '{image_folder}' created.")
    else:
        print(f"{image_folder}' already exists.")

    # create labels folder
    if not os.path.exists(labels_folder):
        os.makedirs(labels_folder)
        print(f"Subfolder '{labels_folder}' created.")
    else:
        print(f"{labels_folder}' already exists.")

    # create train, val, test subfolders within images and labels folder
    subfolders = ['train', 'val', 'test']

    for subfolder in subfolders:
        # define subfolder paths
        path_images = os.path.join(image_folder, subfolder)  
        path_labels = os.path.join(labels_folder, subfolder)  

        if not os.path.exists(path_images):  # create subfolders within images
            os.makedirs(path_images)
            print(f"'{path_images}' created.")
        else:
            print(f"'{path_images}' already exists.")

        if not os.path.exists(path_labels): # create subfolders withing labels
            os.makedirs(path_labels)
            print(f"'{path_labels}' created.")
        else:
            print(f"'{path_labels}' already exists.")
            

def move_to_folder(list_of_files, destination):
    """
    Moves a list of files to the specified destination folder.

    Args:
        list_of_files (list): A list of file paths to move.
        destination (str): The destination folder where files should be moved.
    """
    for f in list_of_files:
        try:
            shutil.copy(f, destination)

        except:
            print(f)
            assert False
