# Step 1: Move kaggle.json to the correct location
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Step 2: Download the dataset
!kaggle datasets download -d aslanahmedov/number-plate-detection

# Step 3: Unzip the dataset
!unzip number-plate-detection.zip

from pathlib import Path
import shutil
from bs4 import BeautifulSoup
import os

def normalized_coordinates(filename, width, height, xmin, ymin, xmax, ymax):
    """Take in image coordinates (unnormalized) as input, return normalized values"""
    xmin, xmax = xmin / width, xmax / width
    ymin, ymax = ymin / height, ymax / height

    width = xmax - xmin
    height = ymax - ymin
    x_center = xmin + (width / 2)
    y_center = ymin + (height / 2)

    return x_center, y_center, width, height

def write_label(filename, x_center, y_center, width, height):
    """Save image's coordinates in text file named 'filename'"""
    with open(filename, mode='w') as outf:
        outf.write(f"{0} {x_center} {y_center} {width} {height}\n")

def parse_xml_tags(data):
    """Parse xml label file, return image file name, and its coordinates as a dictionary"""
    Bs_data = BeautifulSoup(data, "xml")
    return {
        'filename': Bs_data.find('filename').text,
        'width': int(Bs_data.find('width').text),
        'height': int(Bs_data.find('height').text),
        'xmin': int(Bs_data.find('xmin').text),
        'ymin': int(Bs_data.find('ymin').text),
        'xmax': int(Bs_data.find('xmax').text),
        'ymax': int(Bs_data.find('ymax').text)
    }

def build_data(dir_folder, ann_dir, img_dir):
    """Write xml labels to text file with specifications format, save at 'labels' folder.
    Move image to 'images' folder"""
    images_folder = f"{dir_folder}/images"
    labels_folder = f"{dir_folder}/labels"

    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(labels_folder, exist_ok=True)

    ann_file_list = [os.path.join(ann_dir, f) for f in os.listdir(ann_dir) if f.endswith('.xml')]

    for ann_file in ann_file_list:
        with open(ann_file, 'r') as f:
            label = parse_xml_tags(f.read())

        img_file_name = label['filename']
        img_path = os.path.join(img_dir, img_file_name)
        if os.path.exists(img_path):
            x_center, y_center, width, height = normalized_coordinates(**label)

            # Save at 'labels' folder
            write_label(f"{labels_folder}/{os.path.splitext(img_file_name)[0]}.txt", x_center, y_center, width, height)

            # Move image to 'images' folder
            shutil.copy(img_path, os.path.join(images_folder, img_file_name))

# Example usage
dir_folder = '/content/data'  # Specify your directory path
ann_dir = '/content/images'  # Directory containing annotation XML files
img_dir = '/content/images'  # Directory containing image files

build_data(dir_folder, ann_dir, img_dir)