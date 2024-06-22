# License-Plate-Detection

This project is a Streamlit web application that detects and recognizes license plates in uploaded images using YOLOv9 for object detection and EasyOCR for optical character recognition (OCR).

Web App : https://license-plate-detection-newchakri.streamlit.app

![image](https://github.com/NewChakri/License-Plate-Detection/assets/99199609/bb54dcd1-19c8-4297-92f5-268badee6e94)


## Features
**Upload Image** : Upload an image in JPG, JPEG, or PNG format. <br />
**License Plate Detection** : Detect license plates in the uploaded image using a pre-trained YOLOv9 model. <br />
**OCR** : Recognize and extract text from detected license plates using EasyOCR. <br />
**Display Results** : View the uploaded image with detected license plates highlighted and the recognized text displayed. <br />


## File Structure
**build_dataset.py** : Script for preparing the dataset. It converts XML annotations to YOLO format and organizes images and labels into separate directories. <br />
**train_model.py** : Script for training the YOLO model on the prepared dataset. <br />
**detect_license.py** : Contains functions for loading the YOLO model, detecting license plates, and performing OCR. <br />
**streamlit_app.py** : The main Streamlit app script that runs the web application. <br />
**best.pt** : The pre-trained YOLOv9 model weights file used for license plate detection. <br />


## Dataset Information
**Source** <br />
The dataset used in this project is the "Number Plate Detection" dataset from Kaggle. You can find it [here](https://www.kaggle.com/datasets/aslanahmedov/number-plate-detection).

**About the Dataset** <br />
This dataset contains 453 files - images in JPEG format with bounding box annotations of the car license plates within the image. Annotations are provided in the PASCAL VOC format. Pascal VOC (Visual Object Classes) is a format to store annotations for localizer or Object Detection datasets and is used by different annotation editors and tools to annotate, modify, and train Machine Learning models. In PASCAL VOC format, for each image there is an XML annotation file containing image details, bounding box details, classes, rotation, and other data.
