import subprocess
import torch
import cv2 as cv
import easyocr
import numpy as np

def run_yolo_detection(source, name):
    img_size = 640
    device = 0
    weights = '/content/yolov9/runs/train/yolov9-e3/weights/best.pt'
    save_txt = '--save-txt'

    command = [
        'python', 'yolov9/detect.py',
        '--source', source,
        '--img', str(img_size),
        '--device', str(device),
        '--weights', weights,
        '--name', name,
        save_txt
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"YOLOv9 detection failed: {result.stderr}")

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Function to process the license plate and save the text to a file
def read_plate_number(cordinates, frame, reader):
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    detected_text = []

    for row in cordinates:
        cls, x_center, y_center, width, height = row
        xmin = int((x_center - width / 2) * x_shape)
        xmax = int((x_center + width / 2) * x_shape)
        ymin = int((y_center - height / 2) * y_shape)
        ymax = int((y_center + height / 2) * y_shape)

        plate = frame[ymin:ymax, xmin:xmax]

        # Preprocess Plate
        gray = cv.cvtColor(plate, cv.COLOR_BGR2GRAY)
        blurred = cv.bilateralFilter(gray, 11, 17, 17)

        # OCR
        text = reader.readtext(blurred)
        text = ' '.join([t[1] for t in text])
        detected_text.append(text)

        plot_img = frame.copy()
        # Bounding box
        cv.rectangle(plot_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        # Text background
        cv.rectangle(plot_img, (xmin, ymin - 30), (xmax, ymin), (0, 255, 0), -1)
        # Text
        final_img = cv.putText(plot_img, text, (xmin, ymin - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    return final_img, detected_text

# Function to read the coordinates from the .txt file
def get_coordinates(txt_file_path):
    cordinates = []

    with open(txt_file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            parts = [float(part) for part in parts]
            cordinates.append(parts)

    return np.array(cordinates)