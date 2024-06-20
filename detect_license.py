import cv2 as cv
import numpy as np
from ultralytics import YOLO
import easyocr

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

def load_model(model_path):
    return YOLO(model_path)

def detect_and_read_plate(image_path, model, reader):
    results = model(image_path)
    cordinates = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = box.conf[0].item()
            cordinates.append([box.cls[0].item(), (x1 + x2) / 2 / image.shape[1], (y1 + y2) / 2 / image.shape[0], (x2 - x1) / image.shape[1], (y2 - y1) / image.shape[0]])

    cordinates = np.array(cordinates)
    image = cv.imread(image_path)
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    result_img, detected_text = read_plate_number(cordinates, image_rgb, reader)

    return result_img, detected_text

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
