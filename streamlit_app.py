import streamlit as st
import cv2 as cv
import numpy as np
from detect_license import read_plate_number, get_coordinates, run_yolo_detection
import easyocr
import os
import tempfile

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Streamlit app
st.title("License Plate Detection and Recognition")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv.imdecode(file_bytes, 1)
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # Display the uploaded image
    st.image(image_rgb, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    # Use a temporary directory to save the image and handle YOLO output
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save the uploaded image to a temporary file
        img_path = os.path.join(tmpdir, uploaded_file.name)
        cv.imwrite(img_path, image)

        # Run YOLOv9 detection
        name = os.path.splitext(uploaded_file.name)[0]
        try:
            run_yolo_detection(img_path, name)
            st.success("YOLOv9 detection completed successfully.")
        except RuntimeError as e:
            st.error(f"YOLOv9 detection failed: {e}")
            st.stop()

        # Read the coordinates from the .txt file
        txt_file_path = os.path.join(tmpdir, f'yolov9/runs/detect/{name}/labels/{uploaded_file.name.replace("jpeg", "txt").replace("jpg", "txt").replace("png", "txt")}')
        cordinates = get_coordinates(txt_file_path)

        # Process the image and get the result
        result_img, detected_text = read_plate_number(cordinates, image_rgb, reader)

        # Convert result image from BGR to RGB
        result_img_rgb = cv.cvtColor(result_img, cv.COLOR_BGR2RGB)

        # Display the result image
        st.image(result_img_rgb, caption='Result Image with Detected License Plates.', use_column_width=True)

        # Display the detected text
        st.write("Detected Text from License Plates:")
        for text in detected_text:
            st.write(text)
