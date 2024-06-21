import streamlit as st
import cv2 as cv
import numpy as np
import os
import tempfile
from detect_license import load_model, detect_and_read_plate, reader

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

        # Load the YOLO model
        model = load_model('best.pt')

        # Run YOLOv9 detection and OCR
        try:
            result_img, detected_text = detect_and_read_plate(img_path, model, reader)
            st.success("License plate detection and recognition completed successfully.")
        except RuntimeError as e:
            st.error(f"License plate detection and recognition failed: {e}")
            st.stop()

        # Convert result image from BGR to RGB
        #result_img_rgb = cv.cvtColor(result_img, cv.COLOR_BGR2RGB)

        # Display the result image
        #st.image(result_img_rgb, caption='Result Image with Detected License Plates.', use_column_width=True)

        # Display the detected text
        st.write("Detected Text from License Plates:")
        for text in detected_text:
            st.write(text)
