import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision.models import resnet50
import torchvision.transforms as transforms

# Paths
MODEL_PATH = "./models/apple_detection_model.pth"
CLASS_NAMES = [
    "apple_6", "apple_braeburn_1", "apple_crimson_snow_1", "apple_golden_1", "apple_golden_2",
    "apple_golden_3", "apple_granny_smith_1", "apple_hit_1", "apple_pink_lady_1", "apple_red_1",
    "apple_red_2", "apple_red_3", "apple_red_delicios_1", "apple_red_yellow_1", "apple_rooten_1"
]  # Update with your actual class names

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# Transformations for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Detection and distance estimation function
def detect_and_estimate(image, focal_length=615, real_width=7.0):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    image_tensor = transform(pil_image).unsqueeze(0).to(device)

    # Predict the class
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        class_name = CLASS_NAMES[predicted.item()]

    # Detect contours for distance estimation
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    apple_count = 0
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter small contours
            apple_count += 1
            x, y, w, h = cv2.boundingRect(contour)
            distance = (real_width * focal_length) / w  # Distance estimation
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, f"{class_name}, {distance:.2f} cm", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.putText(image, f"Apples detected: {apple_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    return image

# Streamlit App UI
st.title("Apple Detection and Distance Estimation")
st.write("Choose real-time detection or upload an image for processing.")

# Real-time camera feed using Streamlit
if st.button("Start Real-Time Detection"):
    st.info("Press 'Stop' or close the app to end the detection.")
    run_camera = st.checkbox("Stop Camera", value=False)

    # Open webcam feed
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Camera not detected. Please check your webcam.")
    else:
        # Streamlit placeholder for displaying the video frames
        frame_placeholder = st.empty()

        while cap.isOpened() and not run_camera:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to grab a frame from the camera.")
                break

            # Perform detection and estimation
            output_frame = detect_and_estimate(frame)

            # Convert BGR to RGB for Streamlit
            output_frame_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)

            # Display the frame in Streamlit
            frame_placeholder.image(output_frame_rgb, channels="RGB", use_column_width=True)

        cap.release()
