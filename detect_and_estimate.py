import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
import numpy as np
from PIL import Image

# Paths
MODEL_PATH = "./models/apple_detection_model.pth"
CLASS_NAMES = [
    "apple_6", "apple_braeburn_1", "apple_crimson_snow_1", "apple_golden_1", "apple_golden_2", 
    "apple_golden_3", "apple_granny_smith_1", "apple_hit_1", "apple_pink_lady_1", "apple_red_1", 
    "apple_red_2", "apple_red_3", "apple_red_delicios_1", "apple_red_yellow_1", "apple_rooten_1"
]  # Update with your actual class names

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet50(weights=None)  # Updated from `pretrained=False`
model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# Transformation for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to detect and estimate distance
def detect_and_estimate(image, focal_length=615, real_width=7.0):
    """
    Detect apples in an image, estimate their distance, and count the number of apples.
    :param image: Input image in BGR format.
    :param focal_length: Camera focal length in pixels.
    :param real_width: Real-world width of the apple in cm.
    """
    # Convert BGR image to RGB and then to PIL Image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    # Transform the image
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
    apple_count = 0  # Initialize apple count

    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter noise
            apple_count += 1  # Increment apple count
            x, y, w, h = cv2.boundingRect(contour)
            distance = (real_width * focal_length) / w  # Distance formula
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, f"{class_name}, {distance:.2f} cm", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the apple count
    cv2.putText(image, f"Apples detected: {apple_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    return image

# Real-time detection
cap = cv2.VideoCapture(0)  # Change '0' to a video file path if needed
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    output_frame = detect_and_estimate(frame)
    cv2.imshow("Apple Detection and Distance Estimation", output_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
