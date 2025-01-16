# image_processing.py

from PIL import Image
import torch
from torchvision import transforms

def load_image(image_path):
    """Loads an image and converts it to RGB for processing by the model."""
    image = Image.open(image_path).convert("RGB")
    return image

def preprocess_image(image):
    """Transforms the image for the model."""
    transform_image = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_image = transform_image(image).unsqueeze(0)
    return input_image
