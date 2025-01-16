import torch
from image_processing import load_image, preprocess_image
from model import load_model
from blending import postprocess_prediction

import os

# Get the directory where the script is located
folder_path = os.path.dirname(os.path.abspath(__file__))

def main():
    # Get the folder path from the .env file
    
    
    # Construct specific file paths using the folder path
    image_path = os.path.join(folder_path, "black_3.jpg")
    output_path = os.path.join(folder_path, "output.jpg")
    
    overlay_color = (130, 44, 50)  # Desired color overlay
    device = 'cpu'

    # Load and preprocess the image
    image = load_image(image_path)
    input_image = preprocess_image(image).to(device)
    
    # Load model and make prediction
    birefnet = load_model(device)
    with torch.no_grad():
        preds = birefnet(input_image)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    
    # Blend and save the final output
    blended_image = postprocess_prediction(pred, image, overlay_color, output_path)
    
    # Display result (optional)
    blended_image.show()

if __name__ == "__main__":
    main()
