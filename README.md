
# Image Segmentation and Blending

This repository contains Python code that performs image segmentation using a pre-trained BiRefNet model, applies a color overlay to the segmented areas, and saves the resulting image. The system takes an input image, processes it through a deep learning segmentation model, blends the predicted mask with a color overlay, and saves the final image.

## Prerequisites

Before running the code, make sure you have the necessary dependencies installed. You can install them by running:


    pip install -r requirements.txt


### Requirements

- Python 3.7+
- PyTorch
- torchvision
- Pillow (PIL)
- transformers

## Project Files

### `blending.py`

Contains the function `postprocess_prediction` which applies a color overlay to the segmented region of the image. It blends the overlay using the predicted mask and saves the final result.

#### Key function:
- `postprocess_prediction(pred, original_image, overlay_color, output_path)`: Applies the overlay to the segmented areas and saves the final image.

### `image_processing.py`

Handles loading and preprocessing the image for input into the BiRefNet model.

#### Key functions:
- `load_image(image_path)`: Loads an image from the specified path and converts it to RGB.
- `preprocess_image(image)`: Preprocesses the image by resizing, normalizing, and converting it to a tensor.

### `model.py`

Responsible for loading the pre-trained BiRefNet segmentation model from the Hugging Face model hub.

#### Key function:
- `load_model(device='cpu')`: Loads the BiRefNet model, sets it to evaluation mode, and transfers it to the specified device (either `'cpu'` or `'cuda'`).


      import torch
      from transformers import AutoModelForImageSegmentation
      
      def load_model(device='cpu'):
          """Loads the BiRefNet model."""
          birefnet = AutoModelForImageSegmentation.from_pretrained('zhengpeng7/BiRefNet', trust_remote_code=True)
          birefnet.to(device)
          birefnet.eval()
          return birefnet


### `main.py`

This is the main script that combines all the functions to load the image, preprocess it, run the model, apply the blending overlay, and save the output.

#### Main steps in `main()`:
1. Load the input image.
2. Preprocess the image.
3. Load the BiRefNet model.
4. Perform image segmentation and get the mask.
5. Apply a color overlay to the segmented regions.
6. Save and display the final blended image.

### Running the Script

To run the script, execute the `main.py` file:


    python main.py


Ensure that you have an input image (e.g., `black_3.jpg`) in the same directory. The final output will be saved as `output.jpg`.

## Model Setup

The BiRefNet model is automatically downloaded from the Hugging Face model hub during runtime. You don’t need to manually download the model weights. The `load_model` function in `model.py` uses `AutoModelForImageSegmentation.from_pretrained()` to fetch the pre-trained model.

## Customization

- **Overlay Color**: You can change the color of the overlay by modifying the `overlay_color` variable in `main.py`. The color should be in RGB format, e.g., `(130, 44, 50)`.
  
- **Image Path**: Modify the `image_path` variable in `main.py` to point to your desired input image file.

## Notes

- The BiRefNet model used in this project is designed for image segmentation and generates a binary mask.
- The blending is done by combining the original image with the color overlay based on the segmentation mask.
- Images are resized to 224x224 pixels for input to the model, but you can adjust this by modifying the image preprocessing in `image_processing.py`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
