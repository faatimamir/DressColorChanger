# blending.py

import numpy as np
from PIL import Image
from torchvision import transforms

def postprocess_prediction(pred, original_image, overlay_color, output_path):
    """Applies an overlay color to the segmented area and saves the result."""
    # Convert prediction mask to a PIL image and resize it
    pred_pil = transforms.ToPILImage()(pred)
    pred_pil = pred_pil.resize(original_image.size)

    # Convert images to numpy arrays for processing
    original_np = np.array(original_image)
    mask_np = np.array(pred_pil)

    # Create a color overlay with the same size as the original image
    overlay_np = np.full_like(original_np, overlay_color)

    # Normalize mask for blending
    mask_np = mask_np / 255.0

    # Blend the color overlay with the original image using the mask
    blended_np = (overlay_np * mask_np[..., None] * 0.6 + original_np * (1 - mask_np[..., None] * 0.6)).astype(np.uint8)

    # Convert back to a PIL image and save
    blended_image = Image.fromarray(blended_np)
    blended_image.save(output_path, "PNG")
    print(f"Saved the blended image with preserved texture at {output_path}")
    return blended_image
