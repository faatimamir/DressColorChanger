# model.py

import torch
from transformers import AutoModelForImageSegmentation

def load_model(device='cpu'):
    """Loads the BiRefNet model."""
    birefnet = AutoModelForImageSegmentation.from_pretrained('zhengpeng7/BiRefNet', trust_remote_code=True)
    birefnet.to(device)
    birefnet.eval()
    return birefnet




# To use model locally
# Clone the BiRefNet repository
# git clone https://github.com/ZhengPeng7/BiRefNet.git
# cd BiRefNet

# # Install required dependencies
# pip install -r requirements.txt

# Download Model Weights Locally

# Download the model weights (BiRefNet-general-epoch_244.pth) to your local machine.
# bash
# wget https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/BiRefNet-general-epoch_244.pth

# model.py

# import torch
# from models.birefnet import BiRefNet
# from utils import check_state_dict

# def load_model(device='cpu'):
#     """Load the BiRefNet model from local weights."""
#     # Initialize the BiRefNet model
#     birefnet = BiRefNet(bb_pretrained=False)

#     # Load weights
#     weights_path = "./model_weights/BiRefNet-general-epoch_244.pth"
#     state_dict = torch.load(weights_path, map_location=device)
#     state_dict = check_state_dict(state_dict)
#     birefnet.load_state_dict(state_dict)

#     birefnet.to(device)
#     birefnet.eval()
#     return birefnet
