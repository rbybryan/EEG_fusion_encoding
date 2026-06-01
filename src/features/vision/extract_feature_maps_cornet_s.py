"""Extract feature maps from images using the CORnet-S DNN model.

This script loads the CORnet-S network (optionally pretrained), registers
forward hooks on its V1, V2, V4, IT, and decoder stages, and saves the
extracted feature maps for every image found in each partition of the image
set directory.
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable as V
from torchvision import transforms as trn

import cornet


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--sub', type=int, default=1,
                    help='Subject identifier.')
parser.add_argument('--pretrained', default='True', type=str,
                    help="Whether to load pretrained weights ('True'/'False').")
parser.add_argument('--project_dir', default=os.environ.get('EEG_FUSION_DATA', 'data'), type=str,
                    help='Root directory containing the image set and outputs.')
args = parser.parse_args()

args.pretrained = args.pretrained.lower() != 'false'

print('>>> Extract feature maps CORnet-S <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220


# =============================================================================
# Import the model
# =============================================================================
def get_model(pretrained=args.pretrained):
    """Build the CORnet-S model and move it to GPU when available."""
    map_location = None if torch.cuda.is_available() else 'cpu'
    model = cornet.cornet_s(pretrained=pretrained, map_location=map_location)
    if torch.cuda.is_available():
        model = model.cuda()
    return model


# =============================================================================
# Define the image preprocessing
# =============================================================================
centre_crop = trn.Compose([
    trn.Resize((224, 224)),
    trn.ToTensor(),
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# =============================================================================
# Load the images and extract the corresponding feature maps
# =============================================================================
# Extracting the feature maps of (1) training images, (2) test images,
# (3) ILSVRC-2012 validation images, (4) ILSVRC-2012 test images.

# Image directories
img_set_dir = os.path.join(args.project_dir, 'image_set')
img_partitions = os.listdir(img_set_dir)

# Define a hook to save the feature maps
feature_maps = {}


def save_feature_maps(layer_name):
    """Return a forward hook that stores the layer output in ``feature_maps``."""
    def hook(module, input, output):
        feature_maps[layer_name] = output.detach().cpu().numpy()
    return hook


# Register hooks to the layers
model = get_model(pretrained=args.pretrained)

# Access the underlying module if DataParallel is used
if isinstance(model, nn.DataParallel):
    model = model.module

model.V1.register_forward_hook(save_feature_maps('V1'))
model.V2.register_forward_hook(save_feature_maps('V2'))
model.V4.register_forward_hook(save_feature_maps('V4'))
model.IT.register_forward_hook(save_feature_maps('IT'))
model.decoder.register_forward_hook(save_feature_maps('decoder'))

for p in img_partitions:
    part_dir = os.path.join(img_set_dir, p)
    image_list = []
    for root, dirs, files in os.walk(part_dir):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".JPEG"):
                image_list.append(os.path.join(root, file))
    image_list.sort()
    # Create the saving directory if not existing
    save_dir = os.path.join(args.project_dir, 'dnn_feature_maps',
                            'full_feature_maps', 'cornet_s',
                            'pretrained-' + str(args.pretrained), p)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Extract and save the feature maps
    for i, image in enumerate(image_list):
        img = Image.open(image).convert('RGB')
        input_img = V(centre_crop(img).unsqueeze(0))
        if torch.cuda.is_available():
            input_img = input_img.cuda()
        with torch.no_grad():
            _ = model(input_img)
        file_name = p + '_' + format(i + 1, '07')
        np.save(os.path.join(save_dir, file_name), feature_maps)
