#!/usr/bin/env python
"""Extract AlexNet feature maps for a set of images.

This script loads AlexNet (optionally with pretrained weights), registers
forward hooks on selected convolutional layers and the classifier output,
and runs every image found under ``[project_dir]/image_set`` through the
network. The captured feature maps are saved as ``.npy`` files, one per
image, under ``[project_dir]/dnn_feature_maps/full_feature_maps/alexnet``.
"""

import argparse
import os
import os.path as op

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable as V  # (Optional: In modern PyTorch, tensors track gradients by default)
from torchvision import transforms as trn
from torchvision import models
from PIL import Image

# -----------------------------------------------------------------------------
# Set the environment variable for deterministic behavior
# -----------------------------------------------------------------------------
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# -----------------------------------------------------------------------------
# Input arguments
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--sub', type=int, default=1,
                    help='Subject identifier.')
parser.add_argument('--pretrained', default='True', type=str,
                    help="Whether to load pretrained weights ('True'/'False').")
parser.add_argument('--project_dir', default=os.environ.get('EEG_FUSION_DATA', 'data'),
                    type=str,
                    help='Root directory containing the image set.')
args = parser.parse_args()

args.pretrained = args.pretrained.lower() != 'false'

print('>>> Extract feature maps from AlexNet <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# -----------------------------------------------------------------------------
# Set random seed for reproducible results
# -----------------------------------------------------------------------------
seed = 20200220
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.use_deterministic_algorithms(True)

# -----------------------------------------------------------------------------
# Define image preprocessing (centre crop, resize, normalization)
# -----------------------------------------------------------------------------
centre_crop = trn.Compose([
    trn.Resize((224, 224)),
    trn.ToTensor(),
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# -----------------------------------------------------------------------------
# Function to load AlexNet (with optional pretrained weights)
# -----------------------------------------------------------------------------
def get_model(pretrained=args.pretrained):
    """Load AlexNet, move it to GPU if available, and set it to eval mode."""
    model = models.alexnet(pretrained=pretrained)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()  # Set to evaluation mode
    return model


# -----------------------------------------------------------------------------
# Define a global dictionary to hold feature maps for a given forward pass.
# -----------------------------------------------------------------------------
feature_maps = {}


def save_feature_maps(layer_name):
    """Create a forward hook that saves the output of a layer."""
    def hook(module, input, output):
        # If output is a tuple (some layers return tuples), take the first element.
        if isinstance(output, tuple):
            output = output[0]
        # Save the numpy array (detach to avoid tracking gradients).
        feature_maps[layer_name] = output.detach().cpu().numpy()
    return hook


# -----------------------------------------------------------------------------
# Instantiate the model and register forward hooks on selected layers
# -----------------------------------------------------------------------------
model = get_model(pretrained=args.pretrained)

# In case the model is wrapped in DataParallel, get the underlying module.
if isinstance(model, nn.DataParallel):
    model = model.module

# Register hooks on layers of interest.
# AlexNet architecture:
#   model.features: [Conv2d, ReLU, MaxPool2d, Conv2d, ReLU, MaxPool2d,
#                    Conv2d, ReLU, Conv2d, ReLU, Conv2d, ReLU, MaxPool2d]
#   model.classifier: [Dropout, Linear, ReLU, Dropout, Linear, ReLU, Linear]
model.features[0].register_forward_hook(save_feature_maps('conv1'))
model.features[3].register_forward_hook(save_feature_maps('conv2'))
model.features[6].register_forward_hook(save_feature_maps('conv3'))
model.features[8].register_forward_hook(save_feature_maps('conv4'))
model.features[10].register_forward_hook(save_feature_maps('conv5'))
model.classifier[6].register_forward_hook(save_feature_maps('decoder'))

# -----------------------------------------------------------------------------
# Process images partitioned into directories under [project_dir]/image_set
# and save feature maps for each image.
# -----------------------------------------------------------------------------
img_set_dir = os.path.join(args.project_dir, 'image_set')
if not op.isdir(img_set_dir):
    raise FileNotFoundError(f"Image set directory not found: {img_set_dir}")

# Each sub-directory is treated as a partition.
img_partitions = sorted(os.listdir(img_set_dir))

for p in img_partitions:
    part_dir = os.path.join(img_set_dir, p)
    # Gather image files (jpg or JPEG)
    image_list = []
    for root, dirs, files in os.walk(part_dir):
        for file in files:
            if file.lower().endswith(".jpg") or file.lower().endswith(".jpeg"):
                image_list.append(os.path.join(root, file))
    image_list.sort()

    # Create the saving directory if it doesn't exist.
    save_dir = os.path.join(args.project_dir, 'dnn_feature_maps', 'full_feature_maps',
                            'alexnet', 'pretrained-' + str(args.pretrained), p)
    if not op.isdir(save_dir):
        os.makedirs(save_dir)

    print(f"Processing partition '{p}' with {len(image_list)} images...")

    # Process each image in the partition.
    for i, image_path in enumerate(image_list):
        try:
            img = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue

        # Preprocess the image.
        input_tensor = centre_crop(img).unsqueeze(0)  # add batch dimension
        # Optionally wrap in Variable (not required in recent PyTorch versions)
        input_img = V(input_tensor)
        if torch.cuda.is_available():
            input_img = input_img.cuda()

        # Clear the global feature_maps dictionary before each forward pass.
        feature_maps.clear()

        # Forward pass (hooks will capture outputs).
        with torch.no_grad():
            _ = model(input_img)

        # Make a deep copy of the captured feature maps.
        feats = {layer: np.copy(fmap) for layer, fmap in feature_maps.items()}

        # Define a file name using the partition name and image index.
        file_name = p + '_' + format(i + 1, '07') + '.npy'
        save_path = op.join(save_dir, file_name)

        # Save the features as a numpy file.
        np.save(save_path, feats)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(image_list)} images...")

    print(f"Completed processing partition '{p}'.\n")

print("Feature maps extraction completed.")
