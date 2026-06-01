"""Extract per-layer feature maps from state-of-the-art vision models via timm.

Saves raw spatial activation maps in the same format as the existing extraction
scripts (extract_feature_maps_alexnet.py, extract_feature_maps_cornet_s.py),
so they can be processed by the existing PCA pipeline (feature_maps_pca.py).

Output format per image:
    {stage_name: np.ndarray of shape (1, C, H, W), ..., 'decoder': (1, num_classes)}

Supported models:
  - convnext_xlarge     (ImageNet top-1 ~87.3%)
  - tf_efficientnetv2_l (ImageNet top-1 ~85.7%)

Usage
-----
python extract_feature_maps_sota.py --model convnext_xlarge --pretrained True
python extract_feature_maps_sota.py --model convnext_xlarge --pretrained False
python extract_feature_maps_sota.py --model tf_efficientnetv2_l --pretrained True
"""

import argparse
import os
import os.path as op
import numpy as np
import torch
from PIL import Image
import timm

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# =============================================================================
# Arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='convnext_xlarge',
                    choices=['convnext_xlarge', 'tf_efficientnetv2_l'],
                    help='Model name from timm')
parser.add_argument('--pretrained', default='True', type=str)
parser.add_argument('--project_dir', default=os.environ.get('EEG_FUSION_DATA', 'data'),
                    type=str)
args = parser.parse_args()

args.pretrained = args.pretrained.lower() != 'false'

print(f'>>> Extract feature maps: {args.model} <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print(f'{key:16} {val}')

# =============================================================================
# Reproducibility
# =============================================================================
seed = 20200220
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.use_deterministic_algorithms(True)

# =============================================================================
# Model setup
# =============================================================================
# features_only model: returns raw spatial feature maps at each stage
model_feat = timm.create_model(args.model, pretrained=args.pretrained,
                               features_only=True)
# full model: for the classifier (decoder) output
model_full = timm.create_model(args.model, pretrained=args.pretrained)

if torch.cuda.is_available():
    model_feat = model_feat.cuda()
    model_full = model_full.cuda()
model_feat.eval()
model_full.eval()

# Stage names from timm's feature_info
stage_info = model_feat.feature_info
stage_names = [info['module'] for info in stage_info]
print(f'Stages: {stage_names}')
for info in stage_info:
    print(f"  {info['module']}: {info['num_chs']}ch, "
          f"reduction={info['reduction']}x")

# =============================================================================
# Preprocessing — use the model-specific transform from timm
# =============================================================================
data_config = timm.data.resolve_model_data_config(model_full)
transform = timm.data.create_transform(**data_config, is_training=False)
print(f'Transform: {transform}')

# =============================================================================
# Extract features
# =============================================================================
img_set_dir = op.join(args.project_dir, 'image_set')
if not op.isdir(img_set_dir):
    raise FileNotFoundError(f"Image set directory not found: {img_set_dir}")

img_partitions = sorted(os.listdir(img_set_dir))

for p in img_partitions:
    part_dir = op.join(img_set_dir, p)
    image_list = []
    for root, dirs, files in os.walk(part_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg')):
                image_list.append(op.join(root, file))
    image_list.sort()

    save_dir = op.join(args.project_dir, 'dnn_feature_maps', 'full_feature_maps',
                       args.model, f'pretrained-{args.pretrained}', p)
    os.makedirs(save_dir, exist_ok=True)

    print(f"Processing partition '{p}' with {len(image_list)} images...")

    for i, image_path in enumerate(image_list):
        try:
            img = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            continue

        input_tensor = transform(img).unsqueeze(0)
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()

        with torch.no_grad():
            # Per-stage raw spatial feature maps (no pooling)
            stage_outputs = model_feat(input_tensor)

            feats = {}
            for name, out in zip(stage_names, stage_outputs):
                # Keep raw spatial activations with batch dim: (1, C, H, W)
                feats[name] = out.cpu().numpy()

            # Decoder: classifier output, analogous to AlexNet classifier[6]
            logits = model_full(input_tensor)
            feats['decoder'] = logits.cpu().numpy()  # shape (1, num_classes)

        file_name = f'{p}_{i + 1:07d}.npy'
        np.save(op.join(save_dir, file_name), feats)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(image_list)} images...")

    print(f"Completed partition '{p}'.\n")

print("Feature maps extraction completed.")
