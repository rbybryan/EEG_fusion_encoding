#!/bin/bash
# End-to-end example: train the fast batched encoding model and correlate its
# predictions with the recorded EEG.
#
# Set the data location once (defaults to ./data if unset):
#   export EEG_FUSION_DATA=/path/to/encoding/data
PROJECT_DIR="${EEG_FUSION_DATA:-data}"

set -e

# (optional) confirm the fast batched solver matches the per-cell oracle exactly
python src/encoding/validate_fast.py

# Train the vision + language fusion encoding model.
# --score_dtype float64 reproduces the published selection exactly;
# float32 is ~1.7x faster with identical selection in our tests.
python src/encoding/encoding_model.py \
  --sub 4 \
  --vision_model cornet_s \
  --language_model text_embedding_large \
  --fusion \
  --project_dir "$PROJECT_DIR" \
  --score_dtype float64 \
  --tag v1

# Correlate the saved fusion predictions with the recorded test EEG.
python src/encoding/correlation.py \
  --sub 4 \
  --project_dir "$PROJECT_DIR" \
  --data_path_bio "$PROJECT_DIR/eeg_dataset/preprocessed_eeg_data_v1" \
  --file cornet_s_with_text_embedding_large_r2_v1_all
