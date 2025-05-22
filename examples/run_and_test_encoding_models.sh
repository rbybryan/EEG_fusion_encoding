#!/bin/bash
cd src/

python src/encoding_model.py \
  --sub 4 \
  --vision_model cornet_s \
  --language_model text_embedding_large \
  --fusion \
  --project_dir $PROJECT_DIR \
  --tag v1


python src/correlation.py \
  --sub 4 \
  --project_dir /scratch/byrong/encoding/data \
  --data_path_bio /scratch/byrong/encoding/data/eeg_dataset/preprocessed_eeg_data_v1 \
  --file cornet_s_with_text_embedding_large_r2_v1
