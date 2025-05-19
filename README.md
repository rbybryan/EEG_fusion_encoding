# EEG_fusion_encoding

A small collection of Python scripts for:
- Performing ridge regression (single model and fusion) with grid‐search  
- Computing and plotting EEG/model correlations  


---

## 📂 Repository Structure

.
├── README.md
├── LICENSE
├── requirements.txta
├── src/
│ ├── ridge_regression_utils.py
│ ├── correlation.py
│ └── encoding_model.py
├── tests/
│ ├── test_correlation.py
│ └── test_encoding_model.py
└── examples/
└── run_encoding_model.sh


## 🛠 Usage
1. Encoding model

python src/encoding_model.py \
  --sub 4 \
  --vision_model cornet_s \
  --language_model text_embedding_large \
  --fusion \
  --project_dir /scratch/byrong/encoding/data \
  --tag v1 \
  --fusion 


2. Correlation plotting

python src/correlation.py \
  --sub 4 \
  --project_dir /scratch/byrong/encoding/data \
  --data_path_bio /scratch/byrong/encoding/data/eeg_dataset/preprocessed_eeg_data_v1 \
  --file cornet_s_with_text_embedding_large_r2_v1.npy
