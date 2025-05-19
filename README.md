# EEG_fusion_encoding

A small collection of Python scripts for:
- Performing ridge regression (single model and fusion) with grid‐search  
- Computing and plotting EEG/model correlations  



- Encoding models (ridge regression) with grid-search hyperparameter tuning
- Fusion models to combine vision and language features
- Correlation analysis between EEG signals and model predictions

---

Features

- Ridge Regression Utilities:
  - Train single-source or fusion ridge regression models
  - Automatic grid-search over regularization strengths
  - Save and load trained models and performance metrics

- Correlation Analysis:
  - Compute Pearson correlations between EEG recordings and model outputs
  - Visualize results with publication-ready plots

- Modular Codebase:
  - Separate modules for encoding, correlation, and data handling
  - Easy to extend to new models or datasets

---

Installation

1. Clone the repository

   git clone https://github.com/yourusername/eeg-encoding-regression.git
   cd eeg-encoding-regression

2. Create and activate a virtual environment (recommended)

   python3 -m venv venv
   source venv/bin/activate    # Linux/macOS
   venv\Scripts\activate     # Windows

3. Install dependencies

   pip install -r requirements.txt

---

Repository Structure

.
├── README.md
├── LICENSE
├── requirements.txt          # project dependencies
├── src/
│   ├── encoding_model.py     # train encoding/fusion ridge models
│   ├── ridge_regression_utils.py  # core regression functions
│   └── correlation.py        # compute and plot EEG/model correlations
├── tests/
│   ├── test_encoding_model.py
│   └── test_correlation.py
└── examples/
    └── run_encoding_model.sh  # example usage script

---

Usage

1. Train an Encoding (or Fusion) Model

   python src/encoding_model.py \
     --sub 4 \
     --vision_model cornet_s \
     --language_model text_embedding_large \
     --fusion \
     --project_dir /scratch/byrong/encoding/data \
     --tag v1

   - --sub: Subject ID (integer)
   - --vision_model: Name of the vision model (e.g., cornet_s)
   - --language_model: Name of the language model (e.g., text_embedding_large)
   - --fusion: Include to enable feature fusion (vision + language)
   - --project_dir: Directory for input data and output results
   - --tag: Identifier for run/version

   Model outputs (R² scores, best alpha, etc.) will be saved under:
   $PROJECT_DIR/results/sub-<sub>/encoding/<vision_model>[_<language_model>]_r2_<tag>.npy

2. Compute and Plot Correlations

   python src/correlation.py \
     --sub 4 \
     --project_dir /scratch/byrong/encoding/data \
     --data_path_bio /scratch/byrong/encoding/data/eeg_dataset/preprocessed_eeg_data_v1 \
     --file cornet_s_with_text_embedding_large_r2_v1.npy

   - --data_path_bio: Path to preprocessed EEG data
   - --file: Encoding results file (R² scores) to correlate and plot

   This will generate time-resolved correlation plots saved in:
   $PROJECT_DIR/figures/sub-<sub>/correlations_<model>_<tag>.png

