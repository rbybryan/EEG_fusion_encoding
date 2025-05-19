# EEG_fusion_encoding

A small collection of Python scripts for:
- **Encoding models** (ridge regression) with grid-search hyperparameter tuning
- **Fusion encoding models** to combine vision and language features
- **Correlation analysis** between EEG signals and model predictions

---

## 📦 Features

- **Ridge Regression Utilities**:
  - Train single-source or fusion ridge regression models
  - Automatic grid-search over regularization strengths
  - Save and load trained models and performance metrics

- **Correlation Analysis**:
  - Compute Pearson correlations between EEG recordings and model outputs
  - Visualize results with publication-ready plots

---

## 🚀 Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/rbybryan/EEG_fusion_encoding.git
   cd EEG_fusion_encoding
   ```

2. **Create and activate a virtual environment** (recommended)

   ```bash
   python3 -m venv venv
   source venv/bin/activate    # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## 🗂 Repository Structure

```text
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
```

---

## 🔧 Usage

### 1. Train an Encoding (or Fusion) Model

```bash
python src/encoding_model.py \
  --sub 4 \
  --vision_model cornet_s \
  --language_model text_embedding_large \
  --fusion \
  --project_dir $PROJECT_DIR \
  --tag v1
```

- `--sub`: Subject ID (integer)
- `--vision_model`: Name of the vision model (e.g., `cornet_s`)
- `--language_model`: Name of the language model (e.g., `text_embedding_large`)
- `--fusion`: Include to enable feature fusion (vision + language)
- `--project_dir`: Directory for input data and output results
- `--tag`: Identifier for run/version

Encoding results will be saved under:
```
$PROJECT_DIR/linear_results/sub-<sub>/synthetic_eeg_data/<vision_model>_with_<language_model>_r2_<tag>_all.npy
```

### 2. Compute and Plot Correlations

```bash
python src/correlation.py \
  --sub 4 \
  --project_dir /scratch/byrong/encoding/data \
  --data_path_bio /scratch/byrong/encoding/data/eeg_dataset/preprocessed_eeg_data_v1 \
  --file cornet_s_with_text_embedding_large_r2_v1
```

- `--data_path_bio`: Path to recorded EEG data
- `--file`: Results file of predicted EEG data to correlate

Correlation results will be saved under:
```
$PROJECT_DIR/linear_results/sub-<sub>/correlation/correlation_<file>.npy
```
