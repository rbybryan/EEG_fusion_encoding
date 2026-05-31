# EEG_fusion_encoding

Code for predicting EEG responses to natural images from **vision** and
**language** model features, and for combining them in an early-fusion ridge
**encoding model**. The pipeline covers the full path from the raw
THINGS-EEG2 dataset and stimulus images to encoding-ready features, the encoding
model itself, and the downstream statistical analyses.

The language stream describes each image with GPT-4V and embeds the descriptions
with a text-embedding model (headline: OpenAI `text-embedding-3-large`); the
vision stream extracts CORnet-S features (plus AlexNet / ResNet-50 / SOTA
controls). Both streams are PCA-reduced and entered into a per-time-point ridge
encoding model, which is then evaluated with correlation, partial correlation,
RSA, and variance partitioning.

---

## 🗂 Repository structure

```text
.
├── README.md
├── requirement.txt
├── examples/
│   └── run_and_test_encoding_models.sh   # end-to-end example
└── src/
    ├── preprocessing/                     # raw THINGS-EEG2 → preprocessed EEG
    │   ├── preprocess_things_eeg_2.py     # CSD + z-score pipeline (canonical)
    │   └── preprocessing_utils.py         # epoching / zscore / save
    ├── features/
    │   ├── vision/                        # image → DNN feature maps → PCA
    │   │   ├── extract_feature_maps_cornet_s.py   # headline vision model
    │   │   ├── sort_feature_maps_cornet_s.py
    │   │   ├── extract_feature_maps_alexnet.py    # control
    │   │   ├── extract_feature_maps_resnet50.py   # control
    │   │   ├── extract_feature_maps_sota.py       # timm SOTA controls
    │   │   ├── feature_maps_pca.py                # KernelPCA reduction
    │   │   └── feature_maps_pca_comb.py
    │   └── language/                      # image → GPT-4V caption → embedding
    │       ├── generate_gpt4v_captions.py
    │       ├── extract_text_embedding_3_large.py  # headline LLM
    │       ├── extract_embeddings.py              # open-source embedders
    │       ├── extract_e5_mistral_untrained_embeddings.py  # untrained control
    │       ├── build_glove_embeddings.py          # non-contextual baseline
    │       └── README.md
    ├── encoding/                          # ridge / fusion encoding model
    │   ├── encoding_model.py
    │   ├── correlation.py
    │   └── utils.py
    └── analysis/                          # statistics on the predictions
        ├── partial_correlation.py · run_partial_corr.py
        ├── rsa.py · similarity_dnn_llm.py · between_within_category_rsa.py
        ├── layerwise_shared_variance.py · crosslayer_shared_variance.py
        ├── conditional_shared_variance_{tel,mistral}.py
        ├── shared_variance_trained_vs_untrained.py · summarize_unique_r2.py
        ├── tel_pca_dimensionality.py · cluster_stats.py
        └── README.md
```

---

## 🚀 Installation

```bash
git clone https://github.com/rbybryan/EEG_fusion_encoding.git
cd EEG_fusion_encoding
python3 -m venv venv && source venv/bin/activate   # optional
pip install -r requirement.txt
```

CORnet is installed from source:

```bash
pip install git+https://github.com/dicarlolab/CORnet.git
```

The OpenAI-based language scripts read the API key from the `OPENAI_API_KEY`
environment variable; no key is stored in this repository.

---

## 🔧 Pipeline

Most scripts take `--project_dir` (data location) and, in the analysis layer,
honor the `EEG_FUSION_DATA` environment variable for the data root.

### 1. Preprocess the EEG

```bash
python src/preprocessing/preprocess_things_eeg_2.py --sub 1 --project_dir $PROJECT_DIR
```

### 2. Extract features

Vision (CORnet-S) and dimensionality reduction:

```bash
python src/features/vision/extract_feature_maps_cornet_s.py --project_dir $PROJECT_DIR
python src/features/vision/sort_feature_maps_cornet_s.py    --project_dir $PROJECT_DIR
python src/features/vision/feature_maps_pca.py --dnn cornet_s --layers all --n_components 1000 --project_dir $PROJECT_DIR
```

Language (GPT-4V captions → text-embedding-3-large):

```bash
export OPENAI_API_KEY=...
python src/features/language/generate_gpt4v_captions.py        --project_dir $PROJECT_DIR
python src/features/language/extract_text_embedding_3_large.py --project_dir $PROJECT_DIR
```

### 3. Train the encoding model

```bash
python src/encoding/encoding_model.py \
  --sub 4 \
  --vision_model cornet_s \
  --language_model text_embedding_large \
  --fusion \
  --project_dir $PROJECT_DIR \
  --tag v1
```

| Flag | Meaning |
| --- | --- |
| `--sub` | Subject ID (integer) |
| `--vision_model` | Vision model name (e.g. `cornet_s`) |
| `--language_model` | Language model name (e.g. `text_embedding_large`) |
| `--fusion` | Enable vision + language fusion |
| `--vision_pretrained` | Use trained (vs randomly-initialised) vision features |
| `--pca_dir` | Directory of PCA-reduced features |
| `--project_dir` | Data / results directory |
| `--tag` | Run identifier |

Results: `$PROJECT_DIR/linear_results/sub-<sub>/synthetic_eeg_data/<vision_model>_with_<language_model>_r2_<tag>_all.npy`

### 4. Correlate predictions with recorded EEG

```bash
python src/encoding/correlation.py \
  --sub 4 \
  --project_dir $PROJECT_DIR \
  --data_path_bio $PROJECT_DIR/eeg_dataset/preprocessed_eeg_data_v1 \
  --file cornet_s_with_text_embedding_large_r2_v1
```

### 5. Analyses

See [`src/analysis/README.md`](src/analysis/README.md) for partial correlation,
RSA, variance partitioning, and dimensionality analyses, e.g.:

```bash
export EEG_FUSION_DATA=$PROJECT_DIR
python src/analysis/partial_correlation.py --sub 1
python src/analysis/summarize_unique_r2.py
```

---

## 📚 Data

The pipeline expects the [THINGS-EEG2](https://osf.io/3jk45/) dataset and its
stimulus image set under `--project_dir`. See the per-stage READMEs for the
exact input/output file layout.
