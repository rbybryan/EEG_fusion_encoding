# Analysis

Statistical analyses that operate on the trained encoding-model predictions and
the extracted features. Each script writes its summary tables / arrays under
`analysis/` (created at run time).

## Paths

Scripts resolve the repository root automatically and default the data location
to `<repo>/data`. Override the data location with the `EEG_FUSION_DATA`
environment variable:

```bash
export EEG_FUSION_DATA=/path/to/encoding/data
python src/analysis/partial_correlation.py --sub 1
```

## Scripts

### Correlation / partial correlation
| Script | Role |
| --- | --- |
| `partial_correlation.py` | Partial correlation between a predictor model's EEG prediction and recorded EEG, controlling for a second model. |
| `run_partial_corr.py` | Driver that defines the model/layer/subject sweep and launches the partial-correlation runs. |

### Representational similarity (RSA)
| Script | Role |
| --- | --- |
| `rsa.py` | Representational similarity analysis between model and EEG RDMs. |
| `similarity_dnn_llm.py` | DNN–LLM representational-similarity (RDM correlation) between the vision and language feature spaces. |
| `between_within_category_rsa.py` | Between- vs within-category RSA: tests whether the DNN–LLM overlap survives removal of coarse category structure. |

### Variance partitioning
| Script | Role |
| --- | --- |
| `variance_partitioning.py` | Unique (UV) and shared (SV) variance between **any** two models and their fusion, via the Borcard-Legendre three-way partition. Each R² is bias-corrected adjusted R² (`1 − (1 − R²)(n − 1)/(n − p − 1)`, p = 1), applied per split-half iteration so the shared-variance baseline sits at ~0. Pass `--pred_a`, `--pred_b`, `--pred_ab`. |

```bash
python src/analysis/variance_partitioning.py \
    --pred_a cornet_s_r2_v1 \
    --pred_b text_embedding_large_r2_v1 \
    --pred_ab cornet_s_with_text_embedding_large_r2_v1 \
    --tag cornet_tel
```

### Time-frequency
| Script | Role |
| --- | --- |
| `time_frequency_decomposition.py` | Morlet wavelet decomposition of the EEG into per-frequency complex / power / phase representations, for band-resolved analyses. |

### Shared utilities
| Script | Role |
| --- | --- |
| `cluster_stats.py` | Cluster-based permutation testing for 1-D time courses. |
