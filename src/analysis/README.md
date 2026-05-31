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
| `run_partial_corr.py` | Driver that defines the model/layer/subject sweep (LAYERS, SUBJECTS, TRAINED_VS_UNTRAINED) and launches the partial-correlation runs. |

### Representational similarity (RSA)
| Script | Role |
| --- | --- |
| `rsa.py` | Representational similarity analysis between model and EEG RDMs. |
| `similarity_dnn_llm.py` | DNN–LLM representational-similarity (RDM correlation) between the vision and language feature spaces. |
| `between_within_category_rsa.py` | Between- vs within-category RSA: tests whether the DNN–LLM representational overlap survives removal of coarse category structure. |

### Variance partitioning (shared / unique R²)
| Script | Role |
| --- | --- |
| `layerwise_shared_variance.py` | Layer-wise shared variance (DNN ∩ LLM). |
| `crosslayer_shared_variance.py` | Cross-layer shared variance for the manuscript PC-1000 baseline. |
| `conditional_shared_variance_tel.py` | Conditional shared variance, trained CORnet-S vs text-embedding-3-large. |
| `conditional_shared_variance_mistral.py` | Conditional shared variance, trained CORnet-S vs e5-Mistral. |
| `shared_variance_trained_vs_untrained.py` | Shared variance between the trained LLM and trained vs untrained DCNN, isolating learned (convergent) overlap. |
| `summarize_unique_r2.py` | True nested-model ΔR² (full − reduced) for the variance-partitioning contrasts. |

### Dimensionality
| Script | Role |
| --- | --- |
| `tel_pca_dimensionality.py` | PCA dimensionality and variance-preserved report for the text-embedding-3-large features. |

### Shared utilities
| Script | Role |
| --- | --- |
| `cluster_stats.py` | Cluster-based permutation testing for 1-D time courses (imported by the analyses above). |
