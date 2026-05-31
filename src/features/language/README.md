# Language features

Text-feature extractors for the language stream of the fusion encoding model.
The pipeline first describes every stimulus image with GPT-4V, then embeds the
descriptions with a text-embedding model. Embedding files are aligned 1:1 with
the vision features and EEG data via the shared `train_index`.

## Pipeline

1. **Describe images** — `generate_gpt4v_captions.py` runs GPT-4V over the
   stimulus set and writes one description per image. (Five-version descriptions
   are cleaned into `gpt4_features/gpt4_5v_cleaned.csv`, the input to step 2.)
2. **Embed descriptions** — `extract_text_embedding_3_large.py` (headline) and
   the alternative / control extractors below.

## Scripts

| Script | Role | Output |
| --- | --- | --- |
| `generate_gpt4v_captions.py` | **Captioner** — GPT-4V (`gpt-4-vision-preview`) image descriptions of the THINGS-EEG2 stimuli. | `gpt4_features/gpt4_descriptions.csv` |
| `extract_text_embedding_3_large.py` | **Headline LLM** — OpenAI `text-embedding-3-large` (3072-d) over the 5-version descriptions, averaged per image. | `gpt4_features/gpt4_responses_embedded_5v_large_cleaned.npy` |
| `extract_embeddings.py` | Open-source contextual sentence embedders (e5, GTE, NV-Embed, etc.) via `sentence-transformers`. | `gpt4_features/gpt4_features_embedded_by_<model>_cleaned.npy` |
| `extract_e5_mistral_untrained_embeddings.py` | **Control** — randomly-initialised (untrained) `e5-mistral-7b-instruct`, to isolate the contribution of learned language structure. | `gpt4_features/gpt4_features_embedded_by_e5-mistral-7b-instruct_untrained_cleaned_pca.npy` |
| `build_glove_embeddings.py` | **Baseline** — GloVe (non-contextual, mean-of-word-vectors) sentence embeddings, to test whether the LLM advantage depends on contextual rather than lexical structure. | `gpt4_features/gpt4_features_embedded_by_glove_pca.npy` |

The dimensionality-reduction (PCA) step applied to the embeddings before the
encoding model is separate; the encoding model consumes the PCA-reduced version
of the headline file.

## OpenAI API key

`generate_gpt4v_captions.py` and `extract_text_embedding_3_large.py` call the
OpenAI API and read the key from the `OPENAI_API_KEY` environment variable — no
key is stored in the repository:

```bash
export OPENAI_API_KEY=...   # your key
python src/features/language/generate_gpt4v_captions.py --project_dir /path/to/encoding/data
python src/features/language/extract_text_embedding_3_large.py --project_dir /path/to/encoding/data
```
