"""
Extract contextual embeddings for video captions using multiple embedding models,
optimized for lower memory usage.

Supported models:
    - Linq-Embed-Mistral
    - gte-Qwen2-7B-instruct
    - multilingual-e5-large-instruct
    - SFR-Embedding-Mistral
    - GritLM/GritLM-7B
    - GritLM/GritLM-8x7B
    - e5-mistral-7b-instruct
"""

import argparse
import os
import os.path as op

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# Set environment variable to help avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# =============================================================================
# Input arguments; select output directory and encoding batch size
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--project_dir', default=os.environ.get('EEG_FUSION_DATA', 'data'), type=str,
                    help="Root directory containing the input CSV and output folders.")
parser.add_argument('--embedding_model', nargs='+', type=str,
                    help="One or more embedding model names. E.g., "
                         "'Linq-AI-Research/Linq-Embed-Mistral', "
                         "'Alibaba-NLP/gte-Qwen2-7B-instruct', etc.")
parser.add_argument('--encode_batch_size', default=16, type=int,
                    help="Batch size for encoding texts (default: 1 to minimize memory usage)")
args = parser.parse_args()

model_names = [
    # "Linq-AI-Research/Linq-Embed-Mistral",
    # "Alibaba-NLP/gte-Qwen2-7B-instruct",
    # # "intfloat/multilingual-e5-large-instruct",
    # "Salesforce/SFR-Embedding-Mistral",
    # "GritLM/GritLM-7B",
    # "GritLM/GritLM-8x7B",
    # "intfloat/e5-mistral-7b-instruct"

    # 'dunzhang/stella_en_1.5B_v5',                     # Stella model with 1.5B parameters, version v5
    # "Alibaba-NLP/gte-multilingual-base",             # A multilingual base model from the GTE series
    # "HIT-TMG/KaLM-embedding-multilingual-mini-v1",   # A compact multilingual embedding model from KaLM
    # 'infgrad/jasper_en_vision_language_v1',
    # 'Salesforce/SFR-Embedding-2_R',
    # 'Snowflake/snowflake-arctic-embed-l-v2.0',
    # 'sentence-transformers/LaBSE',
    # 'ibm-granite/granite-embedding-107m-multilingual',
    # 'WhereIsAI/UAE-Large-V1',
    # 'jinaai/jina-embeddings-v3'                  # Jina AI's embeddings model, version 3

    # "Alibaba-NLP/gte-Qwen2-7B-instruct",
    # "GritLM/GritLM-7B",
    # "nvidia/NV-Embed-v2",  # NVIDIA's advanced embedding model
    # "Salesforce/SFR-Embedding-Mistral",
    # "intfloat/e5-mistral-7b-instruct",
    # "OrdalieTech/Solon-embeddings-large-0.1",         # Solon-embeddings-large-0.1
    # 'OrdalieTech/Solon-embeddings-base-0.1',
    "infgrad/jasper_en_vision_language_v1",          # jasper_en_vision_language_v1
    # "BAAI/bge-m3",                                   # bge-m3 (update with the correct repo if needed)
    "voyager/voyage-3-lite",                         # voyage-3-lite
    "voyager/voyage-3",                              # voyage-3
    "Lajavaness/bilingual-embedding-large",            # bilingual-embedding-large (replace "your_org" with the actual namespace)
    "intfloat/multilingual-e5-large",                # multilingual-e5-large
    "intfloat/multilingual-e5-base",                  # multilingual-e5-base
    "mxbai/mxbai-embed-large-v1",           # mxbai-embed-large-v1
    "avsolatorio/GIST-large-Embedding-v0",         # GIST-large-Embedding-v0 (verify the namespace/repo)
    "intfloat/e5-large-v2",                 # e5-large-v2 (if available under intfloat or similar)
    "nomic-ai/nomic-embed-text-v1",         # nomic-embed-text-v1
    "nomic-ai/nomic-embed-text-v1-unsupervised",  # nomic-embed-text-v1-unsupervised
    "nomic-ai/nomic-embed-text-v1.5",       # nomic-embed-text-v1.5
    "shibing624/text2vec-base-multilingual",  # text2vec-base-multilingual
    "Mihaiii/Ivysaur"                       # Ivysaur (ensure the correct repository name)
    'avsolatorio/NoInstruct-small-Embedding-v0'
]

encode_batch_size = args.encode_batch_size

# =============================================================================
# Data preparation: Load CSV file and gather text responses
# =============================================================================

df = pd.read_csv(os.path.join(args.project_dir, 'gpt4_features', 'gpt4_5v_cleaned.csv'))
df = df.fillna('')

response = {}
response_test = {}
train_index = []
for key in range(1, 6):
    response[f'v{key}'], response_test[f'v{key}'] = [], []

    # Process training images
    img_train_dir = os.path.join(args.project_dir, 'image_set', 'training_images')
    train_images = []
    for root, dirs, files in os.walk(img_train_dir):
        for file in files:
            if file.endswith(".jpg"):
                train_images.append(os.path.join(root, file))
    train_images.sort()

    for i, image in enumerate(train_images):
        name = image.split('/')[-1]
        if df[df.image == name][f'v{key}'].iloc[0] != '':
            response[f'v{key}'].append(df[df.image == name].iloc[0][f'v{key}'])
            if key == 1:
                train_index.append(i)

    # Process test images
    img_test_dir = os.path.join(args.project_dir, 'image_set', 'test_images')
    test_images = []
    for root, dirs, files in os.walk(img_test_dir):
        for file in files:
            if file.endswith(".jpg"):
                test_images.append(os.path.join(root, file))
    test_images.sort()

    for image in test_images:
        name = image.split('/')[-1]
        if df[df.image == name][f'v{key}'].iloc[0] != '':
            response_test[f'v{key}'].append(df[df.image == name].iloc[0][f'v{key}'])

# =============================================================================
# Embedding extraction function with memory optimizations
# =============================================================================

def encode_embeddings(model, model_name, batch_size):
    """Encode train and test captions and save the embeddings to disk.

    Parameters
    ----------
    model : SentenceTransformer
        Loaded embedding model used to encode the caption texts.
    model_name : str
        Identifier of the model; used to build the output file name.
    batch_size : int
        Batch size passed to ``model.encode`` to control memory usage.
    """
    embeddings = []
    for key, texts in response.items():
        with torch.no_grad():
            # Encode texts with reduced batch size
            emb = model.encode(texts, batch_size=batch_size)
        embeddings.append(emb)
        print(f"{model_name} encoded {len(texts)} texts for key {key}")
        # Clear cache after each key to free memory fragments
        torch.cuda.empty_cache()

    embeddings = np.stack(embeddings)

    embeddings_test = []
    for key, texts in response_test.items():
        with torch.no_grad():
            emb = model.encode(texts, batch_size=batch_size)
        embeddings_test.append(emb)
        torch.cuda.empty_cache()
    embeddings_test = np.stack(embeddings_test)

    print("Train embeddings shape:", embeddings.shape, "Test embeddings shape:", embeddings_test.shape)

    # Save embeddings to file
    data_dict = {
        'embedding_train': embeddings,
        'embedding_test': embeddings_test,
        'train_index': train_index
    }

    save_dir = os.path.join(args.project_dir, 'gpt4_features')
    if not op.exists(save_dir):
        os.makedirs(save_dir)
    model_name = model_name.split('/')[-1]
    file_name = f'gpt4_features_embedded_by_{model_name}_cleaned.npy'
    np.save(os.path.join(save_dir, file_name), data_dict)
    print(f"Embeddings saved to {os.path.join(save_dir, file_name)}")

# =============================================================================
# Process each model with memory optimizations
# =============================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
for model_name in model_names:
    print(f"Processing model: {model_name}")
    try:
        # For models that require executing remote code, set trust_remote_code=True.
        if model_name in ["nvidia/NV-Embed-v2", "Alibaba-NLP/gte-multilingual-base",
        "jinaai/jina-embeddings-v3","GIST/GIST-large-Embedding-v0",'nomic-ai/nomic-embed-text-v1','nomic-ai/nomic-embed-text-v1-unsupervised',
        'nomic-ai/nomic-embed-text-v1.5','Lajavaness/bilingual-embedding-large',
        "Ivysaur/Ivysaur" ]:
            model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        elif model_name == "infgrad/jasper_en_vision_language_v1":
            model = SentenceTransformer(
                model_name,
                trust_remote_code=True,
                device="cpu" if not use_gpu else "cuda",
                model_kwargs={
                    "torch_dtype": torch.bfloat16 if use_gpu else torch.float32,
                    "attn_implementation": "sdpa"
                },
                # vector_dim must be 12288, 1024, 512, 256
                ## 1024 is recommended
                # set is_text_encoder 'True', if you do not encode image
                config_kwargs={"is_text_encoder": True, "vector_dim": 1024},)
        # elif model_name == 'avsolatorio/GIST-large-Embedding-v0':


        else:
            model = SentenceTransformer(model_name, device=device)

        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            encode_embeddings(model, model_name, encode_batch_size)

        # embeddings = np.array(embeddings)
        # print(f"Embeddings shape for {model_name}: {embeddings.shape}\n")
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error loading or encoding with model {model_name}: {e}\n")
        torch.cuda.empty_cache()

print("All embeddings processed!")
