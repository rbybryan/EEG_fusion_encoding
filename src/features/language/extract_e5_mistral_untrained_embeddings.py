import argparse
import os
import os.path as op

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract random-weight e5-mistral embeddings for GPT-4 text features."
    )
    parser.add_argument("--project_dir", default="/scratch/byrong/encoding/data", type=str)
    parser.add_argument("--model_name", default="intfloat/e5-mistral-7b-instruct", type=str)
    parser.add_argument(
        "--output_name",
        default="e5-mistral-7b-instruct_untrained_cleaned",
        type=str,
        help="Suffix used in gpt4_features_embedded_by_<output_name>_pca.npy",
    )
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--max_length", default=4096, type=int)
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype on GPU. CPU always uses float32.",
    )
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def collect_texts(project_dir: str):
    df = pd.read_csv(op.join(project_dir, "gpt4_features", "gpt4_5v_cleaned.csv")).fillna("")

    train_dir = op.join(project_dir, "image_set", "training_images")
    test_dir = op.join(project_dir, "image_set", "test_images")

    train_images = []
    for root, _, files in os.walk(train_dir):
        for file_name in files:
            if file_name.endswith(".jpg"):
                train_images.append(op.join(root, file_name))
    train_images.sort()

    test_images = []
    for root, _, files in os.walk(test_dir):
        for file_name in files:
            if file_name.endswith(".jpg"):
                test_images.append(op.join(root, file_name))
    test_images.sort()

    train_by_view = {}
    test_by_view = {}
    train_index = []

    for key in range(1, 6):
        view_name = f"v{key}"
        train_texts = []
        test_texts = []

        for i, image in enumerate(train_images):
            image_name = op.basename(image)
            value = df.loc[df.image == image_name, view_name].iloc[0]
            if value != "":
                train_texts.append(value)
                if key == 1:
                    train_index.append(i)

        for image in test_images:
            image_name = op.basename(image)
            value = df.loc[df.image == image_name, view_name].iloc[0]
            if value != "":
                test_texts.append(value)

        train_by_view[view_name] = train_texts
        test_by_view[view_name] = test_texts

    return train_by_view, test_by_view, train_index


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
    ]


def encode_texts(model, tokenizer, texts, batch_size, max_length, device):
    outputs = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        batch = tokenizer(
            batch_texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            model_output = model(**batch)
            pooled = last_token_pool(model_output.last_hidden_state, batch["attention_mask"])
            pooled = F.normalize(pooled, p=2, dim=1)
        outputs.append(pooled.detach().cpu().float().numpy())
    return np.concatenate(outputs, axis=0)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    requested_dtype = resolve_dtype(args.dtype)
    model_dtype = requested_dtype if device == "cuda" else torch.float32

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    train_by_view, test_by_view, train_index = collect_texts(args.project_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(args.model_name)
    model = AutoModel.from_config(config)
    model.eval()
    model.to(device=device, dtype=model_dtype)

    train_embeddings = []
    test_embeddings = []

    for view_name in sorted(train_by_view.keys()):
        train_texts = train_by_view[view_name]
        test_texts = test_by_view[view_name]
        print(f"Encoding {view_name}: train={len(train_texts)} test={len(test_texts)}")

        train_emb = encode_texts(
            model=model,
            tokenizer=tokenizer,
            texts=train_texts,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=device,
        )
        test_emb = encode_texts(
            model=model,
            tokenizer=tokenizer,
            texts=test_texts,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=device,
        )

        train_embeddings.append(train_emb)
        test_embeddings.append(test_emb)

        if device == "cuda":
            torch.cuda.empty_cache()

    train_embeddings = np.stack(train_embeddings, axis=0)
    test_embeddings = np.stack(test_embeddings, axis=0)

    averaged_train = train_embeddings.mean(axis=0).astype(np.float32)
    averaged_test = test_embeddings.mean(axis=0).astype(np.float32)

    save_dir = op.join(args.project_dir, "gpt4_features")
    os.makedirs(save_dir, exist_ok=True)
    output_path = op.join(
        save_dir,
        f"gpt4_features_embedded_by_{args.output_name}_pca.npy",
    )

    data_dict = {
        "embedding_train": averaged_train,
        "embedding_test": averaged_test,
        "train_index": train_index,
    }
    np.save(output_path, data_dict)

    print("Saved:", output_path)
    print("embedding_train", averaged_train.shape, averaged_train.dtype)
    print("embedding_test", averaged_test.shape, averaged_test.dtype)
    print("train_index", len(train_index))


if __name__ == "__main__":
    main()
