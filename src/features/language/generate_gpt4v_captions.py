"""Generate GPT-4V image descriptions for the THINGS-EEG2 image set.

Each image in the stimulus set is described by GPT-4V (``gpt-4-vision-preview``)
with the prompt "Describe the image simply.". The descriptions are the raw
material for the language stream of the fusion encoding model: they are later
embedded with a text-embedding model (see ``extract_text_embedding_3_large.py``).

The images are read from::

    <project_dir>/image_set/{training_images,test_images}/<concept>/<image>.jpg

and the descriptions are written, one row per image, to a CSV with columns
``group`` (training/test), ``category`` (THINGS concept folder), ``image``
(file name) and ``feature`` (the GPT-4V description). The CSV is re-saved after
every successful call so the run can be resumed after an interruption.

The OpenAI API key is read from the ``OPENAI_API_KEY`` environment variable.

Example
-------
    export OPENAI_API_KEY=...
    python generate_gpt4v_captions.py --project_dir /path/to/encoding/data
"""

import argparse
import base64
import os
import os.path as op
import time

import pandas as pd
import requests

PROMPT = "Describe the image simply."
MODEL = "gpt-4-vision-preview"
API_URL = "https://api.openai.com/v1/chat/completions"
MAX_TOKENS = 1000


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project_dir', default='/scratch/byrong/encoding/data',
                        type=str, help='Project directory holding image_set/.')
    parser.add_argument('--out_csv', default=None, type=str,
                        help='Output CSV path (default: '
                             '<project_dir>/gpt4_features/gpt4_descriptions.csv).')
    parser.add_argument('--request_delay', default=2.0, type=float,
                        help='Seconds to wait between API calls (rate limiting).')
    return parser.parse_args()


def encode_image(image_path):
    """Return the base64-encoded contents of an image file."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_description(image_path, api_key, request_delay):
    """Query GPT-4V for a description of a single image, with one retry."""
    base64_image = encode_image(image_path)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
        "max_tokens": MAX_TOKENS,
    }

    response = requests.post(API_URL, headers=headers, json=payload).json()
    time.sleep(request_delay)
    # Retry once on an error response.
    if 'error' in response:
        print('  error response, retrying ...')
        time.sleep(request_delay)
        response = requests.post(API_URL, headers=headers, json=payload).json()
    return response['choices'][0]['message']['content']


def build_image_table(image_set_dir):
    """Walk the stimulus set and return a dataframe of all .jpg images."""
    groups, categories, images, features = [], [], [], []
    for group in ['training_images', 'test_images']:
        group_dir = op.join(image_set_dir, group)
        for concept in sorted(os.listdir(group_dir)):
            if concept.startswith('.'):
                continue
            for image in sorted(os.listdir(op.join(group_dir, concept))):
                if image.endswith('.jpg'):
                    groups.append(group)
                    categories.append(concept)
                    images.append(image)
                    features.append('')
    return pd.DataFrame({'group': groups, 'category': categories,
                         'image': images, 'feature': features})


def main():
    args = parse_args()
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise SystemExit('Set the OPENAI_API_KEY environment variable.')

    image_set_dir = op.join(args.project_dir, 'image_set')
    out_csv = args.out_csv or op.join(args.project_dir, 'gpt4_features',
                                      'gpt4_descriptions.csv')
    os.makedirs(op.dirname(out_csv), exist_ok=True)

    # Resume from an existing CSV if present, otherwise build a fresh table.
    if op.exists(out_csv):
        df = pd.read_csv(out_csv).fillna('')
        print(f'Resuming from {out_csv} ({len(df)} rows).')
    else:
        df = build_image_table(image_set_dir)
        print(f'Built image table with {len(df)} images.')

    for i in range(len(df)):
        if df.at[i, 'feature'] != '':
            continue  # already described
        image_path = op.join(image_set_dir, df.at[i, 'group'],
                             df.at[i, 'category'], df.at[i, 'image'])
        print(f'[{i + 1}/{len(df)}] {df.at[i, "image"]}')
        df.at[i, 'feature'] = get_description(image_path, api_key,
                                              args.request_delay)
        df.to_csv(out_csv, index=False)  # checkpoint after every call

    print(f'Done. Descriptions saved to {out_csv}')


if __name__ == '__main__':
    main()
