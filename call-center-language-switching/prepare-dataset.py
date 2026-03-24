#!/usr/bin/env python
"""
Dataset Preparation: Call Center Language Switching SLM
- Downloads 4 train datasets from HuggingFace
- Builds positive (<|im_end|>) and negative (no <|im_end|>) classes
- Tokenizes with Qwen3 chat template
- Multipacks into blocks and writes chinidataset Parquet
- Uploads to S3 bucket (aies-research-datasets)

Usage:
    python prepare-dataset.py
    python prepare-dataset.py --block_size 8192 --n_workers 20 --s3_bucket aies-research-datasets
"""

import os
import json
import random
import argparse
import subprocess
import numpy as np
from glob import glob
from tqdm import tqdm
from datasets import load_dataset
from collections import Counter, defaultdict
from transformers import AutoTokenizer
from chinidataset import ParquetWriter, StreamingDataset
from multiprocess import Pool
import itertools

random.seed(42)
np.random.seed(42)


# ---------------------------------------------------------------------------
# Converters – turn any HF row into unified {"messages": [...], ...} format
# ---------------------------------------------------------------------------

def convert_messages_format(row):
    """OpenAI format: [{'role': 'system', 'content': '...'}, ...]"""
    if 'messages' in row:
        return row['messages']
    return None


def convert_sharegpt_format(row):
    """ShareGPT format: [{'from': 'human', 'value': '...'}, ...]"""
    role_map = {
        'system': 'system', 'human': 'user', 'gpt': 'assistant',
        'user': 'user', 'assistant': 'assistant',
    }
    if 'conversations' in row:
        messages = []
        for turn in row['conversations']:
            role = role_map.get(turn.get('from', turn.get('role', '')), 'user')
            content = turn.get('value', turn.get('content', ''))
            messages.append({'role': role, 'content': content})
        return messages
    return None


def convert_to_unified(row, source_name):
    messages = convert_messages_format(row)
    if messages is None:
        messages = convert_sharegpt_format(row)
    if messages is None:
        return None
    return {
        'messages': messages,
        'language': row.get('language', row.get('lang', 'unknown')),
        'source': source_name,
    }


# ---------------------------------------------------------------------------
# Chat template formatting + positive / negative class builders
# ---------------------------------------------------------------------------

def format_chat_template(messages):
    parts = []
    for msg in messages:
        parts.append(f'<|im_start|>{msg["role"]}\n{msg["content"]}<|im_end|>')
    return '\n'.join(parts)


def build_positive(messages):
    """Complete conversation ending with <|im_end|>."""
    return format_chat_template(messages)


def build_negative(messages):
    """Truncated conversation NOT ending with <|im_end|>."""
    text = format_chat_template(messages)
    if text.endswith('<|im_end|>'):
        text = text[:-len('<|im_end|>')]

    # 50% chance: further truncate within the last assistant response
    if random.random() < 0.5 and len(text) > 100:
        last_assistant_idx = text.rfind('<|im_start|>assistant\n')
        if last_assistant_idx >= 0:
            content_start = last_assistant_idx + len('<|im_start|>assistant\n')
            content = text[content_start:]
            if len(content) > 10:
                keep_ratio = random.uniform(0.2, 0.8)
                keep_len = max(5, int(len(content) * keep_ratio))
                text = text[:content_start + keep_len]

    return text.rstrip()


# ---------------------------------------------------------------------------
# Multipacking helpers
# ---------------------------------------------------------------------------

COLUMNS = {
    'input_ids': 'uint32[]',
    'position_ids': 'uint32[]',
    'attention_mask': 'uint32[]',
    'text': 'str',
}


def collator_pack(batch_ids, batch_position_ids):
    input_ids, position_ids, masks = [], [], []
    for i in range(len(batch_ids)):
        input_ids.extend(batch_ids[i])
        position_ids.extend(batch_position_ids[i])
        masks.append(len(batch_ids[i]))
    return {
        'input_ids': np.array(input_ids, dtype=np.uint32),
        'position_ids': np.array(position_ids, dtype=np.uint32),
        'attention_mask': np.array(masks, dtype=np.uint32),
        'text': '',
    }


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield (lst[i:i + n], i // n)


def multiprocessing_run(data, function, cores=6, returned=True):
    df_split = chunks(data, max(1, len(data) // cores))
    pool = Pool(cores)
    pooled = pool.map(function, df_split)
    pool.close()
    pool.join()
    if returned:
        return list(itertools.chain(*pooled))


# These are set at module level so multiprocess workers can see them
tokenizer = None
BLOCK_SIZE = 8192
OUTPUT_DIR = './parquet-train'


def init_tokenizer(model_name='Qwen/Qwen3-0.6B-Base'):
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def write_parquet_worker(args):
    rows, index = args
    out_root = f'{OUTPUT_DIR}/shard-{index}'

    count, written = 0, 0
    temp_ids, temp_pos = [], []

    with ParquetWriter(out=out_root, columns=COLUMNS, exist_ok=True) as writer:
        for row in tqdm(rows, desc=f'Worker {index}'):
            outputs = tokenizer(row['text'], add_special_tokens=False)
            ids = outputs['input_ids']
            length = len(ids)

            if length == 0 or length > BLOCK_SIZE:
                continue

            if count + length > BLOCK_SIZE:
                o = collator_pack(temp_ids, temp_pos)
                if o['input_ids'].shape[0] > 0:
                    writer.write(o)
                    written += 1
                temp_ids = [ids]
                temp_pos = [list(range(length))]
                count = length
            else:
                temp_ids.append(ids)
                temp_pos.append(list(range(length)))
                count += length

        if temp_ids:
            o = collator_pack(temp_ids, temp_pos)
            if o['input_ids'].shape[0] > 0:
                writer.write(o)
                written += 1

    return [written]


# ---------------------------------------------------------------------------
# S3 upload helper
# ---------------------------------------------------------------------------

def s3_sync(local_dir, bucket, prefix):
    """Upload local_dir to s3://bucket/prefix/ using aws cli."""
    s3_path = f's3://{bucket}/{prefix}/'
    print(f'\nUploading {local_dir} -> {s3_path}')
    cmd = ['aws', 's3', 'sync', local_dir, s3_path, '--no-progress']
    subprocess.run(cmd, check=True)
    print(f'Done: {s3_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Prepare Call Center Language Switching dataset')
    parser.add_argument('--block_size', type=int, default=8192, help='Tokens per packed block')
    parser.add_argument('--n_workers', type=int, default=20, help='Multiprocessing workers')
    parser.add_argument('--s3_bucket', type=str, default='aies-research-datasets', help='S3 bucket name')
    parser.add_argument('--s3_prefix', type=str, default='call-center-language-switching', help='S3 key prefix')
    parser.add_argument('--tokenizer', type=str, default='Qwen/Qwen3-0.6B-Base', help='Tokenizer to use')
    args = parser.parse_args()

    global BLOCK_SIZE, OUTPUT_DIR
    BLOCK_SIZE = args.block_size

    # ----- Step 1: Download datasets -----
    print('=' * 60)
    print('Step 1: Downloading datasets')
    print('=' * 60)

    ds_call_center = load_dataset('Scicom-intl/Call-Center-Language-Switching')
    ds_multiturn = load_dataset('mesolitica/Malaysian-Multiturn-Chat-Assistant')
    ds_speech = load_dataset('mesolitica/Malaysian-Speech-Instructions')

    # Function-Call has multiple configs - load all of them
    FUNCTION_CALL_CONFIGS = [
        'extended_functions', 'extended_functions_v2',
        'functions', 'functions_multilingual_examples',
        'functions_multilingual_examples_v2',
        'multifunctions', 'multifunctions_deep', 'multifunctions_deep_v2',
        'multifunctions_multiturn', 'multifunctions_multiturn_extra',
        'multifunctions_multiturn_language_switching',
        'multifunctions_multiturn_language_switching_extra',
        'multifunctions_multiturn_v2_deep',
        'multifunctions_v2',
        'telco_multifunctions_premium', 'telco_multifunctions_premium_multiturn',
    ]
    ds_function_call_all = {}
    for cfg in FUNCTION_CALL_CONFIGS:
        print(f'  Loading Function-Call/{cfg}...')
        ds_function_call_all[cfg] = load_dataset('Scicom-intl/Function-Call', cfg)

    # ----- Step 2: Convert to unified format -----
    print('\n' + '=' * 60)
    print('Step 2: Converting to unified format')
    print('=' * 60)

    all_data = []

    # Call Center Language Switching
    for name, ds in [
        ('call-center-language-switching', ds_call_center),
        ('multiturn-chat', ds_multiturn),
        ('speech-instructions', ds_speech),
    ]:
        print(f'\nConverting {name}...')
        split = 'train'
        if split not in ds:
            split = list(ds.keys())[0]
        for row in tqdm(ds[split]):
            unified = convert_to_unified(row, name)
            if unified:
                all_data.append(unified)

    # Function-Call (all configs)
    for cfg, ds in ds_function_call_all.items():
        print(f'\nConverting function-call/{cfg}...')
        split = 'train'
        if split not in ds:
            split = list(ds.keys())[0]
        for row in tqdm(ds[split]):
            unified = convert_to_unified(row, f'function-call/{cfg}')
            if unified:
                all_data.append(unified)

    print(f'\nTotal unified conversations: {len(all_data)}')
    source_counts = Counter(d['source'] for d in all_data)
    for src, cnt in source_counts.items():
        print(f'  {src}: {cnt}')

    # ----- Step 3: Build test set (50 per language) -----
    print('\n' + '=' * 60)
    print('Step 3: Building test set (50 conversations per language)')
    print('=' * 60)

    call_center_data = [d for d in all_data if d['source'] == 'call-center-language-switching']
    by_language = defaultdict(list)
    for d in call_center_data:
        by_language[d['language']].append(d)

    print('Languages found:')
    for lang, items in by_language.items():
        print(f'  {lang}: {len(items)} conversations')

    test_set = []
    test_indices = set()
    for lang, items in by_language.items():
        n_sample = min(50, len(items))
        sampled = random.sample(items, n_sample)
        for s in sampled:
            s['split'] = 'test'
            test_set.append(s)
            test_indices.add(id(s))
        print(f'  Test: {lang} -> {n_sample} conversations')

    train_data = [d for d in all_data if id(d) not in test_indices]
    print(f'\nTotal test set:  {len(test_set)} conversations')
    print(f'Total train set: {len(train_data)} conversations')

    # ----- Step 4: Build positive / negative classes -----
    print('\n' + '=' * 60)
    print('Step 4: Building positive / negative samples')
    print('=' * 60)

    train_samples = []
    for d in tqdm(train_data, desc='Building train samples'):
        messages = d['messages']
        if not messages or len(messages) < 2:
            continue
        train_samples.append({'text': build_positive(messages), 'label': 'positive', 'source': d['source']})
        train_samples.append({'text': build_negative(messages), 'label': 'negative', 'source': d['source']})
    random.shuffle(train_samples)

    test_samples = []
    for d in tqdm(test_set, desc='Building test samples'):
        messages = d['messages']
        if not messages or len(messages) < 2:
            continue
        test_samples.append({'text': build_positive(messages), 'label': 'positive', 'language': d['language']})
        test_samples.append({'text': build_negative(messages), 'label': 'negative', 'language': d['language']})

    label_counts = Counter(s['label'] for s in train_samples)
    print(f'\nTrain samples: {len(train_samples)}  (pos={label_counts["positive"]}, neg={label_counts["negative"]})')
    print(f'Test  samples: {len(test_samples)}')

    # ----- Step 5: Tokenize + multipack + write parquet -----
    print('\n' + '=' * 60)
    print(f'Step 5: Tokenize & multipack (block_size={BLOCK_SIZE})')
    print('=' * 60)

    tok = init_tokenizer(args.tokenizer)
    im_end_id = tok.convert_tokens_to_ids('<|im_end|>')
    print(f'Vocab size: {len(tok)}')
    print(f'<|im_end|> token ID: {im_end_id}')

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = multiprocessing_run(train_samples, write_parquet_worker, cores=args.n_workers)
    print(f'Total blocks written across {args.n_workers} workers: {sum(results)}')

    # ----- Step 6: Merge shards -----
    print('\n' + '=' * 60)
    print('Step 6: Merging shards')
    print('=' * 60)

    shard_folders = sorted(
        glob(f'{OUTPUT_DIR}/shard-*'),
        key=lambda x: int(x.split('-')[-1]),
    )
    print(f'Found {len(shard_folders)} shards')

    MERGED_DIR = './parquet-train-merged'
    total = 0
    with ParquetWriter(out=MERGED_DIR, columns=COLUMNS, exist_ok=True) as writer:
        for folder in shard_folders:
            try:
                ds = StreamingDataset(local=folder)
                for i in tqdm(range(len(ds)), desc=folder):
                    writer.write(ds[i])
                    total += 1
            except Exception as e:
                print(f'Error reading {folder}: {e}')
    print(f'Merged dataset: {total} blocks -> {MERGED_DIR}')

    # ----- Step 7: Write test set parquet -----
    print('\n' + '=' * 60)
    print('Step 7: Writing test set parquet')
    print('=' * 60)

    TEST_DIR = './parquet-test'
    total_test = 0
    with ParquetWriter(out=TEST_DIR, columns=COLUMNS, exist_ok=True) as writer:
        for sample in tqdm(test_samples, desc='Writing test parquet'):
            outputs = tok(sample['text'], add_special_tokens=False)
            ids = outputs['input_ids']
            if len(ids) == 0:
                continue
            writer.write({
                'input_ids': np.array(ids, dtype=np.uint32),
                'position_ids': np.array(list(range(len(ids))), dtype=np.uint32),
                'attention_mask': np.array([len(ids)], dtype=np.uint32),
                'text': json.dumps({'label': sample['label'], 'language': sample['language']}),
            })
            total_test += 1
    print(f'Test set: {total_test} samples -> {TEST_DIR}')

    # ----- Step 8: Verify -----
    print('\n' + '=' * 60)
    print('Step 8: Verification')
    print('=' * 60)

    ds_train = StreamingDataset(local=MERGED_DIR)
    print(f'Train dataset: {len(ds_train)} packed blocks')
    sample = ds_train[0]
    print(f'  First block: {sample["input_ids"].shape[0]} tokens, {len(sample["attention_mask"])} sequences')

    ds_test = StreamingDataset(local=TEST_DIR)
    print(f'Test  dataset: {len(ds_test)} samples')

    correct = 0
    for i in range(len(ds_test)):
        s = ds_test[i]
        meta = json.loads(s['text'])
        last_token = int(s['input_ids'][-1])
        if meta['label'] == 'positive' and last_token == im_end_id:
            correct += 1
        elif meta['label'] == 'negative' and last_token != im_end_id:
            correct += 1
    print(f'  Label verification: {correct}/{len(ds_test)} correct ({100 * correct / len(ds_test):.1f}%)')

    # Count total tokens
    total_tokens = 0
    for i in tqdm(range(len(ds_train)), desc='Counting tokens'):
        total_tokens += ds_train[i]['input_ids'].shape[0]

    # ----- Step 9: Upload to S3 -----
    print('\n' + '=' * 60)
    print(f'Step 9: Upload to s3://{args.s3_bucket}/{args.s3_prefix}/')
    print('=' * 60)

    s3_sync(MERGED_DIR, args.s3_bucket, f'{args.s3_prefix}/parquet-train-merged')
    s3_sync(TEST_DIR, args.s3_bucket, f'{args.s3_prefix}/parquet-test')

    # ----- Summary -----
    print('\n' + '=' * 60)
    print('Dataset Summary')
    print('=' * 60)
    print(f'Train blocks:  {len(ds_train)}')
    print(f'Train tokens:  {total_tokens:,}')
    print(f'Train samples: {len(train_samples)} (positive + negative)')
    print(f'Test  samples: {len(test_samples)}')
    print(f'Block size:    {BLOCK_SIZE}')
    print(f'\nLocal paths:')
    print(f'  Train: {MERGED_DIR}')
    print(f'  Test:  {TEST_DIR}')
    print(f'\nS3 paths:')
    print(f'  Train: s3://{args.s3_bucket}/{args.s3_prefix}/parquet-train-merged/')
    print(f'  Test:  s3://{args.s3_bucket}/{args.s3_prefix}/parquet-test/')
    print(f'\nReady for training:')
    print(f'  python train.py --train_file {MERGED_DIR} --block_size {BLOCK_SIZE}')
    print(f'  # or from S3:')
    print(f'  # python train.py --train_file s3://{args.s3_bucket}/{args.s3_prefix}/parquet-train-merged --block_size {BLOCK_SIZE}')


if __name__ == '__main__':
    main()
