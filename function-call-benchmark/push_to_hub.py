"""Push a local parquet file to the Hugging Face Hub as a dataset.

Examples:
    python3 push_to_hub.py \
        --file train-function-merged-00000-of-00001.parquet \
        --dataset Scicom-intl/Function-Call-TaaS \
        --subset train-basic --split train

    # short flags
    python3 push_to_hub.py -f test-function-merged-00000-of-00001.parquet \
        -d Scicom-intl/Function-Call-TaaS -s function --split test
"""

import argparse

import pandas as pd
from datasets import Dataset


def main():
    parser = argparse.ArgumentParser(
        description="Push a parquet file to the Hugging Face Hub as a dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-f", "--file", required=True,
        help="Path to the local parquet file to upload.",
    )
    parser.add_argument(
        "-d", "--dataset", required=True,
        help="Target Hub dataset repo id, e.g. 'username/dataset-name'.",
    )
    parser.add_argument(
        "-s", "--subset", default="default",
        help="Dataset subset / config name.",
    )
    parser.add_argument(
        "--split", default="train",
        help="Split name to write the rows under.",
    )
    parser.add_argument(
        "--private", action="store_true",
        help="Create the dataset repo as private.",
    )
    parser.add_argument(
        "--token", default=None,
        help="HF token (defaults to the cached login / HF_TOKEN env var).",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.file)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns from {args.file}")

    dataset = Dataset.from_pandas(df, preserve_index=False)
    dataset.push_to_hub(
        args.dataset,
        config_name=args.subset,
        split=args.split,
        private=args.private,
        token=args.token,
    )
    print(
        f"Pushed {len(dataset)} rows to "
        f"https://huggingface.co/datasets/{args.dataset} "
        f"(subset={args.subset!r}, split={args.split!r})"
    )


if __name__ == "__main__":
    main()
