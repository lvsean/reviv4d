#!/usr/bin/env python3
"""Download Cosmos Tokenizer checkpoints to a local folder."""

from argparse import ArgumentParser
import os
from pathlib import Path


def parse_args():
    parser = ArgumentParser(description="Download Cosmos Tokenizer checkpoints.")
    parser.add_argument(
        "--repo_id",
        type=str,
        default="nvidia/Cosmos-0.1-Tokenizer-DV4x8x8",
        help="Hugging Face repo id.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./Cosmos/checkpoints/Cosmos-0.1-Tokenizer-DV4x8x8",
        help="Local directory to save checkpoints.",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=["encoder.jit", "decoder.jit"],
        help="Checkpoint filenames to download.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Optional Hugging Face token (for gated/private repos).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency `huggingface_hub`. Install with: pip install huggingface_hub"
        ) from exc

    for filename in args.files:
        try:
            local_path = hf_hub_download(
                repo_id=args.repo_id,
                filename=filename,
                local_dir=str(output_dir),
                local_dir_use_symlinks=False,
                token=args.token,
            )
            print(f"Downloaded: {filename} -> {local_path}")
        except Exception as exc:
            proxy_vars = [
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "http_proxy",
                "https_proxy",
                "ALL_PROXY",
                "all_proxy",
                "NO_PROXY",
                "no_proxy",
            ]
            print("\nDownload failed.")
            print(f"repo_id={args.repo_id}, file={filename}")
            print(f"error: {type(exc).__name__}: {exc}")
            print("\nProxy-related environment:")
            for key in proxy_vars:
                val = os.environ.get(key, "")
                print(f"  {key}={val if val else '<unset>'}")
            print(
                "\nIf this is a proxy/TLS issue, verify your proxy settings or unset invalid proxy vars."
            )
            print(
                "Example: unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy"
            )
            raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
