#!/usr/bin/env python3
from qpenn.utils import ARF_URLS, fetch_arf

if __name__ == "__main__":
    print("Fetching all standard QPE simulation ARFs...\n")
    for key in ARF_URLS.keys():
        fetch_arf(key, target_dir="downloads")
    print("\nAll downloads complete!")