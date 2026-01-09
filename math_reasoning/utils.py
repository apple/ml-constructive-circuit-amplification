# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

import hashlib
import json
import logging
import os
import shutil
import urllib.parse

from filelock import FileLock

# Configure logger
logger = logging.getLogger("gsm_eval")
logger.setLevel(logging.INFO)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(console_handler)

curr_dir = os.path.dirname(os.path.abspath(__file__))
curr_dir = (
    os.path.join(curr_dir, "math_reasoning") if "math_reasoning" not in curr_dir else curr_dir
)


def load_json(fp, storage_options=None, caching=True):
    """Load a json file from a path."""
    # For local files, just use standard open
    if fp.startswith(("http://", "https://")):
        raise ValueError(f"Remote URLs not supported in open source version: {fp}")
    
    with open(fp) as f:
        return json.load(f)


def copy_file_to_local(url: str, dest: str):
    """Copy a file to local destination."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copy(url, dest)


def _is_file_url(url: str) -> bool:
    """Detect if URL points to a file based on file extension heuristics.
    URLs with model file extensions are treated as files, others as directories.
    """
    # Parse the URL to get the path component
    parsed = urllib.parse.urlparse(url)
    path = parsed.path
    
    # Model file extensions
    file_extensions = {'.bin', '.safetensors', '.pt', '.pth', '.pkl'}
    
    # Get the file extension (everything after the last dot)
    _, ext = os.path.splitext(path)
    
    # Return True if we found a recognized file extension
    return ext.lower() in file_extensions


def _get_cached_remote_directory(directory_url: str, cache_dir: str | None = None) -> str:
    """Cache a remote directory. Only supports local paths.
    Returns the path to the local directory.
    """
    # For local paths, just return the path as-is
    if os.path.exists(directory_url):
        return directory_url
    
    raise FileNotFoundError(f"Directory not found: {directory_url}")


def _cache_subpath(filename: str) -> str:
    h = hashlib.sha256(filename.encode("utf-8")).hexdigest()
    return h[:2] + "/" + h + os.path.splitext(filename)[1]


def _get_cached_remote_single_file(filename: str, cache_dir: str | None = None) -> str:
    """Original file caching implementation for single files."""
    if cache_dir is None:
        cache_dir = "/tmp/"
    cache_path = os.path.join(cache_dir, _cache_subpath(filename))
    lock_file_path = cache_path + ".lock"

    # Make sure the directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    with FileLock(lock_file_path):
        if not os.path.exists(cache_path):
            logger.info(f"Downloading {filename} to {cache_path}")
            try:
                copy_file_to_local(filename, cache_path)
            except Exception as e:
                logger.error(f"Error downloading {filename}")
                logger.error(f"{e}")
                try:
                    os.unlink(cache_path)
                except:
                    pass
                raise RuntimeError(f"Error downloading {filename}")
            logger.info(f"{filename} downloaded")
    return cache_path


def get_cached_remote_file(filename: str, cache_dir: str | None = None) -> str:
    """Download and cache a remote file or directory automatically.
    
    Uses file extension heuristics to determine if the URL points to:
    - A single file (.bin, .safetensors, .pt, .pkl) -> downloads single file
    - A directory (no recognized file extension) -> downloads entire directory
    
    Args:
        filename: URL to the remote file or directory
        cache_dir: Local cache directory (defaults to /tmp/)
        
    Returns:
        Path to the cached file or directory

    """
    if _is_file_url(filename):
        logger.info(f"Detected file URL: {filename}")
        return _get_cached_remote_single_file(filename, cache_dir)
    elif "://" in filename:
        logger.info(f"Detected directory URL: {filename}")
        return _get_cached_remote_directory(filename, cache_dir)
    else:
        # this is actually a local directory just return it.
        return filename


def get_best_result_idx(results: dict, update_using_mask: bool) -> int:
    """Get the best result index based on the test accuracy and update using mask."""
    best_result_idx = -1
    best_test_acc = float("-inf")
    for child_id, result in results.items():
        if (
            result[0]["overall_test_accuracy"] > best_test_acc
            and result[0]["update_using_mask"] == update_using_mask
        ):
            best_test_acc = result[0]["overall_test_accuracy"]
            best_result_idx = child_id
    return best_result_idx
