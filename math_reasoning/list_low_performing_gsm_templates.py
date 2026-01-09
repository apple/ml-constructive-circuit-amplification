# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

import glob as glob_module
import os
from pathlib import Path

import click

from math_reasoning.utils import load_json


def find_low_performing_templates(file_path, threshold):
    """Returns the ids of templates whose performance is below the threshold."""
    data = load_json(file_path)

    low_accuracy_keys = []
    for key, value in data.items():
        if isinstance(value, dict) and "accuracy" in value:
            if value["accuracy"] < threshold:
                low_accuracy_keys.append(key)

    return low_accuracy_keys


@click.command()
@click.argument("path", type=str)
@click.option(
    "--threshold",
    default=0.8,
    help="Accuracy threshold to report keys below this value.",
    type=float,
)
def main(path, threshold):
    """Analyzes model evaluation JSON files in a path to find keys with low accuracy.

    The script processes all numbered .json files (e.g., 2.json, 3.json) in the
    specified path.
    """
    try:
        if "://" in path:
            raise ValueError(f"Remote URLs not supported: {path}")
        
        p = Path(path)
        if not p.is_dir():
            click.echo(f"Error: Local directory not found at '{path}'", err=True)
            return
        all_files = [str(f) for f in p.glob("*.json")]

        # Filter for numbered json files
        json_files = [f for f in all_files if Path(f).stem.isdigit()]
        sorted_files = sorted(json_files, key=lambda p: int(Path(p).stem))

    except Exception as e:
        click.echo(f"Error accessing or listing files in path '{path}': {e}", err=True)
        return

    if not sorted_files:
        click.echo(f"No numbered JSON files found in '{path}'.")
        return

    all_low_accuracy_keys = []
    for file_path in sorted_files:
        try:
            low_accuracy_keys = find_low_performing_templates(file_path, threshold)

            if low_accuracy_keys:
                file_name = Path(file_path).name
                click.echo(f"--- Results for {file_name} ---")
                click.echo(f"Templates with accuracy below {threshold}:")
                # Sort keys numerically for consistent output
                sorted_keys = sorted(low_accuracy_keys, key=int)
                # Format output as requested
                click.echo(f"[{', '.join(sorted_keys)}]")
                click.echo()
                all_low_accuracy_keys.extend(low_accuracy_keys)
        except Exception as e:
            click.echo(
                f"Warning: Could not process file {Path(file_path).name}. Error: {e}",
                err=True,
            )
            continue

    assert len(all_low_accuracy_keys) == len(set(all_low_accuracy_keys))

    click.echo(f"--- all ({len(all_low_accuracy_keys)}) low accuracy templates ---")
    click.echo(f"[{', '.join(all_low_accuracy_keys)}]")
    click.echo()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
