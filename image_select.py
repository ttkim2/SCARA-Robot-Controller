"""
Shared image-selection prompt used by main.py and preview_drawing.py.

Scans assets/inputs/ for image files, prints a numbered menu, and
returns the path chosen by the user.
"""

import os
import sys

_INPUTS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'assets', 'inputs')
_IMAGE_EXTS  = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def select_image() -> str:
    """
    Prompt the user to choose an image from assets/inputs/.

    Returns the relative path string (e.g. 'assets/inputs/uiuc_logo.jpg').
    Exits with an error message if the directory is empty or the selection
    is invalid.
    """
    if not os.path.isdir(_INPUTS_DIR):
        print(f"[ERROR] Input directory not found: {_INPUTS_DIR}")
        sys.exit(1)

    images = sorted(
        f for f in os.listdir(_INPUTS_DIR)
        if os.path.splitext(f)[1].lower() in _IMAGE_EXTS
    )

    if not images:
        print(f"[ERROR] No images found in {_INPUTS_DIR}")
        sys.exit(1)

    print("\nAvailable images:")
    for i, name in enumerate(images, 1):
        print(f"  {i}. {name}")

    while True:
        raw = input(f"Select image [1-{len(images)}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(images):
            chosen = os.path.join('assets', 'inputs', images[int(raw) - 1])
            print(f"  → {chosen}")
            return chosen
        print(f"  Please enter a number between 1 and {len(images)}.")
