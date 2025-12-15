import os
import sys

def get_path(filename):
    """Returns absolute path for a file in the project root."""
    # Assuming main.py is in the root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, filename)