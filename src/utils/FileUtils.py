import os
from pathlib import Path


def create_folder_safe(full_path_and_file):
    path = (os.path.dirname(full_path_and_file))
    Path(path).mkdir(parents=True, exist_ok=True)