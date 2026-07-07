
from __future__ import annotations

import sys
from pathlib import Path


def add_project_root_to_sys_path() -> None:
    project_root = Path(__file__).resolve().parent.parent
    project_root_text = str(project_root)
    if project_root_text not in sys.path:
        sys.path.insert(0, project_root_text)
