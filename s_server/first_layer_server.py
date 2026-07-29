from __future__ import annotations

from pathlib import Path
from typing import Any


class FirstLayerServer:
    """Top-level server placeholder.

    The first-layer server is present in the architecture only. It does not
    receive global experience, store libraries, or coordinate agents yet.
    """

    def __init__(
        self,
        name: str,
        data_dir: Path,
        backend: Any | None = None,
        specialty: str = "第一层服务器",
    ) -> None:
        self.name = name
        self.data_dir = data_dir
        self.backend = backend
        self.specialty = specialty
