from __future__ import annotations

from abc import ABC, abstractmethod

from app.core.broll.types import VideoItem


class BrollProvider(ABC):
    @abstractmethod
    def search(self, query: str, orientation: str, per_page: int) -> list[VideoItem]:
        raise NotImplementedError
