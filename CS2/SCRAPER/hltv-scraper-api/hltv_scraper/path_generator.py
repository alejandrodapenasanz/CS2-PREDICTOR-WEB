import os
from abc import ABC, abstractmethod
from pathlib import Path


class FilePathGenerator(ABC):
    @abstractmethod
    def __init__(self, base_path: str) -> None:
        pass

    @abstractmethod
    def generate(self, filename: str) -> str:
        pass


class JsonFilePathGenerator(FilePathGenerator):
    def __init__(self, base_path: str) -> None:
        self.base_path = base_path
        source = Path(__file__).resolve().parent
        if Path(base_path).resolve() == source:
            self.data_dir = str(
                source.parents[3] / "VAULT" / "CS2" / "SCRAPER" / "hltv-scraper-api" / "hltv_scraper" / "data"
            )
        else:
            self.data_dir = os.path.join(base_path, "data")
        os.makedirs(self.data_dir, exist_ok=True)

    def generate(self, filename: str) -> str:
        if filename.endswith(".json"):
            filename = filename[:-5]

        file_dir = os.path.dirname(os.path.join(self.data_dir, filename))

        if file_dir:
            os.makedirs(file_dir, exist_ok=True)

        return f"{self.data_dir}/{filename}.json"
