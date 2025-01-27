from abc import ABC, abstractmethod

class JsonReaderInterface(ABC):
    @abstractmethod
    def read_json(self, file_path: str) -> dict:
        """Читает JSON из указанного файла"""
        pass
