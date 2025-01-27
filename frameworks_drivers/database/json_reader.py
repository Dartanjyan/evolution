import json
from interface_adapters.json_reader_interface import JsonReaderInterface

class JsonReader(JsonReaderInterface):
    def read_json(self, file_path: str) -> dict:
        with open(file_path, 'r') as file:
            return json.load(file)
