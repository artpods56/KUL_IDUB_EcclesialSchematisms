from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
import transformers
transformers.logging.set_verbosity_error()

import torch
import sqlite3
from typing import Literal
from utils import sliding_window
from PIL import Image

class LayoutLMv3Interface:
    def __init__(self, model_path, processor_path, verbose=False):
        self._model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)
        self._processor = LayoutLMv3Processor.from_pretrained(processor_path)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self.id2label = self._model.config.id2label
        self.label2id = self._model.config.label2id
        self.verbose = verbose


    def predict(self, image_path: str):
        if self.verbose:
            print(f"Predicting on image: {image_path}")
        with Image.open(image_path) as img:
            image = img.copy()
        image = image.convert("RGB")

        image_width, image_height = image.size

        if self.verbose:
            print(f"Passing image to processor")

        encoding = self._processor(
            image,
            truncation=True,
            stride=128,
            padding="max_length",
            max_length=512,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        offset_mapping = encoding.pop("offset_mapping")
        overflow_to_sample_mapping = encoding.pop("overflow_to_sample_mapping")
        x = []
        for i in range(0, len(encoding["pixel_values"])):
            ndarray_pixel_values = encoding["pixel_values"][i]
            tensor_pixel_values = ndarray_pixel_values.clone().detach()
            x.append(tensor_pixel_values)

        x = torch.stack(x)
        encoding["pixel_values"] = x

        for k, v in encoding.items():
            encoding[k] = v.clone().detach().to(self._device)



        if self.verbose:
            print(f"Forward pass")

        with torch.no_grad():
            outputs = self._model(**encoding)

        logits = outputs.logits

        predicted_tokens = logits.argmax(-1).squeeze().tolist()
        token_bboxes = encoding.bbox.squeeze().tolist()

        if len(token_bboxes) == 512:
            predicted_tokens = [predicted_tokens]
            token_bboxes = [token_bboxes]

        if self.verbose:
            print(f"Applying sliding window")

        bboxes, predictions, words = sliding_window(
            self._processor,
            token_bboxes,
            predicted_tokens,
            encoding,
            image_width,
            image_height,
        )  

        if self.verbose:
            print(f"Finished predicting")

        return bboxes, predictions, words

    def evaluate(self, image_path, ground_truth_path):
        pass

    def save_results(self, results_path):
        pass







class SQLite3Interface:
    def __init__(self, database: str):
        self.db_connection = sqlite3.connect(database)
        self.cursor = self.db_connection.cursor()

    def __post_init__(self):
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = self.cursor.fetchall()
        if tables:
            print("Połączono")
            print("Dostępne tabele:", tables)
        else:
            print("Problem z połączeniem")
            raise Exception("Problem z połączeniem")
    
    def execute(self, query: str):
        self.cursor.execute(query)
        return self.cursor.fetchall()

    def get_tables(self):
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        return self.cursor.fetchall()
    
    def get_schema(self, table_name: str):
        self.cursor.execute(f"PRAGMA table_info({table_name})")
        return self.cursor.fetchall()
        
    def query(self, select_fields: list, where_query_map: dict, table_name: str):
        
        constrcted_where_query = ""
        for key, value in where_query_map.items():
            constrcted_where_query += f"{key} = {value} AND "

        constrcted_where_query = constrcted_where_query[:-5]
        query = f"SELECT {", ".join(select_fields)} FROM {table_name} WHERE {constrcted_where_query}"

        print(f"Query: {query}")
        self.cursor.execute(query)
        return self.cursor.fetchall()
    
    def get_geojson(self, page_number = None):
        if page_number is None:
            self.cursor.execute(f"SELECT the_geom FROM dane_hasla WHERE skany LIKE 'wloclawek_1872'")
        else:
            self.cursor.execute(f"SELECT the_geom FROM dane_hasla WHERE skany LIKE 'wloclawek_1872' AND strona_p = {page_number}")
        return self.cursor.fetchall()
    
    def get_page_records(self, page_number):
        self.cursor.execute(f"SELECT * FROM dane_hasla WHERE skany LIKE '%wloclawek%' AND strona_p = {page_number}")
        return self.cursor.fetchall()
    