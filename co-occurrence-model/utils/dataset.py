import numpy as np
import pandas as pd
from torch.utils import data

__all__ = ["ObjectCooccurrenceCOCODataset"]

METADATA_COLUMNS = [
    "license",
    "file_name",
    "coco_url",
    "height",
    "width",
    "date_captured",
    "flickr_url",
    "id",
]


class ObjectCooccurrenceCOCODataset(data.Dataset):
    def __init__(self, file_location: str):
        super().__init__()
        self.data = pd.read_csv(file_location)
        self.metadata = self.data[METADATA_COLUMNS]
        self.features = self.data.drop(METADATA_COLUMNS, axis=1)
        self.fetch_mode = "features"

    def set_fetch_mode(self, mode: str):
        if mode not in ["features", "metadata", "all"]:
            raise ValueError("Data fetching type is invalid!")
        self.fetch_mode = mode
        return self

    def __getitem__(self, idx: int) -> np.ndarray:
        if self.fetch_mode == "metadata":
            return self.metadata.iloc[idx]
        if self.fetch_mode == "all":
            return self.data.iloc[idx]

        return self.features.iloc[idx].to_numpy()

    def __len__(self):
        return len(self.data)
