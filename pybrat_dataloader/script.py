import os
import unittest
from dataset import Dataset
from loguru import logger

dataset_path = "../datasets/"
data_root = os.path.join(dataset_path, "DDICorpusBrat")

_TRAIN = "Train"
_TEST = "Test"

splits = {"train": _TRAIN, "test": _TEST}

fmt = "brat"

dataset = Dataset(dataset="ddi", data_root=data_root, split_names=splits, fmt=fmt)


# Get document 3 (python index 2) from the train
doc = dataset.data.train[2]

# Show the text (in form str)
print(doc.text)
print("="*100)

# Show the entities (in form List[Entity])
print(doc.entities)
print("="*100)

# Show the relations (in form List[Relation])
print(doc.relations)
print("="*100)