import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import AutoAugment
from transformers import CLIPProcessor
from skimage.io import imread
from skimage.color import gray2rgb

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from random import randint


def clean_data(data):
    data = data.dropna(axis=0)
    return data.drop_duplicates()


def split_dataset(data, split_date):
    before = data[data['created'] < split_date]
    after = data[data['created'] >= split_date]

    return before, after


def load_data(tsv_path: Path, split_date: datetime) -> dict[str, pd.DataFrame]:
    data = pd.read_csv(tsv_path, delimiter='\t', header=0)

    data = clean_data(data)
    data_train, data_test = split_dataset(data, split_date)

    return {"full": data, "train": data_train, "test": data_test}


class MushroomDataset(Dataset):
    def __init__(
        self, 
        dataframe: pd.DataFrame, 
        images_dir: Path, 
        augment: bool = True
        ):
        self.data = dataframe
        self.dir = images_dir
        self.augment = augment
        self.aug = AutoAugment()


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        try:
            row = self.data.iloc[index]
            filename = row['filename']
            label = row['name']
            image = self.read_file(filename)

            return image, label
        except Exception as e:
            return self.__getitem__(randint(0, self.__len__() - 1))


    def read_file(self, filename):
        image = imread(str(Path(self.dir, filename)))
        if len(image.shape) == 2:
            image = gray2rgb(image)
        elif image.shape[2] == 1:
            image = gray2rgb(image)
        elif image.shape[2] == 4:
            image = image[...,:3]

        if self.augment:
            image = torch.from_numpy(image).permute(2, 0, 1)
            image = self.aug(image)
            image = image.permute(1, 2, 0).numpy()
        
        return image


@dataclass
class CLIPMushroomCollate:
    processor: CLIPProcessor
    return_tensors: str = "pt"
    padding: bool = True
    train: bool = True


    def __call__(self, features):
        images, labels = zip(*features)
        tensors = self.processor(
            text=labels, 
            images=images, 
            return_tensors=self.return_tensors, 
            padding=self.padding
        )
        if self.train:
            tensors["labels"] = self.target_matrix(labels)
        return tensors


    @staticmethod
    def target_matrix(labels):
        target_similarities = torch.eye(len(labels))
        idx = torch.triu_indices(len(labels), len(labels), offset=1)
        for (a, b), idx1, idx2 in zip(combinations(labels, 2), idx[0], idx[1]):
            if a == b:
                target_similarities[idx1, idx2] = 1
                target_similarities[idx2, idx1] = 1
        return target_similarities / target_similarities.sum(dim=-1)
