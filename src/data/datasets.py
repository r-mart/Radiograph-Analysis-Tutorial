import cv2
import torch
from torch.utils.data import Dataset


class ClassificationDataset(Dataset):

    def __init__(self, df, data_path, transforms=None):
        super().__init__()

        self.df = df
        self.data_path = data_path
        self.transforms = transforms

    def __getitem__(self, index: int):
        sample = self.df.iloc[index]

        img_path = self.data_path / sample['rel_image_path']
        image_id = sample["id"]
        img = cv2.imread(str(img_path), 0)
        label = sample["study_label"]

        if self.transforms:
            trans = self.transforms(image=img)
            img = trans['image']
            label = torch.tensor(label)

        return img, label, image_id

    def __len__(self) -> int:
        return self.df.shape[0]
