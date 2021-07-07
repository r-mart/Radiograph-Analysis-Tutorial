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
        # turn into binary classification
        label = int(label > 0)

        if self.transforms:
            trans = self.transforms(image=img)
            img = trans['image']
            label = torch.tensor(label)

        return img, label, image_id

    def __len__(self) -> int:
        return self.df.shape[0]


class DetectionDataset(Dataset):

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

        boxes = sample["boxes"]
        labels = sample["labels"]
        study_label = sample["study_label"]

        boxes = [box for label, box in zip(labels, boxes) if label != 'none']
        labels = [study_label for label in labels if label != 'none']

        target = {}
        target['boxes'] = boxes
        target['labels'] = torch.tensor(labels)
        target['image_id'] = torch.tensor([index])
        target["height"] = torch.tensor([sample["height"]])
        target["width"] = torch.tensor([sample["width"]])

        if self.transforms:
            trans = self.transforms(image=img,
                                    bboxes=target['boxes'],
                                    labels=labels)
            img = trans['image']
            if len(trans['bboxes']) > 0:
                target['boxes'] = torch.stack(
                    tuple(map(torch.tensor, zip(*trans['bboxes'])))).permute(1, 0)
            else:
                target['boxes'] = torch.tensor([[0, 0, 1, 1]])
                target['labels'] = torch.tensor([study_label])

        return img, target, image_id

    def __len__(self) -> int:
        return self.df.shape[0]
