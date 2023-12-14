import os

import torch
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from torch.utils.data import Dataset, DataLoader


class Dataloader(Dataset):
    def __init__(self, data_folder_path, image_set='train', transform=None, subset_percentage=30):
        self.data_folder = data_folder_path
        self.image_set = image_set
        self.transform = transform
        self.voc_dataset = VOCSegmentation(data_folder_path, year='2007', image_set=image_set, download=True)
        subset_size = int(len(self.voc_dataset) * (subset_percentage / 100.0))

        subset_indices = torch.randperm(len(self.voc_dataset))[:subset_size]

        self.voc_dataset = torch.utils.data.Subset(self.voc_dataset, subset_indices)

    def __len__(self):
        return len(self.voc_dataset)

    def __getitem__(self, idx):
        image, target = self.voc_dataset[idx]

        if self.transform:
            image = self.transform(image)
            target = self.transform(target)

        return image, target


def get_data_loaders(data_folder_path, batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = Dataloader(data_folder_path, image_set='train', transform=transform)
    val_dataset = Dataloader(data_folder_path, image_set='val',
                             transform=transforms.ToTensor())  # No random flipping for validation
    test_dataset = Dataloader(data_folder, image_set='test',
                              transform=transforms.ToTensor())  # No random flipping for testing

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# Example usage:
data_folder = os.getcwd() + "/Data_Folder"
train_loader, val_loader, test_loader = get_data_loaders(data_folder)
