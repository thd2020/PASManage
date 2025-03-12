import dill
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class MultiModelDataset(Dataset):
    def __init__(self, clinicals, imgs, labels):
        self.clinicals = clinicals
        self.imgs = imgs
        self.labels = labels
        self.preprocessor = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        clinical = torch.tensor(self.clinicals[idx], dtype=torch.float32)
        img = self.preprocessor(self.imgs[idx])
        img = torch.tensor(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return (clinical, img), label


if __name__ == "__main__":
    with open("/mnt/sda1/hwx/rerun_PAS/code/dataset/123/training.pkl", "rb") as f:
        train_dataset = dill.load(f)

    # sample = train_dataset[0]
    # print(sample[0][0].shape)
    # print(sample[0][1].shape)
    # print(sample[1])

    with open("testing.pkl", "rb") as f:
        test_dataset = dill.load(f)

    # print(len(train_dataset))
    # print(len(test_dataset))

    loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)
    i = 1
    for sample in loader:
        if i == 1:
            print(sample[0][0])
            print(sample[0][1].shape)
        i += 1