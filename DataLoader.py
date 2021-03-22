import os
from PIL import Image
from torchvision import transforms
import torch.utils.data as Data
#from torch.utils.data import Dataset, DataLoader


class dataset_WNet(Data.Dataset):
    def __init__(self, config):
        self.image_path = config.datapath
        self.transform = transforms.Compose([
          transforms.Resize(config.inputsize),
          transforms.ToTensor()
        ])
        self.image_names = os.listdir(self.image_path)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.image_path, self.image_names[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image

def torch_loader(config):
    dataset = dataset_WNet(config)
    return Data.DataLoader(
                      dataset,
                      batch_size = config.BatchSize,
                      shuffle = config.Shuffle,
                      num_workers = config.LoadThread
                      )
