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
          transforms.ToTensor(),
          transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
          )
        ])

        included_extensions = ['jpg','jpeg','png']
        self.image_names = [i for i in os.listdir(self.image_path)
                            if any(i.endswith(ext) for ext in included_extensions)]
        #for clean dataset (ex:VOC2012) this can be substitued with:
        #self.image_names = os.listdir(self.image_path)
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
