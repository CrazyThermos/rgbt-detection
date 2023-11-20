import os
import torch
from PIL import Image
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, root_dir, label_dir):

        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path_list = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path_list[idx]  
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name) 
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)
    


