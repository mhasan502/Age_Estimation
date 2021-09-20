import torch
import numpy as np
from tqdm.notebook import tqdm
from PIL import Image
from parse import parse
from torch.utils.data import DataLoader, Dataset


class AgeDBDataset(Dataset):
    
    def __init__(self, image_list, device, train=False, augment=0, train_transform=None, test_transform=None, **kwargs):
        self.image_list = image_list
        self.device = device
        self.train = train
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.labels = []
        self.images = []
        device = torch.device("cuda")

        if self.train:
            for i in tqdm(range(len(image_list))):

                image = Image.open(self.image_list[i]['image_location']).convert('RGB')
                image_location = self.image_list[i]['image_location']
                age = self.image_list[i]['age']
                gender = self.image_list[i]['gender']

                image = np.array(image)

                for j in range(augment):
                    if j == 0:
                        augmented_images = self.test_transform(image=image)['image']
                    else:
                        augmented_images = self.train_transform(image=image)['image']

                    self.images.append(augmented_images)
                    self.labels.append({
                        'image_location': image_location,
                        'age': age,
                        'gender': gender
                    })

                    
    def __len__(self):

        if self.train:
            return len(self.labels)
        else:
            return len(self.image_list)

        
    def __getitem__(self, index):

        if torch.is_tensor(index):
            index = index.tolist()

        if self.train:
            image = self.images[index]
            labels = {
                'image_location': self.labels[index]['image_location'],
                'age': self.labels[index]['age'],
                'gender': self.labels[index]['gender']
            }

        else:
            image = Image.open(
                self.image_list[index]['image_location']).convert('RGB')
            image = np.array(image)
            image = self.test_transform(image=image)['image']
            labels = {
                'image_location': self.image_list[index]['image_location'],
                'age': self.image_list[index]['age'],
                'gender': self.image_list[index]['gender']
            }

        return image.to(self.device), labels
