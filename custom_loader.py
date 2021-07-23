import os
import sys
import torch
import zipfile
import decouple
from PIL import Image
from parse import parse
from torch.utils.data import DataLoader, Dataset

zip_pass = decouple.config("ZIP_KEY")

class AgeDBHandler:
    def __init__(self, datasets_dir, preload=False, use_preprocessed=True, device: torch.device = torch.device('cpu')):
        self.device = device
        self.preload = preload

        self.datasets_dir = datasets_dir
        self.directory = os.path.join(datasets_dir, 'AgeDB')
        self.zipFile = os.path.join(datasets_dir, 'AgeDB.zip')

        self.trainDataset = None
        self.testDataset = None
        self.validateDataset = None

        self._prepare_on_disk()

    def get_loaders(self, transform, train_size=0.7, validate_size=0.2, test_size=0.1, batch_size=15, **kwargs):
        if round(train_size + validate_size + test_size, 1) > 1.0:
            sys.exit("Sum of the percentages should be less than 1. it's " + str(
                train_size + validate_size + test_size) + " now!")

        dataset = AgeDBDataset(directory=self.directory,
                               transform=transform,
                               preload=self.preload,
                               device=self.device,
                               **kwargs)

        train_len = int(len(dataset) * train_size)
        validate_len = int(len(dataset) * validate_size)
        test_len = int(len(dataset) * test_size)
        others_len = len(dataset) - train_len - validate_len - test_len

        self.trainDataset, self.validateDataset, self.testDataset, _ = torch.utils.data.random_split(
            dataset, [train_len, validate_len, test_len, others_len])

        # noinspection PyUnusedLocal
        dataset = None

        train_loader = DataLoader(self.trainDataset, batch_size=batch_size)
        validate_loader = DataLoader(self.validateDataset, batch_size=batch_size)
        test_loader = DataLoader(self.testDataset, batch_size=batch_size)

        return train_loader, validate_loader, test_loader

    def _prepare_on_disk(self):
        if os.path.exists(self.directory):
            if len(os.listdir(self.directory)) != 0:
                print('AgeDB Already Exists on ' +
                      self.directory + '.We will use it!')
                return
            
        print('Could not find AgeDB on', self.directory)
        print('Looking for ', self.zipFile)
        
        if os.path.exists(self.zipFile):
            
            print(self.zipFile, 'is found. Trying to extract:')
            with zipfile.ZipFile(self.zipFile) as zf:
                zf.extractall(pwd=zip_pass, path=self.datasets_dir)
                
            print('Successfully extracted')
            
        else:
            sys.exit('AgeDB Zip file not found!')
            

class AgeDBDataset(Dataset):
    def __init__(self, directory, transform, preload=False, device: torch.device = torch.device('cpu'), **kwargs):
        self.device = device
        self.directory = directory if kwargs.get('usePreprocessed', 0) == 0 else os.path.join(directory, 'preProcessed')
        self.transform = transform
        gender_to_class_id = {'m': 0, 'f': 1}
        self.labels = []
        self.images = []
        self.preload = preload

        for i, file in enumerate(os.listdir(self.directory)):
            if file.endswith(('_2.jpg', '_3.jpg')):  # TODO
                continue
            file_labels = parse('{}_{}_{age}_{gender}.jpg', file)
            
            if file_labels is None:
                continue
                
            if self.preload:
                image = Image.open(os.path.join(self.directory, file)).convert('RGB')
                
                if self.transform is not None:
                    image = self.transform(image).to(self.device)
            else:
                image = os.path.join(self.directory, file)

            self.images.append(image)
            gender = gender_to_class_id[file_labels['gender']]
            age = int(file_labels['age'])
            self.labels.append({
                'age': age,
                'gender': gender
            })
        pass

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.images[idx]

        if not self.preload:
            image = Image.open(image).convert('RGB')
            if self.transform is not None:
                image = self.transform(image).to(self.device)

        labels = {'age': self.labels[idx]['age'], 'gender': self.labels[idx]['gender']}
        return image.to(self.device), labels