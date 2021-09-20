import torch
import albumentations as A
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2

from utils.handler import DataHandler
from utils.dataset import AgeDBDataset


def get_image_list(train_size, test_size, val_size, directory):
    
    handler = DataHandler(test_size=test_size, val_size=val_size, train_size=train_size)
    handler.imageList(directory)
    handler.findAge()
    handler.TestValTrainNum()
    train_image_list, test_image_list, validate_image_list = handler.test_val_train_list()
    
    return train_image_list, test_image_list, validate_image_list


def get_loader(input_size, train_image_list, test_image_list, validate_image_list, batch_size, train_augment=3):
    
    device = torch.device("cuda")
    
    train_transform = A.Compose([
        A.Resize(input_size, input_size),
        A.ToGray(p=1),
        A.Rotate(limit=10, p=0.9),
        A.HorizontalFlip(p=0.75),
        A.OneOf([
            A.Blur(blur_limit=3),
            A.OpticalDistortion(),
        ], p=1),
        ToTensorV2()
    ])

    test_transform = A.Compose([
        A.Resize(input_size, input_size),
        A.ToGray(p=1),
        ToTensorV2()
    ])


    trainDataset = AgeDBDataset(image_list=train_image_list,
                                device=device,
                                train=True,
                                augment=train_augment,
                                train_transform=train_transform,
                                test_transform=test_transform)
    testDataset = AgeDBDataset(image_list=test_image_list,
                               device=device,
                               train=False,
                               augment=0,
                               train_transform=train_transform,
                               test_transform=test_transform)
    valDataset = AgeDBDataset(image_list=validate_image_list,
                              device=device,
                              train=False,
                              augment=0,
                              train_transform=train_transform,
                              test_transform=test_transform)

    print('Train Set Length: ', len(trainDataset))
    print('Test Set Length: ', len(testDataset))
    print('Validation Set Length: ', len(valDataset))
    print('Total: ', len(trainDataset) + len(testDataset) + len(valDataset))

    train_loader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testDataset, batch_size=batch_size, shuffle=False)
    validation_loader = DataLoader(valDataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, validation_loader