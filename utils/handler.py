import os
import torch
from tqdm.notebook import tqdm
from PIL import Image
from parse import parse
from torch.utils.data import Dataset
import random


class DataHandler():
    
    def __init__(self, test_size, val_size, train_size):
        self.test_size = test_size
        self.val_size = val_size
        self.train_size = train_size

        self.listofzeros = [0] * 102
        self.same_age = [i for i in range(0, 102)]
        self.theta1 = 0.998
        self.theta2 = 0.997


    def imageList(self, directory):
        self.directory = directory
        self.image_list = []
        for i, file in enumerate(sorted(os.listdir(self.directory))):
            file_labels = parse('{}_{}_{age}_{gender}.jpg', file)

            if file_labels is None:
                continue

            image_location = os.path.join(self.directory, file)
            gender_to_class_id = {'m': 0, 'f': 1}
            gender = gender_to_class_id[file_labels['gender']]
            age = int(file_labels['age'])
            self.image_list.append({
                'image_location': image_location,
                'age': age,
                'gender': gender
            })

        random.shuffle(self.image_list)
        return self.image_list  ###################


    def findAge(self):
        self.image_num_list = {}
        self.image_num_list = dict(zip(self.same_age, self.listofzeros))

        for i in range(len(self.image_list)):
            self.image_num_list[self.image_list[i]['age']] += 1

        return self.image_num_list  ##################


    def TestValTrainNum(self):
        self.num_test_imgs = {}
        self.num_val_imgs = {}
        self.num_train_imgs = {}

        self.num_test_imgs = dict(zip(self.same_age, self.listofzeros))
        self.num_val_imgs = dict(zip(self.same_age, self.listofzeros))
        self.num_train_imgs = dict(zip(self.same_age, self.listofzeros))

        for i in range(len(self.image_list)):
            self.num_train_imgs[self.image_list[i]['age']] += 1

        # For Test
        for i in range(len(self.image_num_list)):

            if self.num_train_imgs[i] != 0:

                if round(self.num_train_imgs[i] * self.test_size) <= 0:
                    self.num_test_imgs[i] = 1
                    self.num_train_imgs[i] -= 1

                else:
                    self.num_test_imgs[i] = round(self.image_num_list[i] *
                                                  self.test_size * self.theta1)
                    self.num_train_imgs[
                        i] = self.num_train_imgs[i] - self.num_test_imgs[i]

        # For Validation
        for i in range(len(self.image_num_list)):

            if self.num_train_imgs[i] != 0:

                if round(self.num_train_imgs[i] * self.val_size) <= 0:
                    self.num_val_imgs[i] = 1
                    self.num_train_imgs[i] -= 1

                else:
                    self.num_val_imgs[i] = round(self.image_num_list[i] *
                                                 self.val_size * self.theta2)
                    self.num_train_imgs[
                        i] = self.num_train_imgs[i] - self.num_val_imgs[i]

        # Test, Validation, Train
        return self.num_test_imgs, self.num_val_imgs, self.num_train_imgs  ##############


    def test_val_train_list(self):
        self.test_image_list = []
        self.validate_image_list = []
        self.train_image_list = []

        n_test_imgs = self.num_test_imgs.copy()
        n_val_imgs = self.num_val_imgs.copy()
        n_train_imgs = self.num_train_imgs.copy()

        for i in range(len(self.image_list)):

            if n_test_imgs[self.image_list[i]['age']] > 0:
                self.test_image_list.append(self.image_list[i])
                n_test_imgs[self.image_list[i]['age']] -= 1

            elif n_val_imgs[self.image_list[i]['age']] > 0:
                self.validate_image_list.append(self.image_list[i])
                n_val_imgs[self.image_list[i]['age']] -= 1

            else:
                self.train_image_list.append(self.image_list[i])
                n_train_imgs[self.image_list[i]['age']] -= 1

        return  self.train_image_list, self.test_image_list, self.validate_image_list  ##############
