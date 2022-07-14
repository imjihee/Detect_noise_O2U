from  torch.utils import data
from  PIL import  Image
import numpy as np
from  io import BytesIO
"""
Select data with small loss
self.train_data, self.train_noisy_labels: cleaned data and label
"""
class Mask_Select(data.Dataset): 
    def __init__(self, origin_dataset, mask_index, idx_sorted, curriculum):
        self.transform = origin_dataset.transform
        self.target_transform = origin_dataset.target_transform
        labels=origin_dataset.train_noisy_labels
        dataset=origin_dataset.train_data
        self.dataname=origin_dataset.dataset
        self.origin_dataset=origin_dataset

        if curriculum:
            print("Build Curriculum Dataset")
            self.train_data=[0]*len(labels)
            self.train_noisy_labels=[-1]*len(labels)
            for i,m in enumerate(mask_index): #mask_index: {index:0/1}
                if m<0.5:
                    continue
                #mask=1인 경우
                idx = idx_sorted.index(i) #mask index에 해당하는 데이터의 loss 순서 반환(오름차순)
                self.train_data[idx] = dataset[i] #해당 순서에 data 저장 --> Loss 순서대로 데이터셋 정렬
                self.train_noisy_labels[idx] = labels[i]

            self.train_data = [i for i in self.train_data if 'array' in str(type(i))]
            self.train_noisy_labels = [i for i in self.train_noisy_labels if i != -1]
        else:
            self.train_data = []
            self.train_noisy_labels = []
            for i, m in enumerate(mask_index):
                if m < 0.5:
                    continue
                # mask=1인 경우
                self.train_data.append(dataset[i])
                self.train_noisy_labels.append(labels[i])


        print ("origin set number:%d"%len(labels),"after clean number:%d" % len(self.train_noisy_labels))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if index>45000:
            print(index)
        img, target = self.train_data[index], self.train_noisy_labels[index]

        if self.dataname!='MinImagenet':
            img = Image.fromarray(img)

        #if self.transform is not None:
        #    img = self.transform(img)

        #if self.target_transform is not None:
        #    target = self.target_transform(target)
        img = self.target_transform(img)

        return img, target, index

    def __len__(self):
        return len(self.train_data)
