from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import os

import cv2
import numpy as np
import pickle


class Sequenced_Data (Dataset) :
    def __init__(self, frames_and_flow_path) :
        self.two_frames = True
        self.transform = self.get_transform()

        videos = os.path.join(frames_and_flow_path, "frames")
        flow = os.path.join(frames_and_flow_path, "flow")

        frames = os.listdir(videos)
        frames.sort()
        frames.sort(key=len)
        frames = [os.path.join(videos, f) for f in frames]

        flows = os.listdir(flow)
        flows.sort()
        flows.sort(key=len)
        flows = [os.path.join(flow, f) for f in flows]

        self.frames = frames
        self.flows = flows

    def __len__(self) :
        return len(self.flows) - 2

    def __getitem__(self, idx):
        anchor = self.get_tensor(idx)
        positive = self.get_tensor(idx + 1)
        negative = self.get_tensor(idx + 2)
        return {"anchor": anchor,
                "positive": positive,
                "negative": negative}

    def get_tensor(self, idx):
        frame_anchor = cv2.imread(self.frames[idx])
        if self.two_frames :
            frame2_anchor = cv2.imread(self.frames[idx])
            frame_anchor = np.concatenate((frame_anchor, frame2_anchor), axis=2)
        flow_anchor = cv2.imread(self.flows[idx])[:,:,:2]
        fusion = np.concatenate((frame_anchor, flow_anchor), axis=2)
        tensor = self.transform(fusion)
        return tensor


    def get_transform(self) :
        if self.two_frames :
            img_transform = transforms.Compose([
                transforms.ToTensor()
                , transforms.Normalize(
                    mean=[0.485, 0.456, 0.406, 0.485, 0.456, 0.406, 0.5, 0.5],
                    std=[0.229, 0.224, 0.225, 0.229, 0.224, 0.225, 0.25, 0.25]),
            ])
        else :
            img_transform = transforms.Compose([
             transforms.ToTensor()
            ,transforms.Normalize(
                mean=[0.485, 0.456, 0.406, 0.5, 0.5],
                std=[0.229, 0.224, 0.225, 0.25, 0.25]),
            ])
        return img_transform


def get_dataloaders(frames_and_flow_path, batch_size=16, train_test_ratio=0.8, shuffle=True):
    dataset = Sequenced_Data(frames_and_flow_path)
    if train_test_ratio != 1 :
        train_size = int(train_test_ratio * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        test_dataset.data_augmentation = False
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
        return train_dataloader, test_dataloader
    else :
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader