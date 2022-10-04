from torchvision import transforms
import torch

import cv2
import os
import pickle
import numpy as np
from numpy import concatenate
from sklearn.decomposition import PCA


def get_next_img_path(img_path) :
    no_extension = img_path.split(".")[-2]
    img_nb = int(no_extension.split("/")[-1])
    next_img_nb = img_nb + 1
    next_img_path = str(next_img_nb).join(img_path.rsplit(str(img_nb), 1))
    a = os.path.isfile(img_path)
    return next_img_path


def get_tensor(flw_path, img_path):
    frame_anchor = cv2.imread(img_path)
    img_transform = transforms.Compose([
        transforms.ToTensor()
        , transforms.Normalize(
            mean=[0.485, 0.456, 0.406, 0.485, 0.456, 0.406, 0.5, 0.5],
            std=[0.229, 0.224, 0.225, 0.229, 0.224, 0.225, 0.25, 0.25]),
            # mean = [0.485, 0.456, 0.406, 0.5, 0.5],
            # std = [0.229, 0.224, 0.225, 0.25, 0.25]),
    ])
    frame2_anchor = cv2.imread(get_next_img_path(img_path))
    frame_anchor = np.concatenate((frame_anchor, frame2_anchor), axis=2)

    flow_anchor = cv2.imread(flw_path)[:, :, :2]
    fusion = concatenate((frame_anchor, flow_anchor), axis=2)
    tensor = img_transform(fusion)
    return tensor


def extract_fv(model, flw_path, img_path) :
    tensor_img = get_tensor(flw_path, img_path)
    tensor_img = tensor_img.unsqueeze(0)#.cuda()
    fv = model(tensor_img)
    fv = fv.view(-1)
    return fv


def extract_and_chronologically_stack_features(model, frames_and_flow_path, force_reset, fv_len=32):
    stacked_fv_path = os.path.join(frames_and_flow_path, "feature_vectors.pkl")

    if os.path.isfile(stacked_fv_path) and not force_reset :
        print("Pre-existing feature vectors frames and flow extraction.")
        stacked_fv = pickle.load(open(stacked_fv_path, 'rb'))
        return stacked_fv

    frames_path = os.path.join(frames_and_flow_path, "frames")
    flow_path = os.path.join(frames_and_flow_path, "flow")
    files = os.listdir(flow_path)
    images_nb = len(files)

    model = model.cpu()
    model.eval()
    stacked_tensors = torch.zeros((images_nb, fv_len))#.cuda()
    for root, dirs, files in os.walk(flow_path) :
        files.sort()
        files.sort(key=len, reverse=False)
        for t, file1 in enumerate(files) :
            flw_path = os.path.join(root, file1)
            img_path = flw_path.replace(flow_path, frames_path)
            fv = extract_fv(model, flw_path, img_path).detach()
            stacked_tensors[t] = fv
    stacked_tensors = stacked_tensors.cpu().numpy()
    pickle.dump(stacked_tensors, open(stacked_fv_path, 'wb'))
    print("Feature vectors extraction completed.")
    return stacked_tensors

def perform_PCA(stacked_features):
    pca = PCA(n_components=1)
    signal_1d = pca.fit_transform(stacked_features)
    return signal_1d