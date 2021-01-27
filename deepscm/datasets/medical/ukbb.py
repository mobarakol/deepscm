from torch.utils.data.dataset import Dataset
import numpy as np
import pandas as pd
import os
import nibabel as nib
from nilearn.image import resample_img
import torch

class UKBBDataset(Dataset):
    def __init__(self, csv_path, base_path='/vol/biobank/12579/brain/rigid_to_mni/images', crop_type=None, crop_size=(64, 64, 64), downsample: float = 2.5):#(64, 64, 64)
        super().__init__()
        self.csv_path = csv_path
        df = pd.read_csv(csv_path)
        self.num_items = len(df)
        self.metrics = {col: torch.as_tensor(df[col]).float() for col in df.columns}
        self.base_path = base_path
        self.filename = 'T1_unbiased_brain_rigid_to_mni.nii.gz'
        self.crop_size = np.array(crop_size)
        self.downsample = downsample

    def __len__(self):
        return self.num_items

    def __getitem__(self, index):
        item = {col: values[index] for col, values in self.metrics.items()}
        mri_path = os.path.join(self.base_path,str(int(item['eid'])),self.filename)
        try:
            img = nib.load(mri_path)#.get_data()
        except:
            index += 1
            item = {col: values[index] for col, values in self.metrics.items()}
            mri_path = os.path.join(self.base_path,str(int(item['eid'])),self.filename)
            img = nib.load(mri_path)#.get_data()

        downsampled_nii = resample_img(img, target_affine=np.eye(3)*self.downsample, interpolation='linear')
        img = downsampled_nii.dataobj
        init_pos = np.round(np.array(img.shape)/2-self.crop_size/2).astype(int)
        end_pos = init_pos+self.crop_size
        min_ = np.min(img)
        max_ = np.max(img)
        img = (img - min_) / (max_ - min_)
        item['image'] = np.expand_dims(img[init_pos[0]:end_pos[0],init_pos[1]:end_pos[1],init_pos[2]:end_pos[2]], axis=0)
        return item
