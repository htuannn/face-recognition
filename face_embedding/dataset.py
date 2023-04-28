from torch.utils.data import Dataset
import torch

from PIL import Image
import torchvision
import numpy as np
import os
import logging

class FaceDataset(Dataset):

    def __init__(self, root= 'datasets/preprocessed', transform=None):
        self.root = root
        self.transform = transform

        self.img_paths = None
        self.img_id_labels = None


    def __len__(self):
        return self.get_n_images()


    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError

        img_path = self.img_paths[idx]
        img_label = self.img_id_labels[idx]

        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, img_label


    def get_n_identities(self):
        return len(np.unique(self.img_id_labels))


    def get_n_images(self):
        return len(self.img_paths)


    def reduce_to_N_identities(self, N, shuffle=False):
        assert N > 0 and N <= len(self)

        id_labels = self.img_id_labels
        ids = np.unique(self.img_id_labels)

        chosen_ids = np.random.choice(ids, N, replace=False) if shuffle else ids[:N]

        chosen_idxs = np.where(np.isin(id_labels, chosen_ids))[0]

        label_map = {id: n for n, id in enumerate(chosen_ids)}

        self.img_id_labels = [label_map[self.img_id_labels[idx]] for idx in chosen_idxs]
        self.img_paths = [self.img_paths[idx] for idx in chosen_idxs]


    def reduce_to_sample_idxs(self, idxs):
        chosen_ids = np.unique(np.array(self.img_id_labels)[idxs])
        chosen_idxs = idxs
        label_map = {id: n for n, id in enumerate(chosen_ids)}

        self.img_id_labels = [label_map[self.img_id_labels[idx]] for idx in chosen_idxs]
        self.img_paths = [self.img_paths[idx] for idx in chosen_idxs]

class FaceImageFolderDataset(FaceDataset):

    def __init__(self, auto_initialize=True, **kwargs):
        super(FaceImageFolderDataset, self).__init__(**kwargs)

        self.img_paths = []
        self.img_ids = []
        self.img_id_labels = []
        self.label_map={}

        if auto_initialize:
            self.init_from_directories()


    def init_from_directories(self):
        if not self.dataset_exists():
            logging.warning(f"The dataset does not contain any images under {self.root}")
            return

        logging.info(f"Creating a FaceImageFolderDataset with data from {self.root}.")

        images_dir = self.root

        for label, identity in enumerate(os.listdir(images_dir)):
            id_path = os.path.join(images_dir, identity)
            self.label_map[label]=identity
            for img_file in os.listdir(id_path):
                self.img_paths.append(os.path.join(id_path, img_file))
                self.img_ids.append(identity)
                self.img_id_labels.append(label)

    def get_label_map(self):
        return self.label_map
    def dataset_exists(self):
        images_dir = os.path.join(self.root)
        return os.path.isdir(images_dir) and len(os.listdir(images_dir)) > 0

def train_loader(args):
    if (len(args.folderdataset_dir)>0):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((112, 112)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomHorizontalFlip()
        ])
        train_dataset = FaceImageFolderDataset(
            root= args.folderdataset_dir,
            transform=transform
        )
    else:
        train_trans = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
        ])
        train_dataset = MagTrainDataset(
            args.train_list,
            transform=train_trans
        )
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=(train_sampler is None),
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=(train_sampler is None))

    return train_loader