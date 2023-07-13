import json
import argparse
import os
import pprint
import warnings
import torchvision

from torch.utils.data import DataLoader

import dataset
from models.model import *

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--backbone', default='iresnet18', type=str,
                    help='backbone architechture')
parser.add_argument('--pretrained_backbone', default=False, type=bool,
                    help='Use pretrain backbone')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--folderdataset_dir', default='datasets/raw', type=str,
                    help='Path to Face Image Folder Dataset')
parser.add_argument('--feat_list', default='datasets/face_embdding.txt', type=str,
                    help='Path for saveing features file')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                    'batch size of all GPUs on the current node when '
                    'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--embedding-size', default=512, type=int,
                    help='The embedding feature size')
parser.add_argument('--cpu-mode', action='store_true', help='Use the CPU.')

args=parser.parse_args()

def main(args):
    device=torch.device("cuda" if (not args.cpu_mode) &(torch.cuda.is_available()) else "cpu") 
    
    embd_model= load_dict_inf(args, build_backbone(args).to(device))
    torch.cuda.empty_cache()
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((112, 112)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0., 0., 0.],
                                         std=[1., 1., 1.]),
    ])    

    data_inf = dataset.FaceImageFolderDataset(root= args.folderdataset_dir, transform = transform)

    dataloader_inf=DataLoader(data_inf, 
                              batch_size= args.batch_size, 
                              num_workers= args.workers,
                              pin_memory=False,
                              shuffle=False,
    )

    label_map_path= os.path.join("/".join(args.feat_list.split('/')[:-1]),'label_map.json')
    cprint('=> starting face embdding...', 'green')
    cprint('=> embdding feature will be saved into {}'.format(args.feat_list))
    cprint('=> mapping label will be saved into {}'.format(label_map_path))

    with open(label_map_path,'w') as f:
        json.dump(data_inf.get_label_map(), f)
        
    embd_model.eval()    
    file= open(args.feat_list, 'w')

    with torch.no_grad():
        for i, (imgs, labels) in enumerate(dataloader_inf):
            imgs=imgs.to(device)
            embedding_feats = embd_model(imgs)
            embedding_feats = embedding_feats.data.cpu().numpy()
            
            for feat, label in zip(embedding_feats, labels):
                file.write('{} '.format(label))
                for r in feat:
                    file.write('{} '.format(r))
                file.write('\n')
    file.close()

if __name__ == '__main__':

    pprint.pprint(vars(args))
    main(args)
