import os, argparse, yaml, cv2, torch, torchvision, timm
from torch.nn import *
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
import albumentations as A
from dataset import ObjectLocalizationDataset
from model import Model
import matplotlib.pyplot as plt

def run(args):
    
    # Get train arguments    
    bs = args.batch_size
    device = args.device
    path = args.ims_path
    data_path = args.data_path
    model_name=args.model_name
    model_path = args.model_path
    argstr = yaml.dump(args.__dict__, default_flow_style=False)
    print(f"\nTraining Arguments:\n{argstr}\n")
    
    # Set train variables
    img_size = 140
    num_classes = 4
    
    # Read the data
    df = pd.read_csv(path)
    
    # Split the data into train and validation sets
    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)
        
    # Get train transformations  
    train_augmentations = A.Compose([A.Resize(img_size, img_size), 
                                     A.HorizontalFlip(p=0.5),
                                     A.VerticalFlip(p=0.5),
                                     A.Rotate()
                                     ], bbox_params=A.BboxParams(format='pascal_voc', 
                                     label_fields = ['class_labels']))
    
    # Get validation transformations
    valid_augmentations = A.Compose([A.Resize(img_size, img_size),
                                    ], bbox_params=A.BboxParams(format='pascal_voc', 
                                    label_fields = ['class_labels']))
    
    # Get train and validation datasets
    trainset = ObjectLocalizationDataset(train_df, augmentations=train_augmentations, data_dir=data_path)
    validset = ObjectLocalizationDataset(valid_df, augmentations=valid_augmentations, data_dir=data_path)
    print(f"Number of training samples: {len(trainset)}")
    print(f"Number of validation samples: {len(validset)}\n")
    
    # Get train and validation dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True)
    validloader = torch.utils.data.DataLoader(validset, batch_size=bs, shuffle=False)
    
    # Double check the traindataloader
    for im, bbox in trainloader:
        break;
    print("Shape of one batch images : {}".format(im.shape))
    print("Shape of one batch bboxes : {}".format(bbox.shape))
    
    # Get the train model
    model = Model(model_name, num_classes)
    # from the checkpoint and change to gpu
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    # Turn evaluation mode on
    model.eval();
    
    # Vizualization function    
    def vizualization(images, gt_bboxes, out_bboxes):
        
        """
        Gets images, ground truth bounding boxes, and predicted bounding boxes
        and displays them as comparison: image + ground truth with bounding box
        image + predicted bounding box.

        Arguments:
        images - input images;
        gt_bboxes - ground truth;
        out_bboxes - predicted bounding boxes.
        
        """
        
        
        
        for i, im in enumerate(images):
            xmin, ymin, xmax, ymax = gt_bboxes[i]
            pt1 = (int(xmin), int(ymin))
            pt2 = (int(xmax), int(ymax))
            
            out_xmin, out_ymin, out_xmax, out_ymax = out_bboxes[i][0]
            out_pt1 = (int(out_xmin), int(out_ymin))
            out_pt2 = (int(out_xmax), int(out_ymax))
            
            out_img = cv2.rectangle(im.squeeze().permute(1, 2, 0).cpu().numpy(),pt1, pt2,(0,255,0),2)
            out_img = cv2.rectangle(out_img,out_pt1, out_pt2,(255,0,0),2)
            plt.subplot(2, 4, i+1)      
            plt.imshow(out_img)
            plt.axis('off')
    
    ims, gts, bbs = [], [], []
    for i, batch in enumerate(validset):
        if i == 8:
            break
        with torch.no_grad():
            im, gt = batch
            im = im.unsqueeze(0).to(device)
            pred_bbox = model(im, gt.to(device))
            ims.append(im)
            gts.append(gt)
            bbs.append(pred_bbox[0])
    plt.figure(figsize=(60,40))
    vizualization(ims, gts, bbs)   
     

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Image Localization Training Arguments')
    parser.add_argument("-bs", "--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("-d", "--device", type=str, default='cuda:0', help="GPU device number")
    parser.add_argument("-ip", "--ims_path", type=str, default='./train.csv', help="Path to the images")
    parser.add_argument("-dp", "--data_path", type=str, default='./', help="Path to the data")
    parser.add_argument("-mn", "--model_name", type=str, default='efficientnet_b3a', help="Model name (from timm library (ex. darknet53, ig_resnext101_32x32d))")
    parser.add_argument("-mp", "--model_path", type=str, default='./best_model.pt', help="Path to the trained model")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3, help="Learning rate value") # from find_lr
    args = parser.parse_args() 
    
    run(args) 
