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

def run(args):
    
    # Get the arguments
    bs = args.batch_size
    device = args.device
    path = args.ims_path
    data_path = args.data_path
    model_name=args.model_name
    lr = args.learning_rate
    argstr = yaml.dump(args.__dict__, default_flow_style=False)
    print(f"\nTraining Arguments:\n{argstr}")
    
    # Set train variables
    epochs, img_size, num_classes  = 300, 140, 4
    
    df = pd.read_csv(path)
    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)
        
    train_augmentations = A.Compose([A.Resize(img_size, img_size), 
                                     A.HorizontalFlip(p=0.5),
                                     A.VerticalFlip(p=0.5),
                                     A.Rotate()
                                     ], bbox_params=A.BboxParams(format='pascal_voc', 
                                     label_fields = ['class_labels']))

    valid_augmentations = A.Compose([A.Resize(img_size, img_size),
                                    ], bbox_params=A.BboxParams(format='pascal_voc', 
                                    label_fields = ['class_labels']))
    
    trainset = ObjectLocalizationDataset(train_df, augmentations=train_augmentations, data_dir=data_path)
    validset = ObjectLocalizationDataset(valid_df, augmentations=valid_augmentations, data_dir=data_path)
    print(f"Number of training samples: {len(trainset)}")
    print(f"Number of validation samples: {len(validset)}\n")
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True)
    validloader = torch.utils.data.DataLoader(validset, batch_size=bs, shuffle=False)
    for im, bbox in trainloader:
        break;
    print("Shape of one batch images : {}".format(im.shape))
    print("Shape of one batch bboxes : {}".format(bbox.shape))
    
    model = Model(model_name, num_classes)
    model.to(device)
    
    def train_fn(model, dl, opt):
        
        total_loss = 0.
        model.train()
        
        for batch in tqdm(dl):
            ims, gts = batch
            ims, gts = ims.to(device), gts.to(device)
            bboxes, loss = model(ims, gts)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            
        return total_loss / len(dl)
    
    def eval_fn(model, dl):
        
        total_loss = 0.
        model.eval()
        with torch.no_grad():
            
            for batch in tqdm(dl):
                ims, gts = batch
                ims, gts = ims.to(device), gts.to(device)
                bboxes, loss = model(ims, gts)
                total_loss += loss.item()

        return total_loss / len(dl)
    
    
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_valid_loss = np.Inf
    
    for epoch in range(epochs):
        train_loss = train_fn(model, trainloader, opt)
        valid_loss = eval_fn(model, validloader)
        
        if valid_loss < best_valid_loss:
            torch.save(model.state_dict(), "best_model.pt")
            print("Best weights are saved!")
            best_valid_loss = valid_loss
        print(f"Epoch {epoch + 1} train loss: {train_loss:.3f}")
        print(f"Epoch {epoch + 1} valid loss: {valid_loss:.3f}") 

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Image Localization Training Arguments')
    parser.add_argument("-bs", "--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("-d", "--device", type=str, default='cuda:0', help="GPU device number")
    parser.add_argument("-ip", "--ims_path", type=str, default='./train.csv', help="Path to the images")
    parser.add_argument("-dp", "--data_path", type=str, default='./', help="Path to the data")
    parser.add_argument("-mn", "--model_name", type=str, default='efficientnet_b3a', help="Model name (from timm library (ex. darknet53, ig_resnext101_32x32d))")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3, help="Learning rate value") # from find_lr
    args = parser.parse_args() 
    
    run(args) 
