# Import libraries
import os, argparse, yaml, cv2, torch, torchvision, timm, pandas as pd, numpy as np, albumentations as A
from torch.nn import *
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from dataset import ObjectLocalizationDataset
from model import Model

def run(args):
    
    # Get the arguments
    bs, device, path, data_path, model_name, lr = args.batch_size, args.device, args.ims_path, args.data_path, args.model_name, args.learning_rate
    
    argstr = yaml.dump(args.__dict__, default_flow_style = False)
    print(f"\nTraining Arguments:\n{argstr}")
    
    # Set train variables
    epochs, img_size, num_classes  = 300, 140, 4
    
    # Read the csv data
    df = pd.read_csv(path)
    
    # Split the data into train and validation parts
    train_df, valid_df = train_test_split(df, test_size = 0.2, random_state = 42)
        
    # Initialize train dataset transformations
    train_augmentations = A.Compose([A.Resize(img_size, img_size), 
                                     A.HorizontalFlip(p = 0.5),
                                     A.VerticalFlip(p = 0.5),
                                     A.Rotate()
                                     ], bbox_params = A.BboxParams(format = "pascal_voc", label_fields = ["class_labels"]))
    
    # Initialize validation dataset transformations
    valid_augmentations = A.Compose([A.Resize(img_size, img_size),
                                    ], bbox_params = A.BboxParams(format = "pascal_voc", 
                                    label_fields = ["class_labels"]))
    
    # Get train and validation datasets
    trainset, validset = ObjectLocalizationDataset(train_df, augmentations = train_augmentations, data_dir = data_path), ObjectLocalizationDataset(valid_df, augmentations = valid_augmentations, data_dir = data_path)
    print(f"Number of training samples: {len(trainset)}")
    print(f"Number of validation samples: {len(validset)}\n")
    
    # Initialize train and validation dataloaders
    trainloader, validloader = torch.utils.data.DataLoader(trainset, batch_size = bs, shuffle = True), torch.utils.data.DataLoader(validset, batch_size = bs, shuffle = False)
   
    # Double check the train dataloader
    for im, bbox in trainloader:
        break;
    print("Shape of one batch images : {}".format(im.shape))
    print("Shape of one batch bboxes : {}".format(bbox.shape))
    
    # Double check the validation dataloader
    for im, bbox in validloader:
        break;
    print("Shape of one batch images : {}".format(im.shape))
    print("Shape of one batch bboxes : {}".format(bbox.shape))
    
    # Initialize train model and move it to gpu
    model = Model(model_name, num_classes)
    model.to(device)
    
    # Train function
    def train_fn(model, dl, opt):
        
        """
        
        This function gets train model, train dataloader, and optimizer performs one epoch of training and returns loss value.
        
        Parameters:
        
            model       - train model, timm model object;
            dl          - train dataloader, torch dataloader object;
            opt         - optimizer, torch optimizer object.        
        
        """
        
        # Set initial loss value
        total_loss = 0.
        
        # Turn train model for the model
        model.train()
        
        # Go through train dataloader
        for batch in tqdm(dl):
            
            # Get images and bounding boxes
            ims, gts = batch
            
            # Move them to gpu
            ims, gts = ims.to(device), gts.to(device)
            
            # Get bounding boxes, and loss value
            bboxes, loss = model(ims, gts)
            
            # Zero grad for the optimizer
            opt.zero_grad()
            
            # Backprop and optimizer step
            loss.backward()
            opt.step()
            
            # Add batch loss to the total loss
            total_loss += loss.item()
            
        # Return average loss for the epoch
        return total_loss / len(dl)
    
    # Validation function
    def eval_fn(model, dl):
        
        """
        
        This function gets train model, validation dataloader and performs validation step 
        and returns validation loss value.
        
        Parameters:
        
            model    - a model, timm model object;
            dl       - validation dataloader, torch dataloader object.
        
        """
        
        # Set initial loss value
        total_loss = 0.
        
        # Change to model evaluation mode
        model.eval()
        
        # Turn off gradient calculation
        with torch.no_grad():
            
            # Go through validation dataloader
            for batch in tqdm(dl):
                
                # Get images and ground truth bounding boxes
                ims, gts = batch
                
                # Move them to gou
                ims, gts = ims.to(device), gts.to(device)
                
                # Get predicted bounding boxes and validation loss value
                bboxes, loss = model(ims, gts)
                
                # Add the loss for the mini-batch to the total loss
                total_loss += loss.item()

        # Return average batch for the dataloader
        return total_loss / len(dl)

    # Initialize optimizer
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Set the best validation loss
    best_valid_loss = np.Inf
    
    # Start training
    for epoch in range(epochs):
        
        # Get train and validation losses
        train_loss = train_fn(model, trainloader, opt)
        valid_loss = eval_fn(model, validloader)
        
        # Save the model with the best loss value
        if valid_loss < best_valid_loss:
            torch.save(model.state_dict(), "best_model.pt")
            print("Best weights are saved!")
            best_valid_loss = valid_loss
            
        # Training progress
        print(f"Epoch {epoch + 1} train loss: {train_loss:.3f}")
        print(f"Epoch {epoch + 1} valid loss: {valid_loss:.3f}") 

if __name__ == "__main__":
    
    # Initialize Argument Parser    
    parser = argparse.ArgumentParser(description = "Image Localization Training Arguments")
    
    # Add arguments to the parser
    parser.add_argument("-bs", "--batch_size", type = int, default = 64, help = "Batch size")
    parser.add_argument("-d", "--device", type = str, default = 'cuda:0', help = "GPU device number")
    parser.add_argument("-ip", "--ims_path", type = str, default = './train.csv', help = "Path to the images")
    parser.add_argument("-dp", "--data_path", type = str, default = './', help = "Path to the data")
    parser.add_argument("-mn", "--model_name", type = str, default = 'efficientnet_b3a', help = "Model name (from timm library (ex. darknet53, ig_resnext101_32x32d))")
    parser.add_argument("-lr", "--learning_rate", type = float, default = 1e-3, help = "Learning rate value") # from find_lr
    
    # Parse the arguments
    args = parser.parse_args() 
    
    # Run the script
    run(args) 
