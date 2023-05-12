# Import libraries
import torch, cv2

class ObjectLocalizationDataset(torch.utils.data.Dataset):
    
    """
    
    This class gets several parameters and returns dataset.
    
    Parameters:
    
        df              - input dataframe, pandas object;
        data_dir        - path to directory with data, str;
        augmentations   - augmentations to be applied, albumentations object.
    
    """
    
    # Initialization
    def __init__(self, df, data_dir, augmentations = None):
        
        # Get arguments
        self.df, self.augs, self.data_dir = df, augmentations, data_dir

    # Get length of the dataset
    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        
        # Get an example from the dataset
        example = self.df.iloc[idx]

        # Get the coordinates of a bounding box
        xmin, xmax, ymin, ymax = example.xmin, example.xmax, example.ymin, example.ymax

        # Create bounding box using the coordinates
        bbox = [[xmin, ymin, xmax, ymax]]

        # Get image path
        im_path = self.data_dir + example.img_path
        im = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB)

        # Apply augmentations
        if self.augs:
            
            data = self.augs(image = im, bboxes = bbox, class_labels = [None])
            im, bbox = data["image"], data["bboxes"][0]

        # Transform to tensor
        im, bbox = torch.from_numpy(im).permute(2, 0, 1) / 255., torch.Tensor(bbox)

        return im, bbox
