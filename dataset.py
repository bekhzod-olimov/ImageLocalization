import torch, cv2

class ObjectLocalizationDataset(torch.utils.data.Dataset):
    
    # Initialization
    def __init__(self, df, data_dir, augmentations=None):
        
        # Get arguments
        self.df = df
        self.augs = augmentations
        self.data_dir = data_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        # Get an example from the dataset
        example = self.df.iloc[idx]

        # Get the coordinates of a bounding box
        xmin = example.xmin
        xmax = example.xmax
        ymin = example.ymin
        ymax = example.ymax

        # Create bounding box using the coordinates
        bbox = [[xmin, ymin, xmax, ymax]]

        # Get image path
        im_path = self.data_dir + example.img_path
        im = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB)

        # Apply augmentations
        if self.augs:
            
            data = self.augs(image=im, bboxes=bbox, class_labels=[None])
            im = data['image']
            bbox = data['bboxes'][0]

        # 
        im = torch.from_numpy(im).permute(2, 0, 1) / 255.
        bbox = torch.Tensor(bbox)

        return im, bbox
