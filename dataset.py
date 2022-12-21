import torch, cv2

class ObjectLocalizationDataset(torch.utils.data.Dataset):
    
    def __init__(self, df, data_dir, augmentations=None):
        self.df = df
        self.augs = augmentations
        self.data_dir = data_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        example = self.df.iloc[idx]

        xmin = example.xmin
        xmax = example.xmax
        ymin = example.ymin
        ymax = example.ymax

        bbox = [[xmin, ymin, xmax, ymax]]

        im_path = self.data_dir + example.img_path
        im = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB)

        if self.augs:
            data = self.augs(image=im, bboxes=bbox, class_labels=[None])
            im = data['image']
            bbox = data['bboxes'][0]

        im = torch.from_numpy(im).permute(2, 0, 1) / 255.
        bbox = torch.Tensor(bbox)

        return im, bbox