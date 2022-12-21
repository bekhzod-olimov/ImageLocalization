import torch, timm

class Model(torch.nn.Module):
    def __init__(self, model_name, num_classes):
        super(Model, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        print(f"{model_name} with num_classes of {num_classes} is successfully loaded!")
        
    def forward(self, ims, bboxes=None):
        bbs = self.backbone(ims)
        if bboxes != None:
            bboxes = bboxes.unsqueeze(0)
            loss = torch.nn.MSELoss()(bbs, bboxes)            
            return bbs, loss