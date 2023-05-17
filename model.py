# Import libraries
import torch, timm

class Model(torch.nn.Module):
    
    """
    
    This class gets several parameters and returns model to be trained.
    
    Parameters:
    
        model_name    - name of a model to be trained, str;
        num_classes   - number of classes in the dataset for training, int.
    
    """
    
    # Initialization
    def __init__(self, model_name, num_classes):
        super(Model, self).__init__()
        
        # Backbone for the localization model
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        print(f"{model_name} with num_classes of {num_classes} is successfully loaded!")
        
    def forward(self, ims, bboxes=None):
        
        # Get bounding boxes
        bbs = self.backbone(ims)
        
        # Compute loss
        if bboxes != None:
            
            bboxes = bboxes.unsqueeze(0)
            loss = torch.nn.MSELoss()(bbs, bboxes)       
            
            return bbs, loss
