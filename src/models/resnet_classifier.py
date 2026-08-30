import torch
import torch.nn as nn
from torchvision import models

class ClothingClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ClothingClassifier, self).__init__()
        
        # 1. Load the pre-trained ResNet18 model
        # The 'DEFAULT' weights mean it has already learned to see from millions of images!
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # 2. ResNet18 was originally designed to classify 1000 different object types.
        # We replace the very last layer (the Fully Connected or 'fc' layer) 
        # so it outputs our specific number of clothing classes instead.
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        return self.model(x)

