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

if __name__ == "__main__":
    # Let's test if the model builds correctly. 
    # Assume we will have 5 clothing classes (e.g., Shirt, Pants, Dress, Shoes, Hat)
    print("Building model...")
    test_model = ClothingClassifier(num_classes=5)
    
    # Create a fake, random image to test the network
    # Format: [Batch Size, Color Channels (RGB), Height, Width]
    # Note: 224x224 is the standard required input size for ResNet
    dummy_image = torch.randn(1, 3, 224, 224)
    
    print("Passing fake image through the model...")
    output = test_model(dummy_image)
    
    print(f"Success! Output shape is: {output.shape}") 
    # We expect [1, 5] (1 image, 5 prediction scores)
