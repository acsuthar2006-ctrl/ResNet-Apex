import torch
import torch.nn as nn
import os
from datasets.real_clothing_data import get_clothing_dataloaders
from models.resnet_classifier import ClothingClassifier

def main():
    # Automatically use the Apple Silicon GPU (MPS) or regular CPU
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("\n1. Loading Data...")
    dataset_path = "data/clothing"
    train_loader, test_loader, class_names = get_clothing_dataloaders(dataset_path, batch_size=32)
    num_classes = len(class_names)

    print("\n2. Building Model...")
    # Build our ResNet model with the exact number of classes from our dataset
    model = ClothingClassifier(num_classes=num_classes)
    model = model.to(device)
    
    # We use CrossEntropyLoss for classification tasks
    criterion = nn.CrossEntropyLoss()
    
    # Since ResNet is already pre-trained, we can use a smaller learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    # For Transfer Learning on this dataset, 10 epochs is usually plenty!
    epochs = 10 

    print("\n3. Starting Training Loop...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch_X, batch_y in train_loader:
            # Move the images and labels to the GPU
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            # Standard PyTorch Training Step
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
                
            train_loss += loss.item() 
            
        train_loss /= len(train_loader)
        print(f"Epoch {epoch+1:2d}/{epochs} | Training Loss: {train_loss:.4f}")

    print("\n4. Saving the trained model...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_save_path = os.path.join(script_dir, "..", "models", "resnet_clothing.pth")
    
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to '{model_save_path}'!")
    print("\nModule 2 Complete! 🎉")

if __name__ == "__main__":
    main()
