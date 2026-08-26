import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_clothing_dataloaders(data_dir, batch_size=32):
    # 1. Define how we want to transform the images
    # ResNet requires images to be 224x224 and normalized in a very specific way
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    # 2. Automatically load all images from the folders!
    print(f"Loading images from {data_dir}...")
    full_dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)
    
    print(f"Found {len(full_dataset)} total images belonging to {len(full_dataset.classes)} classes.")
    print(f"Classes: {full_dataset.classes}")

    # 3. Split the data: 80% for training, 20% for testing/validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 4. Create DataLoaders (these feed the images to the AI in batches)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, full_dataset.classes

if __name__ == "__main__":
    # Test the dataloader!
    # Update this path if your dataset is somewhere else
    dataset_path = "data/clothing" 
    
    if not os.path.exists(dataset_path):
        print(f"Error: Could not find the folder at {dataset_path}")
    else:
        train_loader, val_loader, class_names = get_clothing_dataloaders(dataset_path)
        
        # Grab one batch of images to test
        images, labels = next(iter(train_loader))
        print(f"\nSuccess! Loaded a batch of {len(images)} images.")
        print(f"Image tensor shape: {images.shape}")
        print(f"Labels tensor shape: {labels.shape}")
