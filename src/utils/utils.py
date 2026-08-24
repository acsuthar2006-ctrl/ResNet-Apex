import torch
import numpy as np
import matplotlib.pyplot as plt

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Model now handles 3D images directly
            # X_batch_flat = X_batch.view(X_batch.size(0), -1)
            
            logits = model(X_batch)
            # For multi-class, prediction is the index with the highest logit
            preds = logits.argmax(dim=1)

            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    return correct / total

def show_predictions(model, loader, device, num_images=16):
    """Visualizes a grid of test images along with their predicted and true labels."""
    model.eval()
    
    # Get a single batch of test data
    dataiter = iter(loader)
    images, labels = next(dataiter)
    
    # Select the first 'num_images'
    images = images[:num_images]
    labels = labels[:num_images]
    
    # Move to device and get predictions
    images_gpu = images.to(device)
    # Model now handles 3D images directly
    # images_flat = images_gpu.view(images_gpu.size(0), -1)
    
    with torch.no_grad():
        logits = model(images_gpu)
        preds = logits.argmax(dim=1).cpu()

    # Create a plot
    fig = plt.figure(figsize=(8, 8))
    
    for i in range(num_images):
        ax = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])
        
        # Un-normalize the image for display
        # CIFAR-10 stats: mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)
        img = images[i].numpy()
        img = np.transpose(img, (1, 2, 0)) # Convert from (C, H, W) to (H, W, C)
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2470, 0.2435, 0.2616])
        img = (img * std) + mean
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        
        # CIFAR-10 class names
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Set title color: green if correct, red if wrong
        color = 'green' if preds[i] == labels[i] else 'red'
        pred_label = classes[preds[i].item()]
        true_label = classes[labels[i].item()]
        ax.set_title(f"P: {pred_label}\nT: {true_label}", color=color, fontsize=10)
        
    plt.tight_layout()
    plt.show()
