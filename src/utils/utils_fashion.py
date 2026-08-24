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
            
            logits = model(X_batch)
            preds = logits.argmax(dim=1)

            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    return correct / total

def show_predictions(model, loader, device, num_images=16):
    """Visualizes a grid of test images along with their predicted and true labels for Fashion-MNIST."""
    model.eval()
    
    dataiter = iter(loader)
    images, labels = next(dataiter)
    
    images = images[:num_images]
    labels = labels[:num_images]
    
    images_gpu = images.to(device)
    
    with torch.no_grad():
        logits = model(images_gpu)
        preds = logits.argmax(dim=1).cpu()

    fig = plt.figure(figsize=(8, 8))
    
    for i in range(num_images):
        ax = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])
        
        # Un-normalize the image for display
        # Fashion MNIST stats: mean=0.2860, std=0.3530
        img = images[i].numpy()
        img = np.transpose(img, (1, 2, 0)) # Convert from (C, H, W) to (H, W, C)
        mean = np.array([0.2860, 0.2860, 0.2860])
        std = np.array([0.3530, 0.3530, 0.3530])
        img = (img * std) + mean
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        
        # Fashion MNIST class names
        classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        
        color = 'green' if preds[i] == labels[i] else 'red'
        pred_label = classes[preds[i].item()]
        true_label = classes[labels[i].item()]
        ax.set_title(f"P: {pred_label}\nT: {true_label}", color=color, fontsize=10)
        
    plt.tight_layout()
    plt.show()
