import torch
import numpy as np
import os
from sklearn.metrics import classification_report
from models.model import MyNeuralNet

def print_model_metrics(dataset_type="cifar10"):
    """
    Evaluates the model and prints detailed metrics: Precision, Recall, and F1-Score.
    """
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # Select which dataset/model to evaluate based on the argument
    if dataset_type == "cifar10":
        from datasets.data import get_dataloaders
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "..", "models", "cifar10_model_long_run.pth")
        in_channels = 3
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
        print("--- Evaluating CIFAR-10 Model ---")
    elif dataset_type == "fashion":
        from datasets.data_fashion import get_dataloaders
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "..", "models", "fashionmnist_model.pth")
        in_channels = 1
        classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        print("--- Evaluating Fashion-MNIST Model ---")
    else:
        raise ValueError("Invalid dataset_type. Choose 'cifar10' or 'fashion'.")

    # Load Data
    print("Loading test data...")
    _, test_loader = get_dataloaders(batch_size=256)
    
    # Initialize & Load Model
    print(f"Loading weights from {model_path}...")
    model = MyNeuralNet(in_channels=in_channels, num_classes=10)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    print("Running inference on 10,000 test images...")
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            logits = model(batch_X)
            preds = logits.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate and Print Metrics using Scikit-Learn
    print("\n" + "="*55)
    print(f"CLASSIFICATION REPORT ({dataset_type.upper()})")
    print("="*55)
    report = classification_report(all_labels, all_preds, target_names=classes, digits=4)
    print(report)

if __name__ == "__main__":
    # To test Fashion-MNIST instead, simply change this to dataset_type="fashion"
    print_model_metrics(dataset_type="fashion")
