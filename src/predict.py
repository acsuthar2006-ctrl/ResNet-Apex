import torch
from models.model import MyNeuralNet
from datasets.data_fashion import get_dataloaders
from utils.utils_fashion import show_predictions, evaluate

def main():
    print("--- Fashion-MNIST Inference Script ---")
    
    model_path = "../models/fashion_mnist_model.pth"
        
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("Initializing ResNet9 architecture...")
    model = MyNeuralNet(num_classes=10)
    
    print(f"Loading weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.eval()
    model.to(device)
    print("Model successfully loaded and locked into evaluation mode.")
    
    print("\nLoading Fashion-MNIST test data...")
    _, test_loader = get_dataloaders(batch_size=256)
    
    print("Running full evaluation on 10,000 test images (This might take a second)...")
    test_acc = evaluate(model, test_loader, device)
    print(f"--> Final Test Accuracy: {test_acc * 100:.2f}%")
    
    print("\nGenerating visual predictions...")
    print("A window will pop up showing the images. Close the window to exit the script.")
    show_predictions(model, test_loader, device)

if __name__ == "__main__":
    main()
