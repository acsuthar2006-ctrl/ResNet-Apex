import torch
import torch.nn as nn
import os
from datasets.data import get_dataloaders
from models.model import MyNeuralNet
from utils.utils import evaluate

def main():
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("Loading data...")
    train_loader, test_loader = get_dataloaders()

    model = MyNeuralNet(num_classes=10)
    model.init_weights()
    
    # Multi-GPU Support for Kaggle T4x2
    if torch.cuda.device_count() > 1:
        print(f"Detected {torch.cuda.device_count()} GPUs! Distributing work via DataParallel...")
        model = nn.DataParallel(model)
        
    model.to(device)
    
    # Standard Loss
    criterion = nn.CrossEntropyLoss()
    
    # Lower starting learning rate for a long, stable run
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    epochs = 150
    
    # Cosine Annealing slowly decays the learning rate to near-zero over 150 epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None

    # Training Loop
    print(f"Starting long training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                # pyrefly: ignore [missing-attribute]
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
            train_loss += loss.item() 
            
        # For CosineAnnealing, we step the scheduler ONCE per epoch, not per batch!
        scheduler.step()
        train_loss /= len(train_loader)
        
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {train_loss:.4f} | LR: {current_lr:.6f}")

    print("\n--- Evaluation ---")
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item()
            
    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
    
    train_acc = evaluate(model, train_loader, device)
    test_acc = evaluate(model, test_loader, device)
    print(f"Train Acc: {train_acc * 100:.2f}%")
    print(f"Test Acc:  {test_acc * 100:.2f}%")

    print("\nSaving the model parameters...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_save_path = os.path.join(script_dir, "..", "models", "cifar10_model.pth")
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), model_save_path)
    else:
        torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to '{model_save_path}'!")

    print("\nGenerating predictions visualization...")
    from utils.utils import show_predictions
    show_predictions(model, test_loader, device)

if __name__ == "__main__":
    main()
