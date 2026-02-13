import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from model import CNN_Model 


BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 30  
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = "data"
SAVE_PATH = "cnn_model.pth" 

def get_transforms():
    mean, std = (0.1722,), (0.3309,)
    
  
    train_transform = transforms.Compose([
        lambda img: transforms.functional.rotate(img, -90),
        lambda img: transforms.functional.hflip(img),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_transform = transforms.Compose([
        lambda img: transforms.functional.rotate(img, -90),
        lambda img: transforms.functional.hflip(img),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    return train_transform, test_transform

def train():
    print(f"Starting Training on device: {DEVICE}")
    train_tf, test_tf = get_transforms()
    print("Downloading EMNIST Data...")
    train_data = datasets.EMNIST(root=DATA_PATH, split="letters", train=True, download=True, transform=train_tf)
    test_data = datasets.EMNIST(root=DATA_PATH, split="letters", train=False, download=True, transform=test_tf)
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    
    model = CNN_Model(num_classes=26).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    
    best_accuracy = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            labels=labels-1
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted_indices=torch.argmax(outputs,dim=1)
            train_correct+=(predicted_indices==labels).sum().item()
            train_total+=len(labels)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                labels=labels-1
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_acc = 100 * correct / total
        train_acc = 100 * train_correct / train_total
        avg_loss = running_loss / len(train_loader)

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
        scheduler.step(test_acc)
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"New High Score! Model saved to {SAVE_PATH}")

    print(f"\nTraining Complete. Best Accuracy: {best_accuracy:.2f}%")

if __name__ == "__main__":
    train()