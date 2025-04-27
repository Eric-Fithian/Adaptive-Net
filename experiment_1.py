import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import time
import copy
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from adaptive_layer import AdaptiveLayer

# Define Models

class VeryTinyNetwork(nn.Module):
    def __init__(self):
        super(VeryTinyNetwork, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 8)
        self.activation1 = nn.GELU()
        self.fc2 = nn.Linear(8, 4)
        self.activation2 = nn.GELU()
        self.fc3 = nn.Linear(4, 10)


    def forward(self, x):
        x = self.fc1(x)
        x = self.activation1(x)
        x = self.fc2(x)
        x = self.activation2(x)
        x = self.fc3(x)
        
        return x

class TinyNetwork(nn.Module):
    def __init__(self):
        super(TinyNetwork, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 16)
        self.activation1 = nn.GELU()
        self.fc2 = nn.Linear(16, 8)
        self.activation2 = nn.GELU()
        self.fc3 = nn.Linear(8, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation1(x)
        x = self.fc2(x)
        x = self.activation2(x)
        x = self.fc3(x)
        
        return x

class SmallNetwork(nn.Module):
    def __init__(self):
        super(SmallNetwork, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.activation1 = nn.GELU()
        self.fc2 = nn.Linear(128, 64)
        self.activation2 = nn.GELU()
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation1(x)
        x = self.fc2(x)
        x = self.activation2(x)
        x = self.fc3(x)
        
        return x
    
class MediumNetwork(nn.Module):
    def __init__(self):
        super(MediumNetwork, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.activation1 = nn.GELU()
        self.fc2 = nn.Linear(256, 128)
        self.activation2 = nn.GELU()
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation1(x)
        x = self.fc2(x)
        x = self.activation2(x)
        x = self.fc3(x)
        
        return x
    
class LargeNetwork(nn.Module):
    def __init__(self):
        super(LargeNetwork, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.activation1 = nn.GELU()
        self.fc2 = nn.Linear(512, 256)
        self.activation2 = nn.GELU()
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation1(x)
        x = self.fc2(x)
        x = self.activation2(x)
        x = self.fc3(x)
        
        return x
    
class AdaptiveNetwork(nn.Module):
    def __init__(self):
        super(AdaptiveNetwork, self).__init__()
        self.adaptive1 = AdaptiveLayer(in_features=28 * 28, out_features=128, adapt_interval=100, k_split=2.0, k_prune=0.1, activation=nn.GELU())
        self.adaptive2 = AdaptiveLayer(in_features=128, out_features=64, adapt_interval=100, k_split=2.0, k_prune=0.1, activation=nn.GELU())
        self.fc3 = nn.Linear(64, 10)

        # Set the next layer for adaptive layers
        self.adaptive1.set_next_layer(self.adaptive2)
        self.adaptive2.set_next_layer(self.fc3)

    def forward(self, x):
        x = self.adaptive1(x)
        x = self.adaptive2(x)
        x = self.fc3(x)
        
        return x
    
    def backward_step(self):
        """
        Call after each backward pass to increment step count and trigger adaptation.
        """
        self.adaptive1.backward_step()
        self.adaptive2.backward_step()


# CONSTANTS
torch.manual_seed(42)
np.random.seed(42)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 200
LEARNING_RATE = 0.0005

# Load FashionMNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

full_train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_dataset, val_dataset = train_test_split(full_train_dataset, test_size=0.16667, random_state=42)
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Train and Eval Baseline Models (small, medium, large)
def train_and_evaluate_baselines(model, train_loader, val_loader, test_loader, epochs, exp_name='Unknown'):
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.view(images.size(0), -1).to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model.forward(images)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.view(images.size(0), -1).to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        # Save the best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
    
    # Load the best model
    model.load_state_dict(best_model_state)
    
    # Evaluate the best model on test set
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.view(images.size(0), -1).to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    test_accuracy = 100 * test_correct / test_total
    avg_test_loss = test_loss / len(test_loader)
    
    print(f"\nBest model (selected by validation loss):")
    print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    # plot metrics
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, filename=f'{exp_name}_metrics.png')
    
    return train_losses, val_losses, train_accuracies, val_accuracies

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, filename='metrics_plot.png'):
    # Create a figure with two subplots
    plt.figure(figsize=(12, 10))
    
    # Plot losses
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracies
    plt.subplot(2, 1, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy Over Time')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'experiment_1_figs/{filename}')
    plt.show()

# Train and evaluate baseline models
very_tiny_model = VeryTinyNetwork()
very_tiny_train_losses, very_tiny_val_losses, very_tiny_train_acc, very_tiny_val_acc  = train_and_evaluate_baselines(very_tiny_model, train_dataloader, val_dataloader, test_dataloader, EPOCHS, 'very_tiny_model')

'''
Epoch [1/200], Train Loss: 1.3985, Train Acc: 56.03%, Val Loss: 0.9333, Val Acc: 72.96%
Epoch [10/200], Train Loss: 0.4688, Train Acc: 83.82%, Val Loss: 0.4831, Val Acc: 83.02%
Epoch [20/200], Train Loss: 0.4254, Train Acc: 85.10%, Val Loss: 0.4519, Val Acc: 84.15%
Epoch [30/200], Train Loss: 0.4037, Train Acc: 85.87%, Val Loss: 0.4535, Val Acc: 83.94%
Epoch [40/200], Train Loss: 0.3906, Train Acc: 86.20%, Val Loss: 0.4346, Val Acc: 85.03%
Epoch [50/200], Train Loss: 0.3820, Train Acc: 86.57%, Val Loss: 0.4364, Val Acc: 84.85%
Epoch [60/200], Train Loss: 0.3748, Train Acc: 86.75%, Val Loss: 0.4343, Val Acc: 85.01%
Epoch [70/200], Train Loss: 0.3687, Train Acc: 86.97%, Val Loss: 0.4337, Val Acc: 85.10%
Epoch [80/200], Train Loss: 0.3631, Train Acc: 87.18%, Val Loss: 0.4342, Val Acc: 85.05%
Epoch [90/200], Train Loss: 0.3591, Train Acc: 87.20%, Val Loss: 0.4394, Val Acc: 84.93%
Epoch [100/200], Train Loss: 0.3565, Train Acc: 87.41%, Val Loss: 0.4395, Val Acc: 85.06%
Epoch [110/200], Train Loss: 0.3536, Train Acc: 87.42%, Val Loss: 0.4411, Val Acc: 85.13%
Epoch [120/200], Train Loss: 0.3506, Train Acc: 87.57%, Val Loss: 0.4463, Val Acc: 84.92%
Epoch [130/200], Train Loss: 0.3491, Train Acc: 87.63%, Val Loss: 0.4454, Val Acc: 84.81%
Epoch [140/200], Train Loss: 0.3471, Train Acc: 87.73%, Val Loss: 0.4438, Val Acc: 84.55%
Epoch [150/200], Train Loss: 0.3437, Train Acc: 87.76%, Val Loss: 0.4428, Val Acc: 84.93%
Epoch [160/200], Train Loss: 0.3443, Train Acc: 87.75%, Val Loss: 0.4508, Val Acc: 84.77%
Epoch [170/200], Train Loss: 0.3409, Train Acc: 87.96%, Val Loss: 0.4434, Val Acc: 84.89%
Epoch [180/200], Train Loss: 0.3406, Train Acc: 87.91%, Val Loss: 0.4500, Val Acc: 84.69%
Epoch [190/200], Train Loss: 0.3402, Train Acc: 87.88%, Val Loss: 0.4493, Val Acc: 84.84%
Epoch [200/200], Train Loss: 0.3387, Train Acc: 88.01%, Val Loss: 0.4486, Val Acc: 84.85%

Best model (selected by validation loss):
Test Loss: 0.4905, Test Accuracy: 83.84%
'''

# tiny_model = TinyNetwork()
# tiny_train_losses, tiny_val_losses, tiny_train_acc, tiny_val_acc  = train_and_evaluate_baselines(tiny_model, train_dataloader, val_dataloader, test_dataloader, EPOCHS, 'tiny_model')
'''
Epoch [1/200], Train Loss: 0.8723, Train Acc: 69.50%, Val Loss: 0.5661, Val Acc: 80.05%
Epoch [10/200], Train Loss: 0.3761, Train Acc: 86.64%, Val Loss: 0.4093, Val Acc: 85.04%
Epoch [20/200], Train Loss: 0.3385, Train Acc: 88.04%, Val Loss: 0.3841, Val Acc: 86.33%
Epoch [30/200], Train Loss: 0.3180, Train Acc: 88.75%, Val Loss: 0.3857, Val Acc: 86.39%
Epoch [40/200], Train Loss: 0.3028, Train Acc: 89.28%, Val Loss: 0.3798, Val Acc: 86.83%
Epoch [50/200], Train Loss: 0.2919, Train Acc: 89.61%, Val Loss: 0.3897, Val Acc: 86.26%
Epoch [60/200], Train Loss: 0.2821, Train Acc: 89.89%, Val Loss: 0.3840, Val Acc: 86.68%
Epoch [70/200], Train Loss: 0.2733, Train Acc: 90.12%, Val Loss: 0.3991, Val Acc: 86.55%
Epoch [80/200], Train Loss: 0.2659, Train Acc: 90.48%, Val Loss: 0.3945, Val Acc: 86.55%
Epoch [90/200], Train Loss: 0.2592, Train Acc: 90.72%, Val Loss: 0.4188, Val Acc: 86.02%
Epoch [100/200], Train Loss: 0.2556, Train Acc: 90.70%, Val Loss: 0.4138, Val Acc: 86.03%
Epoch [110/200], Train Loss: 0.2500, Train Acc: 90.94%, Val Loss: 0.4220, Val Acc: 86.23%
Epoch [120/200], Train Loss: 0.2447, Train Acc: 91.09%, Val Loss: 0.4300, Val Acc: 85.95%
Epoch [130/200], Train Loss: 0.2417, Train Acc: 91.20%, Val Loss: 0.4368, Val Acc: 85.93%
Epoch [140/200], Train Loss: 0.2361, Train Acc: 91.37%, Val Loss: 0.4410, Val Acc: 85.95%
Epoch [150/200], Train Loss: 0.2327, Train Acc: 91.56%, Val Loss: 0.4368, Val Acc: 85.96%
Epoch [160/200], Train Loss: 0.2308, Train Acc: 91.49%, Val Loss: 0.4570, Val Acc: 85.65%
Epoch [170/200], Train Loss: 0.2246, Train Acc: 91.82%, Val Loss: 0.4649, Val Acc: 85.57%
Epoch [180/200], Train Loss: 0.2241, Train Acc: 91.83%, Val Loss: 0.4939, Val Acc: 85.37%
Epoch [190/200], Train Loss: 0.2234, Train Acc: 91.80%, Val Loss: 0.4765, Val Acc: 85.78%
Epoch [200/200], Train Loss: 0.2183, Train Acc: 92.18%, Val Loss: 0.4928, Val Acc: 85.31%

Best model (selected by validation loss):
Test Loss: 0.5208, Test Accuracy: 85.46%
'''

# small_model = SmallNetwork()
# small_train_losses, small_val_losses, small_train_acc, small_val_acc  = train_and_evaluate_baselines(small_model, train_dataloader, val_dataloader, test_dataloader, EPOCHS, 'small_model')
'''
Epoch [1/200], Train Loss: 0.5511, Train Acc: 80.33%, Val Loss: 0.4258, Val Acc: 84.49%
Epoch [10/200], Train Loss: 0.2399, Train Acc: 91.18%, Val Loss: 0.3487, Val Acc: 87.37%
Epoch [20/200], Train Loss: 0.1624, Train Acc: 94.03%, Val Loss: 0.3305, Val Acc: 89.19%
Epoch [30/200], Train Loss: 0.1102, Train Acc: 96.00%, Val Loss: 0.3937, Val Acc: 88.91%
Epoch [40/200], Train Loss: 0.0781, Train Acc: 97.15%, Val Loss: 0.4722, Val Acc: 88.49%
Epoch [50/200], Train Loss: 0.0541, Train Acc: 98.06%, Val Loss: 0.5285, Val Acc: 88.67%
Epoch [60/200], Train Loss: 0.0396, Train Acc: 98.60%, Val Loss: 0.6551, Val Acc: 88.07%
Epoch [70/200], Train Loss: 0.0354, Train Acc: 98.70%, Val Loss: 0.7269, Val Acc: 88.32%
Epoch [80/200], Train Loss: 0.0308, Train Acc: 98.90%, Val Loss: 0.7842, Val Acc: 88.59%
Epoch [90/200], Train Loss: 0.0282, Train Acc: 99.00%, Val Loss: 0.8055, Val Acc: 88.16%
Epoch [100/200], Train Loss: 0.0184, Train Acc: 99.39%, Val Loss: 0.8487, Val Acc: 88.05%
Epoch [110/200], Train Loss: 0.0255, Train Acc: 99.18%, Val Loss: 0.9055, Val Acc: 88.78%
Epoch [120/200], Train Loss: 0.0144, Train Acc: 99.51%, Val Loss: 0.9464, Val Acc: 88.83%
Epoch [130/200], Train Loss: 0.0139, Train Acc: 99.49%, Val Loss: 0.9927, Val Acc: 88.56%
Epoch [140/200], Train Loss: 0.0320, Train Acc: 99.01%, Val Loss: 0.9523, Val Acc: 88.60%
Epoch [150/200], Train Loss: 0.0260, Train Acc: 99.18%, Val Loss: 1.0468, Val Acc: 88.03%
Epoch [160/200], Train Loss: 0.0162, Train Acc: 99.60%, Val Loss: 1.3351, Val Acc: 86.87%
Epoch [170/200], Train Loss: 0.0208, Train Acc: 99.31%, Val Loss: 0.9968, Val Acc: 88.96%
Epoch [180/200], Train Loss: 0.0249, Train Acc: 99.23%, Val Loss: 1.0498, Val Acc: 88.65%
Epoch [190/200], Train Loss: 0.0233, Train Acc: 99.26%, Val Loss: 1.0931, Val Acc: 88.59%
Epoch [200/200], Train Loss: 0.0144, Train Acc: 99.51%, Val Loss: 1.1801, Val Acc: 88.25%

Best model (selected by validation loss):
Test Loss: 1.2814, Test Accuracy: 88.09%
'''

# medium_model = MediumNetwork()
# medium_train_losses, medium_val_losses, medium_train_acc, medium_val_acc  = train_and_evaluate_baselines(medium_model, train_dataloader, val_dataloader, test_dataloader, EPOCHS, 'medium_model')

'''
Epoch [1/200], Train Loss: 0.5212, Train Acc: 81.24%, Val Loss: 0.4086, Val Acc: 85.00%
Epoch [10/200], Train Loss: 0.2110, Train Acc: 92.11%, Val Loss: 0.3011, Val Acc: 89.19%
Epoch [20/200], Train Loss: 0.1152, Train Acc: 95.65%, Val Loss: 0.3586, Val Acc: 88.88%
Epoch [30/200], Train Loss: 0.0665, Train Acc: 97.52%, Val Loss: 0.4473, Val Acc: 89.23%
Epoch [40/200], Train Loss: 0.0436, Train Acc: 98.40%, Val Loss: 0.5330, Val Acc: 89.16%
Epoch [50/200], Train Loss: 0.0366, Train Acc: 98.68%, Val Loss: 0.6519, Val Acc: 88.97%
Epoch [60/200], Train Loss: 0.0190, Train Acc: 99.30%, Val Loss: 0.7036, Val Acc: 89.71%
Epoch [70/200], Train Loss: 0.0268, Train Acc: 99.07%, Val Loss: 0.7772, Val Acc: 88.94%
Epoch [80/200], Train Loss: 0.0199, Train Acc: 99.30%, Val Loss: 0.7425, Val Acc: 89.54%
Epoch [90/200], Train Loss: 0.0205, Train Acc: 99.28%, Val Loss: 0.8342, Val Acc: 89.25%
Epoch [100/200], Train Loss: 0.0252, Train Acc: 99.14%, Val Loss: 0.8964, Val Acc: 88.56%
Epoch [110/200], Train Loss: 0.0173, Train Acc: 99.41%, Val Loss: 0.8852, Val Acc: 89.58%
Epoch [120/200], Train Loss: 0.0248, Train Acc: 99.23%, Val Loss: 0.9085, Val Acc: 89.32%
Epoch [130/200], Train Loss: 0.0123, Train Acc: 99.56%, Val Loss: 1.0767, Val Acc: 88.90%
Epoch [140/200], Train Loss: 0.0151, Train Acc: 99.51%, Val Loss: 0.9381, Val Acc: 89.87%
Epoch [150/200], Train Loss: 0.0173, Train Acc: 99.44%, Val Loss: 1.0206, Val Acc: 89.20%
Epoch [160/200], Train Loss: 0.0210, Train Acc: 99.31%, Val Loss: 1.0421, Val Acc: 89.56%
Epoch [170/200], Train Loss: 0.0138, Train Acc: 99.56%, Val Loss: 1.1052, Val Acc: 88.88%
Epoch [180/200], Train Loss: 0.0127, Train Acc: 99.64%, Val Loss: 1.1043, Val Acc: 89.66%
Epoch [190/200], Train Loss: 0.0104, Train Acc: 99.68%, Val Loss: 1.1571, Val Acc: 89.33%
Epoch [200/200], Train Loss: 0.0099, Train Acc: 99.71%, Val Loss: 1.1409, Val Acc: 89.44%

Best model (selected by validation loss):
Test Loss: 1.3449, Test Accuracy: 88.66%
'''

# large_model = LargeNetwork()
# large_train_losses, large_val_losses, large_train_acc, large_val_acc  = train_and_evaluate_baselines(large_model, train_dataloader, val_dataloader, test_dataloader, EPOCHS, 'large_model')

'''
Epoch [1/200], Train Loss: 0.5016, Train Acc: 81.66%, Val Loss: 0.4178, Val Acc: 84.20%
Epoch [10/200], Train Loss: 0.1843, Train Acc: 92.99%, Val Loss: 0.3326, Val Acc: 88.73%
Epoch [20/200], Train Loss: 0.0871, Train Acc: 96.77%, Val Loss: 0.3840, Val Acc: 89.72%
Epoch [30/200], Train Loss: 0.0478, Train Acc: 98.24%, Val Loss: 0.5194, Val Acc: 89.31%
Epoch [40/200], Train Loss: 0.0311, Train Acc: 98.89%, Val Loss: 0.6334, Val Acc: 89.28%
Epoch [50/200], Train Loss: 0.0343, Train Acc: 98.76%, Val Loss: 0.6878, Val Acc: 89.13%
Epoch [60/200], Train Loss: 0.0173, Train Acc: 99.42%, Val Loss: 0.7681, Val Acc: 89.37%
Epoch [70/200], Train Loss: 0.0284, Train Acc: 99.07%, Val Loss: 0.8246, Val Acc: 88.35%
Epoch [80/200], Train Loss: 0.0237, Train Acc: 99.23%, Val Loss: 0.8456, Val Acc: 89.19%
Epoch [90/200], Train Loss: 0.0156, Train Acc: 99.50%, Val Loss: 0.8713, Val Acc: 89.52%
Epoch [100/200], Train Loss: 0.0053, Train Acc: 99.84%, Val Loss: 0.9305, Val Acc: 89.73%
Epoch [110/200], Train Loss: 0.0257, Train Acc: 99.22%, Val Loss: 0.9720, Val Acc: 89.23%
Epoch [120/200], Train Loss: 0.0132, Train Acc: 99.58%, Val Loss: 1.0144, Val Acc: 89.63%
Epoch [130/200], Train Loss: 0.0139, Train Acc: 99.58%, Val Loss: 1.0644, Val Acc: 89.01%
Epoch [140/200], Train Loss: 0.0117, Train Acc: 99.62%, Val Loss: 1.0980, Val Acc: 89.25%
Epoch [150/200], Train Loss: 0.0094, Train Acc: 99.69%, Val Loss: 1.0967, Val Acc: 89.14%
Epoch [160/200], Train Loss: 0.0132, Train Acc: 99.61%, Val Loss: 1.1696, Val Acc: 89.51%
Epoch [170/200], Train Loss: 0.0071, Train Acc: 99.80%, Val Loss: 1.1654, Val Acc: 89.23%
Epoch [180/200], Train Loss: 0.0165, Train Acc: 99.48%, Val Loss: 1.2962, Val Acc: 89.48%
Epoch [190/200], Train Loss: 0.0303, Train Acc: 99.21%, Val Loss: 1.3056, Val Acc: 89.04%
Epoch [200/200], Train Loss: 0.0146, Train Acc: 99.58%, Val Loss: 1.3427, Val Acc: 88.64%

Best model (selected by validation loss):
Test Loss: 1.4679, Test Accuracy: 88.63%
'''
