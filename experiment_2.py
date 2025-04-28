import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import time
import copy
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from adaptive_layer import AdaptiveLayer

# Define Models

class VeryTinyNetwork(nn.Module):
    def __init__(self):
        super(VeryTinyNetwork, self).__init__()
        self.fc1 = nn.Linear(8, 3)
        self.activation1 = nn.GELU()
        self.fc2 = nn.Linear(3, 2)
        self.activation2 = nn.GELU()
        self.fc3 = nn.Linear(2, 1)


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
        self.fc1 = nn.Linear(8, 4)
        self.activation1 = nn.GELU()
        self.fc2 = nn.Linear(4, 2)
        self.activation2 = nn.GELU()
        self.fc3 = nn.Linear(2, 1)

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
        self.fc1 = nn.Linear(8, 8)
        self.activation1 = nn.GELU()
        self.fc2 = nn.Linear(8, 4)
        self.activation2 = nn.GELU()
        self.fc3 = nn.Linear(4, 1)

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
        self.fc1 = nn.Linear(8, 16)
        self.activation1 = nn.GELU()
        self.fc2 = nn.Linear(16, 8)
        self.activation2 = nn.GELU()
        self.fc3 = nn.Linear(8, 1)

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
        self.fc1 = nn.Linear(8, 32)
        self.activation1 = nn.GELU()
        self.fc2 = nn.Linear(32, 16)
        self.activation2 = nn.GELU()
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation1(x)
        x = self.fc2(x)
        x = self.activation2(x)
        x = self.fc3(x)
        
        return x
    
class AdaptiveNetwork(nn.Module):
    def __init__(self, layer_1_size=8, layer_2_size=4, adapt_interval=25, k_split=1.5, k_prune=0.2):
        super(AdaptiveNetwork, self).__init__()
        self.adaptive1 = AdaptiveLayer(in_features=8, out_features=layer_1_size, adapt_interval=adapt_interval, k_split=k_split, k_prune=k_prune, activation=nn.GELU())
        self.adaptive2 = AdaptiveLayer(in_features=layer_1_size, out_features=layer_2_size, adapt_interval=adapt_interval, k_split=k_split, k_prune=k_prune, activation=nn.GELU())
        self.fc3 = nn.Linear(layer_2_size, 1)

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
# torch.manual_seed(42)
# np.random.seed(42)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 200
LEARNING_RATE = 0.005
MAX_NETWORK_SIZE = 10000*20

# Load Boston Housing Regression Dataset
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()

full_train_data = housing.data
full_train_labels = housing.target
full_train_dataset = torch.tensor(full_train_data, dtype=torch.float32)
full_train_labels = torch.tensor(full_train_labels, dtype=torch.float32).view(-1, 1)
full_train_dataset = torch.cat((full_train_dataset, full_train_labels), dim=1)

train_dataset, test_dataset = train_test_split(full_train_dataset, test_size=0.2, random_state=42)
train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.2, random_state=42)

standard_scaler = StandardScaler()
train_dataset = standard_scaler.fit_transform(train_dataset)
val_dataset = standard_scaler.transform(val_dataset)
test_dataset = standard_scaler.transform(test_dataset)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Train and Eval Baseline Models (very tiny, tiny, small, medium, large)
def train_and_evaluate_baselines(model, train_loader, val_loader, test_loader, epochs, exp_name='Unknown'):
    model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_losses = []  # MSE
    val_losses = []    # MSE
    train_maes = []    # Mean Absolute Error
    val_maes = []      # Mean Absolute Error
    train_r2s = []     # R-squared
    val_r2s = []       # R-squared
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_predictions = []
        train_targets = []
        
        for batch in train_loader:
            inputs = batch[:, :-1].to(DEVICE).float()
            targets = batch[:, -1:].to(DEVICE).float()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Store predictions and targets for metric calculation
            train_predictions.append(outputs.detach().cpu().numpy())
            train_targets.append(targets.detach().cpu().numpy())
        
        # Calculate metrics
        train_predictions = np.vstack(train_predictions).flatten()
        train_targets = np.vstack(train_targets).flatten()
        
        avg_train_loss = train_loss / len(train_loader)  # MSE
        train_mae = mean_absolute_error(train_targets, train_predictions)
        train_r2 = r2_score(train_targets, train_predictions)
        
        train_losses.append(avg_train_loss)
        train_maes.append(train_mae)
        train_r2s.append(train_r2)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[:, :-1].to(DEVICE).float()
                targets = batch[:, -1:].to(DEVICE).float()
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                # Store predictions and targets for metric calculation
                val_predictions.append(outputs.cpu().numpy())
                val_targets.append(targets.cpu().numpy())
        
        # Calculate metrics
        val_predictions = np.vstack(val_predictions).flatten()
        val_targets = np.vstack(val_targets).flatten()
        
        avg_val_loss = val_loss / len(val_loader)  # MSE
        val_mae = mean_absolute_error(val_targets, val_predictions)
        val_r2 = r2_score(val_targets, val_predictions)
        
        val_losses.append(avg_val_loss)
        val_maes.append(val_mae)
        val_r2s.append(val_r2)
        
        # Save the best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train MSE: {avg_train_loss:.4f}, Train MAE: {train_mae:.4f}, Train R²: {train_r2:.4f}, "
                  f"Val MSE: {avg_val_loss:.4f}, Val MAE: {val_mae:.4f}, Val R²: {val_r2:.4f}")
    
    # Load the best model
    model.load_state_dict(best_model_state)
    
    # Evaluate the best model on test set
    model.eval()
    test_loss = 0.0
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch[:, :-1].to(DEVICE).float()
            targets = batch[:, -1:].to(DEVICE).float()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
            # Store predictions and targets for metric calculation
            test_predictions.append(outputs.cpu().numpy())
            test_targets.append(targets.cpu().numpy())
    
    # Calculate metrics
    test_predictions = np.vstack(test_predictions).flatten()
    test_targets = np.vstack(test_targets).flatten()
    
    avg_test_loss = test_loss / len(test_loader)  # MSE
    test_mae = mean_absolute_error(test_targets, test_predictions)
    test_r2 = r2_score(test_targets, test_predictions)
    
    print(f"\nBest model (selected by validation loss):")
    print(f"Test MSE: {avg_test_loss:.4f}, Test MAE: {test_mae:.4f}, Test R²: {test_r2:.4f}")

    # plot metrics
    plot_metrics(train_losses, val_losses, train_maes, val_maes, train_r2s, val_r2s, 
                filename=f'{exp_name}_metrics.png')
    
    return train_losses, val_losses, train_maes, val_maes, train_r2s, val_r2s

# Corrected function for adaptive networks
def train_and_evaluate_adaptive(model, train_loader, val_loader, test_loader, epochs, exp_name='Unknown'):
    model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_losses = []  # MSE
    val_losses = []    # MSE
    train_maes = []    # Mean Absolute Error
    val_maes = []      # Mean Absolute Error
    train_r2s = []     # R-squared
    val_r2s = []       # R-squared
    
    layer_1_sizes = []
    layer_2_sizes = []
    best_val_loss = float('inf')
    best_model_state = None
    best_model_structure = None
    stop = False

    for epoch in range(epochs):
        if stop:
            break
            
        # Training phase
        model.train()
        train_loss = 0.0
        train_predictions = []
        train_targets = []
        
        for batch in train_loader:
            inputs = batch[:, :-1].to(DEVICE).float()
            targets = batch[:, -1:].to(DEVICE).float()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            old_structure = (model.adaptive1.out_features, model.adaptive2.out_features)
            
            model.backward_step()
            
            new_structure = (model.adaptive1.out_features, model.adaptive2.out_features)
            
            if old_structure != new_structure:
                # Structure changed, recreate optimizer with fresh state
                optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
                print(f"Network structure changed: {old_structure} → {new_structure}. Optimizer reset.")
                
            if (new_structure[0] * new_structure[1]) > MAX_NETWORK_SIZE:
                print(f"New structure exceeds max size: {new_structure}. Stopping training.")
                stop = True
            
            train_loss += loss.item()
            
            # Store predictions and targets for metric calculation
            train_predictions.append(outputs.detach().cpu().numpy())
            train_targets.append(targets.detach().cpu().numpy())
        
        # Calculate metrics
        train_predictions = np.vstack(train_predictions).flatten()
        train_targets = np.vstack(train_targets).flatten()
        
        avg_train_loss = train_loss / len(train_loader)  # MSE
        train_mae = mean_absolute_error(train_targets, train_predictions)
        train_r2 = r2_score(train_targets, train_predictions)
        
        train_losses.append(avg_train_loss)
        train_maes.append(train_mae)
        train_r2s.append(train_r2)
        
        # Store layer sizes
        current_layer_1_size = model.adaptive1.out_features
        current_layer_2_size = model.adaptive2.out_features
        layer_1_sizes.append(current_layer_1_size)
        layer_2_sizes.append(current_layer_2_size)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[:, :-1].to(DEVICE).float()
                targets = batch[:, -1:].to(DEVICE).float()
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                # Store predictions and targets for metric calculation
                val_predictions.append(outputs.cpu().numpy())
                val_targets.append(targets.cpu().numpy())
        
        # Calculate metrics
        val_predictions = np.vstack(val_predictions).flatten()
        val_targets = np.vstack(val_targets).flatten()
        
        avg_val_loss = val_loss / len(val_loader)  # MSE
        val_mae = mean_absolute_error(val_targets, val_predictions)
        val_r2 = r2_score(val_targets, val_predictions)
        
        val_losses.append(avg_val_loss)
        val_maes.append(val_mae)
        val_r2s.append(val_r2)
        
        # Save the best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            best_model_structure = (current_layer_1_size, current_layer_2_size)  # Save structure
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train MSE: {avg_train_loss:.4f}, Train MAE: {train_mae:.4f}, Train R²: {train_r2:.4f}, "
                  f"Val MSE: {avg_val_loss:.4f}, Val MAE: {val_mae:.4f}, Val R²: {val_r2:.4f}")
    
    # Print the best model's structure
    print(f"\nBest model structure: Layer 1 size = {best_model_structure[0]}, Layer 2 size = {best_model_structure[1]}")
    
    # Check if model structure needs to be adjusted to match the best model
    current_structure = (model.adaptive1.out_features, model.adaptive2.out_features)
    if current_structure != best_model_structure:
        print(f"Restoring best model structure: {current_structure} → {best_model_structure}")
        # Create a new model with the best structure
        best_model = AdaptiveNetwork(
            layer_1_size=best_model_structure[0],
            layer_2_size=best_model_structure[1],
            adapt_interval=model.adaptive1.adapt_interval,
            k_split=model.adaptive1.k_split,
            k_prune=model.adaptive1.k_prune
        )
        
        # Load the state dict into the correctly sized model
        best_model.to(DEVICE)
        best_model.load_state_dict(best_model_state)
        model = best_model  # Replace the original model with the best one
    else:
        # Load the best model's state
        model.load_state_dict(best_model_state)
    
    # Evaluate the best model on test set
    model.eval()
    test_loss = 0.0
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch[:, :-1].to(DEVICE).float()
            targets = batch[:, -1:].to(DEVICE).float()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
            # Store predictions and targets for metric calculation
            test_predictions.append(outputs.cpu().numpy())
            test_targets.append(targets.cpu().numpy())
    
    # Calculate metrics
    test_predictions = np.vstack(test_predictions).flatten()
    test_targets = np.vstack(test_targets).flatten()
    
    avg_test_loss = test_loss / len(test_loader)  # MSE
    test_mae = mean_absolute_error(test_targets, test_predictions)
    test_r2 = r2_score(test_targets, test_predictions)
    
    print(f"\nBest model (selected by validation loss):")
    print(f"Test MSE: {avg_test_loss:.4f}, Test MAE: {test_mae:.4f}, Test R²: {test_r2:.4f}")

    # plot metrics with layer sizes
    plot_metrics(train_losses, val_losses, train_maes, val_maes, train_r2s, val_r2s, 
                layer_1_sizes, layer_2_sizes, filename=f'{exp_name}_metrics.png')
    
    return train_losses, val_losses, train_maes, val_maes, train_r2s, val_r2s

# Updated plotting function
def plot_metrics(train_losses, val_losses, train_maes, val_maes, train_r2s, val_r2s, 
                layer_1_sizes=None, layer_2_sizes=None, filename='metrics_plot.png'):
    if layer_1_sizes is not None and layer_2_sizes is not None:
        # Create a figure with four subplots (includes layer sizes)
        plt.figure(figsize=(12, 20))
        n_plots = 4
    else:
        # Create a figure with three subplots
        plt.figure(figsize=(12, 15))
        n_plots = 3
    
    # Plot MSE
    plt.subplot(n_plots, 1, 1)
    plt.plot(train_losses, label='Training MSE')
    plt.plot(val_losses, label='Validation MSE')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.title('Training and Validation MSE Over Time')
    plt.legend()
    plt.grid(True)
    
    # Plot MAE
    plt.subplot(n_plots, 1, 2)
    plt.plot(train_maes, label='Training MAE')
    plt.plot(val_maes, label='Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error')
    plt.title('Training and Validation MAE Over Time')
    plt.legend()
    plt.grid(True)
    
    # Plot R²
    plt.subplot(n_plots, 1, 3)
    plt.plot(train_r2s, label='Training R²')
    plt.plot(val_r2s, label='Validation R²')
    plt.xlabel('Epochs')
    plt.ylabel('R² Score')
    plt.title('Training and Validation R² Over Time')
    plt.legend()
    plt.grid(True)
    
    # Plot layer sizes if provided
    if layer_1_sizes is not None and layer_2_sizes is not None:
        plt.subplot(n_plots, 1, 4)
        plt.plot(layer_1_sizes, label='Layer 1 Size')
        plt.plot(layer_2_sizes, label='Layer 2 Size')
        plt.xlabel('Epochs')
        plt.ylabel('Number of Neurons')
        plt.title('Network Layer Sizes Over Time')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'experiment_2_figs/{filename}')
    plt.show()

# Train and evaluate baseline models
# very_tiny_model = VeryTinyNetwork()
# _, _, _, _, _, _  = train_and_evaluate_baselines(very_tiny_model, train_dataloader, val_dataloader, test_dataloader, EPOCHS, 'very_tiny_model')
'''
Epoch [1/100], Train MSE: 0.5032, Train MAE: 0.5375, Train R²: 0.4968, Val MSE: 0.3322, Val MAE: 0.4150, Val R²: 0.6807
Epoch [10/100], Train MSE: 0.2753, Train MAE: 0.3747, Train R²: 0.7248, Val MSE: 0.3044, Val MAE: 0.3974, Val R²: 0.7075
Epoch [20/100], Train MSE: 0.2741, Train MAE: 0.3742, Train R²: 0.7259, Val MSE: 0.3034, Val MAE: 0.3930, Val R²: 0.7084
Epoch [30/100], Train MSE: 0.2728, Train MAE: 0.3725, Train R²: 0.7272, Val MSE: 0.3087, Val MAE: 0.3959, Val R²: 0.7032
Epoch [40/100], Train MSE: 0.2741, Train MAE: 0.3736, Train R²: 0.7259, Val MSE: 0.2921, Val MAE: 0.3848, Val R²: 0.7193
Epoch [50/100], Train MSE: 0.2722, Train MAE: 0.3725, Train R²: 0.7277, Val MSE: 0.2973, Val MAE: 0.3851, Val R²: 0.7143
Epoch [60/100], Train MSE: 0.2730, Train MAE: 0.3729, Train R²: 0.7269, Val MSE: 0.3124, Val MAE: 0.3940, Val R²: 0.6997
Epoch [70/100], Train MSE: 0.2723, Train MAE: 0.3722, Train R²: 0.7277, Val MSE: 0.3034, Val MAE: 0.3953, Val R²: 0.7084
Epoch [80/100], Train MSE: 0.2724, Train MAE: 0.3722, Train R²: 0.7275, Val MSE: 0.3040, Val MAE: 0.3985, Val R²: 0.7079
Epoch [90/100], Train MSE: 0.2715, Train MAE: 0.3709, Train R²: 0.7288, Val MSE: 0.3029, Val MAE: 0.3995, Val R²: 0.7090
Epoch [100/100], Train MSE: 0.2716, Train MAE: 0.3715, Train R²: 0.7285, Val MSE: 0.2935, Val MAE: 0.3867, Val R²: 0.7179

Best model (selected by validation loss):
Test MSE: 0.2780, Test MAE: 0.3724, Test R²: 0.7187
'''

# tiny_model = TinyNetwork()
# _, _, _, _, _, _  = train_and_evaluate_baselines(tiny_model, train_dataloader, val_dataloader, test_dataloader, EPOCHS, 'tiny_model')
'''
Epoch [1/100], Train MSE: 0.4222, Train MAE: 0.4866, Train R²: 0.5777, Val MSE: 0.3278, Val MAE: 0.4115, Val R²: 0.6849
Epoch [10/100], Train MSE: 0.2759, Train MAE: 0.3732, Train R²: 0.7240, Val MSE: 0.2965, Val MAE: 0.3900, Val R²: 0.7151
Epoch [20/100], Train MSE: 0.2730, Train MAE: 0.3717, Train R²: 0.7269, Val MSE: 0.2916, Val MAE: 0.3892, Val R²: 0.7198
Epoch [30/100], Train MSE: 0.2755, Train MAE: 0.3716, Train R²: 0.7245, Val MSE: 0.2864, Val MAE: 0.3801, Val R²: 0.7248
Epoch [40/100], Train MSE: 0.2689, Train MAE: 0.3679, Train R²: 0.7312, Val MSE: 0.2900, Val MAE: 0.3841, Val R²: 0.7215
Epoch [50/100], Train MSE: 0.2687, Train MAE: 0.3681, Train R²: 0.7315, Val MSE: 0.2828, Val MAE: 0.3802, Val R²: 0.7283
Epoch [60/100], Train MSE: 0.2713, Train MAE: 0.3696, Train R²: 0.7286, Val MSE: 0.2811, Val MAE: 0.3799, Val R²: 0.7299
Epoch [70/100], Train MSE: 0.2673, Train MAE: 0.3663, Train R²: 0.7329, Val MSE: 0.2902, Val MAE: 0.3853, Val R²: 0.7212
Epoch [80/100], Train MSE: 0.2678, Train MAE: 0.3674, Train R²: 0.7323, Val MSE: 0.2833, Val MAE: 0.3811, Val R²: 0.7278
Epoch [90/100], Train MSE: 0.2685, Train MAE: 0.3684, Train R²: 0.7314, Val MSE: 0.2863, Val MAE: 0.3865, Val R²: 0.7249
Epoch [100/100], Train MSE: 0.2678, Train MAE: 0.3671, Train R²: 0.7321, Val MSE: 0.2902, Val MAE: 0.3897, Val R²: 0.7211

Best model (selected by validation loss):
Test MSE: 0.2815, Test MAE: 0.3796, Test R²: 0.7152
'''

# small_model = SmallNetwork()
# _, _, _, _, _, _  = train_and_evaluate_baselines(small_model, train_dataloader, val_dataloader, test_dataloader, EPOCHS, 'small_model')
'''
Epoch [1/100], Train MSE: 0.4278, Train MAE: 0.4462, Train R²: 0.5721, Val MSE: 0.3127, Val MAE: 0.3982, Val R²: 0.6994
Epoch [10/100], Train MSE: 0.2488, Train MAE: 0.3484, Train R²: 0.7512, Val MSE: 0.2721, Val MAE: 0.3707, Val R²: 0.7385
Epoch [20/100], Train MSE: 0.2372, Train MAE: 0.3377, Train R²: 0.7629, Val MSE: 0.2550, Val MAE: 0.3453, Val R²: 0.7549
Epoch [30/100], Train MSE: 0.2297, Train MAE: 0.3302, Train R²: 0.7702, Val MSE: 0.2535, Val MAE: 0.3549, Val R²: 0.7563
Epoch [40/100], Train MSE: 0.2260, Train MAE: 0.3290, Train R²: 0.7740, Val MSE: 0.2451, Val MAE: 0.3426, Val R²: 0.7644
Epoch [50/100], Train MSE: 0.2251, Train MAE: 0.3277, Train R²: 0.7750, Val MSE: 0.2502, Val MAE: 0.3457, Val R²: 0.7595
Epoch [60/100], Train MSE: 0.2253, Train MAE: 0.3282, Train R²: 0.7747, Val MSE: 0.2471, Val MAE: 0.3381, Val R²: 0.7625
Epoch [70/100], Train MSE: 0.2244, Train MAE: 0.3265, Train R²: 0.7757, Val MSE: 0.2430, Val MAE: 0.3398, Val R²: 0.7665
Epoch [80/100], Train MSE: 0.2239, Train MAE: 0.3274, Train R²: 0.7761, Val MSE: 0.2494, Val MAE: 0.3436, Val R²: 0.7603
Epoch [90/100], Train MSE: 0.2231, Train MAE: 0.3256, Train R²: 0.7768, Val MSE: 0.2468, Val MAE: 0.3389, Val R²: 0.7628
Epoch [100/100], Train MSE: 0.2230, Train MAE: 0.3264, Train R²: 0.7769, Val MSE: 0.2461, Val MAE: 0.3362, Val R²: 0.7635

Best model (selected by validation loss):
Test MSE: 0.2412, Test MAE: 0.3309, Test R²: 0.7560
'''

# medium_model = MediumNetwork()
# _, _, _, _, _, _  = train_and_evaluate_baselines(medium_model, train_dataloader, val_dataloader, test_dataloader, EPOCHS, 'medium_model')
'''
Test MSE: 0.2412, Test MAE: 0.3309, Test R²: 0.7560
Epoch [1/100], Train MSE: 0.3597, Train MAE: 0.4231, Train R²: 0.6402, Val MSE: 0.3230, Val MAE: 0.4224, Val R²: 0.6896
Epoch [10/100], Train MSE: 0.2321, Train MAE: 0.3363, Train R²: 0.7678, Val MSE: 0.2499, Val MAE: 0.3545, Val R²: 0.7602
Epoch [20/100], Train MSE: 0.2151, Train MAE: 0.3205, Train R²: 0.7849, Val MSE: 0.2293, Val MAE: 0.3388, Val R²: 0.7798
Epoch [30/100], Train MSE: 0.2074, Train MAE: 0.3133, Train R²: 0.7926, Val MSE: 0.2272, Val MAE: 0.3264, Val R²: 0.7818
Epoch [40/100], Train MSE: 0.2013, Train MAE: 0.3075, Train R²: 0.7988, Val MSE: 0.2244, Val MAE: 0.3241, Val R²: 0.7845
Epoch [50/100], Train MSE: 0.1997, Train MAE: 0.3065, Train R²: 0.8003, Val MSE: 0.2209, Val MAE: 0.3193, Val R²: 0.7878
Epoch [60/100], Train MSE: 0.1998, Train MAE: 0.3060, Train R²: 0.8002, Val MSE: 0.2292, Val MAE: 0.3354, Val R²: 0.7798
Epoch [70/100], Train MSE: 0.1968, Train MAE: 0.3037, Train R²: 0.8033, Val MSE: 0.2276, Val MAE: 0.3231, Val R²: 0.7814
Epoch [80/100], Train MSE: 0.1958, Train MAE: 0.3046, Train R²: 0.8041, Val MSE: 0.2218, Val MAE: 0.3163, Val R²: 0.7870
Epoch [90/100], Train MSE: 0.1996, Train MAE: 0.3056, Train R²: 0.8004, Val MSE: 0.2299, Val MAE: 0.3213, Val R²: 0.7791
Epoch [100/100], Train MSE: 0.1943, Train MAE: 0.3030, Train R²: 0.8056, Val MSE: 0.2156, Val MAE: 0.3108, Val R²: 0.7929

Best model (selected by validation loss):
Test MSE: 0.2015, Test MAE: 0.3014, Test R²: 0.7961
'''

# large_model = LargeNetwork()
# _, _, _, _, _, _  = train_and_evaluate_baselines(large_model, train_dataloader, val_dataloader, test_dataloader, EPOCHS, 'large_model')
'''
Epoch [1/100], Train MSE: 0.3492, Train MAE: 0.4030, Train R²: 0.6506, Val MSE: 0.2877, Val MAE: 0.3827, Val R²: 0.7235
Epoch [10/100], Train MSE: 0.2242, Train MAE: 0.3276, Train R²: 0.7757, Val MSE: 0.2879, Val MAE: 0.3528, Val R²: 0.7232
Epoch [20/100], Train MSE: 0.2085, Train MAE: 0.3141, Train R²: 0.7916, Val MSE: 0.2306, Val MAE: 0.3303, Val R²: 0.7784
Epoch [30/100], Train MSE: 0.1956, Train MAE: 0.3041, Train R²: 0.8045, Val MSE: 0.2188, Val MAE: 0.3201, Val R²: 0.7898
Epoch [40/100], Train MSE: 0.1890, Train MAE: 0.2984, Train R²: 0.8110, Val MSE: 0.2146, Val MAE: 0.3175, Val R²: 0.7938
Epoch [50/100], Train MSE: 0.1846, Train MAE: 0.2959, Train R²: 0.8154, Val MSE: 0.2215, Val MAE: 0.3396, Val R²: 0.7872
Epoch [60/100], Train MSE: 0.1825, Train MAE: 0.2944, Train R²: 0.8174, Val MSE: 0.2140, Val MAE: 0.3127, Val R²: 0.7944
Epoch [70/100], Train MSE: 0.1787, Train MAE: 0.2910, Train R²: 0.8213, Val MSE: 0.2093, Val MAE: 0.3082, Val R²: 0.7989
Epoch [80/100], Train MSE: 0.1793, Train MAE: 0.2927, Train R²: 0.8208, Val MSE: 0.2218, Val MAE: 0.3287, Val R²: 0.7870
Epoch [90/100], Train MSE: 0.1742, Train MAE: 0.2885, Train R²: 0.8257, Val MSE: 0.2209, Val MAE: 0.3185, Val R²: 0.7878
Epoch [100/100], Train MSE: 0.1751, Train MAE: 0.2884, Train R²: 0.8249, Val MSE: 0.2067, Val MAE: 0.3155, Val R²: 0.8014

Best model (selected by validation loss):
Test MSE: 0.2078, Test MAE: 0.3125, Test R²: 0.7898
'''

# Train and evaluate adaptive model
adaptive_model = AdaptiveNetwork(layer_1_size=8, layer_2_size=4, adapt_interval=500, k_split=1.5, k_prune=0.5)
_, _, _, _, _, _  = train_and_evaluate_adaptive(adaptive_model, train_dataloader, val_dataloader, test_dataloader, EPOCHS, 'adaptive_model')
'''
Network structure changed: (8, 4) → (7, 4). Optimizer reset.
Epoch [1/200], Train MSE: 0.4610, Train MAE: 0.4506, Train R²: 0.5388, Val MSE: 0.3188, Val MAE: 0.3911, Val R²: 0.6941
Network structure changed: (7, 4) → (5, 3). Optimizer reset.
Network structure changed: (5, 3) → (5, 2). Optimizer reset.
Epoch [10/200], Train MSE: 0.2664, Train MAE: 0.3641, Train R²: 0.7335, Val MSE: 0.2834, Val MAE: 0.3801, Val R²: 0.7276
Epoch [20/200], Train MSE: 0.2503, Train MAE: 0.3488, Train R²: 0.7496, Val MSE: 0.2643, Val MAE: 0.3592, Val R²: 0.7461
Network structure changed: (5, 2) → (6, 2). Optimizer reset.
Epoch [30/200], Train MSE: 0.2507, Train MAE: 0.3511, Train R²: 0.7493, Val MSE: 0.2668, Val MAE: 0.3613, Val R²: 0.7437
Network structure changed: (6, 2) → (7, 2). Optimizer reset.
Epoch [40/200], Train MSE: 0.2442, Train MAE: 0.3442, Train R²: 0.7557, Val MSE: 0.2647, Val MAE: 0.3582, Val R²: 0.7456
Network structure changed: (7, 2) → (8, 2). Optimizer reset.
Network structure changed: (8, 2) → (7, 2). Optimizer reset.
Epoch [50/200], Train MSE: 0.2447, Train MAE: 0.3440, Train R²: 0.7554, Val MSE: 0.2610, Val MAE: 0.3580, Val R²: 0.7492
Network structure changed: (7, 2) → (8, 2). Optimizer reset.
Network structure changed: (8, 2) → (7, 2). Optimizer reset.
Epoch [60/200], Train MSE: 0.2428, Train MAE: 0.3434, Train R²: 0.7572, Val MSE: 0.2562, Val MAE: 0.3531, Val R²: 0.7537
Network structure changed: (7, 2) → (6, 2). Optimizer reset.
Network structure changed: (6, 2) → (7, 2). Optimizer reset.
Epoch [70/200], Train MSE: 0.2459, Train MAE: 0.3456, Train R²: 0.7542, Val MSE: 0.2683, Val MAE: 0.3643, Val R²: 0.7421
Epoch [80/200], Train MSE: 0.2404, Train MAE: 0.3408, Train R²: 0.7595, Val MSE: 0.2588, Val MAE: 0.3517, Val R²: 0.7513
Network structure changed: (7, 2) → (8, 2). Optimizer reset.
Network structure changed: (8, 2) → (6, 2). Optimizer reset.
Network structure changed: (6, 2) → (7, 2). Optimizer reset.
Epoch [90/200], Train MSE: 0.2434, Train MAE: 0.3434, Train R²: 0.7566, Val MSE: 0.2640, Val MAE: 0.3683, Val R²: 0.7462
Epoch [100/200], Train MSE: 0.2392, Train MAE: 0.3404, Train R²: 0.7608, Val MSE: 0.2554, Val MAE: 0.3510, Val R²: 0.7545
Network structure changed: (7, 2) → (6, 2). Optimizer reset.
Epoch [110/200], Train MSE: 0.2396, Train MAE: 0.3413, Train R²: 0.7604, Val MSE: 0.2530, Val MAE: 0.3497, Val R²: 0.7569
Epoch [120/200], Train MSE: 0.2382, Train MAE: 0.3412, Train R²: 0.7619, Val MSE: 0.2546, Val MAE: 0.3641, Val R²: 0.7553
Epoch [130/200], Train MSE: 0.2363, Train MAE: 0.3406, Train R²: 0.7637, Val MSE: 0.2505, Val MAE: 0.3542, Val R²: 0.7592
Epoch [140/200], Train MSE: 0.2340, Train MAE: 0.3392, Train R²: 0.7660, Val MSE: 0.2665, Val MAE: 0.3562, Val R²: 0.7438
Epoch [150/200], Train MSE: 0.2343, Train MAE: 0.3381, Train R²: 0.7657, Val MSE: 0.2534, Val MAE: 0.3492, Val R²: 0.7565
Epoch [160/200], Train MSE: 0.2350, Train MAE: 0.3386, Train R²: 0.7650, Val MSE: 0.2658, Val MAE: 0.3574, Val R²: 0.7444
Epoch [170/200], Train MSE: 0.2340, Train MAE: 0.3376, Train R²: 0.7660, Val MSE: 0.2511, Val MAE: 0.3462, Val R²: 0.7587
Epoch [180/200], Train MSE: 0.2339, Train MAE: 0.3381, Train R²: 0.7660, Val MSE: 0.2526, Val MAE: 0.3475, Val R²: 0.7573
Epoch [190/200], Train MSE: 0.2345, Train MAE: 0.3378, Train R²: 0.7655, Val MSE: 0.2542, Val MAE: 0.3586, Val R²: 0.7557
Epoch [200/200], Train MSE: 0.2331, Train MAE: 0.3359, Train R²: 0.7668, Val MSE: 0.2528, Val MAE: 0.3453, Val R²: 0.7570

Best model structure: Layer 1 size = 6, Layer 2 size = 2

Best model (selected by validation loss):
Test MSE: 0.2460, Test MAE: 0.3394, Test R²: 0.7511
'''