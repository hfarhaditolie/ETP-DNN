import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Add a small value to denominator to avoid division by zero
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
def r_squared_loss(y_true, y_pred):
    # Calculate SS_res
    ss_res = torch.sum((y_true - y_pred) ** 2)
    
    # Calculate SS_tot
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    
    # Calculate R^2
    r2 = 1 - (ss_res / ss_tot)
    
    # Return negative R^2 for minimization
    return -r2

# Load and prepare the dataset
file_path = 'Data/processed_data_AN.xlsx'
df = pd.read_excel(file_path)
data = np.asarray(df)
names = data[:, 0]  # Sample names or IDs
data = data[:, 1:]

# Define target variable and features
y = data[:, 0]  # First column as target (Thickness)
X = data[:, 2:]  # Remaining columns as features

class Flatten(nn.Module):
    def forward(self, inp):
        return inp.view(inp.size(0), -1)
class SWINTransformer1DBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, window_size, dropout_rate=0.2):
        super(SWINTransformer1DBlock, self).__init__()
        
        # Window-based multi-head self-attention
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
        
        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

        # Shifted window mechanism can be mimicked by applying attention to parts of the input sequence.
        self.window_size = window_size

    def forward(self, x):
        # Split the input into windows (like shifted windows in SWIN)
        B, C, L = x.size()  # Batch, Channels, Length (sequence)
        
        # Change to sequence-first format for attention: (B, L, C) -> (L, B, C)
        x = x.permute(2, 0, 1)  # Now (L, B, C)
        
        # Apply multi-head attention to each window
        attn_output, _ = self.attn(x, x, x)
        
        # Add & Norm (skip connection with input `x`)
        x = x + self.dropout(attn_output)  # Skip connection
        x = self.norm1(x)
        
        # Pass through the feedforward network
        ffn_output = self.ffn(x.permute(1, 0, 2))  # Convert back to (B, L, C) format for FFN
        
        # Add & Norm (skip connection with FFN output)
        x = x.permute(1, 0, 2) + self.dropout(ffn_output)  # Skip connection and back to (B, L, C)
        x = self.norm2(x)
        
        return x.permute(0, 2, 1)  # Return output in (B, C, L) format for compatibility with other layers

# Define the 1D CNN model
class Inception1DBlock(nn.Module):
    def __init__(self, in_channels, dropout_rate=0.2):
        super(Inception1DBlock, self).__init__()
        
        # 1x1 convolution path
        self.conv1x1 = nn.Conv1d(in_channels, 16, kernel_size=1)
        self.bn1x1 = nn.BatchNorm1d(16)  # Batch normalization for the 1x1 conv

        # 1x3 convolution path
        self.conv1x3 = nn.Conv1d(in_channels, 16, kernel_size=3,dilation=4, padding=4)
        self.bn1x3 = nn.BatchNorm1d(16)  # Batch normalization for the 1x3 conv

        # 1x5 convolution path
        self.conv1x5 = nn.Conv1d(in_channels, 16, kernel_size=5,dilation=4, padding=8)
        self.bn1x5 = nn.BatchNorm1d(16)  # Batch normalization for the 1x5 conv

        # Max Pooling path with 1x1 convolution after pooling
        self.pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(in_channels, 16, kernel_size=1)
        self.bn_pool = nn.BatchNorm1d(16)  # Batch normalization for the pooling path
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        path1 = self.dropout(self.bn1x1(self.conv1x1(x)))  # 1x1 conv + BN + Dropout
        path2 = self.dropout(self.bn1x3(self.conv1x3(x)))  # 1x3 conv + BN + Dropout
        path3 = self.dropout(self.bn1x5(self.conv1x5(x)))  # 1x5 conv + BN + Dropout
        path4 = self.dropout(self.bn_pool(self.conv_pool(self.pool(x))))  # Pool + 1x1 conv + BN + Dropout

        # Concatenate outputs along the channel dimension
        outputs = [path1, path2, path3, path4]
        return torch.cat(outputs, dim=1)
class Conv1DInceptionModel(nn.Module):
    def __init__(self):
        super(Conv1DInceptionModel, self).__init__()
        self.inception1 = Inception1DBlock(in_channels=1)  # Inception block
        self.inception2 = Inception1DBlock(in_channels=64)  # Inception block
        self.swin_transformer = SWINTransformer1DBlock(input_dim=64, hidden_dim=128, num_heads=4, window_size=100)
        self.fc1 = nn.Linear(16000, 64)  # Fully connected layer
        self.fc2 = nn.Linear(64, 1)  # Output layer for regression

    def forward(self, x):
        # x_cat=x[:,:,:2]
        # x=x[:,:,2:]
        x = self.inception1(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool1d(kernel_size=2)(x)  # Max pooling layer

        x = self.inception2(x)

        x = nn.ReLU()(x)
        x = nn.MaxPool1d(kernel_size=2)(x)  # Max pooling layer
        x = self.swin_transformer(x)


        #x = x.view(x.size(0), -1)  # Flatten the output
        x = x.reshape(x.size(0),-1)
        # x_cat = x_cat.reshape(x.size(0),-1)
        x = self.fc1(x)
        # x= torch.cat([x,x_cat], dim=1)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x

# Generate sample data for demonstration (e.g., sine wave)

# Prepare the dataset
data = np.hstack((X[:,:2],X[:,3002:])).astype(float)
targets = y.astype(float)
data = data.reshape(-1, 1, 1002) 
# data = data.reshape(60, 4002, 1)  # Reshape for Conv1D: (num_samples, channels, length)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_rmse = []
fold_mae = []
fold_r_squared = []
fold_mape = []

# Perform k-fold cross-validation
for fold, (train_index, test_index) in enumerate(kf.split(data)):
    print(f'Fold {fold + 1}/{kf.n_splits}')
    
    # Split the data
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = targets[train_index], targets[test_index]

    # Create TensorDatasets for training and testing sets
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

    # Create DataLoaders for training and testing sets
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Initialize the model, loss function, and optimizer
    model = Conv1DInceptionModel()
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs.view(-1), labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Test the model
    rmse = []
    mae = []
    r_squared = []
    with torch.no_grad():
        y_true, y_pred = [], []
        for inp, lbl in test_loader:
            prediction = model(inp)
            y_true.append(lbl.item())
            y_pred.append(prediction.item())
        rmse=(np.sqrt(mean_squared_error(np.array(y_true), np.array(y_pred))))
        mae=(mean_absolute_error(np.array(y_true), np.array(y_pred)))
        
        # Calculate R-squared
        r2 = r2_score(np.array(y_true), np.array(y_pred))

    fold_rmse.append(rmse)
    fold_mae.append(mae)
    fold_r_squared.append(r2)
    fold_mape.append(mean_absolute_percentage_error(np.array(y_true), np.array(y_pred)))

    print(f'Fold {fold + 1} RMSE: {fold_rmse[-1]:.4f}, MAE: {fold_mae[-1]:.4f}, R²: {fold_r_squared[-1]:.4f}')

# Calculate the average metrics across all folds
# Calculate the average metrics and their standard deviations across all folds
avg_rmse = np.mean(fold_rmse)
std_rmse = np.std(fold_rmse)

avg_mae = np.mean(fold_mae)
std_mae = np.std(fold_mae)

avg_mape = np.mean(fold_mape)
std_mape = np.std(fold_mape)

avg_r_squared = np.mean(fold_r_squared)
std_r_squared = np.std(fold_r_squared)

print(f'Average MAE across all folds: {avg_mae:.4f} ± {std_mae:.4f}')
print(f'Average MAPE across all folds: {avg_mape:.4f} ± {std_mape:.4f}')

print(f'Average RMSE across all folds: {avg_rmse:.4f} ± {std_rmse:.4f}')

print(f'Average R² across all folds: {avg_r_squared:.4f} ± {std_r_squared:.4f}')
