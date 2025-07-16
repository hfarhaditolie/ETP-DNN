import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
def save_checkpoint(epoch, model, optimizer, loss, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, filename)
# Cathode
file_path = '../Data/data_CAT.xlsx'
df = pd.read_excel(file_path)
data = np.asarray(df)
names = data[:, 0]  # Sample names or IDs
data = data[:-1, 1:]
y_cat = data[:, -3]  # First column as target (Thickness)
X_cat= np.hstack((data[:,-5:-3],data[:,-2:-1],data[:,1:2001]))

# file_path = 'processed_data_AN.xlsx'
# df = pd.read_excel(file_path)
# data = np.asarray(df)
# names = data[:, 0]  # Sample names or IDs
# data = data[:, 1:]
# y= data[:,0]
from scipy.signal import medfilt
min_val, max_val = np.min(y_cat) , np.max(y_cat)
n_classes = 5
interval_size = (max_val - min_val) / n_classes
y_cathode = []
x_cathode = []
for i in range(5):
    #Load the generate synthetic data
    feat = (np.load("../Cathode/Synthetic/synthetic_class_"+str(i)+".npy"))
    for j in range(20):
        feat[j,:]=medfilt(feat[j,:],25)
    tar = min_val + (np.ones(20)*i + 0.5) * interval_size
    y_cathode.append(tar)
    x_cathode.append(feat)
x_cathode=np.concatenate(x_cathode,axis=0)
y_cathode=np.concatenate(y_cathode,axis=0)
shuffle_indices = np.random.permutation(len(y_cathode))

# Apply the permutation to shuffle x_cathode and y_cathode
x_cathode = x_cathode[shuffle_indices]
y_cathode= y_cathode[shuffle_indices]
from scipy.signal import find_peaks, medfilt
# plt.plot(range(2000),medfilt(x_cathode[0,:],31))
# plt.show()
# Define target variable and features
# y_cathode = data[:, 0]  # First column as target (Thickness)
# x_cathode = data[:, 2004:] 

# y_cathode =np.load("synthetic_targets.npy")  # First column as target (Thickness)
# x_cathode = np.load("synthetic_feat.npy") 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split

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
        self.swin_transformer1 = SWINTransformer1DBlock(input_dim=64, hidden_dim=128, num_heads=4, window_size=100)
        self.swin_transformer2 = SWINTransformer1DBlock(input_dim=64, hidden_dim=128, num_heads=4, window_size=100)
        self.fc1 = nn.Linear(16000, 64)  # Fully connected layer
        self.fc2 = nn.Linear(64, 1)  # Output layer for regression

    def forward(self, x):
        x = self.inception1(x)
        #x = self.swin_transformer(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool1d(kernel_size=2)(x)  # Max pooling layer

        x = self.inception2(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool1d(kernel_size=4)(x)  # Max pooling layer
        x = self.swin_transformer2(x)

        # x = x.view(x.size(0), -1)  # Flatten the output
        x = x.reshape(x.size(0),-1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x

# Prepare the dataset
x_cathode = np.hstack((np.zeros((x_cathode.shape[0],3)),x_cathode))
data = x_cathode.astype(float)
targets = y_cathode.astype(float)
data = data.reshape(-1, 1, 2003) 

# Create TensorDatasets for training and testing sets
train_dataset = TensorDataset(torch.tensor(data, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32))
# Create DataLoaders for training and testing sets
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# Initialize the model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Conv1DInceptionModel()
model = model.to(device)
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
num_epochs = 70
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(inputs.to(device))  # Forward pass
        loss = criterion(outputs.view(-1), labels.to(device))  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        save_checkpoint(epoch, model, optimizer, loss, f'checkpoints/cathode_epoch_{epoch+1}.pth')
    # Test the model
    rmse = []
    mae = []
    r_squared = []
    with torch.no_grad():
        y_true, y_pred = [], []
        for inp, lbl in train_loader:
            prediction = model(inp.to(device))

            # Ensure lbl and prediction are lists of values, not scalars
            y_true.extend(lbl.cpu().numpy())   # Append all 16 values from lbl
            y_pred.extend(prediction.cpu().numpy())  # Append all 16 predicted values

        # Convert lists to numpy arrays for metric calculations
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        print(f'RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}')
