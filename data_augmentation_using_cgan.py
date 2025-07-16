import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_blobs
import torch.nn.functional as F
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
latent_dim = 512  # dimension of the latent space
n_samples = 36    # size of our dataset, 60 for anode
n_classes = 5
n_features = 2000 # number of features, 1000 for anode

# Load and preprocess data
file_path = 'Data/data_CAT.xlsx' #change the data path for anode accordingly. Data loading should be similar to the one in mainMLAnode.py
df = pd.read_excel(file_path)
data = np.asarray(df)
names = data[:, 0]  # Sample names or IDs
data = data[:-1, 1:]
y = data[:, -3]     # First column as target (Thickness)
X = data[:, 1:2001]

X = X.astype(float)
y = y.astype(float)

# Convert y to class labels
min_val, max_val = np.min(y), np.max(y)
interval_size = (max_val - min_val) / n_classes
class_labels = ((y - min_val) // interval_size).astype(int)
class_labels[class_labels == n_classes] = n_classes - 1
y = class_labels

# Normalize features
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_X = scaler.fit_transform(X)

print(f'Size of our dataset: {len(X)}')
print(f'Number of features: {X.shape[1]}')
print(f'Classes: {set(y)}')

# Convert data to PyTorch tensors
X_tensor = torch.FloatTensor(scaled_X).to(device)
y_tensor = torch.LongTensor(y).to(device)

# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.label_embedding = nn.Sequential(
            nn.Embedding(n_classes, n_features),
            nn.Flatten()
        )
        
        self.model = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, labels):
        # Conditional discrimination
        label_embedding = self.label_embedding(labels)
        inputs = x * label_embedding
        return self.model(inputs)

# Define the Generator with 1D convolutional and Inception-like architecture


class Generator(nn.Module):
    def __init__(self, latent_dim, n_classes, n_features):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.n_features = n_features

        # Embedding layer for labels
        self.label_embedding = nn.Embedding(n_classes, latent_dim)

        # Inception-like convolutional block
        self.conv1 = nn.Conv1d(1, 64, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm1d(64, momentum=0.8)

        self.conv3 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64, momentum=0.8)

        self.conv5 = nn.Conv1d(1, 64, kernel_size=5, padding=2)
        self.bn5 = nn.BatchNorm1d(64, momentum=0.8)

        self.pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.pool_conv = nn.Conv1d(1, 64, kernel_size=1)
        self.bn_pool = nn.BatchNorm1d(64, momentum=0.8)

        # Post-inception convolution
        self.conv_post = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256, momentum=0.8)
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(256 * latent_dim, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512, momentum=0.8),
            nn.Linear(512, n_features),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Label embedding
        label_embed = self.label_embedding(labels).view(-1, self.latent_dim)

        # Element-wise multiplication of noise and label embedding
        x = noise * label_embed  # shape: (batch_size, latent_dim)
        x = x.unsqueeze(2)       # shape: (batch_size, latent_dim, 1)
        x = x.permute(0, 2, 1)   # shape: (batch_size, 1, latent_dim)

        # Inception-like module with BN
        out1 = self.bn1(F.leaky_relu(self.conv1(x), 0.2))
        out3 = self.bn3(F.leaky_relu(self.conv3(x), 0.2))
        out5 = self.bn5(F.leaky_relu(self.conv5(x), 0.2))
        out_pool = self.bn_pool(F.leaky_relu(self.pool_conv(self.pool(x)), 0.2))

        # Concatenate along channel axis
        x = torch.cat([out1, out3, out5, out_pool], dim=1)

        # Further conv + flatten + FC
        x = self.conv_post(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# Initialize models
discriminator = Discriminator().to(device)
generator = Generator(latent_dim=latent_dim,n_classes=n_classes,n_features=n_features).to(device)

# Optimizers
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Loss function
criterion = nn.BCELoss()

# Training function
def train_gan(generator, discriminator, X, y, n_epochs=100, batch_size=16, log_interval=10):
    # Lists to store training history
    d_losses = []
    g_losses = []
    real_accs = []
    fake_accs = []
    
    for epoch in range(n_epochs):
        # Get random batch
        idx = np.random.randint(0, len(X) - batch_size)
        real_data = X_tensor[idx:idx+batch_size]
        real_labels = y_tensor[idx:idx+batch_size]
        
        # Real data
        real_target = torch.ones(batch_size, 1, device=device)
        
        # Fake data
        noise = torch.rand(batch_size, latent_dim, device=device) * 0.5
        fake_labels = torch.randint(0, n_classes, (batch_size,), device=device)
        fake_data = generator(noise, fake_labels)
        fake_target = torch.zeros(batch_size, 1, device=device)
        
        # Train Discriminator
        discriminator.zero_grad()
        
        # Real loss
        real_output = discriminator(real_data, real_labels)
        d_loss_real = criterion(real_output, real_target)
        
        # Fake loss
        fake_output = discriminator(fake_data.detach(), fake_labels)
        d_loss_fake = criterion(fake_output, fake_target)
        
        # Total discriminator loss
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        d_optimizer.step()
        
        # Train Generator
        generator.zero_grad()
        
        # Try to fool the discriminator
        gen_output = discriminator(fake_data, fake_labels)
        g_loss = criterion(gen_output, real_target)
        g_loss.backward()
        g_optimizer.step()
        
        # Calculate accuracies
        real_acc = (real_output > 0.5).float().mean()
        fake_acc = (fake_output < 0.5).float().mean()
        
        # Store losses and accuracies
        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())
        real_accs.append(real_acc.item())
        fake_accs.append(fake_acc.item())
        
        # Log progress
        if (epoch + 1) % log_interval == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}] '
                  f'D Loss: {d_loss.item():.4f} '
                  f'G Loss: {g_loss.item():.4f} '
                  f'Real Acc: {real_acc.item():.4f} '
                  f'Fake Acc: {fake_acc.item():.4f}')
    
    return d_losses, g_losses, real_accs, fake_accs

# Train the GAN
d_losses, g_losses, real_accs, fake_accs = train_gan(
    generator, discriminator, X_tensor, y_tensor, n_epochs=100, batch_size=16)

# Function to generate samples
def generate_samples(class_for, n_samples=20):
    generator.eval()
    with torch.no_grad():
        noise = torch.rand(n_samples, latent_dim, device=device) * 0.05
        labels = torch.full((n_samples,), class_for, dtype=torch.long, device=device)
        samples = generator(noise, labels).cpu().numpy()
    return scaler.inverse_transform(samples)

# for i in range(n_classes):
#     feats = generate_samples(i)
#     np.save(f"synthetic_class_{i}.npy", feats)

# Example visualization of generated features
features_class_0 = generate_samples(0)
plt.figure(figsize=(15, 6))
plt.plot(range(n_features), features_class_0[0, :])
plt.title('Generated Sample from Class 0')
plt.xlabel('Feature Index')
plt.ylabel('Feature Value')
plt.show()