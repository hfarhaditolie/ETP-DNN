import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from scipy.signal import find_peaks, peak_widths,peak_prominences
from scipy.stats import skew, kurtosis
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a
def highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# Load and prepare the dataset
file_path = 'Data/processed_data_AN.xlsx'
df = pd.read_excel(file_path)
data = np.asarray(df)
names = data[:, 0]  # Sample names or IDs
data = data[:, 1:]
# Define target values (Thickness)
y = data[:, 0]  # First column as target (Thickness)
#Forming the feature set with three categorical parameters including the Calendering Speed and Roll Gap, combined with ultrasonic signals
X = data[:, 2:]

# Extract signal data (specific columns related to signal)
signal_data = X[:,3002:]


N = signal_data.shape[1]  # Length of signal 
# Sampling frequency (Hz)
fs = 400  
cutoff_frequency = 4 
#high pass filtering of the signal
filtered_signal = highpass_filter(signal_data, cutoff_frequency, fs)

#downsampling the signal
downsample_factor = 30
downsampled_signal = filtered_signal[:, ::downsample_factor]

frequency_domain_signals = []
for i in range(60):
    current_signal = downsampled_signal[i, :]
    # Apply FFT
    N = len(current_signal) 
    fft_values = np.fft.fft(current_signal)
    fft_magnitude = np.abs(fft_values)  # Get the magnitude
    frequencies = np.fft.fftfreq(N, d=1/fs)
    # Only keep the positive frequencies
    positive_frequencies = frequencies[:N // 2]
    positive_magnitude = fft_magnitude[:N // 2]

    frequency_domain_signals.append(positive_magnitude)
freq=np.array(frequency_domain_signals)

X = np.hstack((X[:,1:3],downsampled_signal, freq))  # Modify features as needed

mae_list = []
rmse_list = []
r2_list = []
kf = KFold(n_splits=5, shuffle=True, random_state=42)
all_y_true, all_y_pred = [], []
mape_list=[]
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = CatBoostRegressor(depth=4, verbose=0)

    # Train the CatBoost model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r_squared = r2_score(y_test, y_pred)

    mae_list.append(mae)
    rmse_list.append(rmse)
    r2_list.append(r_squared)
    mape_list.append(mean_absolute_percentage_error(y_test,y_pred))
    print(rmse)

    # Calculate the mean and standard deviation of the evaluation metrics
    average_mae = np.mean(mae_list)
    std_mae = np.std(mae_list)

    average_rmse = np.mean(rmse_list)
    std_rmse = np.std(rmse_list)

    average_mape = np.mean(mape_list)
    std_mape = np.std(mape_list)

    average_r2 = np.mean(r2_list)
    std_r2 = np.std(r2_list)
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)

    # Print results
print(f'Average MAE: {average_mae:.3f} ± {std_mae:.3f}')
print(f'Average MAPE: {average_mape:.3f} ± {std_mape:.3f}')
print(f'Average RMSE: {average_rmse:.3f} ± {std_rmse:.3f}')

print(f'Average R2: {average_r2:.3f} ± {std_r2:.3f}')
print("---------------\n\r")