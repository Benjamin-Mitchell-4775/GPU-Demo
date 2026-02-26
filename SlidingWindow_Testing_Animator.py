import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import splrep, BSpline
import joblib

def build_windows_from_struct(inp, out, window_size):
    """
    data: array (n_traj, n_time, n_feat)
    window_size: length of history to use

    returns:
        X -> (n_samples, window_size, n_feat)
        y -> (n_samples,) or (n_samples, k)
    """
    X, y = [], []

    n_traj, n_time, n_feat = inp.shape

    X_padded = np.zeros((n_traj * n_time, window_size, n_feat))
    y        = []

    sample_i = 0
    for traj in range(n_traj):
        for t in range(1, n_time):
            history = inp[traj, max(0, t - window_size):t, :]
            hist_len = history.shape[0]
            # Right-align the available history in the padded window
            X_padded[sample_i, -hist_len:, :] = history
            y.append(out[traj, t])
            sample_i += 1

    X_padded = X_padded[:sample_i]
    y = np.array(y)
    y = y[:, np.newaxis]


    return X_padded, y

model = load_model('lstm_seq2seq_model_smoothed.keras')  # Replace with the actual path to your .h5 file

input_scaler = joblib.load('InputScaler_smoothed.gz')
output_scaler = joblib.load('OutputScaler_smoothed.gz')

# Y_All = np.load('outputVals_LSTM.npy')    
# X_All = np.load("inputVals_LSTM.npy")

# X_All = X_All[:1,:,:]
# Y_All = Y_All[:1,:]

# X_SingleTraj = X_All
# Y_SingleTraj = Y_All

# np.save('X_SingleTraj.npy', X_SingleTraj, allow_pickle=True)
# np.save('Y_SingleTraj.npy', Y_SingleTraj, allow_pickle=True)

Y_All = np.load('Y_SingleTraj.npy')    
X_All = np.load("X_SingleTraj.npy")

# np.savetxt("input100.csv", inputs_padded, delimiter=',', fmt='%g')

print("The shape of input is:" + str(X_All.shape))
print("The shape of output is:" + str(Y_All.shape))

# Example usage
window_size = 15

X, y = build_windows_from_struct(X_All, Y_All, window_size)

print("The shape of input windowed is:" + str(X.shape))
print("The shape of output windowed is:" + str(y.shape))


X_test_normalized = input_scaler.transform(X.reshape(-1, X.shape[2])).reshape(X.shape)
y_test_normalized = output_scaler.transform(y.reshape(-1, 1)).reshape(y.shape)

# print("The shape of output is:" + str(X_All.shape))


# Evaluate the model on the test set
test_loss = model.evaluate(X_test_normalized, y_test_normalized)
print(f"Test loss: {test_loss}")
# Predict the outputs
predictions = model.predict(X_test_normalized)

print("Shape of X_Test_normalized: " + str(X_test_normalized.shape))

# Inverse transform the predictions to original scale
predictions = output_scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(predictions.shape)

actual_outputs = output_scaler.inverse_transform(y_test_normalized.reshape(-1, 1)).reshape(y_test_normalized.shape)
actual_inputs = input_scaler.inverse_transform(X_test_normalized.reshape(-1, 3)).reshape(X_test_normalized.shape)


print("!! Shape of Pre-ML Inputs: " + str(actual_inputs.shape))
print("!! Shape of Pre-ML Outputs: " + str(actual_outputs.shape))
print("!! Shape of ML produced Outputs: " + str(predictions.shape))

np.save("actual_outputs.npy", actual_outputs, allow_pickle=True)
np.save("predictions.npy", predictions, allow_pickle=True)


traj_results = []
actual_traj_results = []

for n in range(0,7113):

    actual_traj_results.append(actual_outputs[n])
    traj_results.append(predictions[n,14])



plt.figure(figsize=(10, 6))
plt.plot(actual_traj_results, label='Actual')
plt.plot(traj_results, label='Predicted')
plt.title('Actual vs Predicted on Test Sequence')
plt.xlabel('Time Step')
plt.ylabel('Control Trajectory')
plt.legend()
#plt.show()
figname = 'testLSTM'+str(00)+'.png'
plt.savefig(figname)

fig, ax = plt.subplots()
xs, ys = [], []

filtered_traj_results = traj_results[::50]

def animate(i):
    # Add new data and limit to the last 20 points for scrolling
    xs.append(i)
    ys.append(filtered_traj_results[i])
    x_data = xs[-7113:]
    y_data = ys[-7113:]
    
    ax.clear()
    ax.plot(x_data, y_data)
    ax.set_title('Real-Time Data')

# Update every 1000ms
ani = animation.FuncAnimation(fig, animate, interval=7113)
plt.show()







