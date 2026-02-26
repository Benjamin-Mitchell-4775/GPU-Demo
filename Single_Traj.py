import numpy as np

Y_All = np.load('outputVals_LSTM.npy')    
X_All = np.load("inputVals_LSTM.npy")

X_All = X_All[:1,:,:]
Y_All = Y_All[:1,:]

X_SingleTraj = X_All
Y_SingleTraj = Y_All

np.save('X_SingleTraj.npy', X_SingleTraj, allow_pickle=True)
np.save('Y_SingleTraj.npy', Y_SingleTraj, allow_pickle=True)
