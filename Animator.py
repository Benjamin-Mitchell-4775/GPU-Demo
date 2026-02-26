import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import numpy as np

fig, ax = plt.subplots()
xs, ys = [], []

actual_outputs = np.load("actual_outputs.npy")
predictions = np.load("predictions.npy")

traj_results = []
actual_traj_results = []

for n in range(0,7113):

    actual_traj_results.append(actual_outputs[n])
    traj_results.append(predictions[n,4])

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

# filtered_traj_results = traj_results[::50]

def animate(i):
    # Add new data and limit to the last 20 points for scrolling
    xs.append(i)
    ys.append(traj_results[i])
    x_data = xs[-7113:]
    y_data = ys[-7113:]
    
    ax.clear()
    ax.plot(x_data, y_data)
    ax.set_title('Real-Time Data')

# Update every 1000ms
ani = animation.FuncAnimation(fig, animate, interval=100)
plt.show()
