fig, ax = plt.subplots()

xs = []
ys_pred = []
ys_actual = []

window = 200   # Number of points visible at once (adjust as desired)

def animate(i):
    if i >= len(traj_results):
        return

    xs.append(i)
    ys_pred.append(traj_results[i])
    ys_actual.append(actual_traj_results[i])

    # Rolling window
    x_data = xs[-window:]
    y_pred = ys_pred[-window:]
    y_actual = ys_actual[-window:]

    ax.clear()
    ax.plot(x_data, y_actual, label='Actual', linewidth=2)
    ax.plot(x_data, y_pred, label='Predicted', linestyle='--')

    ax.set_title('Real-Time Prediction vs Actual')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Control Trajectory')
    ax.legend()
    ax.grid(True)

ani = animation.FuncAnimation(
    fig,
    animate,
    frames=len(traj_results),
    interval=30,   # ~33 FPS animation
    repeat=False
)

plt.show()