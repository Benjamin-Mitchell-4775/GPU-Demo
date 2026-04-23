"""
Integrated Closed-Loop Entry Guidance Simulation

This script runs a single-trajectory simulation using vehicle dynamics and a
sliding-window LSTM controller. It supports three controller modes:
    - LSTM only
    - Apollo (baseline) only
    - COMPARE (runs both for evaluation)

Key Features:
    * Single deterministic initial condition generation
    * Real-time sliding-window LSTM querying
    * Physics-based dynamics propagation
    * Optional controller comparison plots
    * Configurable vehicle parameters (e.g., reference area)

The structure is intentionally compact for live visualization, debugging,
and demonstration purposes.
"""

# ============================================================
# IMPORTS
# ============================================================

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib

# ============================================================
# LOAD RESET VARIABLES (Single Source of Truth)
# ============================================================

"""
All constants from the MATLAB reset vars file should be defined in a
Python module named `reset_vars.py`.

This script imports those variables directly so that the dynamics and
controllers always use the same physical constants as the validated
MATLAB implementation.
"""

try:
    from reset_vars import *
except ImportError:
    print("WARNING: reset_vars.py not found. Using fallback placeholders.")

# Map commonly used simulation names to reset vars equivalents
# (keeps physics consistent while allowing readable variable names)

try:
    R_PLANET = RE
    MU = MUE
except NameError:
    pass

# ============================================================
# VARIABLES REQUIRED BUT NOT DEFINED IN RESET VARS
# ============================================================

"""
Anything listed here must be defined either in:

1) reset_vars.py
OR
2) the main simulation configuration

This makes missing dependencies explicit and prevents silent failures.
"""

REQUIRED_EXTERNAL_VARIABLES = [
    "SIM_STEPS",
    "WINDOW_SIZE",
    "MAX_BANK_RAD",
    "PRINT_EVERY",
    "LIVE_PLOT_EVERY",
    "VEHICLE_AREA",
    "MASS_SC"
]


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

# ============================================================
# CONTROLLER MODE SWITCH
# ============================================================


CONTROLLER_MODE = "COMPARE"

# ============================================================
# VEHICLE PARAMETERS
# ============================================================


def generate_initial_condition():

    h0 = R_PLANET + 120000.0

    lat = np.deg2rad(-12.7)
    lon = np.deg2rad(122.9)

    x_offset = 0.0
    y_offset = 0.0
    z_offset = 0.0

    r = np.array([
        h0 * np.cos(lat) * np.cos(lon) + x_offset,
        h0 * np.cos(lat) * np.sin(lon) + y_offset,
        h0 * np.sin(lat) + z_offset
    ])

    v = np.array([
        0.0,
        7500.0,
        0.0
    ])

    target = np.array([
        0.0,
        0.0,
        0.0
    ])

    bank = 0.0

    return r, v, target, bank

# ============================================================
# ATMOSPHERE MODEL
# ============================================================

def atmosphere(h):

    rho0 = 1.225
    H = 8500.0

    density = rho0 * np.exp(-h / H)

    speed_sound = 340.0

    return density, speed_sound

# ============================================================
# LIFT / DRAG MODEL
# ============================================================

def lift_and_drag(mach):

    cl = 0.3
    cd = 1.2

    return cl, cd

# ============================================================
# ROTATION
# ============================================================

def rotation(f_lift, f_drag, bank):

    c = np.cos(bank)
    s = np.sin(bank)

    rot_matrix = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])

    f_lift = rot_matrix @ f_lift

    return f_lift, f_drag

# ============================================================
# DYNAMICS PROPAGATION
# ============================================================

def propagate_dynamics(r, v, bank):

    altitude = np.linalg.norm(r) - R_PLANET

    density, speed_sound = atmosphere(altitude)

    velocity_mag = np.linalg.norm(v)

    mach = velocity_mag / speed_sound

    cl, cd = lift_and_drag(mach)

    f_grav = (
        -MU * MASS_SC
        / (np.linalg.norm(r) ** 3)
        * r
    )

    drag_dir = -v / velocity_mag

    f_drag = (
        0.5
        * cd
        * VEHICLE_AREA
        * density
        * velocity_mag ** 2
        * drag_dir
    )

    lift_vec = r - (
        np.dot(r, v)
        / velocity_mag ** 2
    ) * v

    lift_vec = lift_vec / np.linalg.norm(lift_vec)

    f_lift = (
        0.5
        * cl
        * VEHICLE_AREA
        * density
        * velocity_mag ** 2
        * lift_vec
    )

    f_lift, f_drag = rotation(
        f_lift,
        f_drag,
        bank
    )

    total_force = f_grav + f_drag + f_lift

    acc = total_force / MASS_SC

    v = v + acc * DT
    r = r + v * DT

    return r, v

# ============================================================
# APOLLO / BASELINE CONTROLLER
# ============================================================

def apollo_controller(error_vector):

    K = 1e-6

    bank = K * np.linalg.norm(error_vector)

    bank = max(
        min(bank, MAX_BANK_RAD),
        -MAX_BANK_RAD
    )

    return bank

# ============================================================
# LSTM CONTROLLER
# ============================================================

def lstm_controller(model,
                    input_scaler,
                    output_scaler,
                    history):

    if model is None:
        return 0.0

    hist_len = history.shape[0]

    padded = np.zeros((WINDOW_SIZE, 3))
    padded[-hist_len:, :] = history

    normed = input_scaler.transform(padded)

    model_input = normed[np.newaxis, :, :]

    prediction = model.predict(
        model_input,
        verbose=0
    )

    bank_norm = prediction[0, -1, 0]

    bank = output_scaler.inverse_transform(
        np.array([[bank_norm]])
    )[0, 0]

    bank = float(bank)

    bank = max(
        min(bank, MAX_BANK_RAD),
        -MAX_BANK_RAD
    )

    return bank

# ============================================================
# PLOTTING
# ============================================================

def init_plots():

    plt.ion()

    fig = plt.figure(figsize=(12, 8))

    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    line_lstm, = ax1.plot([], [])
    line_apollo, = ax1.plot([], [])

    err_line, = ax2.plot([], [])

    ax1.set_title("Bank Command Comparison")
    ax1.set_ylabel("Degrees")
    ax1.grid(True)

    ax2.set_title("Position Error Magnitude")
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Meters")
    ax2.grid(True)

    return fig, ax1, ax2, line_lstm, line_apollo, err_line

# ============================================================
# MAIN SIMULATION
# ============================================================

def run_simulation():

    model, input_scaler, output_scaler = load_lstm()

    r, v, target, bank = generate_initial_condition()

    history_buffer = np.zeros((0, 3))

    bank_lstm_hist = np.zeros(SIM_STEPS)
    bank_apollo_hist = np.zeros(SIM_STEPS)

    error_hist = np.zeros(SIM_STEPS)

    fig, ax1, ax2, line_lstm, line_apollo, err_line = init_plots()

    print("Simulation starting...")

    for step in range(SIM_STEPS):

        error_vector = r - target

        history_buffer = np.vstack([
            history_buffer,
            error_vector
        ])

        if history_buffer.shape[0] > WINDOW_SIZE:

            history_buffer = history_buffer[-WINDOW_SIZE:]

        if step >= WINDOW_SIZE:

            bank_lstm = lstm_controller(
                model,
                input_scaler,
                output_scaler,
                history_buffer
            )

        else:

            bank_lstm = 0.0

        bank_apollo = apollo_controller(
            error_vector
        )

        if CONTROLLER_MODE == "LSTM":

            bank = bank_lstm

        elif CONTROLLER_MODE == "APOLLO":

            bank = bank_apollo

        elif CONTROLLER_MODE == "COMPARE":

            bank = bank_lstm

        r, v = propagate_dynamics(
            r,
            v,
            bank
        )

        error_mag = np.linalg.norm(
            error_vector
        )

        bank_lstm_hist[step] = bank_lstm
        bank_apollo_hist[step] = bank_apollo
        error_hist[step] = error_mag

        altitude = np.linalg.norm(r) - R_PLANET

        if altitude < 7620:

            print("Termination altitude reached")
            break

        if step % PRINT_EVERY == 0:

            print(
                f"Step {step:6d} | "
                f"Alt {altitude:10.2f} m | "
                f"Err {error_mag:10.2f} m"
            )

        if step % LIVE_PLOT_EVERY == 0:

            line_lstm.set_data(
                np.arange(step),
                np.degrees(
                    bank_lstm_hist[:step]
                )
            )

            if CONTROLLER_MODE == "COMPARE":

                line_apollo.set_data(
                    np.arange(step),
                    np.degrees(
                        bank_apollo_hist[:step]
                    )
                )

            err_line.set_data(
                np.arange(step),
                error_hist[:step]
            )

            ax1.relim()
            ax1.autoscale_view()

            ax2.relim()
            ax2.autoscale_view()

            fig.canvas.draw()
            fig.canvas.flush_events()

    plt.ioff()

    print("Simulation complete.")

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_simulation()
