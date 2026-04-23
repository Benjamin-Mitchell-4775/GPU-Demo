"""
SlidingWindow_Testing_V2.py
===========================
LSTM GNC Sliding-Window Inference — Live Per-Step Control Query
---------------------------------------------------------------
At every timestep t, this script:
  1. Grabs the raw position-error history  X[traj, max(0,t-W):t, :]
  2. Normalises it with input_scaler
  3. Feeds it into the LSTM → gets bank_angle prediction (rad)
  4. Inverse-transforms the output back to physical units
  5. Compares against the ground-truth bank angle

This mirrors exactly how the model would be queried in a live
closed-loop simulation — one step at a time, with only history
available up to the current moment.

Inputs (3 features per timestep, from inputVals_LSTM.npy):
    [x_error (m),  y_error (m),  z_error (m)]
    — position error vector from vehicle to target

Output (scalar, from outputVals_LSTM.npy):
    bank_angle (rad)  range ≈ [-π, π]

Usage
-----
    python SlidingWindow_Testing_V2.py

All file paths are in the CONFIG block below.
"""

import os
import warnings
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────
#  CONFIG  –  edit paths here, nothing else
# ─────────────────────────────────────────────
MODEL_PATH         = "lstm_seq2seq_model_smoothed.keras"
INPUT_SCALER_PATH  = "InputScaler_smoothed.gz"
OUTPUT_SCALER_PATH = "OutputScaler_smoothed.gz"
INPUT_DATA_PATH    = "inputVals_LSTM.npy"    # raw (n_traj, n_time, 3)
OUTPUT_DATA_PATH   = "outputVals_LSTM.npy"   # raw (n_traj, n_time)  bank angles

WINDOW_SIZE        = 15       # history window fed to model — must match training
TRAJ_INDEX         = 0        # which trajectory to run (0 = first)
PRINT_EVERY        = 500      # console log interval (timesteps)
FIGURE_DPI         = 150
OUTPUT_LABEL       = "Bank Angle (rad)"
# ─────────────────────────────────────────────


# ══════════════════════════════════════════════
#  1.  LOAD ASSETS
# ══════════════════════════════════════════════

def load_assets():
    """Load model, scalers, and raw trajectory arrays."""
    print("[1/4] Loading model and scalers …")
    model         = tf.keras.models.load_model(MODEL_PATH)
    input_scaler  = joblib.load(INPUT_SCALER_PATH)
    output_scaler = joblib.load(OUTPUT_SCALER_PATH)

    print("[2/4] Loading raw trajectory data …")
    inputVals_LSTM  = np.load(INPUT_DATA_PATH)   # (n_traj, n_time, 3)
    outputVals_LSTM = np.load(OUTPUT_DATA_PATH)  # (n_traj, n_time)

    n_traj, n_time, n_feat = inputVals_LSTM.shape
    print(f"      inputVals_LSTM  shape : {inputVals_LSTM.shape}")
    print(f"      outputVals_LSTM shape : {outputVals_LSTM.shape}")
    print(f"      Running trajectory    : {TRAJ_INDEX} / {n_traj - 1}")
    print(f"      Timesteps             : {n_time}")
    print(f"      Features (x,y,z err)  : {n_feat}")

    return model, input_scaler, output_scaler, inputVals_LSTM, outputVals_LSTM


# ══════════════════════════════════════════════
#  2.  PER-STEP LSTM QUERY
# ══════════════════════════════════════════════

def query_lstm(model, input_scaler, output_scaler, history_window):
    """
    Query the LSTM model with a single sliding window of position errors.

    Parameters
    ----------
    history_window : np.ndarray  (hist_len, 3)
        Raw (un-normalised) position error history up to current step.
        hist_len <= WINDOW_SIZE.  If shorter, it is right-aligned (zero-padded).

    Returns
    -------
    bank_angle : float   (radians)
    """
    hist_len = history_window.shape[0]

    # --- build padded window (WINDOW_SIZE, 3) ---
    padded = np.zeros((WINDOW_SIZE, 3), dtype=np.float32)
    padded[-hist_len:, :] = history_window          # right-align history

    # --- normalise inputs ---
    normed = input_scaler.transform(padded)          # (WINDOW_SIZE, 3)

    # --- reshape for model: (1, WINDOW_SIZE, 3) ---
    model_input = normed[np.newaxis, :, :]

    # --- query model → (1, WINDOW_SIZE, 1) seq2seq output ---
    raw_output = model.predict(model_input, verbose=0)

    # --- take last timestep → normalised bank angle ---
    bank_norm = raw_output[0, -1, 0]

    # --- inverse transform to physical units ---
    bank_angle = output_scaler.inverse_transform(
        np.array([[bank_norm]])
    )[0, 0]

    return float(bank_angle)


# ══════════════════════════════════════════════
#  3.  SIMULATION LOOP
# ══════════════════════════════════════════════

def run_sliding_window_inference(model, input_scaler, output_scaler,
                                  inputVals_LSTM, outputVals_LSTM):
    """
    Step through every timestep of one trajectory.
    At each step:
      - Slice the raw position-error history up to t
      - Call query_lstm() to get the bank angle command
      - Record predicted vs ground-truth bank angle
    """
    print("[3/4] Running per-step sliding window inference …\n")

    traj_inputs  = inputVals_LSTM[TRAJ_INDEX]   # (n_time, 3)  — x_err, y_err, z_err
    traj_outputs = outputVals_LSTM[TRAJ_INDEX]  # (n_time,)    — ground truth bank angle

    n_time = traj_inputs.shape[0]

    predictions   = np.zeros(n_time, dtype=np.float32)
    actual_values = traj_outputs.copy()

    for t in range(1, n_time):
        # Grab position-error history available up to (but not including) t
        # This is what the controller would actually have access to in real time
        history = traj_inputs[max(0, t - WINDOW_SIZE):t, :]  # (hist_len, 3)

        # ── LSTM QUERY ──────────────────────────────────────────────────
        try:
            bank_angle = query_lstm(model, input_scaler, output_scaler, history)
        except Exception as e:
            print(f"  [WARN] Model query failed at t={t}: {e} — defaulting to 0.0")
            bank_angle = 0.0
        # ────────────────────────────────────────────────────────────────

        predictions[t] = bank_angle

        if t % PRINT_EVERY == 0:
            x_err, y_err, z_err = traj_inputs[t]
            dist = np.linalg.norm([x_err, y_err, z_err])
            print(f"  t={t:>5d}  |err|={dist:>12.1f} m  "
                  f"bank_pred={np.degrees(bank_angle):>8.3f}°  "
                  f"bank_true={np.degrees(traj_outputs[t]):>8.3f}°")

    print()
    return predictions, actual_values


# ══════════════════════════════════════════════
#  4.  METRICS
# ══════════════════════════════════════════════

def compute_metrics(actual, predicted):
    """Compute and print evaluation metrics (skip t=0, no prediction there)."""
    print("[4/4] Computing metrics …\n")

    a = actual[1:]     # skip t=0 (no history available)
    p = predicted[1:]

    mae   = np.mean(np.abs(a - p))
    rmse  = np.sqrt(np.mean((a - p) ** 2))
    ss_res = np.sum((a - p) ** 2)
    ss_tot = np.sum((a - np.mean(a)) ** 2)
    r2    = 1.0 - ss_res / (ss_tot + 1e-12)

    print("╔══════════════════════════════════╗")
    print("║       EVALUATION RESULTS         ║")
    print("╠══════════════════════════════════╣")
    print(f"║  MAE        : {mae:>8.4f} rad             ║")
    print(f"║  MAE        : {np.degrees(mae):>8.4f} deg             ║")
    print(f"║  RMSE       : {rmse:>8.4f} rad             ║")
    print(f"║  R²         : {r2:>8.6f}                 ║")
    print("╚══════════════════════════════════╝\n")

    return {"mae": mae, "rmse": rmse, "r2": r2}


# ══════════════════════════════════════════════
#  5.  VISUALISATION
# ══════════════════════════════════════════════

def plot_results(actual, predicted, inputVals_LSTM, metrics):
    """4-panel diagnostic figure."""
    t = np.arange(len(actual))
    act_deg = np.degrees(actual)
    prd_deg = np.degrees(predicted)
    err_deg = prd_deg - act_deg

    # Position error magnitude over time (for context)
    pos_err_mag = np.linalg.norm(inputVals_LSTM[TRAJ_INDEX], axis=1) / 1000.0  # km

    fig = plt.figure(figsize=(16, 11), dpi=FIGURE_DPI)
    fig.suptitle(
        f"LSTM GNC Controller — Live Per-Step Bank Angle Query  "
        f"(Trajectory {TRAJ_INDEX})",
        fontsize=13, fontweight="bold", y=0.98
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.32)

    # ── Panel 1: Actual vs Predicted bank angle ────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t, act_deg, lw=1.0,  color="#2196F3", label="Ground Truth",   alpha=0.9)
    ax1.plot(t, prd_deg, lw=0.85, color="#FF5722", label="LSTM Prediction",
             alpha=0.75, linestyle="--")
    ax1.set_title("Bank Angle: Ground Truth vs LSTM Output (per-step query)")
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Bank Angle (degrees)")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.5)
    _annotate_metrics(ax1, metrics)

    # ── Panel 2: Prediction error ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t[1:], err_deg[1:], lw=0.8, color="#9C27B0", alpha=0.8)
    ax2.axhline(0, color="k", lw=0.6, linestyle="--")
    ax2.fill_between(t[1:], err_deg[1:], 0, alpha=0.15, color="#9C27B0")
    ax2.set_title("Bank Angle Error  (pred − truth)")
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Error (degrees)")
    ax2.grid(True, linestyle=":", alpha=0.5)

    # ── Panel 3: Position error magnitude (model input context) ───────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(t, pos_err_mag, lw=0.9, color="#4CAF50", alpha=0.85)
    ax3.set_title("Position Error Magnitude (model input context)")
    ax3.set_xlabel("Timestep")
    ax3.set_ylabel("|error| (km)")
    ax3.grid(True, linestyle=":", alpha=0.5)

    plt.savefig("SlidingWindow_Results_V2.png", bbox_inches="tight", dpi=FIGURE_DPI)
    print("Figure saved → SlidingWindow_Results_V2.png")
    plt.show()


def _annotate_metrics(ax, m):
    txt = (f"MAE={np.degrees(m['mae']):.3f}°  "
           f"RMSE={np.degrees(m['rmse']):.3f}°  "
           f"R²={m['r2']:.4f}")
    ax.text(0.01, 0.02, txt, transform=ax.transAxes,
            fontsize=8, color="dimgray",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))


# ══════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════

def main():
    print("\n━━━  LSTM GNC — Sliding Window Per-Step Inference  ━━━\n")

    model, input_scaler, output_scaler, inputVals_LSTM, outputVals_LSTM = load_assets()

    predictions, actual_values = run_sliding_window_inference(
        model, input_scaler, output_scaler, inputVals_LSTM, outputVals_LSTM
    )

    metrics = compute_metrics(actual_values, predictions)

    plot_results(actual_values, predictions, inputVals_LSTM, metrics)

    print("Done. ✓\n")


if __name__ == "__main__":
    main()
