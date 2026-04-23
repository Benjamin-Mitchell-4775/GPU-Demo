"""
simulate_lstm_guidance.py
=========================
Closed-loop LSTM Entry Guidance Simulation
Apollo 10 / Earth re-entry scenario

CONTROLLER MODES:
    "LSTM"    — LSTM model drives bank angle every timestep
    "APOLLO"  — Apollo guidance law only (baseline)
    "COMPARE" — Runs LSTM but plots both for side-by-side comparison

HOW TO RUN:
    python simulate_lstm_guidance.py

REQUIRED FILES (same directory):
    reset_vars.py                       — guidance constants (auto-generated)
    lstm_seq2seq_model_smoothed.keras   — trained LSTM weights
    InputScaler_smoothed.gz             — sklearn input scaler
    OutputScaler_smoothed.gz            — sklearn output scaler
"""

# ============================================================
# IMPORTS
# ============================================================

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("WARNING: TensorFlow not found. LSTM controller will fall back to bank_angle=0.")

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    print("WARNING: joblib not found. Scalers cannot be loaded.")

# ============================================================
# LOAD RESET VARS  (single source of truth for all constants)
# ============================================================

from reset_vars import (
    # Physics
    ATK, GS, HS, J, KWE, MUE, RE, VSAT, WIE,
    # Guidance gains
    C1, C16, C17, C18, C19, C20, CHOOK, CH1,
    D0MAX, GMAX, KA, KB1, KB2, KDMIN,
    KLAT, LATBIAS, KTETA, K44,
    LAD, L_DCMINR, LEWD, LOD,
    Q2, Q3, Q5, Q6, Q7F,
    VFINAL, VLMIN, VRCONTRL, VCORLIM, NM_25, VQUIT,
    # Convenience helper
    get_guidance_state,
)

# Rename to match Python simulation naming convention
R_PLANET = RE * 0.3048          # Convert ft → m  (6,371,084 m ≈ 6371 km ✓)
MU       = MUE * 1e14           # 3.986e14 m^3/s^2

# ============================================================
# SIMULATION CONFIGURATION  — edit these
# ============================================================

CONTROLLER_MODE  = "COMPARE"    # "LSTM" | "APOLLO" | "COMPARE"

SIM_STEPS        = 10000        # max integration steps
DT_SIM           = 0.1         # timestep [s]  (matches MATLAB dt=0.1)
WINDOW_SIZE      = 15           # LSTM sliding-window length

MASS_SC          = 5498.22      # kg  (Apollo 10 pre-entry mass)
VEHICLE_AREA     = 12.017       # m^2 (Apollo 10 reference area)

MAX_BANK_RAD     = np.deg2rad(180.0)   # hard clamp on bank output

PRINT_EVERY      = 200          # console print cadence [steps]
LIVE_PLOT_EVERY  = 100          # live plot refresh cadence [steps]

TERM_ALTITUDE_M  = 7620 * 0.3048  # ~2.32 km  (matches MATLAB rPlanet+7620 ft)

MODEL_PATH       = "lstm_seq2seq_model_smoothed.keras"
IN_SCALER_PATH   = "InputScaler_smoothed.gz"
OUT_SCALER_PATH  = "OutputScaler_smoothed.gz"

# ============================================================
# ATMOSPHERE MODEL  (NASA Standard Atmosphere)
# Matches Atmosphere() function in Trans_MainJones
# ============================================================

def compute_atmosphere(h_m):
    """
    Returns (density [kg/m^3], speed_of_sound [m/s]) for altitude h_m in metres.
    Uses the NASA Standard Atmosphere model (same logic as MATLAB source).
    Falls back to exponential model above 80 km for continuity.
    """
    if h_m > 25000:
        temp_c   = -131.21 + 0.00299 * h_m
        pressure = 2.488 * ((temp_c + 273.1) / 216.6) ** (-11.388)
    elif h_m > 11000:
        temp_c   = -56.46
        pressure = 22.65 * np.exp(1.73 - 0.000157 * h_m)
    else:
        temp_c   = 15.04 - 0.00649 * h_m
        pressure = 101.29 * ((temp_c + 273.1) / 288.08) ** 5.256

    temp_k      = temp_c + 273.1
    density     = pressure / (0.2869 * temp_k)
    speed_sound = np.sqrt(1.4 * 286.0 * (temp_c + 273.15))

    # Safety floor — avoids divide-by-zero at extreme altitudes
    density     = max(density, 1e-12)
    speed_sound = max(speed_sound, 1.0)

    return density, speed_sound


# ============================================================
# LIFT / DRAG MODEL  (Apollo 11 lookup table)
# Matches liftAndDrag() in Trans_MainJones
# ============================================================

# Apollo 11 aerodynamic data  (page 379 of Apollo document)
_MACH_TABLE = np.array([0.4, 0.7, 0.9, 1.1, 1.2, 1.35, 1.65,
                         2.0, 2.4, 3.0, 4.0, 10.0, 29.5])
_CL_TABLE   = np.array([0.24465, 0.26325, 0.32074, 0.49373, 0.47853,
                         0.56282, 0.55002, 0.53247, 0.50740, 0.47883,
                         0.44147, 0.42856, 0.38773])
_CD_TABLE   = np.array([0.853,   0.98542, 1.10652, 1.1697,  1.156,
                         1.2788,  1.2657,  1.2721,  1.2412,  1.2167,
                         1.2148,  1.2246,  1.2891])

def compute_lift_drag(mach):
    """
    Returns (CL, CD) interpolated from the Apollo 11 lookup table.
    Clamps to table bounds — no extrapolation errors.
    """
    mach = np.clip(mach, _MACH_TABLE[0], _MACH_TABLE[-1])
    cl   = float(np.interp(mach, _MACH_TABLE, _CL_TABLE))
    cd   = float(np.interp(mach, _MACH_TABLE, _CD_TABLE))
    return cl, cd


# ============================================================
# ROTATION  (Arbitrary-axis rotation about drag vector)
# Matches rotation() in Trans_MainJones exactly
# ============================================================

def rotate_lift_vector(f_lift, f_drag, bank_rad):
    """
    Rotates f_lift about the f_drag axis by bank_rad.
    Uses the arbitrary-axis rotation formula from Trans_MainJones.
    Returns rotated f_lift; f_drag is unchanged.
    """
    drag_sq = np.dot(f_drag, f_drag)
    if drag_sq < 1e-20:
        return f_lift, f_drag          # near-zero drag — no rotation

    dot_fd_fl = np.dot(f_drag, f_lift)
    sqrt_dsq  = np.sqrt(drag_sq)
    c         = np.cos(bank_rad)
    s         = np.sin(bank_rad)

    L = np.array([
        (f_drag[0] * dot_fd_fl * (1 - c)
         + drag_sq * f_lift[0] * c
         + sqrt_dsq * (-f_drag[2] * f_lift[1] + f_drag[1] * f_lift[2]) * s) / drag_sq,

        (f_drag[1] * dot_fd_fl * (1 - c)
         + drag_sq * f_lift[1] * c
         + sqrt_dsq * ( f_drag[2] * f_lift[0] - f_drag[0] * f_lift[2]) * s) / drag_sq,

        (f_drag[2] * dot_fd_fl * (1 - c)
         + drag_sq * f_lift[2] * c
         + sqrt_dsq * (-f_drag[1] * f_lift[0] + f_drag[0] * f_lift[1]) * s) / drag_sq,
    ])
    return L, f_drag


# ============================================================
# DYNAMICS PROPAGATION
# Preserves all physics from Trans_MainJones inner loop
# ============================================================

def propagate_dynamics(r, v, bank_rad):
    """
    One Euler integration step of spacecraft entry dynamics.

    Inputs
    ------
    r        : position vector [m, ECEF]
    v        : velocity vector [m/s, ECEF]
    bank_rad : bank angle command from controller [rad]

    Returns
    -------
    r_new, v_new, acc_no_grav
    acc_no_grav is the non-gravitational acceleration (needed by Apollo guidance)
    """
    r_norm   = np.linalg.norm(r)
    v_norm   = np.linalg.norm(v)

    if v_norm < 1e-6:
        v_norm = 1e-6   # guard against stationary vehicle at t=0

    altitude = r_norm - R_PLANET
    density, speed_sound = compute_atmosphere(altitude)
    mach                 = v_norm / speed_sound
    cl, cd               = compute_lift_drag(mach)

    # Gravity
    f_grav = (-MU * MASS_SC / r_norm ** 3) * r

    # Drag  (opposes velocity)
    drag_dir = -v / v_norm
    f_drag   = 0.5 * cd * VEHICLE_AREA * density * v_norm ** 2 * drag_dir

    # Lift direction: component of r perpendicular to v
    lift_dir = r - (np.dot(r, v) / v_norm ** 2) * v
    lift_norm = np.linalg.norm(lift_dir)
    if lift_norm < 1e-10:
        lift_dir = np.array([0.0, 0.0, 1.0])
    else:
        lift_dir = lift_dir / lift_norm

    f_lift = 0.5 * cl * VEHICLE_AREA * density * v_norm ** 2 * lift_dir

    # Apply bank rotation only when drag is non-trivial
    if np.linalg.norm(f_drag) > 1e-6:
        f_lift, f_drag = rotate_lift_vector(f_lift, f_drag, bank_rad)

    total_force = f_grav + f_drag + f_lift
    acc         = total_force / MASS_SC
    acc_no_grav = (f_drag + f_lift) / MASS_SC

    v_new = v + acc * DT_SIM
    r_new = r + v_new * DT_SIM

    return r_new, v_new, acc_no_grav


# ============================================================
# APOLLO GUIDANCE CONTROLLER
# Full port of att_ctrl() from Trans_MainJones
# State is kept in a plain dict so it can be reset easily
# ============================================================

def make_apollo_state():
    """Returns a fresh Apollo guidance state dict (mirrors get_guidance_state)."""
    s = get_guidance_state()
    # Override THETNM with correct initial value
    s["THETNM"] = 2000.0
    return s


def apollo_guidance(r_SC, v_SC, acc_no_grav, target, state, dt=DT_SIM):
    """
    Apollo entry guidance law.  Full port of att_ctrl() from Trans_MainJones.

    All units are internally converted to feet/fps (as the original algorithm
    requires), then the output bank angle (rotSC[0]) is returned in radians.

    Parameters
    ----------
    r_SC       : position [m]
    v_SC       : velocity [m/s]
    acc_no_grav: non-grav acceleration [m/s^2]
    target     : target position [m]
    state      : dict of guidance state variables (mutated in place)
    dt         : timestep [s]

    Returns
    -------
    bank_rad : bank angle command [rad]
    state    : updated guidance state dict
    """
    # -------- Unpack state --------
    s = state  # shorthand alias

    # -------- Unit conversions (m → ft, m/s → fps) --------
    m2ft  = 3.28084
    V     = np.linalg.norm(v_SC) * m2ft          # speed [fps]
    VSQ   = (V / VSAT) ** 2
    LEQ   = (VSQ - 1) * GS

    unit_R = r_SC / np.linalg.norm(r_SC)
    RDOT   = np.dot(v_SC, unit_R) * m2ft          # [fps]

    temp   = np.cross(v_SC, unit_R)
    temp_n = np.linalg.norm(temp)
    UNI_   = temp / temp_n if temp_n > 1e-10 else np.array([0., 0., 1.])

    D      = np.linalg.norm(acc_no_grav) * m2ft   # non-grav decel [fpss]

    URT_   = target / np.linalg.norm(target)
    LATANG = np.dot(URT_, UNI_)

    dot_urt_r = np.dot(URT_, unit_R)
    dot_urt_r = np.clip(dot_urt_r, -1.0, 1.0)
    THETA  = np.arccos(dot_urt_r)

    # -------- Initialise K2ROLL once --------
    if s["intitK2"] == 0:
        s["K2ROLL"]  = -np.sign(LATANG) if abs(LATANG) > 1e-10 else 1.0
        s["intitK2"] = 1

    # Store derived quantities we'll need in sub-functions
    s["V"]      = V
    s["VSQ"]    = VSQ
    s["LEQ"]    = LEQ
    s["RDOT"]   = RDOT
    s["UNI_"]   = UNI_
    s["D"]      = D
    s["URT_"]   = URT_
    s["LATANG"] = LATANG
    s["THETA"]  = THETA
    s["rSC"]    = r_SC
    s["target"] = target

    # -------- MODE SELECTOR --------
    sel = s["SELECTOR"]
    if   sel == 1:  _kep2(s)
    elif sel == 2:  _huntest(s)
    elif sel == 3:  _predict3(s)
    elif sel == 4:  _upcontrol(s)
    else:           _init_roll(s)      # selector == 0

    # -------- ThreeEighty: slew rotSC toward ROLLC --------
    _three_eighty(s, dt)

    bank_rad = s["rotSC"]
    return bank_rad, s


# ---- Apollo sub-functions (nested in MATLAB, flat here) ----

def _final_phase_lookup(V):
    lv     = [0, 337, 1080, 2103, 3922, 6295, 8531, 10101, 14014, 15951, 18357, 20829, 23090, 23500, 35000]
    lF2    = [0, 0, 0.002591, 0.003582, 0.007039, 0.01446, 0.02479, 0.03391, 0.06139, 0.07683, 0.09982, 0.1335, 0.2175, 0.3046, 0.3046]
    lF1    = [-0.02695, -0.02695, -0.03629, -0.05551, -0.09034, -0.1410, -0.1978, -0.2372, -0.3305, -0.3605, -0.4956, -0.6483, -2.021, -3.354, -3.354]
    lRTOGO = [0, 0, 2.7, 8.9, 22.1, 46.3, 75.4, 99.9, 170.9, 210.3, 266.8, 344.3, 504.8, 643.0, 643.0]
    lF3    = [1, 1, 6.44*2, 10.91*2, 21.64*2, 48.35*2, 93.72*2, 141.1*2, 329.4, 465.5, 682.7, 980.5, 1385, 1508, 1508]

    F1    = float(np.interp(V, lv, lF1))
    F2    = float(np.interp(V, lv, lF2))
    F3    = float(np.interp(V, lv, lF3))
    RTOGO = float(np.interp(V, lv, lRTOGO))
    return F1, F2, F3, RTOGO


def _three_ten2(s):
    L_D  = s["L_D"]
    LAD_ = LAD
    if abs(L_D / LAD_) - 1 > 0:
        L_D = LAD_ * np.sign(L_D)
    s["L_D"]  = L_D
    s["ROLLC"] = s["K2ROLL"] * np.arccos(np.clip(L_D / LAD_, -1, 1)) + 2 * np.pi * s["K1ROLL"]


def _three_eighty(s, dt):
    ROLLC = s.get("ROLLC", 0.0)
    rotSC = s.get("rotSC", 0.0)
    if abs(rotSC - ROLLC) < 20 * dt:
        rotSC = ROLLC
    elif ROLLC - rotSC > rotSC - ROLLC:
        rotSC = rotSC + 20 * dt
    else:
        rotSC = rotSC - 20 * dt
    s["rotSC"] = rotSC


def _three_ten(s):
    V    = s["V"]
    VSQ  = s["VSQ"]
    LATANG = s["LATANG"]
    L_D  = s["L_D"]

    if s["GONEPAST"] == 0:
        Y = KLAT * VSQ + LATBIAS / 3440.0
        if abs(L_D) - L_DCMINR < 0:
            if s["K2ROLL"] * LATANG - Y > 0:
                s["K2ROLL"] = -s["K2ROLL"]
                if L_D < 0:
                    s["K1ROLL"] = s["K1ROLL"] - s["K2ROLL"]
        else:
            Y = Y / 2.0
            if s["K2ROLL"] * LATANG > 0:
                if s["K2ROLL"] * LATANG - Y > 0:
                    s["K2ROLL"] = -s["K2ROLL"]
                    if L_D < 0:
                        s["K1ROLL"] = s["K1ROLL"] - s["K2ROLL"]
            else:
                s["L_D"] = L_DCMINR * np.sign(L_D)
    _three_ten2(s)


def _negtest(s):
    if (s["L_D"] < 0) and (s["D"] - C20 > 0):
        s["L_D"] = 0.0
    _three_ten(s)


def _glimiter(s):
    D    = s["D"]
    RDOT = s["RDOT"]
    V    = s["V"]
    if GMAX / 2 - D > 0:
        _three_ten(s)
    else:
        if GMAX - D > 0:
            X = np.sqrt(abs(2 * HS * (GMAX - D) * (s["LEQ"] / GMAX + LAD) + (2 * HS * GMAX / V) ** 2))
            if RDOT + X > 0:
                _three_ten(s)
            else:
                s["L_D"] = LAD
                _three_ten(s)
        else:
            s["L_D"] = LAD
            _three_ten(s)


def _constd(s):
    D    = s["D"]
    D0   = s["D0"]
    RDOT = s["RDOT"]
    V    = s["V"]
    s["L_D"] = (-s["LEQ"] / D0
                + C16 * (D - D0)
                - C17 * (RDOT + 2 * HS * D0 / V))
    _negtest(s)


def _huntest1_and_range_prediction(s):
    """
    Iterative replacement for the mutually recursive _huntest1 / _range_prediction pair.

    In the original MATLAB the two nested functions call each other to converge on
    a range prediction.  In Python that blows the call stack.  The logic is
    identical — we just run it in a while-loop instead.

    Loop exits when one of these conditions is met:
        • |DIFF| < 25  → hand off to UPCONTROL  (SELECTOR=4)
        • VL < VLMIN   → hand off to PREDICT3   (SELECTOR=3)
        • VL > VSAT    → hand off to CONSTD      (SELECTOR=2)
        • HIND==1 on a second pass  (correction applied, done)
        • Safety cap of MAX_ITER iterations
    """
    MAX_ITER = 20

    for _ in range(MAX_ITER):

        # ---- HUNTEST1 block ----
        V1    = s["V1"]
        A0    = s["A0"]
        ALP   = (2 * C1 * A0 * HS) / (LEWD * V1 ** 2) if V1 > 0 else 0
        FACT1 = V1 / (1 - ALP) if abs(1 - ALP) > 1e-10 else V1
        FACT2 = ALP * (ALP - 1) / A0 if A0 > 0 else 0
        inner = FACT2 * s["Q7"] + ALP
        VL    = FACT1 * (1 - np.sqrt(max(inner, 0)))

        s["ALP"]   = ALP
        s["FACT1"] = FACT1
        s["FACT2"] = FACT2
        s["VL"]    = VL

        # Exit branches that don't continue the loop
        if VL - VLMIN < 0:
            s["SELECTOR"] = 3
            s["EGSW"]     = 1
            _predict3(s)
            return

        if VL - VSAT > 0:
            s["SELECTOR"] = 2
            _constd(s)
            return

        # Compute GAMMAL / GAMMAL1
        VS1     = VSAT if V1 > VSAT else V1
        DVL     = VS1 - VL
        DHOOK   = ((1 - VS1 / FACT1) ** 2 - ALP) / FACT2 if abs(FACT2) > 1e-10 else 0
        AHOOK   = CHOOK * (DHOOK / s["Q7"] - 1) / DVL if (abs(DVL) > 1e-10 and s["Q7"] > 0) else 0
        GAMMAL1 = LEWD * (V1 - VL) / VL if VL > 0 else 0
        GAMMAL  = (GAMMAL1
                   - (CH1 * GS * DVL ** 2 * (1 + AHOOK * DVL))
                   / (DHOOK * VL ** 2)
                   if (abs(DHOOK) > 1e-10 and VL > 0) else GAMMAL1)

        s.update({"VS1": VS1, "DVL": DVL, "DHOOK": DHOOK,
                  "AHOOK": AHOOK, "GAMMAL1": GAMMAL1, "GAMMAL": GAMMAL})

        if GAMMAL < 0:
            denom = (LEWD - (3 * AHOOK * DVL ** 2 + 2 * DVL)
                     * (CH1 * GS / (DHOOK * VL))
                     if abs(DHOOK * VL) > 1e-10 else LEWD)
            VL = VL + (GAMMAL * VL) / denom if abs(denom) > 1e-10 else VL
            s["Q7"]    = ((1 - VL / FACT1) ** 2 - ALP) / FACT2 if abs(FACT2) > 1e-10 else s["Q7"]
            s["GAMMAL"] = 0.0
            s["VL"]    = VL

        # ---- RANGE_PREDICTION block ----
        VL      = s["VL"]
        V1      = s["V1"]
        RDOT    = s["RDOT"]
        V       = s["V"]
        A0      = s["A0"]
        GAMMAL1 = s["GAMMAL1"]

        VBARS = VL ** 2 / VSAT ** 2
        COSG  = 1 - s["GAMMAL"] ** 2 / 2
        E_val = np.sqrt(abs(1 + (VBARS - 2) * COSG ** 2 * VBARS))

        ASKEP  = 2 * ATK * np.arcsin(np.clip(VBARS * COSG * s["GAMMAL"] / E_val, -1, 1))
        ASP1   = Q2 + Q3 * VL
        ASPUP  = ((ATK / (RE * 0.3048)) * (HS * 0.3048 / GAMMAL1)
                  * np.log(A0 * VL ** 2 / (s["Q7"] * V1 ** 2))
                  if (s["Q7"] * V1 ** 2) > 0 else 0)
        ASP3   = Q5 * (Q6 - s["GAMMAL"])
        ASPDWN = -RDOT * V * ATK / (A0 * LAD * (RE * 0.3048)) if A0 > 0 else 0
        ASP    = ASKEP + ASP1 + ASPUP + ASP3 + ASPDWN

        s.update({"VBARS": VBARS, "COSG": COSG, "ASKEP": ASKEP,
                  "ASP1": ASP1, "ASPUP": ASPUP, "ASP3": ASP3,
                  "ASPDWN": ASPDWN, "ASP": ASP})

        DIFF      = s["THETNM"] - ASP
        s["DIFF"] = DIFF

        # Converged — hand off to UPCONTROL
        if abs(DIFF) - 25 < 0:
            s["SELECTOR"] = 4
            _upcontrol(s)
            return

        # First pass with HIND==0: apply initial correction or seed CONSTD
        if s["HIND"] == 0:
            if DIFF < 0:
                s["DIFFOLD"] = DIFF
                s["V1OLD"]   = V1
                _constd(s)
                return
            else:
                s["VCORR"] = V1 - s["V1OLD"]

        # Compute V1 correction and loop back into HUNTEST1
        denom_diff = s["DIFFOLD"] - DIFF
        VCORR = (s["VCORR"] * DIFF) / denom_diff if abs(denom_diff) > 1e-10 else 0.0
        VCORR = min(VCORR, VCORLIM)
        if VSAT - VL - VCORR < 0:
            VCORR = VCORR / 2

        s["V1"]      = V1 + VCORR
        s["HIND"]    = 1
        s["DIFFOLD"] = DIFF
        # loop continues → back to HUNTEST1 block at top

    # Safety fallback: if loop exhausted without converging, go to CONSTD
    _constd(s)


# Keep the old names as thin wrappers so call-sites (_huntest, _upcontrol, etc.) don't change
def _range_prediction(s):
    _huntest1_and_range_prediction(s)

def _huntest1(s):
    _huntest1_and_range_prediction(s)


def _huntest(s):
    V    = s["V"]
    RDOT = s["RDOT"]
    D    = s["D"]

    if RDOT < 0:
        V1 = V + RDOT / LAD
        A0 = (V1 / V) ** 2 * (D + RDOT ** 2 / (2 * HS * LAD)) if V > 0 else 0
    else:
        V1 = V + RDOT / LEWD
        A0 = (V1 / V) ** 2 * (D + RDOT ** 2 / (2 * HS * LEWD)) if V > 0 else 0

    s["V1"] = V1
    s["A0"] = A0

    if s["HUNTIND"] == 0:
        s["HUNTIND"] = 1
        s["DIFFOLD"] = 0.0
        s["V1OLD"]   = V1 + C18
        s["Q7"]      = Q7F
        if A0 - C19 < 0:
            s["D0"] = C19
        elif A0 - D0MAX > 0:
            s["D0"] = D0MAX
        else:
            s["D0"] = A0
    _huntest1(s)


def _predict3(s):
    V     = s["V"]
    RDOT  = s["RDOT"]
    D     = s["D"]
    URT_  = s["URT_"]
    rSC   = s["rSC"]
    UNI_  = s["UNI_"]

    if V - VQUIT > 0:
        if s["GONEPAST"] == 0:
            temp  = np.cross(URT_, rSC)
            temp2 = np.dot(temp, UNI_)
            if temp2 > 0:
                F1, F2, F3, RTOGO = _final_phase_lookup(V)
                PREDANGL = RTOGO + F2 * (RDOT - s.get("RDOTREF", 0)) + F1 * (D - s.get("DREFR", 0))
                L_D = LOD + 4 * (s["THETNM"] - PREDANGL) / F3 if F3 != 0 else LOD
                s.update({"F1": F1, "F2": F2, "F3": F3, "RTOGO": RTOGO,
                           "PREDANGL": PREDANGL, "L_D": L_D})
                _glimiter(s)
            else:
                s["GONEPAST"] = 1
                s["L_D"]      = -LAD
                _glimiter(s)
        else:
            s["L_D"] = -LAD
            _glimiter(s)
    else:
        _three_eighty(s, DT_SIM)


def _upcontrol(s):
    V    = s["V"]
    V1   = s["V1"]
    RDOT = s["RDOT"]
    D    = s["D"]
    A0   = s["A0"]

    if V - V1 > 0:
        RDTR = LAD * (V1 - V)
        DR   = (V / V1) ** 2 * A0 - RDTR ** 2 / (2 * HS * LAD) if V1 > 0 else 0
        s["L_D"]   = LAD + C16 * (D - DR) - C17 * (RDOT - RDTR)
        s["RDTR"]  = RDTR
        s["DR"]    = DR
        _three_ten(s)
    else:
        if D - s["Q7"] > 0:
            if RDOT < 0 and V - s["VL"] - C18 < 0:
                s["SELECTOR"] = 3
                s["EGSW"]     = 1
                _predict3(s)
            else:
                if A0 - D < 0:
                    s["L_D"] = LAD
                    _three_ten(s)
                else:
                    VREF = s["FACT1"] * (1 - np.sqrt(max(s["FACT2"] * D + s["ALP"], 0)))
                    if VREF - VSAT > 0:
                        RDOTREF = LEWD * (V1 - VREF)
                    else:
                        RDOTREF = (LEWD * (V1 - VREF)
                                   - CH1 * GS * (s["VS1"] - VREF) ** 2
                                   * (1 + s["AHOOK"] * (s["VS1"] - VREF))
                                   / (s["DHOOK"] * VREF)
                                   if (abs(s["DHOOK"]) > 1e-10 and abs(VREF) > 1e-10) else LEWD * (V1 - VREF))
                    FACTOR = (D - s["Q7"]) / (A0 - s["Q7"]) if abs(A0 - s["Q7"]) > 1e-10 else 0
                    s["L_D"] = (LEWD
                                - KB2 * FACTOR * (KB1 * FACTOR * (RDOT - RDOTREF) + V - VREF))
                    s["VREF"]    = VREF
                    s["RDOTREF"] = RDOTREF
                    s["FACTOR"]  = FACTOR
                    _negtest(s)
        else:
            s["SELECTOR"] = 1
            _kep2(s)


def _kep2(s):
    D  = s["D"]
    Q7 = s["Q7"]
    if D - (Q7 + KDMIN) > 0:
        s["EGSW"]     = 1
        s["SELECTOR"] = 3
        _predict3(s)


def _init_roll(s):
    D    = s["D"]
    V    = s["V"]
    RDOT = s["RDOT"]

    if s["INRLSW"] == 0:
        if D - 0.5 * GS > 0:
            s["INRLSW"] = 1
        else:
            _three_ten(s)
            return

        if V - VFINAL < 0:
            s["SELECTOR"] = 1
            s["L_D"]      = LAD
            _three_ten(s)
        else:
            s["L_D"] = -LAD if (V - VFINAL + K44 * (RDOT / V) ** 3 > 0) else LAD
            _three_ten(s)
    else:
        if D - KA > 0:
            s["L_D"] = LAD
        if RDOT + VRCONTRL < 0:
            _three_ten(s)
        else:
            s["SELECTOR"] = 2
            _huntest(s)


# ============================================================
# LSTM CONTROLLER
# ============================================================

def load_lstm():
    """Loads the LSTM model and scalers. Returns (model, in_scaler, out_scaler)."""
    if not TF_AVAILABLE or not JOBLIB_AVAILABLE:
        print("LSTM dependencies missing — controller will return bank=0.")
        return None, None, None

    model       = None
    in_scaler   = None
    out_scaler  = None

    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"  Model loaded  : {MODEL_PATH}")
        except Exception as e:
            print(f"  WARNING: Could not load model ({e}). bank_angle=0 fallback active.")
    else:
        print(f"  WARNING: Model file not found: {MODEL_PATH}. bank_angle=0 fallback active.")

    if os.path.exists(IN_SCALER_PATH):
        try:
            in_scaler = joblib.load(IN_SCALER_PATH)
            print(f"  Input scaler  : {IN_SCALER_PATH}")
        except Exception as e:
            print(f"  WARNING: Could not load input scaler ({e}).")

    if os.path.exists(OUT_SCALER_PATH):
        try:
            out_scaler = joblib.load(OUT_SCALER_PATH)
            print(f"  Output scaler : {OUT_SCALER_PATH}")
        except Exception as e:
            print(f"  WARNING: Could not load output scaler ({e}).")

    return model, in_scaler, out_scaler


def predict_bank_angle(model, in_scaler, out_scaler, history_buffer):
    """
    Query the LSTM with the current sliding-window history.

    Parameters
    ----------
    history_buffer : np.ndarray, shape (<=WINDOW_SIZE, 3)  — position error history

    Returns
    -------
    bank_rad : float [rad]
    """
    if model is None or in_scaler is None or out_scaler is None:
        return 0.0

    try:
        hist_len = history_buffer.shape[0]
        padded   = np.zeros((WINDOW_SIZE, 3))
        padded[-hist_len:, :] = history_buffer

        normed      = in_scaler.transform(padded)
        model_input = normed[np.newaxis, :, :]   # (1, WINDOW_SIZE, 3)

        prediction  = model.predict(model_input, verbose=0)
        bank_norm   = prediction[0, -1, 0]

        bank_rad = float(out_scaler.inverse_transform(
            np.array([[bank_norm]])
        )[0, 0])

        bank_rad = np.clip(bank_rad, -MAX_BANK_RAD, MAX_BANK_RAD)

        if not np.isfinite(bank_rad):
            return 0.0

        return bank_rad

    except Exception as e:
        print(f"  LSTM predict error: {e}  → bank=0")
        return 0.0


# ============================================================
# INITIAL CONDITIONS  (Apollo 10 / Trans_MainJones values)
# ============================================================

def generate_initial_condition():
    """
    Single deterministic initial condition taken directly from Trans_MainJones.
    No Monte Carlo offsets applied.
    """
    r_planet_m = RE * 0.3048            # Earth radius in metres

    h0    = 420000 * 0.3048 + r_planet_m   # ~128 km  (120000 ft AGL)
    lat0  = np.deg2rad(-12.7)
    lon0  = np.deg2rad(122.9)

    r_SC = np.array([
        h0 * np.cos(lat0) * np.cos(lon0),
        h0 * np.cos(lat0) * np.sin(lon0),
        h0 * np.sin(lat0),
    ])

    v_SC = np.array([-5001.6939822274662,
                     -2062.7765069170787,
                      9616.2295172411814])

    target = np.array([-4752753.07, 3769998.46, 1971101.98])

    return r_SC, v_SC, target


# ============================================================
# LIVE PLOT INITIALISATION
# ============================================================

def init_live_plots():
    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("LSTM Entry Guidance — Live Simulation", fontsize=13)

    ax_bank = axes[0, 0]
    ax_err  = axes[0, 1]
    ax_alt  = axes[1, 0]
    ax_vel  = axes[1, 1]

    for ax, title, ylabel in [
        (ax_bank, "Bank Angle Command",      "Bank [deg]"),
        (ax_err,  "Position Error Magnitude", "Error [m]"),
        (ax_alt,  "Altitude",                 "Altitude [km]"),
        (ax_vel,  "Speed",                    "Speed [m/s]"),
    ]:
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.4)

    line_lstm,   = ax_bank.plot([], [], label="LSTM",   color="tab:blue")
    line_apollo, = ax_bank.plot([], [], label="Apollo", color="tab:orange", linestyle="--")
    ax_bank.legend(fontsize=8)

    line_err,    = ax_err.plot([], [], color="tab:red")
    line_alt,    = ax_alt.plot([], [], color="tab:green")
    line_vel,    = ax_vel.plot([], [], color="tab:purple")

    plt.tight_layout()
    return fig, ax_bank, ax_err, ax_alt, ax_vel, line_lstm, line_apollo, line_err, line_alt, line_vel


def update_live_plots(fig, axes_lines, step,
                      bank_lstm_hist, bank_apollo_hist,
                      error_hist, alt_hist, vel_hist):
    (ax_bank, ax_err, ax_alt, ax_vel,
     line_lstm, line_apollo, line_err, line_alt, line_vel) = axes_lines

    t = np.arange(step)
    line_lstm.set_data(t, np.degrees(bank_lstm_hist[:step]))
    line_apollo.set_data(t, np.degrees(bank_apollo_hist[:step]))
    line_err.set_data(t, error_hist[:step])
    line_alt.set_data(t, alt_hist[:step] * 1e-3)
    line_vel.set_data(t, vel_hist[:step])

    for ax in [ax_bank, ax_err, ax_alt, ax_vel]:
        ax.relim()
        ax.autoscale_view()

    fig.canvas.draw()
    fig.canvas.flush_events()


# ============================================================
# MAIN SIMULATION LOOP
# ============================================================

def run_simulation():
    print("\n" + "=" * 60)
    print("  LSTM Entry Guidance Simulation")
    print(f"  Mode       : {CONTROLLER_MODE}")
    print(f"  Max steps  : {SIM_STEPS}   dt={DT_SIM}s   window={WINDOW_SIZE}")
    print("=" * 60)

    # --- Load LSTM ---
    print("\nLoading LSTM model...")
    model, in_scaler, out_scaler = load_lstm()

    # --- Initial conditions ---
    r_SC, v_SC, target = generate_initial_condition()
    alt0 = np.linalg.norm(r_SC) - R_PLANET
    print(f"\nInitial altitude : {alt0/1000:.2f} km")
    print(f"Initial speed    : {np.linalg.norm(v_SC):.2f} m/s")
    print(f"Target (ECEF)    : {target}")

    # --- Guidance state ---
    apollo_state = make_apollo_state()
    apollo_state["rotSC"] = 0.0     # initial bank angle

    # --- Sliding window buffer for LSTM ---
    history_buffer = np.zeros((0, 3))

    # --- History arrays ---
    bank_lstm_hist  = np.zeros(SIM_STEPS)
    bank_apollo_hist = np.zeros(SIM_STEPS)
    error_hist      = np.zeros(SIM_STEPS)
    alt_hist        = np.zeros(SIM_STEPS)
    vel_hist        = np.zeros(SIM_STEPS)
    r_hist          = np.zeros((SIM_STEPS, 3))
    v_hist          = np.zeros((SIM_STEPS, 3))

    # --- Live plot setup ---
    (fig, ax_bank, ax_err, ax_alt, ax_vel,
     line_lstm, line_apollo, line_err, line_alt, line_vel) = init_live_plots()
    axes_lines = (ax_bank, ax_err, ax_alt, ax_vel,
                  line_lstm, line_apollo, line_err, line_alt, line_vel)

    # Dummy acc_no_grav for first step
    acc_no_grav = np.zeros(3)

    print("\nSimulation starting...\n")

    final_step = SIM_STEPS
    for step in range(SIM_STEPS):

        # ---- State quantities ----
        r_norm    = np.linalg.norm(r_SC)
        v_norm    = np.linalg.norm(v_SC)
        altitude  = r_norm - R_PLANET
        error_vec = r_SC - target
        error_mag = np.linalg.norm(error_vec)

        # ---- Termination check ----
        if altitude < TERM_ALTITUDE_M:
            print(f"\nTermination altitude reached at step {step} ({altitude:.1f} m)")
            final_step = step
            break

        # ---- Update LSTM sliding window (position error) ----
        history_buffer = np.vstack([history_buffer, error_vec])
        if history_buffer.shape[0] > WINDOW_SIZE:
            history_buffer = history_buffer[-WINDOW_SIZE:]

        # ---- LSTM bank angle ----
        if step >= WINDOW_SIZE:
            bank_lstm = predict_bank_angle(model, in_scaler, out_scaler, history_buffer)
        else:
            bank_lstm = 0.0

        # ---- Apollo bank angle ----
        bank_apollo, apollo_state = apollo_guidance(
            r_SC, v_SC, acc_no_grav, target, apollo_state, dt=DT_SIM
        )

        # ---- Select active bank command ----
        if CONTROLLER_MODE == "LSTM":
            active_bank = bank_lstm
        elif CONTROLLER_MODE == "APOLLO":
            active_bank = bank_apollo
        else:  # COMPARE — LSTM drives dynamics, Apollo logged separately
            active_bank = bank_lstm

        # ---- Propagate dynamics ----
        r_SC, v_SC, acc_no_grav = propagate_dynamics(r_SC, v_SC, active_bank)

        # ---- Log ----
        bank_lstm_hist[step]   = bank_lstm
        bank_apollo_hist[step] = bank_apollo
        error_hist[step]       = error_mag
        alt_hist[step]         = altitude
        vel_hist[step]         = v_norm
        r_hist[step]           = r_SC
        v_hist[step]           = v_SC

        # ---- Console output ----
        if step % PRINT_EVERY == 0:
            print(
                f"  Step {step:6d} | "
                f"t={step*DT_SIM:8.1f}s | "
                f"Alt={altitude/1000:8.3f} km | "
                f"V={v_norm:9.2f} m/s | "
                f"Bank(LSTM)={np.degrees(bank_lstm):7.2f}° | "
                f"Bank(Apollo)={np.degrees(bank_apollo):7.2f}°"
            )

        # ---- Live plot ----
        if step % LIVE_PLOT_EVERY == 0 and step > 0:
            update_live_plots(fig, axes_lines, step,
                              bank_lstm_hist, bank_apollo_hist,
                              error_hist, alt_hist, vel_hist)

    # ============================================================
    # POST-SIMULATION PLOTS
    # ============================================================
    plt.ioff()

    n = final_step
    t_arr = np.arange(n) * DT_SIM

    miss  = np.linalg.norm(r_SC - target)
    print(f"\n{'='*60}")
    print(f"  Simulation complete.")
    print(f"  Steps run   : {n}")
    print(f"  Final alt   : {alt_hist[max(n-1,0)]/1000:.3f} km")
    print(f"  Final speed : {vel_hist[max(n-1,0)]:.2f} m/s")
    print(f"  Miss dist   : {miss/1000:.3f} km")
    print(f"{'='*60}\n")

    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 9))
    fig2.suptitle("LSTM Entry Guidance — Final Results", fontsize=13)

    # 3-D trajectory
    ax3d = fig2.add_subplot(2, 2, 1, projection='3d')
    ax3d.plot(r_hist[:n, 0]*1e-6, r_hist[:n, 1]*1e-6, r_hist[:n, 2]*1e-6,
              color="tab:blue", linewidth=1.2, label="Trajectory")
    ax3d.scatter(*target*1e-6, color="red", s=60, zorder=5, label="Target")
    ax3d.set_title("3-D Trajectory [×10⁶ m]", fontsize=10)
    ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
    ax3d.legend(fontsize=8)

    ax_a = fig2.add_subplot(2, 2, 2)
    ax_a.plot(t_arr, alt_hist[:n]*1e-3, color="tab:green")
    ax_a.set_title("Altitude vs Time"); ax_a.set_xlabel("Time [s]"); ax_a.set_ylabel("Alt [km]"); ax_a.grid(True, alpha=0.4)

    ax_v2 = fig2.add_subplot(2, 2, 3)
    ax_v2.plot(t_arr, vel_hist[:n], color="tab:purple")
    ax_v2.set_title("Speed vs Time"); ax_v2.set_xlabel("Time [s]"); ax_v2.set_ylabel("Speed [m/s]"); ax_v2.grid(True, alpha=0.4)

    ax_b2 = fig2.add_subplot(2, 2, 4)
    ax_b2.plot(t_arr, np.degrees(bank_lstm_hist[:n]),   label="LSTM",   color="tab:blue")
    ax_b2.plot(t_arr, np.degrees(bank_apollo_hist[:n]), label="Apollo", color="tab:orange", linestyle="--")
    ax_b2.set_title("Bank Angle vs Time"); ax_b2.set_xlabel("Time [s]"); ax_b2.set_ylabel("Bank [deg]")
    ax_b2.legend(fontsize=8); ax_b2.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig("simulation_results.png", dpi=150)
    print("  Final plots saved → simulation_results.png")
    plt.show()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_simulation()