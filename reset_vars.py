"""
reset_vars.py
=============
Direct Python conversion of reset_vars.m

All units preserved from the original MATLAB file.
Apollo GNC guidance constants — do not modify unless
you know what you're changing and why.
"""

import numpy as np

# ============================================================
# GUIDANCE STATE FLAGS
# ============================================================
intitK2    = 0
SELECTOR   = 0
GONEPAST   = 0
RELVELSW   = 0
EGSW       = 0
HUNTIND    = 0
HIND       = 0
INRLSW     = 0

# ============================================================
# GUIDANCE GAINS AND LIMITS
# ============================================================
C1       = 1.25
C16      = 0.01
C17      = 0.001
C18      = 500.0          # FPS
C19      = 130.0          # FPSS
C20      = 175.0          # FPSS
CHOOK    = 0.25
CH1      = 0.75
D0MAX    = 175.0          # FPSS
DT       = 2.0            # SEC
GMAX     = 322.0          # FPSS
KA       = 64.4
KB1      = 3.4
KB2      = 0.0034
KDMIN    = 0.5            # FPSS
KLAT     = 0.0125
LATBIAS  = 0.4            # NM
KTETA    = 1000.0
K44      = 44389312.0     # FPS
LAD      = 0.3
L_DCMINR = 0.2895
LEWD     = 0.1
LOD      = 0.18
Q2       = -1002.0        # NM
Q3       = 0.07           # NM/FPS
Q5       = 7050.0         # NM/RAD
Q6       = 0.0349         # RAD
Q7F      = 6.0            # FPSS
VFINAL   = 25000.0        # FPS
VLMIN    = 18000.0        # FPS
VRCONTRL = 700.0          # FPS
VCORLIM  = 1000.0         # FPS
NM_25    = 25.0           # NM
VQUIT    = 1000.0         # FPS

# ============================================================
# PLANETARY / PHYSICAL CONSTANTS
# ============================================================
ATK  = 3437.7468          # NM/RAD
GS   = 32.2               # FPSS  (ft/s^2)
HS   = 28500.0            # FT    (atmospheric scale height)
J    = 0.00162345
KWE  = 1546.70168         # FPS
MUE  = 3.986032233        # E14 m^3/s^2  (Earth grav. param.)
RE   = 21202900.0         # FT   (Earth radius)
VSAT = 25766.1973         # FPS  (circular orbital speed)
WIE  = 0.0000729211505    # RAD/S (Earth rotation rate)

# ============================================================
# VECTOR STATE VARIABLES  (initialised to zero)
# ============================================================
URT0_ = np.zeros(3)
UZ_   = np.zeros(3)
R_    = np.zeros(3)
VI_   = np.zeros(3)
RTE_  = np.zeros(3)
UTR_  = np.zeros(3)
URT_  = np.zeros(3)
UNI_  = np.zeros(3)
DELV_ = np.zeros(3)

# ============================================================
# SCALAR GUIDANCE STATE VARIABLES  (initialised to zero)
# ============================================================
A0       = 0.0
AHOOK    = 0.0
AHOOKDV  = 0.0
ALP      = 0.0
ASKEP    = 0.0
ASP1     = 0.0
ASPUP    = 0.0
ASP3     = 0.0
ASPDWN   = 0.0
ASP      = 0.0
COSG     = 0.0
D        = 0.0
D0       = 0.0
DHOOK    = 0.0
DIFF     = 0.0
DIFFOLD  = 0.0
DR       = 0.0
DREFR    = 0.0
DVL      = 0.0
E        = 0.0
F1       = 0.0
F2       = 0.0
F3       = 0.0
FACT1    = 0.0
FACT2    = 0.0
FACTOR   = 0.0
GAMMAL   = 0.0
GAMMAL1  = 0.0
K1ROLL   = 0.0
K2ROLL   = 0.0
LATANG   = 0.0
LEQ      = 0.0
L_D      = 0.0
PREDANGL = 0.0
Q7       = 0.0
RDOT     = 0.0
RDOTREF  = 0.0
RDTR     = 0.0
ROLLC    = 0.0
RTOGO    = 0.0
SL       = 0.0
T        = 0.0
THETA    = 0.0
THETNM   = 2000.0         # NOTE: non-zero initial value from original MATLAB
V        = 0.0
V1       = 0.0
V1OLD    = 0.0
VCORR    = 0.0
VL       = 0.0
VREF     = 0.0
VS1      = 0.0
VBARS    = 0.0
VSQ      = 0.0
WT       = 0.0
X        = 0.0
Y        = 0.0

def get_guidance_state():
    """
    Returns a dict of all mutable guidance state variables.
    Useful for initialising / resetting state at the start of a run.
    """
    return dict(
        intitK2=intitK2, SELECTOR=SELECTOR, GONEPAST=GONEPAST,
        RELVELSW=RELVELSW, EGSW=EGSW, HUNTIND=HUNTIND,
        HIND=HIND, INRLSW=INRLSW,
        URT0_=URT0_.copy(), UZ_=UZ_.copy(), R_=R_.copy(),
        VI_=VI_.copy(), RTE_=RTE_.copy(), UTR_=UTR_.copy(),
        URT_=URT_.copy(), UNI_=UNI_.copy(), DELV_=DELV_.copy(),
        A0=A0, AHOOK=AHOOK, AHOOKDV=AHOOKDV, ALP=ALP,
        ASKEP=ASKEP, ASP1=ASP1, ASPUP=ASPUP, ASP3=ASP3,
        ASPDWN=ASPDWN, ASP=ASP, COSG=COSG,
        D=D, D0=D0, DHOOK=DHOOK, DIFF=DIFF, DIFFOLD=DIFFOLD,
        DR=DR, DREFR=DREFR, DVL=DVL, E=E,
        F1=F1, F2=F2, F3=F3, FACT1=FACT1, FACT2=FACT2, FACTOR=FACTOR,
        GAMMAL=GAMMAL, GAMMAL1=GAMMAL1,
        K1ROLL=K1ROLL, K2ROLL=K2ROLL,
        LATANG=LATANG, LEQ=LEQ, L_D=L_D, PREDANGL=PREDANGL,
        Q7=Q7, RDOT=RDOT, RDOTREF=RDOTREF, RDTR=RDTR,
        ROLLC=ROLLC, RTOGO=RTOGO, SL=SL, T=T, THETA=THETA,
        THETNM=THETNM,
        V=V, V1=V1, V1OLD=V1OLD, VCORR=VCORR, VL=VL, VREF=VREF,
        VS1=VS1, VBARS=VBARS, VSQ=VSQ, WT=WT, X=X, Y=Y,
    )
