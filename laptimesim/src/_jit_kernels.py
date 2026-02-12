"""
Numba JIT-compiled kernels for the lap time simulation hot path.

All functions are standalone @njit(cache=True) functions operating on packed
parameter arrays to avoid Python dict lookups and object overhead inside tight loops.

Falls back to plain Python if numba is not installed.
"""

import math

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

# ======================================================================================
# Parameter array index constants
# ======================================================================================

# params indices (1D float64 array)
_M = 0           # mass [kg]
_RHO = 1         # air density [kg/m^3]
_CWA = 2         # drag coeff * area [m^2]
_FROLL = 3       # rolling resistance coeff [-]
_DRS = 4         # DRS drag reduction factor [-]
_LF = 5          # front axle distance [m]
_LR = 6          # rear axle distance [m]
_TOPO = 7        # topology: 0=AWD, 1=FWD, 2=RWD
_EXP = 8         # tire model exponent [-]
_MUX_F = 9       # front tire mu_x [-]
_MUY_F = 10      # front tire mu_y [-]
_DMUX_F = 11     # front dmu_x/dFz [1/N]
_DMUY_F = 12     # front dmu_y/dFz [1/N]
_FZ0_F = 13      # front reference load [N]
_MUX_R = 14      # rear tire mu_x [-]
_MUY_R = 15      # rear tire mu_y [-]
_DMUX_R = 16     # rear dmu_x/dFz [1/N]
_DMUY_R = 17     # rear dmu_y/dFz [1/N]
_FZ0_R = 18      # rear reference load [N]
PARAMS_SIZE = 19

# fz_data row indices (2D float64 array, shape (6, 4))
_STAT = 0        # static load
_TLONG = 1       # longitudinal transfer magnitude
_TLONG_S = 2     # longitudinal transfer sign
_TLAT = 3        # lateral transfer magnitude
_TLAT_S = 4      # lateral transfer sign
_AERO = 5        # aero downforce


# ======================================================================================
# JIT-compiled kernel functions
# ======================================================================================

@njit(cache=True)
def _tire_force_pots(vel, a_x, a_y, mu, params, fz_data):
    """
    Calculate tire force potentials for all 4 wheels.
    Returns tuple of 12 floats:
        (f_x_pot_fl, f_y_pot_fl, f_z_fl,
         f_x_pot_fr, f_y_pot_fr, f_z_fr,
         f_x_pot_rl, f_y_pot_rl, f_z_rl,
         f_x_pot_rr, f_y_pot_rr, f_z_rr)
    """
    vel_sq = vel * vel

    # tire parameters
    mux_f = params[_MUX_F]
    muy_f = params[_MUY_F]
    dmux_f = params[_DMUX_F]
    dmuy_f = params[_DMUY_F]
    fz0_f = params[_FZ0_F]
    mux_r = params[_MUX_R]
    muy_r = params[_MUY_R]
    dmux_r = params[_DMUX_R]
    dmuy_r = params[_DMUY_R]
    fz0_r = params[_FZ0_R]

    # compute tire loads for each wheel (FL=0, FR=1, RL=2, RR=3)
    f_z_fl = (fz_data[_STAT, 0]
              + a_x * fz_data[_TLONG_S, 0] * fz_data[_TLONG, 0]
              + a_y * fz_data[_TLAT_S, 0] * fz_data[_TLAT, 0]
              + vel_sq * fz_data[_AERO, 0])
    if f_z_fl < 30.0:
        f_z_fl = 30.0

    f_z_fr = (fz_data[_STAT, 1]
              + a_x * fz_data[_TLONG_S, 1] * fz_data[_TLONG, 1]
              + a_y * fz_data[_TLAT_S, 1] * fz_data[_TLAT, 1]
              + vel_sq * fz_data[_AERO, 1])
    if f_z_fr < 30.0:
        f_z_fr = 30.0

    f_z_rl = (fz_data[_STAT, 2]
              + a_x * fz_data[_TLONG_S, 2] * fz_data[_TLONG, 2]
              + a_y * fz_data[_TLAT_S, 2] * fz_data[_TLAT, 2]
              + vel_sq * fz_data[_AERO, 2])
    if f_z_rl < 30.0:
        f_z_rl = 30.0

    f_z_rr = (fz_data[_STAT, 3]
              + a_x * fz_data[_TLONG_S, 3] * fz_data[_TLONG, 3]
              + a_y * fz_data[_TLAT_S, 3] * fz_data[_TLAT, 3]
              + vel_sq * fz_data[_AERO, 3])
    if f_z_rr < 30.0:
        f_z_rr = 30.0

    # force potentials - front wheels
    f_x_pot_fl = mu * (mux_f + dmux_f * (f_z_fl - fz0_f)) * f_z_fl
    f_y_pot_fl = mu * (muy_f + dmuy_f * (f_z_fl - fz0_f)) * f_z_fl

    f_x_pot_fr = mu * (mux_f + dmux_f * (f_z_fr - fz0_f)) * f_z_fr
    f_y_pot_fr = mu * (muy_f + dmuy_f * (f_z_fr - fz0_f)) * f_z_fr

    # force potentials - rear wheels
    f_x_pot_rl = mu * (mux_r + dmux_r * (f_z_rl - fz0_r)) * f_z_rl
    f_y_pot_rl = mu * (muy_r + dmuy_r * (f_z_rl - fz0_r)) * f_z_rl

    f_x_pot_rr = mu * (mux_r + dmux_r * (f_z_rr - fz0_r)) * f_z_rr
    f_y_pot_rr = mu * (muy_r + dmuy_r * (f_z_rr - fz0_r)) * f_z_rr

    return (f_x_pot_fl, f_y_pot_fl, f_z_fl,
            f_x_pot_fr, f_y_pot_fr, f_z_fr,
            f_x_pot_rl, f_y_pot_rl, f_z_rl,
            f_x_pot_rr, f_y_pot_rr, f_z_rr)


@njit(cache=True)
def _calc_lat_forces(a_y, m, lf, lr):
    """Calculate lateral forces on front and rear axle. Returns (f_y_f, f_y_r)."""
    f_y = m * a_y
    l_tot = lf + lr
    f_y_f = f_y * lr / l_tot
    f_y_r = f_y * lf / l_tot
    return f_y_f, f_y_r


@njit(cache=True)
def _calc_f_x_pot(f_x_pot_fl, f_x_pot_fr, f_x_pot_rl, f_x_pot_rr,
                  f_y_pot_f, f_y_pot_r, f_y_f, f_y_r,
                  topology, exp, force_all_wheels, lbs_flag):
    """
    Calculate remaining tire potential for longitudinal force.
    topology: 0=AWD, 1=FWD, 2=RWD
    lbs_flag: 0=None, 1=FA, 2=RA, 3=all (limit_braking_weak_side)
    """
    inv_exp = 1.0 / exp

    # determine axle potentials based on weak side limiting
    if lbs_flag == 1:  # FA
        f_x_pot_f = 2.0 * min(f_x_pot_fl, f_x_pot_fr)
        f_x_pot_r = f_x_pot_rl + f_x_pot_rr
    elif lbs_flag == 2:  # RA
        f_x_pot_f = f_x_pot_fl + f_x_pot_fr
        f_x_pot_r = 2.0 * min(f_x_pot_rl, f_x_pot_rr)
    elif lbs_flag == 3:  # all
        f_x_pot_f = 2.0 * min(f_x_pot_fl, f_x_pot_fr)
        f_x_pot_r = 2.0 * min(f_x_pot_rl, f_x_pot_rr)
    else:  # None
        f_x_pot_f = f_x_pot_fl + f_x_pot_fr
        f_x_pot_r = f_x_pot_rl + f_x_pot_rr

    # calculate radicands
    radicand_f = 1.0 - (abs(f_y_f) / f_y_pot_f) ** exp
    radicand_r = 1.0 - (abs(f_y_r) / f_y_pot_r) ** exp

    if radicand_f < 0.0:
        radicand_f = 0.0
    if radicand_r < 0.0:
        radicand_r = 0.0

    # calculate remaining force potential based on topology
    if topology == 0 or force_all_wheels:  # AWD or braking with all wheels
        f_x_poss_f = f_x_pot_f * radicand_f ** inv_exp
        f_x_poss_r = f_x_pot_r * radicand_r ** inv_exp
    elif topology == 1:  # FWD
        f_x_poss_f = f_x_pot_f * radicand_f ** inv_exp
        f_x_poss_r = 0.0
    else:  # RWD (topology == 2)
        f_x_poss_f = 0.0
        f_x_poss_r = f_x_pot_r * radicand_r ** inv_exp

    return f_x_poss_f + f_x_poss_r


@njit(cache=True)
def _air_res(vel, drs, rho_air, c_w_a, drs_factor):
    """Calculate air resistance force in N."""
    vel_sq = vel * vel
    if drs:
        return 0.5 * (1.0 - drs_factor) * c_w_a * rho_air * vel_sq
    else:
        return 0.5 * c_w_a * rho_air * vel_sq


@njit(cache=True)
def _roll_res(f_z_tot, f_roll):
    """Calculate rolling resistance force in N."""
    return f_z_tot * f_roll


@njit(cache=True)
def _v_max_cornering(kappa, mu, vel_subtr_corner, params, fz_data):
    """
    Find maximum cornering velocity using binary search.
    Entire binary search runs in compiled code.
    """
    no_steps = 546
    vel_max = 110.0

    # build velocity range inline (avoid np.linspace allocation)
    vel_step = (vel_max - 1.0) / (no_steps - 1)

    m = params[_M]
    lf = params[_LF]
    lr = params[_LR]
    topology = int(params[_TOPO])
    exp = params[_EXP]
    rho_air = params[_RHO]
    c_w_a = params[_CWA]
    f_roll = params[_FROLL]

    ind_first = 0
    ind_last = no_steps - 1
    ind_mid = (ind_first + ind_last + 1) // 2

    while ind_first != ind_last:
        vel_mid = 1.0 + ind_mid * vel_step

        # lateral acceleration and forces
        a_y = vel_mid * vel_mid * kappa
        f_y_f, f_y_r = _calc_lat_forces(a_y, m, lf, lr)

        # tire force potentials (a_x = 0.0 at maximum cornering)
        (f_x_pot_fl, f_y_pot_fl, f_z_fl,
         f_x_pot_fr, f_y_pot_fr, f_z_fr,
         f_x_pot_rl, f_y_pot_rl, f_z_rl,
         f_x_pot_rr, f_y_pot_rr, f_z_rr) = _tire_force_pots(vel_mid, 0.0, a_y, mu, params, fz_data)

        # check if potential is left
        if (abs(f_y_f) < f_y_pot_fl + f_y_pot_fr
                and abs(f_y_r) < f_y_pot_rl + f_y_pot_rr):

            f_x_poss = _calc_f_x_pot(
                f_x_pot_fl, f_x_pot_fr, f_x_pot_rl, f_x_pot_rr,
                f_y_pot_fl + f_y_pot_fr, f_y_pot_rl + f_y_pot_rr,
                f_y_f, f_y_r,
                topology, exp, False, 0)

            f_z_tot = f_z_fl + f_z_fr + f_z_rl + f_z_rr
            f_x_drag = _air_res(vel_mid, False, rho_air, c_w_a, 0.0) + _roll_res(f_z_tot, f_roll)

            if f_x_poss < f_x_drag:
                potential_exceeded = True
            else:
                potential_exceeded = False
        else:
            potential_exceeded = True

        if not potential_exceeded:
            ind_first = ind_mid
        else:
            ind_last = ind_mid - 1

        ind_mid = (ind_first + ind_last + 1) // 2

    vel_result = 1.0 + ind_mid * vel_step
    return vel_result - vel_subtr_corner


@njit(cache=True)
def _calc_max_ax(vel, a_y, mu, f_y_f, f_y_r, params, fz_data):
    """
    Calculate maximum longitudinal acceleration using binary search.
    Entire binary search runs in compiled code.
    """
    no_steps = 101
    a_x_max = 25.0

    a_x_step = a_x_max / (no_steps - 1)

    ind_first = 0
    ind_last = no_steps - 1
    ind_mid = (ind_first + ind_last + 1) // 2

    abs_f_y_f = abs(f_y_f)
    abs_f_y_r = abs(f_y_r)

    while ind_first != ind_last:
        a_x_mid = ind_mid * a_x_step

        _, f_y_pot_fl, _, _, f_y_pot_fr, _, _, f_y_pot_rl, _, _, f_y_pot_rr, _ = (
            _tire_force_pots(vel, a_x_mid, a_y, mu, params, fz_data))

        if (abs_f_y_f <= f_y_pot_fl + f_y_pot_fr
                and abs_f_y_r <= f_y_pot_rl + f_y_pot_rr):
            ind_first = ind_mid
        else:
            ind_last = ind_mid - 1

        ind_mid = (ind_first + ind_last + 1) // 2

    return ind_mid * a_x_step
