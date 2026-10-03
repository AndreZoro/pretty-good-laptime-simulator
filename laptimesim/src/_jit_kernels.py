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
_DIFF_LOCK = 19  # differential lock ratio [-] (0=open, 1=locked)
PARAMS_SIZE = 20

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
                  topology, exp, force_all_wheels, lbs_flag, diff_lock_ratio):
    """
    Calculate remaining tire potential for longitudinal force.
    topology: 0=AWD, 1=FWD, 2=RWD
    lbs_flag: 0=None, 1=FA, 2=RA, 3=all (limit_braking_weak_side)
    diff_lock_ratio: 0.0=open diff (weak side limited), 1.0=locked/spool (full sum)
    """
    inv_exp = 1.0 / exp

    # determine axle potentials based on weak side limiting (braking) or diff model (acceleration)
    if force_all_wheels:
        # braking: use lbs_flag as before
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
    else:
        # acceleration: apply differential model to driven axle(s)
        if topology == 0:  # AWD — diff on both axles
            f_x_pot_f_open = 2.0 * min(f_x_pot_fl, f_x_pot_fr)
            f_x_pot_f_locked = f_x_pot_fl + f_x_pot_fr
            f_x_pot_f = f_x_pot_f_open + diff_lock_ratio * (f_x_pot_f_locked - f_x_pot_f_open)
            f_x_pot_r_open = 2.0 * min(f_x_pot_rl, f_x_pot_rr)
            f_x_pot_r_locked = f_x_pot_rl + f_x_pot_rr
            f_x_pot_r = f_x_pot_r_open + diff_lock_ratio * (f_x_pot_r_locked - f_x_pot_r_open)
        elif topology == 1:  # FWD — diff on front axle only
            f_x_pot_f_open = 2.0 * min(f_x_pot_fl, f_x_pot_fr)
            f_x_pot_f_locked = f_x_pot_fl + f_x_pot_fr
            f_x_pot_f = f_x_pot_f_open + diff_lock_ratio * (f_x_pot_f_locked - f_x_pot_f_open)
            f_x_pot_r = f_x_pot_rl + f_x_pot_rr
        else:  # RWD — diff on rear axle only
            f_x_pot_f = f_x_pot_fl + f_x_pot_fr
            f_x_pot_r_open = 2.0 * min(f_x_pot_rl, f_x_pot_rr)
            f_x_pot_r_locked = f_x_pot_rl + f_x_pot_rr
            f_x_pot_r = f_x_pot_r_open + diff_lock_ratio * (f_x_pot_r_locked - f_x_pot_r_open)

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
def _cornering_feasible(vel, kappa, mu, params, fz_data):
    """Can the car hold this curvature at this velocity (pure cornering, a_x = 0)?

    True when both axles can transmit the lateral force AND the remaining longitudinal
    potential still covers drag plus rolling resistance.
    """
    m = params[_M]
    lf = params[_LF]
    lr = params[_LR]
    topology = int(params[_TOPO])
    exp = params[_EXP]
    rho_air = params[_RHO]
    c_w_a = params[_CWA]
    f_roll = params[_FROLL]
    diff_lock_ratio = params[_DIFF_LOCK]

    a_y = vel * vel * kappa
    f_y_f, f_y_r = _calc_lat_forces(a_y, m, lf, lr)

    (f_x_pot_fl, f_y_pot_fl, f_z_fl,
     f_x_pot_fr, f_y_pot_fr, f_z_fr,
     f_x_pot_rl, f_y_pot_rl, f_z_rl,
     f_x_pot_rr, f_y_pot_rr, f_z_rr) = _tire_force_pots(vel, 0.0, a_y, mu, params, fz_data)

    if not (abs(f_y_f) < f_y_pot_fl + f_y_pot_fr
            and abs(f_y_r) < f_y_pot_rl + f_y_pot_rr):
        return False

    f_x_poss = _calc_f_x_pot(
        f_x_pot_fl, f_x_pot_fr, f_x_pot_rl, f_x_pot_rr,
        f_y_pot_fl + f_y_pot_fr, f_y_pot_rl + f_y_pot_rr,
        f_y_f, f_y_r,
        topology, exp, False, 0, diff_lock_ratio)

    f_z_tot = f_z_fl + f_z_fr + f_z_rl + f_z_rr
    f_x_drag = _air_res(vel, False, rho_air, c_w_a, 0.0) + _roll_res(f_z_tot, f_roll)

    return f_x_poss >= f_x_drag


@njit(cache=True)
def _v_max_cornering(kappa, mu, vel_subtr_corner, params, fz_data):
    """Maximum cornering velocity [m/s], by continuous bisection on the feasibility test.

    This used to bisect a FIXED GRID of 546 velocities between 1 and 110 m/s, i.e. it returned
    the ceiling quantised to 0.2 m/s. That quantisation was visible in the solver: through a
    gradually tightening bend the ceiling fell in 0.2 m/s steps, and shedding 0.2 m/s inside a
    single 1 m step needs about -13 m/s^2, so the velocity profile showed a hold/brake sawtooth
    with exactly that amplitude. Bisecting the interval itself instead makes the ceiling smooth,
    and costs ~27 cheap iterations instead of ~10.
    """
    vel_lo = 1.0    # assumed feasible (matches the old grid's floor)
    vel_hi = 110.0  # assumed infeasible

    if not _cornering_feasible(vel_lo, kappa, mu, params, fz_data):
        return vel_lo - vel_subtr_corner

    if _cornering_feasible(vel_hi, kappa, mu, params, fz_data):
        return vel_hi - vel_subtr_corner

    # 27 halvings take the 109 m/s bracket below 1e-6 m/s
    for _ in range(27):
        vel_mid = 0.5 * (vel_lo + vel_hi)
        if _cornering_feasible(vel_mid, kappa, mu, params, fz_data):
            vel_lo = vel_mid
        else:
            vel_hi = vel_mid

    return vel_lo - vel_subtr_corner


@njit(cache=True)
def _v_max_cornering_arr(kappa, mu, vel_subtr_corner, params, fz_data, out):
    """Maximum cornering velocity for every point, written into 'out'.

    Same result as calling _v_max_cornering() per point -- it exists so the solver can hold the
    cornering ceiling as an array and enforce it in the forward pass, instead of evaluating it
    reactively once the lateral grip check has already failed.
    """
    for i in range(kappa.shape[0]):
        out[i] = _v_max_cornering(kappa[i], mu[i], vel_subtr_corner, params, fz_data)
    return out


@njit(cache=True)
def _find_gear(vel, circ_ref, i_trans, n_shift):
    """Gear (zero based) and engine rev [1/s] at this velocity. vel in m/s, circ_ref in m.

    Picks the lowest gear whose theoretical rev is still below its shift rev, and stays in the
    final gear once even that one is over its shift rev.

    Replaces an array formulation that built three temporaries per call (an 8-element divide,
    a boolean mask, then np.all/np.argmax over it) for what is scalar work. find_gear is called
    once per forward step and once per recalculated point, 157k times in a single MVRC lap, so
    the temporaries dominated it. Bit-identical to the array form: the arithmetic per gear is
    the same and the selection rule picks the same index.
    """
    circ = circ_ref * (1.0 + (vel * 3.6 - 60.0) * (0.045 / 200.0))

    gear_ind = n_shift.shape[0] - 1  # -1 due to zero based indexing
    for k in range(n_shift.shape[0]):
        if vel / (circ * i_trans[k]) < n_shift[k]:
            gear_ind = k
            break

    return gear_ind, vel / (circ * i_trans[gear_ind])


@njit(cache=True)
def _brake_back_vel(vel_known, a_x, kappa_prev, mu_prev, stepsize, tol, max_iters,
                    params, fz_data, lbs_flag, tire_loads_out):
    """Velocity at the point before 'vel_known' under maximum braking, as a fixed point.

    This is the backward-sweep inner loop of Lap.__fbplus, moved into one compiled kernel:
    it used to run in Python and cross the Numba boundary six times per iteration, which
    dominated the solver's run time because the iteration count per point is in the hundreds
    (the fixed point is approached by a running average -- see the loop body).

    The arithmetic is unchanged, operation for operation, so the result is bit-identical to
    the Python version.

    Returns (vel_tmp, a_x, counter); tire_loads_out[0:4] holds the loads at the converged
    velocity. counter > max_iters signals non-convergence, which the caller reports.
    """
    m = params[_M]
    exp = params[_EXP]
    diff_lock_ratio = params[_DIFF_LOCK]
    topology = int(params[_TOPO])
    rho_air = params[_RHO]
    c_w_a = params[_CWA]
    drs_factor = params[_DRS]
    f_roll = params[_FROLL]
    lf = params[_LF]
    lr = params[_LR]

    vel_tmp = vel_known
    vel_tmp_old = 0.0
    vel_sum = 0.0
    vel_count = 0
    counter = 0

    while abs(vel_tmp - vel_tmp_old) > tol:
        counter += 1
        vel_tmp_old = vel_tmp

        if counter > max_iters:
            return vel_tmp, a_x, counter

        a_y = vel_tmp * vel_tmp * kappa_prev
        f_y_f, f_y_r = _calc_lat_forces(a_y, m, lf, lr)

        (f_x_pot_fl, f_y_pot_fl, tire_loads_out[0],
         f_x_pot_fr, f_y_pot_fr, tire_loads_out[1],
         f_x_pot_rl, f_y_pot_rl, tire_loads_out[2],
         f_x_pot_rr, f_y_pot_rr, tire_loads_out[3]) = _tire_force_pots(
            vel_tmp, a_x, a_y, mu_prev, params, fz_data)

        f_x_poss = _calc_f_x_pot(
            f_x_pot_fl, f_x_pot_fr, f_x_pot_rl, f_x_pot_rr,
            f_y_pot_fl + f_y_pot_fr, f_y_pot_rl + f_y_pot_rr, f_y_f, f_y_r,
            topology, exp, True, lbs_flag, diff_lock_ratio)

        f_z_tot = (tire_loads_out[0] + tire_loads_out[1]
                   + tire_loads_out[2] + tire_loads_out[3])

        a_x = -(f_x_poss
                + _air_res(vel_tmp, False, rho_air, c_w_a, drs_factor)
                + _roll_res(f_z_tot, f_roll)) / m

        vel_sum += math.sqrt(vel_known * vel_known + 2 * -a_x * stepsize)
        vel_count += 1

        # applied velocity as the average of all iterates (robust convergence characteristic)
        vel_tmp = vel_sum / vel_count

    return vel_tmp, a_x, counter


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
