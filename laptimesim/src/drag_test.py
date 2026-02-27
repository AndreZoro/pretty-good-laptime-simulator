import math
import numpy as np

# Standard drag-test distances
EIGHTH_MILE_M = 201.168
QUARTER_MILE_M = 402.336
ONE_KM_M = 1000.0

DRAG_DISTANCES = {
    "1/8 mile": EIGHTH_MILE_M,
    "1/4 mile": QUARTER_MILE_M,
    "1 km":     ONE_KM_M,
}


def _max_motor_torque(car, n: float, vel: float) -> float:
    """Return the maximum combined powertrain torque [Nm] at shaft speed n [1/s]."""
    if car.powertrain_type == "electric":
        return car.torque_e_motor(n=n)
    else:  # hybrid / combustion
        return car.torque(n=n) + car.torque_e_motor(n=n, vel=vel)


def run_drag_test(car, distance_m: float, mu: float = 1.0, dt: float = 0.001) -> dict:
    """
    Simulate a standing-start drag test on a flat, straight road.

    The car starts from rest (v = 0) and accelerates at full throttle.
    Physics at each step:
      - powertrain force  = max motor torque × eta_g / (i_trans × r_tire)
      - traction limit    = longitudinal tire potential (AWD/RWD/FWD as configured)
      - resistances       = aerodynamic drag + rolling resistance
      - effective mass    = vehicle mass × rotational-inertia factor e_i[gear]

    Parameters
    ----------
    car : CarElectric or CarHybrid
        Fully initialised vehicle object.
    distance_m : float
        Distance to simulate [m].
    mu : float
        Track friction coefficient (1.0 = dry).
    dt : float
        Euler integration time step [s].

    Returns
    -------
    dict with keys:
        t     : np.ndarray  cumulative time [s]
        vel   : np.ndarray  velocity [m/s]
        dist  : np.ndarray  distance covered [m]
        gear  : np.ndarray  gear index (0-based)
        a_x   : np.ndarray  longitudinal acceleration [m/s²]
    """
    t_list    = [0.0]
    vel_list  = [0.0]
    dist_list = [0.0]
    gear_list = [0]
    ax_list   = [0.0]

    vel  = 0.0
    dist = 0.0
    t    = 0.0

    while dist < distance_m:
        # Use a minimum velocity for lookup functions to avoid divide-by-zero at standstill.
        # At this speed, torque_e_motor returns torque_max (power limit >> torque limit).
        vel_calc = max(vel, 0.5)

        # Gear and shaft speed [1/s]
        gear, n = car.find_gear(vel=vel_calc)

        # Maximum torque the powertrain can produce
        m_max = _max_motor_torque(car, n=n, vel=vel_calc)

        # Drive force at the tire contact patch
        r_tire = car.r_driven_tire(vel=vel_calc)
        f_drive_powertrain = (
            m_max * car.pars_gearbox["eta_g"]
            / (car.pars_gearbox["i_trans"][gear] * r_tire)
        )

        # Traction limit — no lateral force on a drag strip (a_y = 0)
        (f_x_pot_fl, f_y_pot_fl, f_z_fl,
         f_x_pot_fr, f_y_pot_fr, f_z_fr,
         f_x_pot_rl, f_y_pot_rl, f_z_rl,
         f_x_pot_rr, f_y_pot_rr, f_z_rr) = car.tire_force_pots(
            vel=vel_calc, a_x=0.0, a_y=0.0, mu=mu
        )

        f_trac_limit = car.calc_f_x_pot(
            f_x_pot_fl, f_x_pot_fr, f_x_pot_rl, f_x_pot_rr,
            f_y_pot_fl + f_y_pot_fr, f_y_pot_rl + f_y_pot_rr,
            0.0, 0.0,                       # f_y_f, f_y_r = 0 on a straight
            force_use_all_wheels=False,
            limit_braking_weak_side=None,
        )

        # Net drive force (limited by whichever is lower)
        f_drive = min(f_drive_powertrain, f_trac_limit)

        # Resistances
        f_z_tot = f_z_fl + f_z_fr + f_z_rl + f_z_rr
        f_drag  = car.air_res(vel=vel_calc, drs=False)
        f_roll  = car.roll_res(f_z_tot=f_z_tot)

        # Effective mass accounts for drivetrain rotational inertia
        m_eff = car.pars_general["m"] * car.pars_gearbox["e_i"][gear]
        a_x   = (f_drive - f_drag - f_roll) / m_eff

        # Euler integration
        vel   = max(vel + a_x * dt, 0.0)
        dist += vel * dt
        t    += dt

        t_list.append(t)
        vel_list.append(vel)
        dist_list.append(dist)
        gear_list.append(gear)
        ax_list.append(a_x)

    return {
        "t":    np.array(t_list),
        "vel":  np.array(vel_list),
        "dist": np.array(dist_list),
        "gear": np.array(gear_list, dtype=int),
        "a_x":  np.array(ax_list),
    }
