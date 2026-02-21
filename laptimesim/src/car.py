import math

import matplotlib.pyplot as plt
import numpy as np

from laptimesim.src._jit_kernels import (
    _tire_force_pots,
    _calc_lat_forces,
    _calc_f_x_pot,
    _air_res,
    _roll_res,
    _v_max_cornering,
    _calc_max_ax,
    PARAMS_SIZE,
)


def _build_jit_params(pars_general, pars_engine, pars_tires, pars_gearbox):
    """Build the packed 1D float64 parameter array for JIT kernels."""
    params = np.zeros(PARAMS_SIZE, dtype=np.float64)
    params[0] = pars_general["m"]           # _M
    params[1] = pars_general["rho_air"]     # _RHO
    params[2] = pars_general["c_w_a"]       # _CWA
    params[3] = pars_general["f_roll"]      # _FROLL
    params[4] = pars_general.get("drs_factor",
                    pars_general.get("active_aero_drag_reduction", 0.0))  # _DRS
    params[5] = pars_general["lf"]          # _LF
    params[6] = pars_general["lr"]          # _LR

    # topology: AWD=0, FWD=1, RWD=2
    topo = pars_engine["topology"]
    if topo == "AWD":
        params[7] = 0.0
    elif topo == "FWD":
        params[7] = 1.0
    elif topo == "RWD":
        params[7] = 2.0
    else:
        raise RuntimeError("Powertrain topology unknown: %s" % topo)

    params[8] = pars_tires["tire_model_exp"]  # _EXP
    params[9] = pars_tires["f"]["mux"]        # _MUX_F
    params[10] = pars_tires["f"]["muy"]       # _MUY_F
    params[11] = pars_tires["f"]["dmux_dfz"]  # _DMUX_F
    params[12] = pars_tires["f"]["dmuy_dfz"]  # _DMUY_F
    params[13] = pars_tires["f"]["fz_0"]      # _FZ0_F
    params[14] = pars_tires["r"]["mux"]       # _MUX_R
    params[15] = pars_tires["r"]["muy"]       # _MUY_R
    params[16] = pars_tires["r"]["dmux_dfz"]  # _DMUX_R
    params[17] = pars_tires["r"]["dmuy_dfz"]  # _DMUY_R
    params[18] = pars_tires["r"]["fz_0"]      # _FZ0_R
    params[19] = pars_gearbox.get("diff_lock_ratio", 1.0)  # _DIFF_LOCK
    return params


def _build_jit_fz_data(f_z_calc_stat):
    """Build the packed 2D float64 array (6, 4) for JIT kernels from f_z_calc_stat dict."""
    fz_data = np.zeros((6, 4), dtype=np.float64)
    fz_data[0, :] = f_z_calc_stat["stat_load"]
    fz_data[1, :] = f_z_calc_stat["trans_long"]
    fz_data[2, :] = f_z_calc_stat["trans_long_sign"]
    fz_data[3, :] = f_z_calc_stat["trans_lat"]
    fz_data[4, :] = f_z_calc_stat["trans_lat_sign"]
    fz_data[5, :] = f_z_calc_stat["aero"]
    return fz_data


# Map limit_braking_weak_side string to int flag for JIT
_LBS_MAP = {None: 0, "FA": 1, "RA": 2, "all": 3}


class Car(object):
    """
    author:
    Alexander Heilmeier (based on the term thesis of Maximilian Geisslinger)

    date:
    23.12.2018

    .. description::
    The file provides functions related to the vehicle, e.g. power and torque calculations.
    Vehicle coordinate system: x - front, y - left, z - up.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # SLOTS ------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    __slots__ = (
        "__powertrain_type",
        "__pars_general",
        "__pars_engine",
        "__pars_gearbox",
        "__pars_tires",
        "__f_z_calc_stat",
        "_jit_params",
        "_jit_fz_data",
        "_jit_fz_data_aa",
    )

    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(
        self,
        powertrain_type: str,
        pars_general: dict,
        pars_engine: dict,
        pars_gearbox: dict,
        pars_tires: dict,
    ):
        self.powertrain_type = powertrain_type
        self.pars_general = pars_general
        self.pars_engine = pars_engine
        self.pars_gearbox = pars_gearbox
        self.pars_tires = pars_tires

        # calculate static parts of tire load calculation
        self.f_z_calc_stat = {}

        g = self.pars_general["g"]
        m = self.pars_general["m"]
        l_tot = self.pars_general["lf"] + self.pars_general["lr"]
        h_cog = self.pars_general["h_cog"]

        # static load
        self.f_z_calc_stat["stat_load"] = np.zeros(4)
        self.f_z_calc_stat["stat_load"][0] = (
            0.5 * m * g * self.pars_general["lr"] / l_tot
        )
        self.f_z_calc_stat["stat_load"][1] = (
            0.5 * m * g * self.pars_general["lr"] / l_tot
        )
        self.f_z_calc_stat["stat_load"][2] = (
            0.5 * m * g * self.pars_general["lf"] / l_tot
        )
        self.f_z_calc_stat["stat_load"][3] = (
            0.5 * m * g * self.pars_general["lf"] / l_tot
        )

        # longitudinal load transfer
        self.f_z_calc_stat["trans_long"] = np.zeros(4)
        self.f_z_calc_stat["trans_long"][0] = 0.5 * m * h_cog / l_tot
        self.f_z_calc_stat["trans_long"][1] = 0.5 * m * h_cog / l_tot
        self.f_z_calc_stat["trans_long"][2] = 0.5 * m * h_cog / l_tot
        self.f_z_calc_stat["trans_long"][3] = 0.5 * m * h_cog / l_tot

        # lateral load transfer
        self.f_z_calc_stat["trans_lat"] = np.zeros(4)
        self.f_z_calc_stat["trans_lat"][0] = (
            m * self.pars_general["lr"] / l_tot * h_cog / self.pars_general["sf"]
        )
        self.f_z_calc_stat["trans_lat"][1] = (
            m * self.pars_general["lr"] / l_tot * h_cog / self.pars_general["sf"]
        )
        self.f_z_calc_stat["trans_lat"][2] = (
            m * self.pars_general["lf"] / l_tot * h_cog / self.pars_general["sr"]
        )
        self.f_z_calc_stat["trans_lat"][3] = (
            m * self.pars_general["lf"] / l_tot * h_cog / self.pars_general["sr"]
        )

        # aero downforce
        self.f_z_calc_stat["aero"] = np.zeros(4)
        self.f_z_calc_stat["aero"][0] = (
            0.5 * 0.5 * self.pars_general["c_z_a_f"] * self.pars_general["rho_air"]
        )
        self.f_z_calc_stat["aero"][1] = (
            0.5 * 0.5 * self.pars_general["c_z_a_f"] * self.pars_general["rho_air"]
        )
        self.f_z_calc_stat["aero"][2] = (
            0.5 * 0.5 * self.pars_general["c_z_a_r"] * self.pars_general["rho_air"]
        )
        self.f_z_calc_stat["aero"][3] = (
            0.5 * 0.5 * self.pars_general["c_z_a_r"] * self.pars_general["rho_air"]
        )

        # sign arrays for vectorized tire load calculation: FL=-a_x, FR=-a_x, RL=+a_x, RR=+a_x
        self.f_z_calc_stat["trans_long_sign"] = np.array([-1.0, -1.0, 1.0, 1.0])
        # FL=-a_y, FR=+a_y, RL=-a_y, RR=+a_y
        self.f_z_calc_stat["trans_lat_sign"] = np.array([-1.0, 1.0, -1.0, 1.0])

        # build packed parameter arrays for JIT kernels
        self._jit_params = _build_jit_params(pars_general, pars_engine, pars_tires, pars_gearbox)
        self._jit_fz_data = _build_jit_fz_data(self.f_z_calc_stat)

        # active aero fz_data: aero rows scaled by downforce reduction fractions
        dz_f = pars_general.get("active_aero_dz_f", 0.0)
        dz_r = pars_general.get("active_aero_dz_r", 0.0)
        if dz_f != 0.0 or dz_r != 0.0:
            fz_aa = self._jit_fz_data.copy()
            fz_aa[5, 0] *= (1.0 - dz_f)  # FL aero
            fz_aa[5, 1] *= (1.0 - dz_f)  # FR aero
            fz_aa[5, 2] *= (1.0 - dz_r)  # RL aero
            fz_aa[5, 3] *= (1.0 - dz_r)  # RR aero
            self._jit_fz_data_aa = fz_aa
        else:
            self._jit_fz_data_aa = self._jit_fz_data

    # ------------------------------------------------------------------------------------------------------------------
    # GETTERS / SETTERS ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __get_powertrain_type(self) -> str:
        return self.__powertrain_type

    def __set_powertrain_type(self, x: str) -> None:
        self.__powertrain_type = x

    powertrain_type = property(__get_powertrain_type, __set_powertrain_type)

    def __get_pars_general(self) -> dict:
        return self.__pars_general

    def __set_pars_general(self, x: dict) -> None:
        self.__pars_general = x

    pars_general = property(__get_pars_general, __set_pars_general)

    def __get_pars_engine(self) -> dict:
        return self.__pars_engine

    def __set_pars_engine(self, x: dict) -> None:
        self.__pars_engine = x

    pars_engine = property(__get_pars_engine, __set_pars_engine)

    def __get_pars_gearbox(self) -> dict:
        return self.__pars_gearbox

    def __set_pars_gearbox(self, x: dict) -> None:
        self.__pars_gearbox = x

    pars_gearbox = property(__get_pars_gearbox, __set_pars_gearbox)

    def __get_pars_tires(self) -> dict:
        return self.__pars_tires

    def __set_pars_tires(self, x: dict) -> None:
        self.__pars_tires = x

    pars_tires = property(__get_pars_tires, __set_pars_tires)

    def __get_f_z_calc_stat(self) -> dict:
        return self.__f_z_calc_stat

    def __set_f_z_calc_stat(self, x: dict) -> None:
        self.__f_z_calc_stat = x

    f_z_calc_stat = property(__get_f_z_calc_stat, __set_f_z_calc_stat)

    # ------------------------------------------------------------------------------------------------------------------
    # METHODS (CALCULATIONS) -------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def tire_force_pots(self, vel: float, a_x: float, a_y: float, mu: float,
                        active_aero: bool = False) -> tuple:
        """
        The function is used to calculate the transmitted tire forces depending on the current longitudinal and lateral
        accelerations and velocity.
        Velocity input in m/s, accelerations in m/s^2. Calculates the currently acting tire loads f_z (considering
        dynamic load transfers) and the force potentials f_t of all four tires. Vehicle coordinate system: x - front,
        y - left, z - up. The tire model includes the reduction of the force potential with rising tire loads as
        tire_par2 are negativ.
        active_aero: when True, uses reduced-downforce fz_data (2026 active aero open position).
        """
        fz = self._jit_fz_data_aa if active_aero else self._jit_fz_data
        return _tire_force_pots(vel, a_x, a_y, mu, self._jit_params, fz)

    def plot_tire_characteristics(self) -> None:
        # calculate relevant data
        f_z_range = np.arange(500.0, 13000.0, 500.0)

        f_x_f = (
            self.pars_tires["f"]["mux"]
            + self.pars_tires["f"]["dmux_dfz"]
            * (f_z_range - self.pars_tires["f"]["fz_0"])
        ) * f_z_range
        f_y_f = (
            self.pars_tires["f"]["muy"]
            + self.pars_tires["f"]["dmuy_dfz"]
            * (f_z_range - self.pars_tires["f"]["fz_0"])
        ) * f_z_range
        f_x_r = (
            self.pars_tires["r"]["mux"]
            + self.pars_tires["r"]["dmux_dfz"]
            * (f_z_range - self.pars_tires["r"]["fz_0"])
        ) * f_z_range
        f_y_r = (
            self.pars_tires["r"]["muy"]
            + self.pars_tires["r"]["dmuy_dfz"]
            * (f_z_range - self.pars_tires["r"]["fz_0"])
        ) * f_z_range

        # plot
        plt.figure()

        plt.plot(f_z_range, f_x_f)
        plt.plot(f_z_range, f_y_f)
        plt.plot(f_z_range, f_x_r)
        plt.plot(f_z_range, f_y_r)

        plt.grid()
        plt.title("Tire force characteristics")
        plt.xlabel("F_z in N")
        plt.ylabel("Forces F_x and F_y in N")
        plt.legend(["F_x front", "F_y front", "F_x rear", "F_y rear"])

        plt.show()

    def __circumref_driven_tire(self, vel: float) -> float:
        """Velocity input in m/s. Reference speed for the circumreference calculation is 60 km/h. Output is in m."""

        if self.pars_engine["topology"] == "FWD":
            tire_circ_ref = self.pars_tires["f"]["circ_ref"]

        elif self.pars_engine["topology"] == "RWD":
            tire_circ_ref = self.pars_tires["r"]["circ_ref"]

        elif self.pars_engine["topology"] == "AWD":
            # use average circumreference in this case
            tire_circ_ref = 0.5 * (
                self.pars_tires["f"]["circ_ref"] + self.pars_tires["r"]["circ_ref"]
            )

        else:
            raise RuntimeError("Powertrain topology unknown!")

        return tire_circ_ref * (1 + (vel * 3.6 - 60.0) * (0.045 / 200.0))

    def r_driven_tire(self, vel: float) -> float:
        """Velocity input in m/s. Output is in m."""

        return self.__circumref_driven_tire(vel=vel) / (2 * math.pi)

    def air_res(self, vel: float, drs: bool) -> float:
        """Velocity input in m/s. Output is in N."""
        drag_reduction = self.pars_general.get(
            "drs_factor", self.pars_general.get("active_aero_drag_reduction", 0.0))
        return _air_res(vel, drs, self.pars_general["rho_air"],
                        self.pars_general["c_w_a"], drag_reduction)

    def roll_res(self, f_z_tot: float) -> float:
        """Output is in N."""
        return _roll_res(f_z_tot, self.pars_general["f_roll"])

    def calc_lat_forces(self, a_y: float) -> tuple:
        """Lateral acceleration input in m/s^2. Output forces in N."""
        return _calc_lat_forces(a_y, self.pars_general["m"],
                                self.pars_general["lf"], self.pars_general["lr"])

    def v_max_cornering(
        self, kappa: float, mu: float, vel_subtr_corner: float = 0.5
    ) -> float:
        """
        Curvature input in rad/m, vel_subtr_corner in m/s. This method determines the maximum drivable velocity for the
        pure cornering case, i.e. without the application of longitudinal acceleration. However, it is considered that
        the tires must be able to transmit enough longitudinal forces to overcome drag and rolling resistances. The
        calculation neglects the available force in the powertrain. The determined velocity is mostly used as velocity
        after deceleration phases. Using binary search technique to decrease calculation time. vel_subtr_corner is
        subtracted from the found cornering velocity because drivers in reality will not hit the maximum perfectly.
        """
        return _v_max_cornering(kappa, mu, vel_subtr_corner, self._jit_params, self._jit_fz_data)

    def calc_f_x_pot(
        self,
        f_x_pot_fl: float,
        f_x_pot_fr: float,
        f_x_pot_rl: float,
        f_x_pot_rr: float,
        f_y_pot_f: float,
        f_y_pot_r: float,
        f_y_f: float,
        f_y_r: float,
        force_use_all_wheels: bool = False,
        limit_braking_weak_side: None or str = None,
    ) -> float:
        """Calculate remaining tire potential for longitudinal force transmission considering driven axle(s). All forces
        in N. 'force_use_all_wheels' flag can be set to use this function also for braking with all four wheels.
        limit_braking_weak_side can be None, 'FA', 'RA', 'all'. This determines if the possible braking force should be
        determined based on the weak side, e.g. when braking into a corner. Can be set separately for both axles. This
        is not necessary during acceleration since a limited slip differential overcomes this problem."""

        if limit_braking_weak_side is not None and not force_use_all_wheels:
            print(
                "WARNING: It seems like the function is used for braking (because limit_braking_weak_side is set)"
                " but force_use_all_wheels is not set True!"
            )

        lbs_flag = _LBS_MAP[limit_braking_weak_side]
        diff_lock_ratio = self.pars_gearbox.get("diff_lock_ratio", 1.0)
        return _calc_f_x_pot(
            f_x_pot_fl, f_x_pot_fr, f_x_pot_rl, f_x_pot_rr,
            f_y_pot_f, f_y_pot_r, f_y_f, f_y_r,
            int(self._jit_params[7]), self.pars_tires["tire_model_exp"],
            force_use_all_wheels, lbs_flag, diff_lock_ratio,
        )

    def calc_max_ax(
        self, vel: float, a_y: float, mu: float, f_y_f: float, f_y_r: float
    ) -> float:
        """Calculate maximum longitudinal acceleration at which the car stays on the track. vel in m/s, a_y in m/s^2,
        f_y_f and f_y_r in N. Using binary search technique to decrease calculation time."""
        return _calc_max_ax(vel, a_y, mu, f_y_f, f_y_r, self._jit_params, self._jit_fz_data)

    def find_gear(self, vel: float) -> tuple:
        """Velocity input in m/s. Output is the gear used for that velocity (zero based) as well as the corresponding
        engine rev in 1/s."""

        # calculate theoretical engine revs for all the gears
        n_gears = vel / (
            self.__circumref_driven_tire(vel=vel) * self.pars_gearbox["i_trans"]
        )  # [1/s]

        # find largest gear below shift revs
        shift_bool = n_gears < self.pars_gearbox["n_shift"]

        if np.all(~shift_bool):
            # if max rev in final gear is reached do not shift up
            gear_ind = (
                self.pars_gearbox["n_shift"].size - 1
            )  # -1 due to zero based indexing
        else:
            # find first True value (zero based indexing of gears)
            gear_ind = int(np.argmax(shift_bool))

        return gear_ind, n_gears[gear_ind]

    def calc_m_requ(self, f_x: float, vel: float) -> float:
        """Function to calculate required powertrain torque to reach a specific longitudinal acceleration force f_x at
        the current velocity. Input f_x in N, vel in m/s. Output is the rquired powertrain torque in Nm."""

        # get gear at velocity
        gear = self.find_gear(vel=vel)[0]

        # calculate powertrain torque (pure mechanical conversion, e_i is handled in the equation of motion)
        m_requ = (
            f_x
            * self.r_driven_tire(vel=vel)
            * self.pars_gearbox["i_trans"][gear]
            / self.pars_gearbox["eta_g"]
        )

        return m_requ


# ----------------------------------------------------------------------------------------------------------------------
# TESTING --------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    pass
