import numpy as np
import math
import matplotlib.pyplot as plt
import ast
from laptimesim.src.car import Car
import configparser


class CarHybrid(Car):
    """
    author:
    Alexander Heilmeier (based on the term thesis of Maximilian Geisslinger)

    date:
    23.12.2018

    .. description::
    The file provides functions related to the vehicle, e.g. power and torque calculations.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # SLOTS ------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # the _pow_* / _fuel_* entries cache what __power_engine needs on its scalar fast path,
    # so a per-point power lookup costs no dict lookups and allocates nothing
    __slots__ = ("__z_pow_engine", "_pow_c3", "_pow_c2", "_pow_c1", "_pow_c0",
                 "_n_clip_lo", "_n_clip_hi", "_fuel_lim_pars",
                 "_pow_e_motor_c", "_torque_e_motor_max_c", "_ers_speed_limit_c",
                 "_vel_min_e_motor_c", "_eta_e_motor_c")

    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, parfilepath: str = None, pars_veh: dict = None):

        # load vehicle parameters from file or use provided dict
        if pars_veh is not None:
            # Use provided parameters directly (deep copy to avoid modifying original)
            import copy
            pars_veh_tmp = copy.deepcopy(pars_veh)
        elif parfilepath is not None:
            parser = configparser.ConfigParser()
            if not parser.read(parfilepath):
                raise RuntimeError('Specified config file does not exist or is empty!')
            pars_veh_tmp = ast.literal_eval(parser.get('VEH_PARS', 'veh_pars'))
        else:
            raise RuntimeError('Either parfilepath or pars_veh must be provided!')

        # unit conversions
        pars_veh_tmp["engine"]["n_begin"] /= 60.0   # [1/min] -> [1/s]
        pars_veh_tmp["engine"]["n_max"] /= 60.0     # [1/min] -> [1/s]
        pars_veh_tmp["engine"]["n_end"] /= 60.0     # [1/min] -> [1/s]
        pars_veh_tmp["engine"]["be_max"] /= 3600.0  # [kg/h] -> [kg/s]

        for i, item in enumerate(pars_veh_tmp["gearbox"]["n_shift"]):
            pars_veh_tmp["gearbox"]["n_shift"][i] = item / 60.0  # [1/min] -> [1/s]

        # convert gearbox arrays to numpy arrays
        pars_veh_tmp["gearbox"]["i_trans"] = np.array(pars_veh_tmp["gearbox"]["i_trans"])
        pars_veh_tmp["gearbox"]["n_shift"] = np.array(pars_veh_tmp["gearbox"]["n_shift"])
        pars_veh_tmp["gearbox"]["e_i"] = np.array(pars_veh_tmp["gearbox"]["e_i"])

        # initialize base class object
        Car.__init__(self,
                     powertrain_type=pars_veh_tmp["powertrain_type"],
                     pars_general=pars_veh_tmp["general"],
                     pars_engine=pars_veh_tmp["engine"],
                     pars_gearbox=pars_veh_tmp["gearbox"],
                     pars_tires=pars_veh_tmp["tires"])

        # calculate ICE power curve LES coefficients (z_pow_engine)
        pow_max = self.pars_engine["pow_max"]
        pow_diff = self.pars_engine["pow_diff"]
        n_begin = self.pars_engine["n_begin"]
        n_max = self.pars_engine["n_max"]
        n_end = self.pars_engine["n_end"]
        pow_begend = pow_max - pow_diff

        a = np.array([[math.pow(n_begin, 3), math.pow(n_begin, 2), n_begin, 1],
                      [3 * math.pow(n_max, 2), 2 * n_max, 1, 0],
                      [math.pow(n_max, 3), math.pow(n_max, 2), n_max, 1],
                      [math.pow(n_end, 3), math.pow(n_end, 2), n_end, 1]])
        b = np.array([[pow_begend], [0], [pow_max], [pow_begend]])
        self.z_pow_engine = np.linalg.solve(a, b)

        # scalar copies of everything __power_engine needs, so the per-point path below does no
        # dict lookups and builds no arrays. pars_engine and z_pow_engine are not written after
        # construction anywhere, same as the packed JIT parameter arrays in Car.
        z = self.z_pow_engine  # shape (4, 1) -- it solves against a column vector
        self._pow_c3 = float(z[0, 0])
        self._pow_c2 = float(z[1, 0])
        self._pow_c1 = float(z[2, 0])
        self._pow_c0 = float(z[3, 0])
        self._n_clip_lo = 0.75 * self.pars_engine["n_begin"]
        self._n_clip_hi = 1.2 * self.pars_engine["n_end"]

        ef_max_ = self.pars_engine.get("fuel_energy_flow_max")
        if ef_max_ is None:
            self._fuel_lim_pars = None
        else:
            self._fuel_lim_pars = (
                ef_max_,
                self.pars_engine.get("fuel_ef_slope", 0.27),
                self.pars_engine.get("fuel_ef_offset", 165.0),
                self.pars_engine.get("fuel_ef_n_ref", 10500.0),
                self.pars_engine.get("eta_thermal", 0.48),
            )

        # engine parameters read on the per-point torque path (torque_e_motor,
        # calc_torque_distr, power_demand_e_motor_drive), cached for the same reason
        self._pow_e_motor_c = self.pars_engine["pow_e_motor"]
        self._torque_e_motor_max_c = self.pars_engine["torque_e_motor_max"]
        self._ers_speed_limit_c = self.pars_engine.get("ers_speed_limit", False)
        self._vel_min_e_motor_c = self.pars_engine["vel_min_e_motor"]
        self._eta_e_motor_c = self.pars_engine["eta_e_motor"]

    # ------------------------------------------------------------------------------------------------------------------
    # GETTERS / SETTERS ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __get_z_pow_engine(self) -> np.ndarray: return self.__z_pow_engine
    def __set_z_pow_engine(self, x: np.ndarray) -> None: self.__z_pow_engine = x
    z_pow_engine = property(__get_z_pow_engine, __set_z_pow_engine)

    # ------------------------------------------------------------------------------------------------------------------
    # METHODS (CALCULATIONS) -------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __power_engine(self, n: float or np.ndarray):
        """
        Power curve is approximated by a peak power pow_max at n_max and equal drops on both sides at n_begin and n_end.
        Rev input is in 1/s, output is in W.
        """

        # Scalar fast path. The solver asks for the power at a single rev ~58k times per lap,
        # and the array formulation below allocates six temporaries per call to do it. The
        # arithmetic here is chosen to match numpy bit for bit: npy_pow special-cases an
        # exponent of 2 into a multiply but not 3, so the cubic term must go through math.pow
        # and the square must NOT (x*x*x differs from pow(x, 3) for ~26 % of inputs, and
        # math.pow(x, 2) differs from x*x for some).
        if type(n) is float or type(n) is np.float64:
            n_use = n
            if n_use < self._n_clip_lo:
                n_use = self._n_clip_lo
            if n_use > self._n_clip_hi:
                n_use = self._n_clip_hi

            p_eng = (self._pow_c3 * math.pow(n_use, 3) + self._pow_c2 * (n_use * n_use)
                     + self._pow_c1 * n_use + self._pow_c0)
            if p_eng < 0.0:
                p_eng = 0.0

            if self._fuel_lim_pars is not None:
                ef_max, ef_slope, ef_offset, n_ref, eta_thermal = self._fuel_lim_pars
                n_rpm = n_use * 60.0
                if n_rpm < n_ref:
                    ef = min(ef_slope * n_rpm + ef_offset, ef_max)
                else:
                    ef = ef_max
                p_limit = eta_thermal * ef * 1e6 / 3600.0
                if p_limit < p_eng:
                    p_eng = p_limit

            return p_eng

        # get relevant data
        n_begin = self.pars_engine["n_begin"]
        n_end = self.pars_engine["n_end"]

        # handle both scalar and array inputs
        scalar_input = np.isscalar(n)
        n_use = np.atleast_1d(np.copy(n))

        # limit engine speed to valid range of power curve
        n_use[n_use < 0.75 * n_begin] = 0.75 * n_begin
        n_use[n_use > 1.2 * n_end] = 1.2 * n_end

        # calculate power
        p_eng = (self.z_pow_engine[0] * np.power(n_use, 3) + self.z_pow_engine[1] * np.power(n_use, 2)
                 + self.z_pow_engine[2] * n_use + self.z_pow_engine[3])
        p_eng[p_eng < 0.0] = 0.0  # assure that no negativ powers appear

        # cap by the fuel energy flow limit; no-op for configs that declare no limit, so
        # pre-2026 vehicles are unaffected
        p_eng = np.minimum(p_eng, self.__fuel_power_limit(n_use))

        # return scalar if input was scalar
        if scalar_input:
            return p_eng[0]
        return p_eng

    def __fuel_power_limit(self, n_use):
        """Maximum crankshaft power permitted by the fuel energy flow rules, in W.

        C5.2.3 caps fuel energy flow at fuel_energy_flow_max (3000 MJ/h for 2026).
        C5.2.4 lowers it below fuel_ef_n_ref rpm to  EF = fuel_ef_slope * N(rpm) + fuel_ef_offset
        (0.27 * N + 165 for 2026); the two curves meet exactly at 10500 rpm.
        Fuel energy is converted to crankshaft power with the brake thermal efficiency
        eta_thermal, chosen per car so the declared pow_max is reachable at the flow limit.

        C5.2.5 (the partial load curve EF <= 9.78 * P + 869) is not enforced separately: it is
        satisfied for any eta_thermal above ~0.10, since deriving power from fuel flow cannot
        produce the low-power/high-flow combination that rule exists to prevent.

        Rev input in 1/s. Returns inf when the config declares no limit.
        """
        ef_max = self.pars_engine.get("fuel_energy_flow_max")
        if ef_max is None:
            return np.inf

        n_rpm = n_use * 60.0
        ef_low = (self.pars_engine.get("fuel_ef_slope", 0.27) * n_rpm
                  + self.pars_engine.get("fuel_ef_offset", 165.0))
        n_ref = self.pars_engine.get("fuel_ef_n_ref", 10500.0)
        ef = np.where(n_rpm < n_ref, np.minimum(ef_low, ef_max), ef_max)  # [MJ/h]

        return self.pars_engine.get("eta_thermal", 0.48) * ef * 1e6 / 3600.0

    def plot_power_engine(self) -> None:
        # plot
        n_range = np.arange(7000.0, 15100.0, 100.0) / 60.0  # [1/s]

        plt.figure()
        plt.plot(n_range * 60.0, self.__power_engine(n=n_range) / 1000.0 * 1.36)
        plt.title("Engine power characteristics")
        plt.xlabel("n in 1/min")
        plt.ylabel("P in PS")

        plt.show()

    def torque(self, n: float) -> float:
        """Rev input in 1/s. Output is the maximum torque in Nm."""

        return float(self.__power_engine(n=n)) / (2 * math.pi * n)

    def pow_e_motor_max(self, vel: float, override: bool = False) -> float:
        """Velocity in m/s. Returns max ERS-K deploy power in W per C5.2.8."""
        v_kph = vel * 3.6
        p_max = self._pow_e_motor_c  # absolute cap (350 kW)

        if not override:
            # Normal mode (C5.2.8.i)
            if v_kph < 340.0:
                p_limit = (1800.0 - 5.0 * v_kph) * 1e3
            elif v_kph < 345.0:
                p_limit = (6900.0 - 20.0 * v_kph) * 1e3
            else:
                p_limit = 0.0
        else:
            # Override mode (C5.2.8.ii)
            if v_kph < 355.0:
                p_limit = (7100.0 - 20.0 * v_kph) * 1e3
            else:
                p_limit = 0.0

        return max(0.0, min(p_max, p_limit))

    def torque_e_motor(self, n: float, vel: float = None) -> float:
        """Rev input in 1/s. Output is the maximum torque in Nm."""

        if vel is not None and self._ers_speed_limit_c:
            pow_avail = self.pow_e_motor_max(vel)
        else:
            pow_avail = self._pow_e_motor_c

        torque_tmp = pow_avail / (2 * math.pi * n)

        if torque_tmp > self._torque_e_motor_max_c:
            torque_tmp = self._torque_e_motor_max_c

        return torque_tmp

    def fuel_cons(self, t_cl: np.ndarray, n_cl: np.ndarray, m_eng: np.ndarray) -> np.ndarray:
        """Rev input in 1/s, torque input in Nm. Output is the consumed fuel mass until the current point in kg
        (closed)."""

        be_kgs = self.__injectionmap(n=n_cl[:-1],
                                     m_eng=m_eng)  # [kg/s]

        # integrate
        consumpt_kg_part = np.diff(t_cl) * be_kgs
        consumpt_kg_cl = np.insert(np.cumsum(consumpt_kg_part), 0, 0.0)  # [kg]

        return consumpt_kg_cl

    def __injectionmap(self, n: np.ndarray, m_eng: np.ndarray) -> np.ndarray:
        """Rev input in 1/s, torque input in Nm. Output is in kg/s. Model of the engine fuel consumption."""

        pow_actual = 2 * math.pi * n * m_eng  # [W]
        pow_max = self.__power_engine(n=n)    # [W]
        be = np.sqrt(pow_actual / pow_max) * self.pars_engine["be_max"]  # [kg/s]

        return be

    def e_cons(self, t_cl: np.ndarray, n_cl: np.ndarray, m_e_motor: np.ndarray) -> np.ndarray:
        """Rev input in 1/s, torque input in Nm. Output is the consumed energy in J until the current point(closed).
        Calculates used energy including the efficiency."""

        be_w = self.power_demand_e_motor_drive(n=n_cl[:-1],
                                               m_e_motor=m_e_motor)  # [W]

        # integrate
        e_consumpt_j_part = np.diff(t_cl) * be_w  # [J]
        e_consumpt_j_cl = np.insert(np.cumsum(e_consumpt_j_part), 0, 0.0)  # [J]

        return e_consumpt_j_cl

    def power_demand_e_motor_drive(self, n: np.ndarray, m_e_motor: np.ndarray) -> np.ndarray:
        """Rev input in 1/s, torque input in Nm. Output is in W. Calculates used power including the efficiency."""

        return (2 * math.pi * n * m_e_motor) / self._eta_e_motor_c

    def calc_torque_distr(self, n: float, m_requ: float, throttle_pos: float, es: float,
                          em_boost_use: bool, vel: float) -> tuple:
        """n in 1/s, torque_req in Nm, es in J. Function returns torques delivered by engine and e motor in
        Nm."""

        # get torque potential of engine and e motor
        eng_torque_max = self.torque(n=n)
        e_motor_torque_max = self.torque_e_motor(n=n, vel=vel)

        if m_requ <= eng_torque_max:  # ICE only
            m_eng = throttle_pos * m_requ
            m_e_motor = 0.0

        elif m_requ <= eng_torque_max + e_motor_torque_max:  # ICE + e motor (partly)
            m_eng = throttle_pos * eng_torque_max

            if es > 0.0 and em_boost_use and vel >= self._vel_min_e_motor_c:
                m_e_motor = throttle_pos * (m_requ - eng_torque_max)
            else:
                m_e_motor = 0.0

        else:  # ICE + e motor (fully)
            m_eng = throttle_pos * eng_torque_max

            if es > 0.0 and em_boost_use and vel >= self._vel_min_e_motor_c:
                m_e_motor = throttle_pos * e_motor_torque_max
            else:
                m_e_motor = 0.0

        return m_eng, m_e_motor

    def calc_torque_distr_f_x(self, f_x: float, n: float, throttle_pos: float, es: float,
                              em_boost_use: bool, vel: float) -> tuple:
        """n in 1/s, torque_req in Nm, es in J. Function returns torques delivered by engine and e motor in
        Nm."""

        # calculate required torque to reach f_x
        m_requ = self.calc_m_requ(f_x=f_x,
                                  vel=vel)

        # get torque potential of engine and e motor
        m_eng, m_e_motor = self.calc_torque_distr(n=n,
                                                  m_requ=m_requ,
                                                  throttle_pos=throttle_pos,
                                                  es=es,
                                                  em_boost_use=em_boost_use,
                                                  vel=vel)

        return m_requ, m_eng, m_e_motor


# ----------------------------------------------------------------------------------------------------------------------
# TESTING --------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    pass
