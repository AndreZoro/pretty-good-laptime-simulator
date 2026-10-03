import math

import matplotlib.pyplot as plt
import numpy as np

from laptimesim.src._jit_kernels import _air_res, _brake_back_vel
from laptimesim.src.car import active_aero_kappa_threshold, _LBS_MAP
from laptimesim.src.driver import Driver, ay_max_deploy, ay_max_harvest
from laptimesim.src.track import Track


class Lap(object):
    """
    author:
    Alexander Heilmeier (based on the term thesis of Maximilian Geisslinger)

    date:
    23.12.2018

    .. description::
    The function provides the solver required to calculate the velocity profile and various other data
    step by step. The solver is based on a moving point mass instead of a kinematic model. However, the longitudinal
    and lateral load transfers are considered for the (steady-state) tire load calculations of every wheel. The tire
    model models the effect of reduced force potential with rising tire loads. The solver does not calculate a proper
    start velocity. This has to be ensured by re-running the solver a second time.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # SLOTS ------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    __slots__ = (
        "__driverobj",
        "__trackobj",
        "__t_cl",
        "__vel_cl",
        "__n_cl",
        "__m_eng",
        "__m_e_motor",
        "__m_requ",
        "__es_cl",
        "__gear_cl",
        "__e_rec_e_motor",
        "__a_x_final",
        "__e_rec_e_motor_max",
        "__p_rec_max",
        "__p_harvest_straight",
        "__pars_solver",
        "__debug_opts",
        "__fuel_cons_cl",
        "__e_cons_cl",
        "__tire_loads",
        "__e_es_to_e_motor_max",
        "__es_max",
    )

    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(
        self, driverobj: Driver, trackobj: Track, pars_solver: dict, debug_opts: dict
    ):
        # save driver and track objects
        self.driverobj = driverobj
        self.trackobj = trackobj

        # save solver parameters and debug options
        self.pars_solver = pars_solver
        self.debug_opts = debug_opts

        # adjust solver parameters
        if self.trackobj.pars_track["use_pit"]:
            # v_start is defined by pit speed limit
            self.pars_solver["v_start"] = self.trackobj.pars_track["pitspeed"]
            self.pars_solver["find_v_start"] = False

        # initialize lap variables
        self.t_cl = np.zeros(
            trackobj.no_points_cl
        )  # [s] lap time at the beginning of a step
        self.vel_cl = np.zeros(
            trackobj.no_points_cl
        )  # [m/s] velocity at the beginning of a step
        self.n_cl = np.zeros(
            trackobj.no_points_cl
        )  # [1/s] rev at the beginning of a step
        self.m_eng = np.zeros(
            trackobj.no_points
        )  # [Nm] used engine torque during current step, positive values
        self.m_e_motor = np.zeros(
            trackobj.no_points
        )  # [Nm] used e motor torque during current step, positive values
        self.m_requ = np.zeros(
            trackobj.no_points
        )  # [Nm] requested torque during current step, positive values
        self.es_cl = np.zeros(
            trackobj.no_points_cl
        )  # [J] energy storage state at the beginning of a step
        self.gear_cl = np.zeros(
            trackobj.no_points_cl, dtype=int
        )  # [-] gear during current step, zero based in solver
        # [J] energy recuperated during the current step available at the beginning of next step
        self.e_rec_e_motor = np.zeros(trackobj.no_points)
        self.a_x_final = 0.0
        self.fuel_cons_cl = np.zeros(
            trackobj.no_points_cl
        )  # [kg] consumed fuel mass until current point
        self.e_cons_cl = np.zeros(
            trackobj.no_points_cl
        )  # [J] consumed energy until current point
        self.tire_loads = np.zeros(
            (trackobj.no_points, 4)
        )  # [N] tire loads [FL, FR, RL, RR]

        # [J/lap] maximum harvestable energy: min of vehicle limit and per-track FIA limit
        _vehicle_rec_max = self.driverobj.carobj.pars_engine.get("e_rec_e_motor_max", np.inf)
        self.e_rec_e_motor_max = min(_vehicle_rec_max, self.__track_e_rec_max())
        _vehicle_p_rec = self.driverobj.carobj.pars_engine.get("pow_e_motor", 350e3)
        _track_p_rec = self.trackobj.pars_track.get("p_rec_max", 350e3)
        self.p_rec_max = min(_vehicle_p_rec, _track_p_rec)
        # [W] power cap for active MGU-K harvest on straights — separate from braking regen which
        # always uses p_rec_max. Defaults to p_rec_max if not specified in the vehicle config.
        self.p_harvest_straight = self.driverobj.carobj.pars_engine.get(
            "p_harvest_straight", self.p_rec_max
        )
        self.e_es_to_e_motor_max = self.driverobj.carobj.pars_engine.get(
            "e_es_to_e_motor_max", np.inf
        )

        # [J] maximum energy storage capacity
        self.es_max = self.driverobj.carobj.pars_engine.get(
            "max_e_energy_storage", np.inf
        )

    # ------------------------------------------------------------------------------------------------------------------
    # GETTERS / SETTERS ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __get_driverobj(self) -> Driver:
        return self.__driverobj

    def __set_driverobj(self, x: Driver) -> None:
        self.__driverobj = x

    driverobj = property(__get_driverobj, __set_driverobj)

    def __get_trackobj(self) -> Track:
        return self.__trackobj

    def __set_trackobj(self, x: Track) -> None:
        self.__trackobj = x

    trackobj = property(__get_trackobj, __set_trackobj)

    def __get_t_cl(self) -> np.ndarray:
        return self.__t_cl

    def __set_t_cl(self, x: np.ndarray) -> None:
        self.__t_cl = x

    t_cl = property(__get_t_cl, __set_t_cl)

    def __get_vel_cl(self) -> np.ndarray:
        return self.__vel_cl

    def __set_vel_cl(self, x: np.ndarray) -> None:
        self.__vel_cl = x

    vel_cl = property(__get_vel_cl, __set_vel_cl)

    def __get_n_cl(self) -> np.ndarray:
        return self.__n_cl

    def __set_n_cl(self, x: np.ndarray) -> None:
        self.__n_cl = x

    n_cl = property(__get_n_cl, __set_n_cl)

    def __get_m_eng(self) -> np.ndarray:
        return self.__m_eng

    def __set_m_eng(self, x: np.ndarray) -> None:
        self.__m_eng = x

    m_eng = property(__get_m_eng, __set_m_eng)

    def __get_m_e_motor(self) -> np.ndarray:
        return self.__m_e_motor

    def __set_m_e_motor(self, x: np.ndarray) -> None:
        self.__m_e_motor = x

    m_e_motor = property(__get_m_e_motor, __set_m_e_motor)

    def __get_m_requ(self) -> np.ndarray:
        return self.__m_requ

    def __set_m_requ(self, x: np.ndarray) -> None:
        self.__m_requ = x

    m_requ = property(__get_m_requ, __set_m_requ)

    def __get_es_cl(self) -> np.ndarray:
        return self.__es_cl

    def __set_es_cl(self, x: np.ndarray) -> None:
        self.__es_cl = x

    es_cl = property(__get_es_cl, __set_es_cl)

    def __get_gear_cl(self) -> np.ndarray:
        return self.__gear_cl

    def __set_gear_cl(self, x: np.ndarray) -> None:
        self.__gear_cl = x

    gear_cl = property(__get_gear_cl, __set_gear_cl)

    def __get_e_rec_e_motor(self) -> np.ndarray:
        return self.__e_rec_e_motor

    def __set_e_rec_e_motor(self, x: np.ndarray) -> None:
        self.__e_rec_e_motor = x

    e_rec_e_motor = property(__get_e_rec_e_motor, __set_e_rec_e_motor)

    def __get_a_x_final(self) -> float:
        return self.__a_x_final

    def __set_a_x_final(self, x: float) -> None:
        self.__a_x_final = x

    a_x_final = property(__get_a_x_final, __set_a_x_final)

    def __get_e_rec_e_motor_max(self) -> float:
        return self.__e_rec_e_motor_max

    def __set_e_rec_e_motor_max(self, x: float) -> None:
        self.__e_rec_e_motor_max = x

    e_rec_e_motor_max = property(__get_e_rec_e_motor_max, __set_e_rec_e_motor_max)

    def __get_p_rec_max(self) -> float:
        return self.__p_rec_max

    def __set_p_rec_max(self, x: float) -> None:
        self.__p_rec_max = x

    p_rec_max = property(__get_p_rec_max, __set_p_rec_max)

    def __get_p_harvest_straight(self) -> float:
        return self.__p_harvest_straight

    def __set_p_harvest_straight(self, x: float) -> None:
        self.__p_harvest_straight = x

    p_harvest_straight = property(__get_p_harvest_straight, __set_p_harvest_straight)

    def __get_es_max(self) -> float:
        return self.__es_max

    def __set_es_max(self, x: float) -> None:
        self.__es_max = x

    es_max = property(__get_es_max, __set_es_max)

    def __get_pars_solver(self) -> dict:
        return self.__pars_solver

    def __set_pars_solver(self, x: dict) -> None:
        self.__pars_solver = x

    pars_solver = property(__get_pars_solver, __set_pars_solver)

    def __get_debug_opts(self) -> dict:
        return self.__debug_opts

    def __set_debug_opts(self, x: dict) -> None:
        self.__debug_opts = x

    debug_opts = property(__get_debug_opts, __set_debug_opts)

    def __get_fuel_cons_cl(self) -> np.ndarray:
        return self.__fuel_cons_cl

    def __set_fuel_cons_cl(self, x: np.ndarray) -> None:
        self.__fuel_cons_cl = x

    fuel_cons_cl = property(__get_fuel_cons_cl, __set_fuel_cons_cl)

    def __get_e_cons_cl(self) -> np.ndarray:
        return self.__e_cons_cl

    def __set_e_cons_cl(self, x: np.ndarray) -> None:
        self.__e_cons_cl = x

    e_cons_cl = property(__get_e_cons_cl, __set_e_cons_cl)

    def __get_tire_loads(self) -> np.ndarray:
        return self.__tire_loads

    def __set_tire_loads(self, x: np.ndarray) -> None:
        self.__tire_loads = x

    tire_loads = property(__get_tire_loads, __set_tire_loads)

    def __get_e_es_to_e_motor_max(self) -> float:
        return self.__e_es_to_e_motor_max

    def __set_e_es_to_e_motor_max(self, x: float) -> None:
        self.__e_es_to_e_motor_max = x

    e_es_to_e_motor_max = property(__get_e_es_to_e_motor_max, __set_e_es_to_e_motor_max)

    # ------------------------------------------------------------------------------------------------------------------
    # METHODS (CALCULATIONS) -------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def reset_lap(self):
        # reset driver object
        self.driverobj.reset_driver(trackobj=self.trackobj)

        # reset lap variables
        self.t_cl = np.zeros(self.trackobj.no_points_cl)
        self.vel_cl = np.zeros(self.trackobj.no_points_cl)
        self.n_cl = np.zeros(self.trackobj.no_points_cl)
        self.m_eng = np.zeros(self.trackobj.no_points)
        self.m_e_motor = np.zeros(self.trackobj.no_points)
        self.m_requ = np.zeros(self.trackobj.no_points)
        self.es_cl = np.zeros(self.trackobj.no_points_cl)
        self.gear_cl = np.zeros(self.trackobj.no_points_cl, dtype=int)
        self.e_rec_e_motor = np.zeros(self.trackobj.no_points)
        self.a_x_final = 0.0
        self.fuel_cons_cl = np.zeros(self.trackobj.no_points_cl)
        self.e_cons_cl = np.zeros(self.trackobj.no_points_cl)
        self.tire_loads = np.zeros((self.trackobj.no_points, 4))

        _vehicle_rec_max = self.driverobj.carobj.pars_engine.get("e_rec_e_motor_max", np.inf)
        self.e_rec_e_motor_max = min(_vehicle_rec_max, self.__track_e_rec_max())
        _vehicle_p_rec = self.driverobj.carobj.pars_engine.get("pow_e_motor", 350e3)
        _track_p_rec = self.trackobj.pars_track.get("p_rec_max", 350e3)
        self.p_rec_max = min(_vehicle_p_rec, _track_p_rec)
        self.p_harvest_straight = self.driverobj.carobj.pars_engine.get(
            "p_harvest_straight", self.p_rec_max
        )
        self.es_max = self.driverobj.carobj.pars_engine.get(
            "max_e_energy_storage", np.inf
        )

    def __track_e_rec_max(self) -> float:
        """[J/lap] per-track Recharge limit for the session being simulated (FIA C5.2.10).

        The baseline is 8.5 MJ per lap, but the FIA may reduce it per Competition -- to 7 MJ,
        and for Qualifying/Sprint Qualifying to no less than 4 MJ. Tracks can therefore carry
        separate `e_rec_max_qualy` and `e_rec_max_race` values in track_pars.ini; the session
        is selected with the `session` track option ("qualifying" by default). A plain
        `e_rec_max` still works as a session-independent value.
        """
        pars = self.trackobj.pars_track
        session = str(pars.get("session", "qualifying")).lower()
        key = "e_rec_max_qualy" if session.startswith("q") else "e_rec_max_race"
        return pars.get(key, pars.get("e_rec_max", 8.5e6))

    def simulate_lap(self):
        """
        Main method used to simulate a lap. It includes the required solver calls and hybrid system calculations.

        Solver calls appear in the following order (EM = LBP/LS):
        1) Initial call without EM
        2) Second call without EM starting with the proper start velocity based on previous result
        3) Loop -> at first the EM strategy is calculated based on remaining energy at end of lap from previous result,
            then solver runs. This is done until either the maximum number of iterations is reached or no more changes
            in the energy storage state appear. In the first runs we never use all the energy due to two reasons:
            1) The boost will increase the velocity and thereby reduce the time consumed in one track segment and
            therefore reduce the amount of energy consumed there. 2) Additional recuperated energy due to the increased
            velocity. This additional energy remains at the end of the lap and can be used to recalculate the EM boost
            points for the next run. We only add new boost points and do net remove old ones to avoid endless loops (we
            do not know how much energy is available for EM strategy otherwise).
            This process helps to get a good approximation without too many loop iterations, i.e. calculation time.

        Solver calls appear in the following order (EM = FCFB):
        1) Initial call with EM
        2) Second call with EM starting with the proper start velocity based on previous result
        3) If lift&coast is used: Based on the previous velocity profile, the brake points are determined. Virtual
            accelerator pedal is set to zero in front of those points. lift&coast points are determined only once
            because brake points do not change depending on the velocity profile (recalculation would lead to premature
            brake points and therefore to a new acceleration phase in between). Furthermore, this mode is meant to just
            calculate the lap time loss compared to the energy savings if enough energy is available. Otherwise it
            will of course improve the lap time instead to decrease it.
        """

        # --------------------------------------------------------------------------------------------------------------
        # CALL SOLVER (WITHOUT EM) -------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # initial solver run
        if self.debug_opts["use_print"]:
            print("-" * 50)
            print("Starting solver run (1)")

        self.__fbplus(v_start=self.pars_solver["v_start"], a_x_start=0.0)

        # use previous result to rerun the simulation with proper start velocity (if desired by user)
        if self.pars_solver["find_v_start"]:
            if self.debug_opts["use_print"]:
                print("Starting solver run (2) (considering new start velocity)")

            self.__fbplus(v_start=self.vel_cl[-1], a_x_start=self.a_x_final)

        # --------------------------------------------------------------------------------------------------------------
        # CALL SOLVER (WITH EM) ----------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # recalculation only required if EM strategy is not "pure FCFB"
        if self.driverobj.pars_driver["em_strategy"] in ["LBP", "LS", "ERSO", "QUALY"]:
            """Due to the mutual influence between velocity profile and EM strategy we need some iterations until an
            equilibrium was found."""

            # create required loop variables
            i = 0
            es_prev = 0.0
            e_rec_prev = 0.0

            strategy = self.driverobj.pars_driver["em_strategy"]
            # ERSO and QUALY use total recovery as the convergence signal because they actively control
            # harvest, so es_cl[-1] doesn't change monotonically between iterations.
            is_erso_or_qualy = strategy in ("ERSO", "QUALY")

            # ERSO/QUALY plan <-> velocity profile feedback can settle into a short limit cycle
            # instead of a fixed point. Detect a revisited plan (same deploy/harvest masks as a
            # recent iteration), stop, and re-apply the best plan seen — every plan's lap time is
            # measured under its own solved profile, so the minimum is a genuine result.
            plan_hist = []  # [(boost_mask, harvest_mask), ...] of the last few iterations
            best_t = np.inf
            best_plan = None  # (boost_mask, harvest_mask, v_start, a_x_start) for exact replay
            plan_cycled = False

            def _not_converged():
                if plan_cycled:
                    return False
                if is_erso_or_qualy:
                    return math.fabs(float(self.e_rec_e_motor.sum()) - e_rec_prev) > self.pars_solver["es_diff_max"]
                return math.fabs(self.es_cl[-1] - es_prev) > self.pars_solver["es_diff_max"]

            while _not_converged() and i < self.pars_solver["max_no_em_iters"]:
                i += 1
                es_prev = self.es_cl[-1]
                e_rec_prev = float(self.e_rec_e_motor.sum())

                if self.debug_opts["use_print"]:
                    print("Starting recalculation considering hybrid system (%i)" % i)

                # electric turbocharger recovery of the previous run (charges the ES from the
                # ICE, not subject to the MGU-K recovery cap) — sustainable deploy income
                if self.driverobj.pars_driver["use_recuperation"]:
                    e_rec_etc = (self.driverobj.carobj.pars_engine["eta_etc_re"] * 2.0 * math.pi
                                 * float(np.sum(self.n_cl[:-1] * self.m_eng * np.diff(self.t_cl))))
                else:
                    e_rec_etc = 0.0

                # calculate hybrid system application (boost) based on the previous result
                self.driverobj.calc_em_boost_use(
                    t_cl=self.t_cl,
                    vel_cl=self.vel_cl,
                    n_cl=self.n_cl,
                    m_requ=self.m_requ,
                    es_final=self.es_cl[-1],
                    e_rec_max=self.e_rec_e_motor_max,
                    p_rec_max=self.p_rec_max,
                    e_rec_actual=float(self.e_rec_e_motor.sum()),
                    # braking regen only (recovery outside the previous active-harvest mask) —
                    # exogenous income for the self-consistent ERSO harvest/deploy plan
                    e_rec_braking=float(np.sum(self.e_rec_e_motor[~self.driverobj.em_harvest_use])),
                    e_rec_etc=e_rec_etc,
                    es_initial=self.driverobj.pars_driver["initial_energy"] if strategy == "QUALY" else 0.0,
                    # both QUALY and ERSO need the curvature to keep active harvest out of
                    # corners where the driven tires have no spare grip (see ay_max_harvest)
                    kappa=self.trackobj.kappa,
                    ers_harvest_speed=self.driverobj.carobj.pars_engine.get("ers_harvest_speed_kmh", 250.0) / 3.6
                                      if strategy == "QUALY" else None,
                    # per-point recovery from the previous run, so QUALY can check that its plan
                    # is payable as the lap unfolds and not just in total
                    e_rec_profile=self.e_rec_e_motor.copy() if strategy == "QUALY" else None,
                )

                # rerun solver to get new velocity profile
                if self.pars_solver["find_v_start"]:
                    v_start_used = self.vel_cl[-1]
                    a_x_start_used = self.a_x_final
                else:
                    v_start_used = self.pars_solver["v_start"]
                    a_x_start_used = 0.0

                self.__fbplus(v_start=v_start_used, a_x_start=a_x_start_used)

                if self.debug_opts["use_print"]:
                    print("Remaining energy in ES: %.0f kJ" % (self.es_cl[-1] / 1000.0))

                if is_erso_or_qualy:
                    boost_mask = np.copy(self.driverobj.em_boost_use)
                    harvest_mask = np.copy(self.driverobj.em_harvest_use)

                    if self.t_cl[-1] < best_t:
                        best_t = float(self.t_cl[-1])
                        best_plan = (boost_mask, harvest_mask, v_start_used, a_x_start_used)

                    # revisited plan -> period-1 (converged) or short limit cycle; either way,
                    # further iterations only replay already-seen states
                    plan_cycled = any(
                        np.array_equal(boost_mask, b) and np.array_equal(harvest_mask, h)
                        for b, h in plan_hist
                    )
                    plan_hist.append((boost_mask, harvest_mask))
                    if len(plan_hist) > 3:
                        plan_hist.pop(0)

            # if the loop ended on a state other than the best visited plan, replay the best
            # plan with its original solver inputs so all outputs reproduce it exactly
            if is_erso_or_qualy and best_plan is not None and self.t_cl[-1] > best_t:
                if self.debug_opts["use_print"]:
                    print("Re-applying best EM plan (lap time %.3f s)" % best_t)

                boost_mask, harvest_mask, v_start_used, a_x_start_used = best_plan
                self.driverobj.em_boost_use = boost_mask
                self.driverobj.em_harvest_use = harvest_mask
                self.__fbplus(v_start=v_start_used, a_x_start=a_x_start_used)

        elif (
            self.driverobj.pars_driver["em_strategy"] == "FCFB"
            and self.driverobj.pars_driver["use_lift_coast"]
        ):
            """Case FCFB + lift&coast."""

            if self.debug_opts["use_print"]:
                print("Starting recalculation considering hybrid system (1)")

            # calculate hybrid system application (boost) based on the previous result
            self.driverobj.calc_em_boost_use(
                t_cl=self.t_cl,
                vel_cl=self.vel_cl,
                n_cl=self.n_cl,
                m_requ=self.m_requ,
                es_final=self.es_cl[-1],
                e_rec_max=self.e_rec_e_motor_max,
                p_rec_max=self.p_rec_max,
            )

            # rerun solver to get new velocity profile
            if self.pars_solver["find_v_start"]:
                self.__fbplus(v_start=self.vel_cl[-1], a_x_start=self.a_x_final)
            else:
                self.__fbplus(v_start=self.pars_solver["v_start"], a_x_start=0.0)

        if self.debug_opts["use_print"]:
            print("Finished solver calculations")

        # --------------------------------------------------------------------------------------------------------------
        # FUEL/ENERGY CONSUMPTION --------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        if not self.driverobj.carobj.powertrain_type == "electric":
            self.fuel_cons_cl = self.driverobj.carobj.fuel_cons(
                t_cl=self.t_cl, n_cl=self.n_cl, m_eng=self.m_eng
            )

        self.e_cons_cl = self.driverobj.carobj.e_cons(
            t_cl=self.t_cl, n_cl=self.n_cl, m_e_motor=self.m_e_motor
        )

    def __fbplus(self, v_start: float, a_x_start: float = 0.0):
        """
        Returned arrays t_cl, vel_cl, n_cl, es_cl and gear_cl are closed, the rest is unclosed.
        """

        # --------------------------------------------------------------------------------------------------------------
        # USER INPUT ---------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # loop options
        tol = self.pars_solver.get(
            "vel_tol", 1e-4
        )  # [m/s] termination criterion (must be greater than force_conv)
        # force_conv = 5e-3   # [m/s] velocity malus per iteration to force convergence (should be about half of 'tol')

        # --------------------------------------------------------------------------------------------------------------
        # INITIALIZE VARIABLES -----------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # Velocity ceiling for the forward pass: the driver's own limits (pit lane, global cap)
        # combined with the cornering limit of every point. Folding the cornering limit in here
        # is what lets the forward pass TRACK a falling ceiling through a tightening bend.
        # Previously it was only evaluated reactively in CASE 2, once the lateral grip check had
        # already failed, so the solver accelerated until it broke grip, got clamped by a
        # maximum-braking backward sweep, accelerated again -- a sawtooth of roughly +0.5 and
        # -24 m/s^2 every few metres instead of the smooth ~-6 m/s^2 lift the corner asks for.
        vel_max_corn = self.driverobj.carobj.v_max_cornering_arr(
            kappa=self.trackobj.kappa,
            mu=self.trackobj.mu,
            vel_subtr_corner=self.driverobj.pars_driver["vel_subtr_corner"],
        )
        vel_lim_pt = np.minimum(self.trackobj.vel_lim, vel_max_corn)
        vel_lim_cl = np.append(vel_lim_pt, vel_lim_pt[0])
        self.e_rec_e_motor[:] = 0.0  # must be reset for every run

        # Running total of e_rec_e_motor, maintained incrementally so the recovery-budget test
        # below does not have to re-sum the whole array. The test is asked ~121k times per lap
        # and the array is ~4600 long, which made it the second largest cost in the solver.
        #
        # The running total is NOT used as the answer: summing incrementally accumulates
        # rounding error in a different order than ndarray.sum(), so using it directly could
        # flip the comparison. It is used only to decide the question cheaply when the answer is
        # not in doubt, and the exact sum is still taken inside a guard band around the cap:
        #   running < cap - EPS  -> under budget for certain
        #   running > cap + EPS  -> over budget for certain
        #   otherwise            -> fall back to the exact e_rec_e_motor.sum()
        # so the branch taken is always the one the exact sum would have given.
        #
        # EPS = 1 kJ. Bounding the drift analytically as m * u * max|total| -- m the updates in
        # one solver pass (~2.7e4), u the double epsilon (1.1e-16), max|total| the cap (8 MJ) --
        # gives ~2.4e-5 J; measured over 271607 tests on Barcelona/MVRC_2026 the worst observed
        # drift was 4.2e-8 J, so 1 kJ leaves a margin of ~1e10. Over those same tests the fast
        # decision never disagreed with the exact one, and 0.04-1.9 % of tests fell inside the
        # band and took the exact sum. If the cap ever becomes small enough to be comparable to
        # EPS, shrink EPS rather than trusting the band.
        e_rec_running = 0.0
        e_rec_max_c = self.e_rec_e_motor_max
        e_rec_lo = e_rec_max_c - 1000.0
        e_rec_hi = e_rec_max_c + 1000.0

        # --------------------------------------------------------------------------------------------------------------
        # SET START CONDITIONS -----------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        self.vel_cl[0] = v_start
        self.es_cl[0] = self.driverobj.pars_driver["initial_energy"]

        # find gear at start
        self.gear_cl[0], self.n_cl[0] = self.driverobj.carobj.find_gear(
            vel=self.vel_cl[0]
        )

        # --------------------------------------------------------------------------------------------------------------
        # LOOP THROUGH ALL THE POINTS ----------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # cache frequently accessed attributes as local variables for performance
        vel_cl = self.vel_cl
        t_cl = self.t_cl
        n_cl = self.n_cl
        es_cl = self.es_cl
        gear_cl = self.gear_cl
        m_eng = self.m_eng
        m_e_motor = self.m_e_motor
        m_requ = self.m_requ
        e_rec_e_motor = self.e_rec_e_motor
        tire_loads = self.tire_loads
        carobj = self.driverobj.carobj
        driverobj = self.driverobj
        kappa = self.trackobj.kappa
        mu = self.trackobj.mu
        stepsize = self.trackobj.stepsize
        drs = self.trackobj.drs
        # active aero allowed: DRS zone AND curvature below threshold (closes in corners and during braking)
        # braking is handled separately — backward sweep always uses active_aero=False
        aa_kappa_threshold = active_aero_kappa_threshold(carobj.pars_general)
        active_aero = drs & (np.abs(kappa) <= aa_kappa_threshold)
        vel_lim = vel_lim_pt  # combined driver + cornering ceiling (see where it is built above)
        no_points = self.trackobj.no_points
        pars_general_m = carobj.pars_general["m"]
        pars_gearbox_e_i = carobj.pars_gearbox["e_i"]
        t_shift = carobj.pars_gearbox.get("t_shift", 0.0)
        powertrain_type = carobj.powertrain_type
        pars_solver = self.pars_solver
        pars_driver = driverobj.pars_driver
        throttle_pos = driverobj.throttle_pos
        em_boost_use = driverobj.em_boost_use
        em_harvest_use = driverobj.em_harvest_use
        pars_engine_eta_e_motor_re = carobj.pars_engine["eta_e_motor_re"]
        es_max = self.es_max
        free_harvest = pars_driver.get("use_free_harvest", True)
        # ceilings on the lateral acceleration at which active harvest / ERS deployment may be
        # applied. The strategy masks are planned from the previous iteration's velocities, so
        # they can ask for either at a point that turns out to be grip limited this time round;
        # these are the authoritative checks, against the a_y the car actually reaches.
        ay_max_harv = ay_max_harvest(pars_driver)
        ay_max_dep = ay_max_deploy(pars_driver)
        # resistance coefficients and the packed tire/aero arrays, hoisted so the resistance
        # terms and the backward-braking kernel can be called without going through the Car
        # wrappers -- they sit on the per-point path and are called 160k+ times per lap
        jit_params = carobj._jit_params
        jit_fz_data = carobj._jit_fz_data
        rho_air_c = jit_params[1]    # _RHO
        c_w_a_c = jit_params[2]      # _CWA
        f_roll_c = jit_params[3]     # _FROLL
        drs_factor_c = jit_params[4] # _DRS

        i = 0
        a_x = a_x_start

        while i < no_points:
            # calculate currently acting lateral acceleration and forces
            a_y = vel_cl[i] * vel_cl[i] * kappa[i]
            f_y_f, f_y_r = carobj.calc_lat_forces(a_y=a_y)

            # deployment is suppressed where the car is loaded up laterally: the extra force
            # only pushes it past the cornering limit and gets braked off again (see
            # ay_max_deploy). Evaluated here, against the a_y actually reached this iteration,
            # because the strategy mask was planned on the previous one.
            boost_ok = em_boost_use[i] and abs(a_y) <= ay_max_dep

            # calculate tire force potentials (using a_x = 0.0 (maximum cornering) to find out if we can stay on track)
            (
                f_x_pot_fl,
                f_y_pot_fl,
                tire_loads[i, 0],
                f_x_pot_fr,
                f_y_pot_fr,
                tire_loads[i, 1],
                f_x_pot_rl,
                f_y_pot_rl,
                tire_loads[i, 2],
                f_x_pot_rr,
                f_y_pot_rr,
                tire_loads[i, 3],
            ) = carobj.tire_force_pots(
                vel=vel_cl[i], a_x=0.0, a_y=a_y, mu=mu[i], active_aero=False
            )

            # ----------------------------------------------------------------------------------------------------------
            # CASE 1: some tire potential is left and no velocity limit is set -> accelerate ---------------------------
            # ----------------------------------------------------------------------------------------------------------

            """Axis-wise consideration of the lateral force potential makes sense because it is better to assume that
            the outer tire gets as worse as the inner tire gets better than to assume that they have to transfer the
            lateral forces according to the wheel load distribution. This would lead to an underestimation of the
            maximum possible cornering speed. A more exact model is not possible without considering slip angles, i.e.
            a kinematic vehicle model."""

            if (
                abs(f_y_f) <= f_y_pot_fl + f_y_pot_fr
                and abs(f_y_r) <= f_y_pot_rl + f_y_pot_rr
                and vel_cl[i] <= vel_lim[i]
            ):
                """Since it is obviously possible to stay on track assuming no longitudinal acceleration, we need to
                get a proper assumption of the maximum longitudinal acceleration a_x that can be handled without
                leaving the track (the connection originates in the a_x influence to the wheel loads)."""

                # obtain maximum longitudinal acceleration
                a_x_max = carobj.calc_max_ax(
                    vel=vel_cl[i],
                    a_y=a_y,
                    mu=mu[i],
                    f_y_f=f_y_f,
                    f_y_r=f_y_r,
                )

                # use maximum longitudinal acceleration as initial estimate (no carry-over from previous step)
                a_x = a_x_max

                # recalculate tire force potentials based on approximated a_x
                (
                    f_x_pot_fl,
                    f_y_pot_fl,
                    tire_loads[i, 0],
                    f_x_pot_fr,
                    f_y_pot_fr,
                    tire_loads[i, 1],
                    f_x_pot_rl,
                    f_y_pot_rl,
                    tire_loads[i, 2],
                    f_x_pot_rr,
                    f_y_pot_rr,
                    tire_loads[i, 3],
                ) = carobj.tire_force_pots(
                    vel=vel_cl[i],
                    a_x=a_x,
                    a_y=a_y,
                    mu=mu[i],
                    active_aero=active_aero[i],
                )

                # calculate remaining tire potential at front and rear axle for longitudinal force transmission
                f_x_poss = carobj.calc_f_x_pot(
                    f_x_pot_fl=f_x_pot_fl,
                    f_x_pot_fr=f_x_pot_fr,
                    f_x_pot_rl=f_x_pot_rl,
                    f_x_pot_rr=f_x_pot_rr,
                    f_y_pot_f=f_y_pot_fl + f_y_pot_fr,
                    f_y_pot_r=f_y_pot_rl + f_y_pot_rr,
                    f_y_f=f_y_f,
                    f_y_r=f_y_r,
                    force_use_all_wheels=False,
                    limit_braking_weak_side=None,
                )

                # calculate torque distribution within the hybrid system (trying to reach the possible force f_x)
                m_requ[i], m_eng[i], m_e_motor[i] = carobj.calc_torque_distr_f_x(
                    f_x=f_x_poss,
                    n=n_cl[i],
                    throttle_pos=throttle_pos[i],
                    es=es_cl[i],
                    em_boost_use=boost_ok,
                    vel=vel_cl[i],
                )

                # active harvesting: MGU-K as generator at high-speed points. Skipped where the
                # car is cornering hard — the generator torque goes through the driven axle and
                # would break the friction ellipse, which the solver would answer with a
                # grip-limited braking sweep (a far bigger lap time loss than the energy gained).
                if (
                    em_harvest_use[i]
                    and abs(a_y) <= ay_max_harv
                    and m_e_motor[i] == 0.0
                    and powertrain_type == "hybrid"
                    and pars_driver["use_recuperation"]
                    and (e_rec_running < e_rec_lo
                         or (e_rec_running <= e_rec_hi and e_rec_e_motor.sum() < e_rec_max_c))
                    and es_cl[i] < es_max
                ):
                    harvest_torque = min(carobj.torque_e_motor(n=n_cl[i]),
                                        self.p_harvest_straight / (2 * math.pi * n_cl[i]))
                    m_e_motor[i] = -harvest_torque

                # ICE-covered ("free") harvesting: where the car is grip limited the ICE has
                # spare capability. Driving the MGU-K as a generator with that surplus charges
                # the battery without costing wheel force, because the extra ICE torque cancels
                # the generator torque in the sum below. Two effects, in order:
                #   cover — the ICE picks up an active harvest that is already running, undoing
                #           its cost in longitudinal force
                #   extra — whatever headroom is left drives the generator on top, force-neutral
                # This buys energy with fuel instead of lap time, so unlike active harvesting it
                # needs no strategy decision and no economic gate.
                if (
                    free_harvest
                    and powertrain_type == "hybrid"
                    and m_e_motor[i] <= 0.0
                    and m_eng[i] > 0.0
                    and pars_driver["use_recuperation"]
                    and (e_rec_running < e_rec_lo
                         or (e_rec_running <= e_rec_hi and e_rec_e_motor.sum() < e_rec_max_c))
                    and es_cl[i] < es_max
                ):
                    spare = carobj.torque(n=n_cl[i]) - m_eng[i]
                    if spare > 0.0:
                        gen_cap = min(carobj.torque_e_motor(n=n_cl[i]),
                                      self.p_harvest_straight / (2 * math.pi * n_cl[i]))
                        h = -m_e_motor[i]
                        cover = min(spare, h)
                        m_eng[i] += cover
                        extra = min(spare - cover, gen_cap - h)
                        if extra > 0.0:
                            m_eng[i] += extra
                            m_e_motor[i] -= extra

                # calculate available acceleration force at the tire
                f_x_powert = (
                    carobj.pars_gearbox["eta_g"]
                    * (m_eng[i] + m_e_motor[i])
                    / (
                        carobj.pars_gearbox["i_trans"][gear_cl[i]]
                        * carobj.r_driven_tire(vel=vel_cl[i])
                    )
                )

                # calculate reached longitudinal acceleration (e_i accounts for rotational inertia)
                a_x_start = (
                    f_x_powert
                    - _air_res(vel_cl[i], active_aero[i], rho_air_c, c_w_a_c, drs_factor_c)
                    - (tire_loads[i, 0] + tire_loads[i, 1] + tire_loads[i, 2] + tire_loads[i, 3]) * f_roll_c
                ) / (pars_general_m * pars_gearbox_e_i[gear_cl[i]])

                # --- Heun's method: lightweight predictor-corrector ---
                # predictor: Euler step using a_x at start of interval
                v_pred = math.sqrt(max(0.0, vel_cl[i] * vel_cl[i] + 2 * a_x_start * stepsize))

                # corrector: re-evaluate only aero/rolling resistance at predicted speed
                # (skip expensive powertrain re-evaluate — use same f_x_powert)
                if i + 1 < no_points:
                    a_x_end = (
                        f_x_powert
                        - _air_res(v_pred, active_aero[i], rho_air_c, c_w_a_c, drs_factor_c)
                        - (tire_loads[i, 0] + tire_loads[i, 1] + tire_loads[i, 2] + tire_loads[i, 3]) * f_roll_c
                    ) / (pars_general_m * pars_gearbox_e_i[gear_cl[i]])

                    a_x = 0.5 * (a_x_start + a_x_end)
                else:
                    a_x = a_x_start

                # calculate velocity in the next point using corrected acceleration
                vel_cl[i + 1] = math.sqrt(max(0.0, vel_cl[i] * vel_cl[i] + 2 * a_x * stepsize))

                # consider velocity limit if reaching it during this step
                """This if statement is intended to prevent unnecessary backward loops. Therefore it should only come
                into operation if the velocity limit is reached from below (i.e. with acceleration) during the current
                step."""
                if vel_cl[i] <= vel_lim_cl[i + 1] < vel_cl[i + 1]:
                    # calculate a_x required to reach the velocity limit
                    a_x = (
                        vel_lim_cl[i + 1] * vel_lim_cl[i + 1] - vel_cl[i] * vel_cl[i]
                    ) / (2 * stepsize)

                    f_x_target = (
                        _air_res(vel_cl[i], False, rho_air_c, c_w_a_c, drs_factor_c)
                        + (tire_loads[i, 0] + tire_loads[i, 1] + tire_loads[i, 2] + tire_loads[i, 3]) * f_roll_c
                        + pars_general_m * pars_gearbox_e_i[gear_cl[i]] * a_x
                    )

                    # calculate torque distribution within the hybrid system (trying to reach the possible force f_x)
                    m_requ[i], m_eng[i], m_e_motor[i] = carobj.calc_torque_distr_f_x(
                        f_x=f_x_target,
                        n=n_cl[i],
                        throttle_pos=throttle_pos[i],
                        es=es_cl[i],
                        em_boost_use=boost_ok,
                        vel=vel_cl[i],
                    )

                    # set velocity accordingly
                    vel_cl[i + 1] = vel_lim_cl[i + 1]

                elif vel_lim_cl[i + 1] < vel_cl[i + 1]:
                    """The ceiling at the next point is BELOW the current velocity, i.e. the
                    corner is tightening and the car has to give speed back. Lifting is usually
                    enough for that -- through a fast sweeper the ceiling falls at roughly the
                    rate drag and rolling resistance slow the car anyway -- so track it by
                    coasting. Without this the point simply broke the lateral grip check and got
                    clamped by a maximum-braking backward sweep, which overshot below the
                    ceiling and let the next step accelerate again: the mid-corner sawtooth.
                    If coasting is NOT enough the ceiling stays unreachable here and the point
                    falls through to CASE 2, where real braking is planned as before."""
                    # coasting means no drive torque; any generator torque the harvest logic
                    # already placed on m_e_motor stays, so its drag counts towards the lift
                    f_x_coast = (
                        carobj.pars_gearbox["eta_g"]
                        * min(m_e_motor[i], 0.0)
                        / (
                            carobj.pars_gearbox["i_trans"][gear_cl[i]]
                            * carobj.r_driven_tire(vel=vel_cl[i])
                        )
                    )
                    a_x_coast = (
                        f_x_coast
                        - _air_res(vel_cl[i], active_aero[i], rho_air_c, c_w_a_c, drs_factor_c)
                        - (tire_loads[i, 0] + tire_loads[i, 1] + tire_loads[i, 2] + tire_loads[i, 3]) * f_roll_c
                    ) / (pars_general_m * pars_gearbox_e_i[gear_cl[i]])

                    a_x_req = (
                        vel_lim_cl[i + 1] * vel_lim_cl[i + 1] - vel_cl[i] * vel_cl[i]
                    ) / (2 * stepsize)

                    if a_x_req >= a_x_coast:
                        # closed throttle: drag, rolling resistance and any generator load do
                        # the decelerating. m_e_motor is left alone so harvest still accounts.
                        m_eng[i] = 0.0
                        m_requ[i] = min(m_e_motor[i], 0.0)
                        a_x = a_x_req
                        vel_cl[i + 1] = vel_lim_cl[i + 1]

                # check shifting -> calculate gear and rev in the next point
                gear_cl[i + 1], n_cl[i + 1] = carobj.find_gear(vel=vel_cl[i + 1])

                # analytical shift model: 25% torque during shift, velocity loss + time penalty
                # only penalize genuine upshifts that stick (not boundary oscillation)
                gear_before = gear_cl[i]
                gear_shifted = False

                if gear_cl[i + 1] > gear_before and t_shift > 0.0:
                    # net force during shift: 25% powertrain minus drag and rolling resistance
                    f_net_shift = (
                        0.25 * f_x_powert
                        - _air_res(vel_cl[i + 1], active_aero[i], rho_air_c, c_w_a_c, drs_factor_c)
                        - (tire_loads[i, 0] + tire_loads[i, 1] + tire_loads[i, 2] + tire_loads[i, 3]) * f_roll_c
                    )
                    a_shift = f_net_shift / (
                        pars_general_m * pars_gearbox_e_i[gear_before]
                    )

                    # apply velocity change over shift duration
                    vel_shifted = vel_cl[i + 1] + a_shift * t_shift

                    # check if the shift actually sticks (car stays in the higher gear)
                    new_gear, new_n = carobj.find_gear(vel=vel_shifted)
                    if new_gear > gear_before:
                        # genuine shift — apply penalty
                        vel_cl[i + 1] = max(vel_shifted, 0.1)
                        gear_cl[i + 1] = new_gear
                        n_cl[i + 1] = new_n
                        gear_shifted = True

                # calculate time at start of next point (includes shift penalty)
                t_cl[i + 1] = t_cl[i] + 2 * stepsize / (vel_cl[i] + vel_cl[i + 1])
                if gear_shifted:
                    t_cl[i + 1] += t_shift

                # calculate energy recuperated during current step by electric turbocharger in [J] (only during acc.)
                if powertrain_type == "hybrid" and pars_driver["use_recuperation"]:
                    e_rec_etc = (
                        carobj.pars_engine["eta_etc_re"]
                        * n_cl[i]
                        * m_eng[i]
                        * 2
                        * math.pi
                        * (t_cl[i + 1] - t_cl[i])
                    )
                else:
                    e_rec_etc = 0.0

                # calculate energy used/harvested by e motor during current step in [J]
                if m_e_motor[i] >= 0.0:
                    # deployment
                    e_cons_e_motor = carobj.power_demand_e_motor_drive(
                        n=n_cl[i], m_e_motor=m_e_motor[i]
                    ) * (t_cl[i + 1] - t_cl[i])
                    e_harvest = 0.0
                else:
                    # active harvesting (MGU-K as generator)
                    e_cons_e_motor = 0.0
                    e_harvest = (
                        pars_engine_eta_e_motor_re
                        * abs(m_e_motor[i])
                        * 2
                        * math.pi
                        * n_cl[i]
                        * (t_cl[i + 1] - t_cl[i])
                    )
                    e_rec_running += e_harvest - e_rec_e_motor[i]
                    e_rec_e_motor[i] = e_harvest

                # calculate changes in the hybrid energy storage [J]
                es_cl[i + 1] = es_cl[i] + e_rec_etc + e_harvest - e_cons_e_motor

                if powertrain_type != "electric" and es_cl[i + 1] < 0.0:
                    es_cl[i + 1] = 0.0

                if es_cl[i + 1] > es_max:
                    es_cl[i + 1] = es_max

                # increment
                i += 1

            # ----------------------------------------------------------------------------------------------------------
            # CASE 2: lateral force is greater than poss. tot. tire force of an axle or speed limit must be kept > brake
            # ----------------------------------------------------------------------------------------------------------

            else:
                # check if start velocity is too high
                if i == 1:
                    raise RuntimeError(
                        "Reduce start velocity! (it could be that braking would affect points within the"
                        + " previous lap)"
                    )

                # get maximum current velocity depending on speed limit or lateral acceleration
                # limit due to curvature -- both already combined into the ceiling array
                vel_cl[i] = vel_lim[i]

                # ------------------------------------------------------------------------------------------------------
                # BACKWARD ITERATIONS -> MAXIMUM CURRENT VELOCITY SHOULD BE KEPT AT CURRENT POINT i --------------------
                # ------------------------------------------------------------------------------------------------------

                j = 0
                a_x = 0.0  # reset longitudinal acceleration (almost zero during maximum cornering)
                lbs_flag = _LBS_MAP[pars_solver["limit_braking_weak_side"]]

                tire_loads_tmp = np.zeros(4)  # [N] tire loads [FL, FR, RL, RR]
                max_inner_iters = pars_solver.get("max_inner_iters", 5000)

                while True:
                    """Walk backwards from the current point, replacing each previous velocity by
                    the one maximum braking can come down from, until a point is reached that was
                    already slow enough. The velocity at each point is a fixed point of the
                    braking equation (the deceleration depends on the velocity through downforce
                    and drag) and is solved inside the _brake_back_vel() kernel."""

                    vel_tmp, a_x, counter = _brake_back_vel(
                        vel_cl[i - j], a_x, kappa[i - j - 1], mu[i - j - 1],
                        stepsize, tol, max_inner_iters,
                        jit_params, jit_fz_data, lbs_flag, tire_loads_tmp,
                    )

                    if counter > max_inner_iters:
                        raise RuntimeError(
                            f"Solver did not converge at track point i={i} (dist={i * stepsize:.1f}m)."
                            f" Inner braking loop exceeded {max_inner_iters} iterations."
                            f" Check active aero gating or car/track parameters."
                        )

                    # check if the calculated velocity is greater than the original one -> break the loop
                    if vel_tmp >= vel_cl[i - j - 1]:
                        break  # without incrementing -> i - j - 1 is last unchanged point
                    else:
                        vel_cl[i - j - 1] = vel_tmp
                        tire_loads[i - j - 1, :] = tire_loads_tmp

                    # increment to previous point
                    j += 1

                    if i - j - 1 < 0:
                        raise RuntimeError(
                            "Reduce start velocity (it could be that braking would affect points within"
                            " the previous lap)!"
                        )

                # ------------------------------------------------------------------------------------------------------
                # MODIFIED VELOCITY PROFILE IS KNOWN -> RECALCULATION OF REMAINING DATA ON THIS BASIS ------------------
                # ------------------------------------------------------------------------------------------------------

                # recalculate gears and revs for all changed points including current point
                for k in range(i - j, i + 1):
                    gear_cl[k], n_cl[k] = carobj.find_gear(vel=vel_cl[k])

                # recalculate lap times starting from the last unchanged point i - j - 1
                for k in range(i - j - 1, i):
                    t_cl[k + 1] = t_cl[k] + 2 * stepsize / (vel_cl[k + 1] + vel_cl[k])

                # recalculate energy related quantities starting from the last unchanged point i - j - 1
                for k in range(i - j - 1, i):
                    # calculate resistance force (in first point, DRS is set as it is prescribed by the track to avoid
                    # a sudden rise of the drag resistance which cannot be overcome by the available powertrain torque
                    # in some cases where the DRS was active in the forward loop, in all other points it is deactivated
                    # to consider the maximal possible deceleration that was used to calculate the velocity profile also
                    # in the energy calculations)
                    if k == i - j - 1:
                        drs_tmp = active_aero[k]
                    else:
                        drs_tmp = False

                    f_x_resi = _air_res(vel_cl[k], drs_tmp, rho_air_c, c_w_a_c, drs_factor_c) + (tire_loads[k, 0] + tire_loads[k, 1] + tire_loads[k, 2] + tire_loads[k, 3]) * f_roll_c

                    # calculate the longitudinal acceleration and force required for the given velocities
                    a_x_requ = (
                        vel_cl[k + 1] * vel_cl[k + 1] - vel_cl[k] * vel_cl[k]
                    ) / (2 * stepsize)
                    f_x_requ = pars_general_m * a_x_requ

                    # calculate force that must be provided by the powertrain (or brakes) to reach this acc. force
                    f_x_powert = f_x_requ + f_x_resi

                    # check for the two cases "engine demanded" and "engine not demanded"
                    if f_x_powert > 0.0:
                        """Engine demanded (this is the case if resistances must be overcome or if the car is
                        accelerating). Therefore, we have to recalculate the torque distribution and the energy storage
                        state."""
                        e_rec_running -= e_rec_e_motor[k]
                        e_rec_e_motor[k] = 0.0

                        # calculate torque distribution within the hybrid system (trying to reach possible force f_x)
                        m_requ[k], m_eng[k], m_e_motor[k] = (
                            carobj.calc_torque_distr_f_x(
                                f_x=f_x_powert,
                                n=n_cl[k],
                                throttle_pos=throttle_pos[k],
                                es=es_cl[k],
                                # same lateral-load gate as the forward pass, so the energy
                                # accounting here matches what was actually deployed
                                em_boost_use=(em_boost_use[k]
                                              and abs(vel_cl[k] * vel_cl[k] * kappa[k]) <= ay_max_dep),
                                vel=vel_cl[k],
                            )
                        )

                        # active harvesting in backward recalculation (energy accounting only, velocity already fixed)
                        if (
                            em_harvest_use[k]
                            and m_e_motor[k] == 0.0
                            and powertrain_type == "hybrid"
                            and pars_driver["use_recuperation"]
                            and (e_rec_running < e_rec_lo
                         or (e_rec_running <= e_rec_hi and e_rec_e_motor.sum() < e_rec_max_c))
                            and es_cl[k] < es_max
                        ):
                            harvest_torque = min(carobj.torque_e_motor(n=n_cl[k]),
                                                self.p_harvest_straight / (2 * math.pi * n_cl[k]))
                            m_e_motor[k] = -harvest_torque

                        # ICE-covered ("free") harvesting — same rule as the forward sweep, so
                        # the two passes agree on the energy balance (velocity is already fixed
                        # here, this is accounting only)
                        if (
                            free_harvest
                            and powertrain_type == "hybrid"
                            and m_e_motor[k] <= 0.0
                            and m_eng[k] > 0.0
                            and pars_driver["use_recuperation"]
                            and (e_rec_running < e_rec_lo
                         or (e_rec_running <= e_rec_hi and e_rec_e_motor.sum() < e_rec_max_c))
                            and es_cl[k] < es_max
                        ):
                            spare = carobj.torque(n=n_cl[k]) - m_eng[k]
                            if spare > 0.0:
                                gen_cap = min(carobj.torque_e_motor(n=n_cl[k]),
                                              self.p_harvest_straight / (2 * math.pi * n_cl[k]))
                                h = -m_e_motor[k]
                                cover = min(spare, h)
                                m_eng[k] += cover
                                extra = min(spare - cover, gen_cap - h)
                                if extra > 0.0:
                                    m_eng[k] += extra
                                    m_e_motor[k] -= extra

                        # check torques provided and requested
                        if m_e_motor[k] >= 0.0 and not math.isclose(
                            m_eng[k] + m_e_motor[k], m_requ[k]
                        ):
                            print(
                                "WARNING: It seems like if the requested torque could not be supplied by the"
                                + " powertrain (maybe because of energy storage changes during recalculation or because"
                                + " the throttle position was set to 0.0 during EM strategy calculation). Be aware"
                                + " that this fact is not considered and the further calculation is processed as if the"
                                + " torque was supplied! The difference amounts to %.1f Nm."
                                % (m_requ[k] - (m_eng[k] + m_e_motor[k]))
                            )

                        # calculate energy recuperated during current step by el. turbocharger in [J] (only during acc.)
                        if (
                            powertrain_type == "hybrid"
                            and pars_driver["use_recuperation"]
                        ):
                            e_rec_etc = (
                                carobj.pars_engine["eta_etc_re"]
                                * n_cl[k]
                                * m_eng[k]
                                * 2
                                * math.pi
                                * (t_cl[k + 1] - t_cl[k])
                            )
                        else:
                            e_rec_etc = 0.0

                        # calculate energy used/harvested by e motor during current step in [J]
                        if m_e_motor[k] >= 0.0:
                            e_cons_e_motor = carobj.power_demand_e_motor_drive(
                                n=n_cl[k], m_e_motor=m_e_motor[k]
                            ) * (t_cl[k + 1] - t_cl[k])
                            e_harvest = 0.0
                        else:
                            e_cons_e_motor = 0.0
                            e_harvest = (
                                pars_engine_eta_e_motor_re
                                * abs(m_e_motor[k])
                                * 2
                                * math.pi
                                * n_cl[k]
                                * (t_cl[k + 1] - t_cl[k])
                            )
                            e_rec_running += e_harvest - e_rec_e_motor[k]
                            e_rec_e_motor[k] = e_harvest

                        # calculate changes in the hybrid energy storage [J]
                        es_cl[k + 1] = es_cl[k] + e_rec_etc + e_harvest - e_cons_e_motor

                        if powertrain_type != "electric" and es_cl[k + 1] < 0.0:
                            es_cl[k + 1] = 0.0

                        if es_cl[k + 1] > es_max:
                            es_cl[k + 1] = es_max

                    else:
                        """Engine not demanded -> no powertrain torques required. However, we have to calculate kinetic
                        energy recuperation."""
                        m_eng[k] = 0.0
                        m_e_motor[k] = 0.0
                        m_requ[k] = 0.0

                        # energy recuperation by e motor in [J] under the assumption of e motor being able to recuperate
                        # all kinetic energy remaining after subtraction of the resistances
                        if (
                            (e_rec_running < e_rec_lo
                             or (e_rec_running <= e_rec_hi
                                 and e_rec_e_motor.sum() < e_rec_max_c))
                            and pars_driver["use_recuperation"]
                        ):
                            e_rec_old = e_rec_e_motor[k]
                            e_rec_e_motor[k] = min(
                                carobj.pars_engine["eta_e_motor_re"]
                                * abs(f_x_powert)
                                * stepsize,
                                self.p_rec_max * stepsize / vel_cl[k],
                            )
                            e_rec_running += e_rec_e_motor[k] - e_rec_old

                        else:
                            e_rec_running -= e_rec_e_motor[k]
                            e_rec_e_motor[k] = 0.0

                        # update energy storage (no energy harvested in el. turbocharger while engine is not demanded)
                        es_cl[k + 1] = es_cl[k] + e_rec_e_motor[k]

                        if es_cl[k + 1] > es_max:
                            es_cl[k + 1] = es_max

                # reset longitudinal acceleration for next step (almost zero during maximum cornering)
                a_x = 0.0

        # --------------------------------------------------------------------------------------------------------------
        # PREPARE FOR RETURN -------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # return real instead of index based gear values
        self.gear_cl = self.gear_cl + 1

        # save final a_x value for a possible recalculation of the lap
        self.a_x_final = a_x

    # ------------------------------------------------------------------------------------------------------------------
    # METHODS (ANALYSIS) -----------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def plot_lat_acc(self):
        a_y_tmp = np.power(self.vel_cl[:-1], 2) * self.trackobj.kappa
        if self.driverobj.carobj.powertrain_type in ("hybrid", "combustion"):
            a_y_valid = 50.0
        else:
            a_y_valid = 30.0

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(self.trackobj.dists_cl[:-1], a_y_tmp)
        ax.axhline(
            y=-a_y_valid, color="k", linestyle="--", linewidth=3.0
        )  # valid lateral acceleration limit
        ax.axhline(
            y=a_y_valid, color="k", linestyle="--", linewidth=3.0
        )  # valid lateral acceleration limit
        ax.set_title("Lateral acceleration profile")
        ax.set_xlabel("distance s in m")
        ax.set_ylabel("lateral acceleration ay in m/s2")
        plt.grid()
        plt.show()

    def plot_torques(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(self.trackobj.dists_cl[:-1], self.m_eng)
        plt.plot(self.trackobj.dists_cl[:-1], self.m_e_motor)
        plt.plot(self.trackobj.dists_cl[:-1], self.m_eng + self.m_e_motor)
        plt.plot(self.trackobj.dists_cl[:-1], self.m_requ)
        ax.set_title("Provided and requested (i.e. transmittable by the tires) torque")
        ax.set_xlabel("distance s in m")
        ax.set_ylabel("torque in Nm")
        plt.legend(
            ["combustion engine", "electric motor", "powertrain total", "requested"]
        )
        plt.grid()
        plt.show()

    def plot_tire_loads(self):
        f_z_stat_avg = (
            0.25
            * self.driverobj.carobj.pars_general["m"]
            * self.driverobj.carobj.pars_general["g"]
        )
        if self.driverobj.carobj.powertrain_type in ("hybrid", "combustion"):
            f_z_dyn_valid = f_z_stat_avg * 5.0
            legend_text = "5 * avg. static"
        else:
            f_z_dyn_valid = f_z_stat_avg * 3.0
            legend_text = "3 * avg. static"

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(self.trackobj.dists_cl[:-1], self.tire_loads[:, 0])
        plt.plot(self.trackobj.dists_cl[:-1], self.tire_loads[:, 1])
        plt.plot(self.trackobj.dists_cl[:-1], self.tire_loads[:, 2])
        plt.plot(self.trackobj.dists_cl[:-1], self.tire_loads[:, 3])
        # plt.plot(self.trackobj.dists_cl[:-1], np.sum(self.tire_loads, axis=1))
        ax.axhline(
            y=f_z_stat_avg, color="k", linestyle="--", linewidth=3.0
        )  # valid tire load range
        ax.axhline(
            y=f_z_dyn_valid, color="k", linestyle="--", linewidth=3.0
        )  # valid tire load range
        ax.set_title("Tire loads")
        ax.set_xlabel("distance s in m")
        ax.set_ylabel("tire load F_z in N")
        plt.legend(
            [
                "front left",
                "front right",
                "rear left",
                "rear right",
                "avg. static",
                legend_text,
            ]
        )
        # plt.legend(["front left", "front right", "rear left", "rear right", "total", "avg. static",
        #             "7 * avg. static"])
        plt.grid()
        plt.show()

    def plot_aero_forces(self):
        c_z_a_f = self.driverobj.carobj.pars_general["c_z_a_f"]
        c_z_a_r = self.driverobj.carobj.pars_general["c_z_a_r"]
        c_w_a = self.driverobj.carobj.pars_general["c_w_a"]
        rho_air = self.driverobj.carobj.pars_general["rho_air"]

        f_x_aero = 0.5 * c_w_a * rho_air * np.power(self.vel_cl[:-1], 2)
        f_z_aero = 0.5 * (c_z_a_f + c_z_a_r) * rho_air * np.power(self.vel_cl[:-1], 2)

        if self.driverobj.carobj.powertrain_type in ("hybrid", "combustion"):
            f_z_aero_valid = 18e3
        else:
            f_z_aero_valid = 6e3

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(self.trackobj.dists_cl[:-1], f_x_aero)
        plt.plot(self.trackobj.dists_cl[:-1], f_z_aero)
        ax.axhline(
            y=f_z_aero_valid, color="k", linestyle="--", linewidth=3.0
        )  # valid downforce range
        ax.set_title("Aero forces (no DRS considered!)")
        ax.set_xlabel("distance s in m")
        ax.set_ylabel("amplitude in N")
        plt.legend(["drag", "downforce", "valid downforce range"])
        plt.grid()
        plt.show()

    def plot_enginespeed_gears(self):
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
        fig.suptitle("Engine speed and gear selection")

        ax1.plot(self.trackobj.dists_cl[:-1], self.n_cl[:-1] * 60.0)
        ax1.set_xlabel("distance s in m")
        ax1.set_ylabel("engine speed in 1/min")
        ax1.grid()

        ax2.plot(self.trackobj.dists_cl[:-1], self.gear_cl[:-1])
        ax2.set_xlabel("distance s in m")
        ax2.set_ylabel("gear in -")
        ax2.grid()

        plt.show()

    def plot_overview(self):
        # set bigger font size
        plt.rcParams["font.size"] = 16.0

        # create figure
        plt.figure(1, figsize=(12.0, 9.0))

        # --------------------------------------------------------------------------------------------------------------
        # VELOCITY PROFILE ---------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # plot velocity profile (two columns)
        plt.subplot(2, 1, 1)
        plt.plot(self.trackobj.dists_cl, self.vel_cl * 3.6, "k-")
        plt.xlabel("distance in m")
        plt.ylabel("velocity in km/h")
        plt.grid()

        # plot global title
        title = "lap time: %.3f s" % self.t_cl[-1]
        plt.title(title)

        # --------------------------------------------------------------------------------------------------------------
        # TRACK MAP ----------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        ax1 = plt.subplot(2, 2, 3)

        # plot raceline
        ax1.plot(self.trackobj.raceline[:, 0], self.trackobj.raceline[:, 1], "k-")

        # plot DRS / active aero zones
        for a_idx, d_idx in zip(
            self.trackobj.zone_inds["drs_zone_act"],
            self.trackobj.zone_inds["drs_zone_deact"],
        ):
            if a_idx < d_idx:
                ax1.plot(
                    self.trackobj.raceline[a_idx:d_idx, 0],
                    self.trackobj.raceline[a_idx:d_idx, 1],
                    "g--",
                    linewidth=3.0,
                )
            else:
                # zone wraps around start/finish line
                ax1.plot(
                    self.trackobj.raceline[a_idx:, 0],
                    self.trackobj.raceline[a_idx:, 1],
                    "g--",
                    linewidth=3.0,
                )
                ax1.plot(
                    self.trackobj.raceline[:d_idx, 0],
                    self.trackobj.raceline[:d_idx, 1],
                    "g--",
                    linewidth=3.0,
                )

        # plot yellow sectors
        if self.driverobj.pars_driver["yellow_s1"]:
            ax1.plot(
                self.trackobj.raceline[: self.trackobj.zone_inds["s12"], 0],
                self.trackobj.raceline[: self.trackobj.zone_inds["s12"], 1],
                "y--",
            )
        if self.driverobj.pars_driver["yellow_s2"]:
            ax1.plot(
                self.trackobj.raceline[
                    self.trackobj.zone_inds["s12"] : self.trackobj.zone_inds["s23"], 0
                ],
                self.trackobj.raceline[
                    self.trackobj.zone_inds["s12"] : self.trackobj.zone_inds["s23"], 1
                ],
                "y--",
            )
        if self.driverobj.pars_driver["yellow_s3"]:
            ax1.plot(
                self.trackobj.raceline[self.trackobj.zone_inds["s23"] :, 0],
                self.trackobj.raceline[self.trackobj.zone_inds["s23"] :, 1],
                "y--",
            )

        # plot pit
        if self.trackobj.pars_track["use_pit"]:
            ax1.plot(
                self.trackobj.raceline[: self.trackobj.zone_inds["pit_out"], 0],
                self.trackobj.raceline[: self.trackobj.zone_inds["pit_out"], 1],
                "r--",
                linewidth=3.0,
            )
            ax1.plot(
                self.trackobj.raceline[self.trackobj.zone_inds["pit_in"] :, 0],
                self.trackobj.raceline[self.trackobj.zone_inds["pit_in"] :, 1],
                "r--",
                linewidth=3.0,
            )

        # plot arrow showing the driving direction
        ax1.arrow(
            self.trackobj.raceline[0, 0],
            self.trackobj.raceline[0, 1],
            self.trackobj.raceline[10, 0] - self.trackobj.raceline[0, 0],
            self.trackobj.raceline[10, 1] - self.trackobj.raceline[0, 1],
            head_width=30.0,
            width=10.0,
        )

        # plot dots at start/finish and at the sector boundaries
        ax1.plot(
            self.trackobj.raceline[0, 0],
            self.trackobj.raceline[0, 1],
            "k.",
            markersize=13.0,
        )
        ax1.plot(
            self.trackobj.raceline[self.trackobj.zone_inds["s12"], 0],
            self.trackobj.raceline[self.trackobj.zone_inds["s12"], 1],
            "k.",
            markersize=13.0,
        )
        ax1.plot(
            self.trackobj.raceline[self.trackobj.zone_inds["s23"], 0],
            self.trackobj.raceline[self.trackobj.zone_inds["s23"], 1],
            "k.",
            markersize=13.0,
        )

        # plot velocity dependent colors into track map
        cmap = plt.get_cmap("RdYlGn")
        normalize = plt.Normalize(
            vmin=-np.amax(self.vel_cl), vmax=-np.amin(self.vel_cl)
        )
        colors = [cmap(normalize(cur_vel)) for cur_vel in -self.vel_cl[:-1]]
        ax1.scatter(
            self.trackobj.raceline[:, 0], self.trackobj.raceline[:, 1], c=colors, s=5
        )

        plt.axis("equal")
        plt.title("track map: " + self.trackobj.pars_track["trackname"])
        plt.xlabel("x in m")
        plt.ylabel("y in m")
        plt.grid()

        # --------------------------------------------------------------------------------------------------------------
        # ENERGY CONSUMPTION -------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        ax1 = plt.subplot(2, 2, 4)
        ax1.plot(self.trackobj.dists_cl, self.es_cl / 1000.0, "k-")  # [J] -> [kJ]

        if not self.driverobj.carobj.powertrain_type == "electric":
            ax2 = ax1.twinx()
            ax2.plot(self.trackobj.dists_cl, self.fuel_cons_cl, "r-")
            ax2.set_ylabel("cumulated fuel consumption in kg", color="r")
            ax2.tick_params("y", colors="r")

        plt.title("consumption")
        ax1.set_xlabel("distance in m")
        ax1.set_ylabel("energy in kJ")
        plt.grid()

        # set tight plot layout and show plot
        plt.tight_layout()
        plt.show()

        # reset font size
        plt.rcParams["font.size"] = 10.0

    def plot_revs_gears(self):
        # --------------------------------------------------------------------------------------------------------------
        # REVS AND GEARS -----------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        fig, ax1 = plt.subplots(figsize=(12.0, 9.0))

        ax1.plot(self.trackobj.dists_cl, self.n_cl * 60.0)
        ax1.set_xlabel("distance in m")
        ax1.set_ylabel("rev in 1/min")

        ax2 = ax1.twinx()
        ax2.plot(self.trackobj.dists_cl, self.gear_cl, "r")
        ax2.set_ylabel("gear")

        fig.tight_layout()
        plt.show()

    def plot_energy_management(self):
        dists = self.trackobj.dists_cl[:-1]

        # e-motor power: positive = deploy, negative = harvest
        p_e_motor = 2 * math.pi * self.n_cl[:-1] * self.m_e_motor  # [W]

        # cumulative harvested energy (from e_rec_e_motor which tracks both braking and active harvest)
        e_harvest_cum = np.cumsum(self.e_rec_e_motor) / 1000.0  # [kJ]

        fig, (ax1, ax2, ax3) = plt.subplots(
            nrows=3, ncols=1, sharex=True, figsize=(12.0, 9.0)
        )
        fig.suptitle("Energy management")

        # subplot 1: energy storage state
        ax1.plot(self.trackobj.dists_cl, self.es_cl / 1000.0, "k-")
        ax1.set_ylabel("ES state in kJ")
        ax1.grid()

        # subplot 2: e-motor power with deploy/harvest coloring
        ax2.fill_between(
            dists,
            p_e_motor / 1000.0,
            0,
            where=p_e_motor >= 0,
            color="C0",
            alpha=0.7,
            label="deploy",
        )
        ax2.fill_between(
            dists,
            p_e_motor / 1000.0,
            0,
            where=p_e_motor < 0,
            color="C3",
            alpha=0.7,
            label="harvest",
        )
        ax2.set_ylabel("e-motor power in kW")
        ax2.legend(loc="upper right")
        ax2.grid()

        # subplot 3: cumulative harvested energy
        ax3.plot(dists, e_harvest_cum, "C3-")
        ax3.set_xlabel("distance in m")
        ax3.set_ylabel("cumulated harvest in kJ")
        ax3.grid()

        plt.tight_layout()
        plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# TESTING --------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    pass
