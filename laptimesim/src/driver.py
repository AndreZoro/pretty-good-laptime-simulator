import numpy as np
import math
import heapq
from laptimesim.src.car_hybrid import CarHybrid
from laptimesim.src.car_electric import CarElectric
from laptimesim.src.track import Track


class Driver(object):
    """
    author:
    Alexander Heilmeier (based on the term thesis of Maximilian Geisslinger)

    date:
    25.12.2018

    .. description::
    The file provides functions related to the energy management strategy. Therefore, it determines when the hybrid
    system is used during a lap.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # SLOTS ------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    __slots__ = ("__carobj",
                 "__pars_driver",
                 "__em_boost_use",
                 "__em_harvest_use",
                 "__throttle_pos",
                 "__no_points_lac")

    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, carobj: CarHybrid or CarElectric, pars_driver: dict, trackobj: Track,
                 stepsize: float = 5.0):
        """stepsize must only be supplied for lift and coast strategy."""

        # save car object and parameters
        self.carobj = carobj
        self.pars_driver = pars_driver

        # --------------------------------------------------------------------------------------------------------------
        # ENERGY MANAGEMENT --------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # set initial energy management strategy -> em_boost_use contains where e_motor boost can be applied
        if self.pars_driver["em_strategy"] == "FCFB":
            self.em_boost_use = np.full(trackobj.no_points, True)
        elif self.pars_driver["em_strategy"] in ["LBP", "LS", "ERSO", "QUALY", "NONE"]:
            self.em_boost_use = np.full(trackobj.no_points, False)
        else:
            raise IOError("Unknown energy management strategy!")

        # em_harvest_use contains where active harvesting (MGU-K as generator) should be applied
        self.em_harvest_use = np.full(trackobj.no_points, False)

        # calculate number of points in front of a braking point without throttle (lac = lift and coast)
        if self.pars_driver["use_lift_coast"]:
            self.no_points_lac = max(int(round(self.pars_driver["lift_coast_dist"] / stepsize)), 1)
        else:
            self.no_points_lac = 0

        # --------------------------------------------------------------------------------------------------------------
        # THROTTLE POSITION --------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # initialize array containing the throttle actuation for the consideration of yellow flags. Furthermore, it is
        # used for "lift and coast" consideration later on.
        self.throttle_pos = np.ones(trackobj.no_points)

        # set reduced throttle for yellow flags
        if any((self.pars_driver["yellow_s1"], self.pars_driver["yellow_s2"], self.pars_driver["yellow_s3"])):
            self.__set_yellow_throttle(trackobj=trackobj)

    # ------------------------------------------------------------------------------------------------------------------
    # GETTERS / SETTERS ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __get_carobj(self) -> CarHybrid or CarElectric: return self.__carobj
    def __set_carobj(self, x: CarHybrid or CarElectric) -> None: self.__carobj = x
    carobj = property(__get_carobj, __set_carobj)

    def __get_pars_driver(self) -> dict: return self.__pars_driver
    def __set_pars_driver(self, x: dict) -> None: self.__pars_driver = x
    pars_driver = property(__get_pars_driver, __set_pars_driver)

    def __get_em_boost_use(self) -> np.ndarray: return self.__em_boost_use
    def __set_em_boost_use(self, x: np.ndarray) -> None: self.__em_boost_use = x
    em_boost_use = property(__get_em_boost_use, __set_em_boost_use)

    def __get_em_harvest_use(self) -> np.ndarray: return self.__em_harvest_use
    def __set_em_harvest_use(self, x: np.ndarray) -> None: self.__em_harvest_use = x
    em_harvest_use = property(__get_em_harvest_use, __set_em_harvest_use)

    def __get_throttle_pos(self) -> np.ndarray: return self.__throttle_pos
    def __set_throttle_pos(self, x: np.ndarray) -> None: self.__throttle_pos = x
    throttle_pos = property(__get_throttle_pos, __set_throttle_pos)

    def __get_no_points_lac(self) -> int: return self.__no_points_lac
    def __set_no_points_lac(self, x: int) -> None: self.__no_points_lac = x
    no_points_lac = property(__get_no_points_lac, __set_no_points_lac)

    # ------------------------------------------------------------------------------------------------------------------
    # METHODS (CALCULATIONS) -------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def reset_driver(self, trackobj: Track):
        """Deployed into a function to be able to also reset the driver object during the simulations."""

        if self.pars_driver["em_strategy"] == "FCFB":
            self.em_boost_use = np.full(trackobj.no_points, True)
        elif self.pars_driver["em_strategy"] in ["LBP", "LS", "ERSO", "QUALY", "NONE"]:
            self.em_boost_use = np.full(trackobj.no_points, False)
        else:
            raise IOError("Unknown energy management strategy!")

        self.em_harvest_use = np.full(trackobj.no_points, False)
        self.throttle_pos = np.ones(trackobj.no_points)

        if any((self.pars_driver["yellow_s1"], self.pars_driver["yellow_s2"], self.pars_driver["yellow_s3"])):
            self.__set_yellow_throttle(trackobj=trackobj)

    def __set_yellow_throttle(self, trackobj: Track):
        if self.pars_driver["yellow_s1"]:
            self.throttle_pos[:trackobj.zone_inds["s12"]] = self.pars_driver["yellow_throttle"]

        if self.pars_driver["yellow_s2"]:
            self.throttle_pos[trackobj.zone_inds["s12"]:trackobj.zone_inds["s23"]] = self.pars_driver["yellow_throttle"]

        if self.pars_driver["yellow_s3"]:
            self.throttle_pos[trackobj.zone_inds["s23"]:] = self.pars_driver["yellow_throttle"]

    def calc_em_boost_use(self, t_cl: np.ndarray, vel_cl: np.ndarray, n_cl: np.ndarray, m_requ: np.ndarray,
                          es_final: float, e_rec_max: float = 8e6, p_rec_max: float = 350e3,
                          e_rec_actual: float = None, e_rec_braking: float = None, e_rec_etc: float = 0.0,
                          es_initial: float = 0.0,
                          kappa: np.ndarray = None, ers_harvest_speed: float = None,
                          e_rec_profile: np.ndarray = None):
        if self.pars_driver["em_strategy"] == "LBP":
            self.__strategy_lbp(t_cl=t_cl,
                                vel_cl=vel_cl,
                                n_cl=n_cl,
                                m_requ=m_requ,
                                es_final=es_final)

        elif self.pars_driver["em_strategy"] == "LS":
            self.__strategy_ls(t_cl=t_cl,
                               vel_cl=vel_cl,
                               n_cl=n_cl,
                               m_requ=m_requ,
                               es_final=es_final)

        elif self.pars_driver["em_strategy"] == "ERSO":
            self.__strategy_erso(t_cl=t_cl,
                                 vel_cl=vel_cl,
                                 n_cl=n_cl,
                                 m_requ=m_requ,
                                 es_final=es_final,
                                 e_rec_max=e_rec_max,
                                 p_rec_max=p_rec_max,
                                 e_rec_actual=e_rec_actual,
                                 e_rec_braking=e_rec_braking,
                                 e_rec_etc=e_rec_etc)

        elif self.pars_driver["em_strategy"] == "QUALY":
            self.__strategy_qualy(t_cl=t_cl,
                                  vel_cl=vel_cl,
                                  n_cl=n_cl,
                                  m_requ=m_requ,
                                  es_initial=es_initial,
                                  e_rec_max=e_rec_max,
                                  p_rec_max=p_rec_max,
                                  e_rec_actual=e_rec_actual,
                                  e_rec_etc=e_rec_etc,
                                  kappa=kappa,
                                  ers_harvest_speed=ers_harvest_speed,
                                  e_rec_profile=e_rec_profile)

        elif self.pars_driver["em_strategy"] == "FCFB" and self.pars_driver["use_lift_coast"]:
            # set array where throttle is 0.0 when driving in lift and coast condition
            self.__lift_coast(vel_cl=vel_cl,
                              n_lac=self.no_points_lac)

        else:
            raise IOError("EM strategy not considered!")

    def __strategy_lbp(self, t_cl: np.ndarray, vel_cl: np.ndarray, n_cl: np.ndarray, m_requ: np.ndarray,
                       es_final: float):
        """lbp = longest time to (next) brakepoint. The approximation of the ES state during this calcultion should be
        on the conservative side as the recalculated velocity profil will be faster and therefore the times of
        appliance will get shorter."""

        # input check: energy store
        if es_final < 0.0:
            print("WARNING: ES charge state already negative when entering EM strategy calculation!")

        # find indices of brake points
        inds_brake = np.where(np.diff(vel_cl) < 0.0)[0]

        # calculate time until next brake point for every point (0.0 for brake points themself)
        no_points = t_cl.size - 1  # - 1 to get number of points for unclosed lap
        t_until_brake = np.zeros(no_points)

        for i in range(no_points):
            if i <= inds_brake[-1]:
                ind_brake_rel = inds_brake[np.searchsorted(inds_brake, i)]
                t_until_brake[i] = t_cl[ind_brake_rel] - t_cl[i]
            else:
                ind_brake_rel = inds_brake[0]
                t_until_brake[i] = t_cl[-1] - t_cl[i] + t_cl[ind_brake_rel]

        # sort t_until_brake and get indices (minus sign to sort in a descending order)
        inds_sorted = np.argsort(-t_until_brake)
        sorted_idx = 0

        while es_final > 0.0 and sorted_idx < len(inds_sorted):
            # get current index
            ind_cur = inds_sorted[sorted_idx]
            sorted_idx += 1

            # check if a brake point would be used (case when too much energy available) -> if so break
            if math.isclose(t_until_brake[ind_cur], 0.0):
                self.em_boost_use = np.full(no_points, True)
                break

            # apply boost if boost was not applied here so far
            if not self.em_boost_use[ind_cur]:
                # apply boost here
                self.em_boost_use[ind_cur] = True

                # calculate torque distribution within the hybrid system
                m_e_motor = self.carobj.calc_torque_distr(n=n_cl[ind_cur],
                                                          m_requ=m_requ[ind_cur],
                                                          throttle_pos=self.throttle_pos[ind_cur],
                                                          es=np.inf,
                                                          em_boost_use=True,
                                                          vel=vel_cl[ind_cur])[1]

                # update energy store status (approximation because velocity profile is influenced obviously)
                es_final -= (self.carobj.power_demand_e_motor_drive(n=n_cl[ind_cur],
                                                                    m_e_motor=np.array(m_e_motor))
                             * (t_cl[ind_cur + 1] - t_cl[ind_cur]))

    def __strategy_ls(self, t_cl: np.ndarray, vel_cl: np.ndarray, n_cl: np.ndarray, m_requ: np.ndarray,
                      es_final: float):
        """ls = lowest speed. The approximation of the ES state during this calcultion should be on the conservative
        side as the recalculated velocity profil will be faster and therefore the times of appliance will get
        shorter."""

        # input check: energy store
        if es_final < 0.0:
            print("WARNING: ES charge state already negative when entering EM strategy calculation!")

        # sort vel and get indices
        inds_sorted = np.argsort(vel_cl[:-1])
        sorted_idx = 0

        while es_final > 0.0 and sorted_idx < len(inds_sorted):
            # get current index
            ind_cur = inds_sorted[sorted_idx]
            sorted_idx += 1

            # apply boost if boost was not applied here so far
            if not self.em_boost_use[ind_cur]:
                # apply boost here
                self.em_boost_use[ind_cur] = True

                # calculate torque distribution within the hybrid system
                m_e_motor = self.carobj.calc_torque_distr(n=n_cl[ind_cur],
                                                          m_requ=m_requ[ind_cur],
                                                          throttle_pos=self.throttle_pos[ind_cur],
                                                          es=np.inf,
                                                          em_boost_use=True,
                                                          vel=vel_cl[ind_cur])[1]

                # update energy store status (approximation because velocity profile is influenced obviously)
                es_final -= (self.carobj.power_demand_e_motor_drive(n=n_cl[ind_cur],
                                                                    m_e_motor=np.array(m_e_motor))
                             * (t_cl[ind_cur + 1] - t_cl[ind_cur]))

    @staticmethod
    def __time_to_next_event(t_cl: np.ndarray, event_mask: np.ndarray) -> np.ndarray:
        """Time until the next braking/corner event for every unclosed point (lap is cyclic).

        This is the persistence horizon of any force applied at a point: a speed gain
        (deploy) or deficit (harvest drag) is carried until the car brakes or enters a
        sharp corner anyway, after which it is absorbed. The per-Joule lap-time value of
        force at point i is therefore ~ tau_i / v_i^2 — this single quantity prices both
        deployment (gain) and active harvest (cost)."""

        no_points = event_mask.size
        t_u = t_cl[:no_points]
        t_lap = float(t_cl[-1])
        evt_inds = np.flatnonzero(event_mask)
        if evt_inds.size == 0:
            return np.full(no_points, t_lap)

        evt_times = t_u[evt_inds]
        pos = np.searchsorted(evt_times, t_u)
        next_evt_t = np.where(pos < evt_inds.size,
                              evt_times[np.minimum(pos, evt_inds.size - 1)],
                              evt_times[0] + t_lap)
        return next_evt_t - t_u

    def __strategy_erso(self, t_cl: np.ndarray, vel_cl: np.ndarray, n_cl: np.ndarray, m_requ: np.ndarray,
                        es_final: float, e_rec_max: float = 8e6, p_rec_max: float = 350e3,
                        e_rec_actual: float = None, e_rec_braking: float = None, e_rec_etc: float = 0.0):
        """erso = ERS-optimized, charge-sustaining race strategy. Every point is priced by its
        persistence-weighted per-Joule value tau / v^2 (see __time_to_next_event). A single
        bisection finds the price threshold T at which planned spending equals planned income:

            E_deploy(T) = min(e_rec_braking + E_harvest(T), e_rec_max)

        Deploy where value >= T; actively harvest at eligible non-deploy points where the drag
        cost (the same tau / v^2) is below eta_e_motor_re * T, i.e. the recovered energy
        redeploys at a profit. E_deploy falls and E_harvest rises with T, so the equilibrium is
        unique. Braking regen (e_rec_braking, from the previous solver run) is free income — it
        happens anyway; the active-harvest income is planned self-consistently WITHIN this call
        rather than read back from the previous run, which removes the lagged budget feedback
        between EM iterations that caused lap-time oscillation. If braking regen alone covers
        deployment everywhere, active harvest is worthless and is skipped entirely."""

        no_points = t_cl.size - 1
        dt = np.diff(t_cl)

        has_speed_limit = hasattr(self.carobj, 'pow_e_motor_max') and \
            self.carobj.pars_engine.get("ers_speed_limit", False)
        pow_e = self.carobj.pars_engine["pow_e_motor"]

        # available ERS power (zero above the speed limit -> point cannot deploy)
        if has_speed_limit:
            pow_avail = np.array([self.carobj.pow_e_motor_max(vel_cl[i]) for i in range(no_points)])
        else:
            pow_avail = np.full(no_points, pow_e)

        # persistence horizon: force effects last until the next braking event
        # (decel well beyond aero drag, ~1 g)
        is_braking = (np.diff(vel_cl) / dt) < -15.0
        tau = self.__time_to_next_event(t_cl, is_braking)

        # per-Joule value of force at each point: a speed change is worth ~ tau_i / v_i^2 s/J.
        # The same quantity prices deployment (gain) and active harvest drag (cost).
        vel_u = np.maximum(vel_cl[:no_points], 1.0)
        value = tau / vel_u ** 2
        deploy_score = np.where(pow_avail > 0.0, value, 0.0)

        # precompute actual energy consumed per step if deploying (uses real motor torque model)
        e_deploy_step = np.zeros(no_points)
        for i in range(no_points):
            if deploy_score[i] > 0.0:
                m_e_motor_val = self.carobj.calc_torque_distr(n=n_cl[i],
                                                              m_requ=m_requ[i],
                                                              throttle_pos=self.throttle_pos[i],
                                                              es=np.inf,
                                                              em_boost_use=True,
                                                              vel=vel_cl[i])[1]
                e_deploy_step[i] = (self.carobj.power_demand_e_motor_drive(n=n_cl[i],
                                                                            m_e_motor=np.array(m_e_motor_val))
                                    * dt[i])

        # harvest eligibility
        harvest_speed_min = self.pars_driver.get("ers_harvest_speed_min", 150.0 / 3.6)  # [m/s]
        harvest_eligible = vel_cl[:no_points] >= harvest_speed_min

        # planned battery energy per step when actively harvesting (mirrors the solver's
        # harvest model: MGU-K as generator at p_harvest_straight, recovered at eta_e_motor_re)
        eta_re = self.carobj.pars_engine.get("eta_e_motor_re", 0.0)
        p_harvest = self.carobj.pars_engine.get("p_harvest_straight", p_rec_max)
        e_harvest_step = np.where(harvest_eligible, eta_re * p_harvest * dt, 0.0)

        # free income: braking regen (MGU-K, subject to e_rec_max) plus electric turbocharger
        # recovery (charges the ES from the ICE, not capped) — both from the previous solver
        # run, recovered regardless of this plan. Fall back to total recovery / es_final if
        # the regen split is not available.
        if e_rec_braking is not None:
            e_regen = min(e_rec_braking, e_rec_max)
        elif e_rec_actual is not None:
            e_regen = min(e_rec_actual, e_rec_max)
        else:
            e_regen = max(0.0, es_final)

        # deploy saturated: free income alone covers deployment at every scoring point, so
        # actively harvested energy could never be spent — deploy everywhere, no harvest
        if float(np.sum(e_deploy_step)) <= e_regen + e_rec_etc:
            self.em_boost_use[:no_points] = deploy_score > 0.0
            self.em_harvest_use[:no_points] = False
            return

        # bisection on the price threshold T for the self-consistent equilibrium
        # E_deploy(T) = min(e_regen + E_harvest(T), e_rec_max). Deploy points (value >= T) are
        # corner exits (low speed, long persistence); harvest points (value <= eta_re * T) are
        # late-straight stretches just before braking (short persistence, cheap drag). E_deploy
        # falls and income rises with T, so the balance is monotone and the root unique.
        def _masks(T):
            d_mask = deploy_score >= T
            h_mask = harvest_eligible & ~d_mask & (value <= eta_re * T)
            return d_mask, h_mask

        lo = 0.0
        hi = float(deploy_score.max()) if deploy_score.max() > 0.0 else 1.0

        for _ in range(60):
            T_mid = (lo + hi) * 0.5
            d_mask, h_mask = _masks(T_mid)
            E_dep = float(np.sum(e_deploy_step[d_mask]))
            income = min(e_regen + float(np.sum(e_harvest_step[h_mask])), e_rec_max) + e_rec_etc
            if E_dep > income:
                lo = T_mid  # price too low — over-deploying / under-harvesting
            else:
                hi = T_mid  # affordable — try deploying more

        # apply final threshold (hi guarantees E_deploy <= income)
        deploy_mask, harvest_mask = _masks(hi)

        self.em_boost_use[:no_points] = deploy_mask
        self.em_harvest_use[:no_points] = harvest_mask

    @staticmethod
    def __enforce_es_feasibility(deploy_mask: np.ndarray, deploy_score: np.ndarray,
                                 e_deploy_step: np.ndarray, e_rec_profile: np.ndarray,
                                 e_rec_etc: float, dt: np.ndarray,
                                 es_initial: float, es_max: float) -> np.ndarray:
        """Drop planned deployments that the energy store cannot actually pay for.

        Simulates the ES forward against the plan. On the first point where the charge would go
        negative, the cheapest deployments planned so far (lowest tau/v^2 value) are removed
        until the deficit is covered, then the walk restarts -- returning energy to an earlier
        point changes everything downstream, and the ES ceiling means the effect is not simply
        additive. Removing the lowest-value points first preserves the ranking the price
        threshold established.

        e_rec_profile is the per-point recovery measured in the previous solver run, so like the
        budget itself this carries one iteration of lag. ETC recovery is spread over the lap in
        proportion to time, since the strategy has no per-point breakdown of it.
        """
        n = deploy_mask.size
        mask = deploy_mask.copy()

        gain = e_rec_profile.astype(float).copy()
        if e_rec_etc > 0.0 and float(np.sum(dt)) > 0.0:
            gain = gain + e_rec_etc * dt / float(np.sum(dt))

        # bounded: every restart removes at least one deployment
        for _ in range(int(np.count_nonzero(mask)) + 1):
            es = es_initial
            heap = []  # (value, index) of deployments encountered so far, cheapest first
            deficit_at = -1
            for i in range(n):
                if mask[i]:
                    heapq.heappush(heap, (deploy_score[i], i))
                    es -= e_deploy_step[i]
                es += gain[i]
                if es > es_max:
                    es = es_max
                if es < 0.0:
                    deficit_at = i
                    break

            if deficit_at < 0:
                return mask
            if not heap:
                return mask  # deficit is not caused by deployment; nothing to give back

            returned = 0.0
            need = -es
            while heap and returned < need:
                _, j = heapq.heappop(heap)
                mask[j] = False
                returned += e_deploy_step[j]

        return mask

    def __strategy_qualy(self, t_cl: np.ndarray, vel_cl: np.ndarray, n_cl: np.ndarray, m_requ: np.ndarray,
                         es_initial: float, e_rec_max: float = 8e6, p_rec_max: float = 350e3,
                         e_rec_actual: float = None, e_rec_etc: float = 0.0, kappa: np.ndarray = None,
                         ers_harvest_speed: float = None, e_rec_profile: np.ndarray = None):
        """qualy = Qualifying lap strategy. Jointly optimizes harvest and deployment for a single lap
        where the car starts with a known battery charge (es_initial) and may end with any state
        in [0, es_max]. Budget = es_initial + energy recovered this lap.

        Deploy: acceleration phases where throttle >= 95% and |kappa| < kappa_max_deploy,
                ranked by persistence-weighted per-Joule value tau / v^2 (see
                __time_to_next_event) with a binary search on the price threshold T.
        Harvest: high-speed low-curvature phases, not already deploying, where the drag cost
                 (the same tau / v^2) is below eta_e_motor_re * T, i.e. the recovered energy
                 redeploys at a profit. No active harvest at all if the budget already covers
                 deployment everywhere.
        Braking regen: handled automatically by the solver — not part of this strategy mask.

        Deployment budget = es_initial + total e-motor recovery from the previous solver run.
        On the first call e_rec_actual is None, so only es_initial is available."""

        no_points = t_cl.size - 1
        dt = np.diff(t_cl)

        has_speed_limit = hasattr(self.carobj, 'pow_e_motor_max') and \
            self.carobj.pars_engine.get("ers_speed_limit", False)
        pow_e = self.carobj.pars_engine["pow_e_motor"]

        # curvature gate: deploy/harvest only on low-curvature sections (straights and gentle bends)
        kappa_max = self.pars_driver.get("kappa_max_deploy", 0.01)  # [1/m] default ≈ 100 m radius
        # harvesting tolerates far more curvature than deployment: the generator load is carried
        # by spare ICE torque or by drag, neither of which needs the grip that deployment does,
        # and long medium-speed corners are exactly where the ICE has headroom to spare
        kappa_max_harv = self.pars_driver.get("kappa_max_harvest", 0.03)  # ≈ 33 m radius
        if kappa is not None:
            is_low_curv = np.abs(kappa[:no_points]) < kappa_max
            is_low_curv_harv = np.abs(kappa[:no_points]) < kappa_max_harv
        else:
            is_low_curv = np.ones(no_points, dtype=bool)
            is_low_curv_harv = np.ones(no_points, dtype=bool)

        # throttle gate: deploy/harvest only at full throttle (filters yellow-flag zones)
        is_full_throttle = self.throttle_pos[:no_points] >= 0.95

        # acceleration gate: deploy/harvest only where the car is accelerating
        vel_diff = np.diff(vel_cl[:no_points + 1])
        is_acc = vel_diff > 0

        # combined deploy eligibility
        deploy_eligible = is_acc & is_full_throttle & is_low_curv

        # available ERS power (zero above the speed limit -> point cannot deploy)
        if has_speed_limit:
            pow_avail = np.array([self.carobj.pow_e_motor_max(vel_cl[i]) for i in range(no_points)])
        else:
            pow_avail = np.full(no_points, pow_e)

        # persistence horizon: force effects last until the next braking point or sharp corner
        # (the same events that release the harvest latch below).
        # braking_threshold separates aero+harvest drag (~0.5-0.7 m/s/step) from braking (>1.5).
        braking_threshold = self.pars_driver.get("qualy_braking_threshold", 1.5)  # [m/s per step]
        is_braking = vel_diff < -braking_threshold
        tau = self.__time_to_next_event(t_cl, is_braking | ~is_low_curv)

        # per-Joule deploy value: a speed gain at point i is worth ~ tau_i / v_i^2 s/J,
        # zero at ineligible points
        vel_u = np.maximum(vel_cl[:no_points], 1.0)
        deploy_score = np.where(deploy_eligible & (pow_avail > 0.0), tau / vel_u ** 2, 0.0)

        # precompute actual energy consumed per step if deploying (uses real motor torque model)
        e_deploy_step = np.zeros(no_points)
        for i in range(no_points):
            if deploy_score[i] > 0.0:
                m_e_motor_val = self.carobj.calc_torque_distr(n=n_cl[i],
                                                              m_requ=m_requ[i],
                                                              throttle_pos=self.throttle_pos[i],
                                                              es=np.inf,
                                                              em_boost_use=True,
                                                              vel=vel_cl[i])[1]
                e_deploy_step[i] = (self.carobj.power_demand_e_motor_drive(n=n_cl[i],
                                                                            m_e_motor=np.array(m_e_motor_val))
                                    * dt[i])

        # harvest speed threshold [m/s]
        harvest_speed_min = ers_harvest_speed if ers_harvest_speed is not None \
            else self.pars_driver.get("ers_harvest_speed_min", 250.0 / 3.6)  # [m/s]

        # deployment budget: starting charge plus all energy recovered this lap (MGU-K plus
        # uncapped electric turbocharger recovery). e_rec_max caps only the MGU-K part (per-lap
        # FIA recovery limit) — the starting charge is limited by the battery capacity instead.
        es_max = self.carobj.pars_engine.get("max_e_energy_storage", np.inf)
        e_deploy_budget = min(es_initial, es_max) + e_rec_etc
        if e_rec_actual is not None:
            e_deploy_budget += min(e_rec_actual, e_rec_max)

        # deploy saturated: budget covers deployment at every eligible point, so additional
        # recovered energy could never be spent — deploy everywhere, no active harvest
        if float(np.sum(e_deploy_step)) <= e_deploy_budget:
            self.em_boost_use[:no_points] = deploy_score > 0.0
            self.em_harvest_use[:no_points] = False
            return

        # binary search for threshold T: find lowest T where total planned deploy <= budget
        lo = 0.0
        hi = float(deploy_score.max()) if deploy_score.max() > 0.0 else 1.0

        for _ in range(60):
            T_mid = (lo + hi) * 0.5
            d_mask = deploy_score >= T_mid
            E_deploy = float(np.sum(e_deploy_step[d_mask]))
            if E_deploy > e_deploy_budget:
                lo = T_mid
            else:
                hi = T_mid

        T_final = hi
        deploy_mask = deploy_score >= T_final

        # Temporal feasibility: the bisection above only guarantees that TOTAL planned spending
        # fits the budget. The ES is a buffer bounded by [0, es_max], so a plan can respect the
        # total and still call for deployment at points where the charge has not been recovered
        # yet. The solver then silently skips those points, leaving the plan mis-priced (it
        # believes it bought time it never got). Walk the ES forward against the plan and drop
        # the least valuable deployments until the trace stays non-negative.
        # Off by default: it makes the ES trace far more plausible but costs a little lap time
        # on balance and does not resolve the worst case (Albert Park), so it stays opt-in until
        # the lag in e_rec_profile is dealt with. See notes above.
        if self.pars_driver.get("use_es_feasibility", False) and e_rec_profile is not None:
            deploy_mask = self.__enforce_es_feasibility(
                deploy_mask=deploy_mask,
                deploy_score=deploy_score,
                e_deploy_step=e_deploy_step,
                e_rec_profile=np.asarray(e_rec_profile[:no_points]),
                e_rec_etc=e_rec_etc,
                dt=dt,
                es_initial=min(es_initial, es_max),
                es_max=es_max)

        # Harvest latch:
        #   START  — speed >= threshold, full throttle, low curvature, profitable (see below;
        #            no is_acc gate here: gentle deceleration from aero + generator drag is
        #            expected and intentional)
        #   CONTINUE — latch holds as long as not braking and not in sharp corner
        #   STOP   — car hits brakes (vel drop > braking_threshold per 5 m step),
        #            OR deploy zone resumes (corner exit), OR sharp corner entered.
        # After braking the braking regen path runs automatically; harvest restarts only
        # when speed exceeds the threshold again on the next straight.
        #
        # Cost-benefit gate: harvesting 1 J at the wheels costs tau/v^2 lap time and returns
        # eta_e_motor_re J redeployed at marginal value T_final. Points price out into deploy
        # (value >= T), harvest (value <= eta_re * T) or neither (the dead band between). tau
        # shrinks toward each event, so once the trigger fires the rest of the latch window is
        # profitable too.
        eta_re = self.carobj.pars_engine.get("eta_e_motor_re", 0.0)
        econ_ok = tau / vel_u ** 2 <= eta_re * T_final
        harvest_trigger = is_full_throttle & is_low_curv_harv & econ_ok \
            & (vel_cl[:no_points] >= harvest_speed_min)

        harvest_mask = np.zeros(no_points, dtype=bool)
        in_harvest = False
        for i in range(no_points):
            if is_braking[i] or deploy_mask[i] or not is_low_curv_harv[i]:
                in_harvest = False
            elif harvest_trigger[i]:
                in_harvest = True
            harvest_mask[i] = in_harvest and not deploy_mask[i]

        self.em_boost_use[:no_points] = deploy_mask
        self.em_harvest_use[:no_points] = harvest_mask

    def __lift_coast(self, vel_cl: np.ndarray, n_lac: int):
        """Velocity input in m/s, n_lac is the number of points without throttle in front of a brake point."""

        no_points = vel_cl.size - 1
        vel_diffs = np.diff(vel_cl)
        inds_neg_vel_diff = np.where(vel_diffs < 0.0)[0]

        for i in inds_neg_vel_diff:
            # catch case that lift&coast starts in front of start/finish line
            if i - n_lac < 0:
                self.throttle_pos[no_points + i - n_lac:] = 0.0
                # current point is not included to allow acceleration directly after last brakepoint
                self.throttle_pos[0:i] = 0.0
            else:
                # current point is not included to allow acceleration directly after last brakepoint
                self.throttle_pos[i - n_lac:i] = 0.0


# ----------------------------------------------------------------------------------------------------------------------
# TESTING --------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    pass
