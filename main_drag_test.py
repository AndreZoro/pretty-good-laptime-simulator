"""
Standing-start drag test simulation.

Accelerates a car from rest on a flat straight and reports elapsed time and
trap speed at three standard drag-racing distances:
  - 1/8 mile  (201.168 m)
  - 1/4 mile  (402.336 m)
  - 1 km      (1000.0 m)

Configuration is set in the section below.
"""

import numpy as np
import matplotlib.pyplot as plt

from laptimesim.src.car_electric import CarElectric
from laptimesim.src.car_hybrid import CarHybrid
from laptimesim.src.drag_test import run_drag_test, DRAG_DISTANCES

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

VEHICLE_FILE     = "laptimesim/input/vehicles/EH_Zapovic_Breeze.ini"
POWERTRAIN_TYPE  = "electric"   # "electric" or "hybrid"
MU               = 1.0          # track friction coefficient (1.0 = dry)
DT               = 0.001        # integration time step [s]

# ----------------------------------------------------------------------
# Load vehicle
# ----------------------------------------------------------------------

if POWERTRAIN_TYPE == "electric":
    car = CarElectric(parfilepath=VEHICLE_FILE)
else:
    car = CarHybrid(parfilepath=VEHICLE_FILE)

# ----------------------------------------------------------------------
# Run drag test up to the longest standard distance
# ----------------------------------------------------------------------

max_dist = max(DRAG_DISTANCES.values())
result   = run_drag_test(car, distance_m=max_dist, mu=MU, dt=DT)

t    = result["t"]
vel  = result["vel"]
dist = result["dist"]


# ----------------------------------------------------------------------
# Interpolation helpers
# ----------------------------------------------------------------------

def interp_at_dist(d_target):
    """Return (time_s, vel_mps) at the exact target distance."""
    idx = np.searchsorted(dist, d_target)
    if idx == 0 or idx >= len(dist):
        return None, None
    alpha  = (d_target - dist[idx - 1]) / (dist[idx] - dist[idx - 1])
    t_val  = t[idx - 1]   + alpha * (t[idx]   - t[idx - 1])
    v_val  = vel[idx - 1] + alpha * (vel[idx] - vel[idx - 1])
    return t_val, v_val


def interp_at_vel(v_target):
    """Return elapsed time [s] when the car first reaches v_target [m/s]."""
    idx = np.searchsorted(vel, v_target)
    if idx == 0 or idx >= len(vel):
        return None
    alpha = (v_target - vel[idx - 1]) / (vel[idx] - vel[idx - 1])
    return t[idx - 1] + alpha * (t[idx] - t[idx - 1])


# ----------------------------------------------------------------------
# Print results table
# ----------------------------------------------------------------------

series = car.pars_engine.get("series", "Unknown")

print(f"\n{'=' * 52}")
print(f"  Drag test  —  {series}")
print(f"  Vehicle : {VEHICLE_FILE}")
print(f"  mu = {MU:.2f}   dt = {DT * 1000:.1f} ms")
print(f"{'=' * 52}")

print(f"\n  {'Distance':<15} {'Time':>9} {'Trap speed':>13}")
print(f"  {'':15} {'(s)':>9} {'(km/h)':>13}")
print(f"  {'-' * 39}")

for name, d_m in sorted(DRAG_DISTANCES.items(), key=lambda x: x[1]):
    t_val, v_val = interp_at_dist(d_m)
    if t_val is not None:
        print(f"  {name:<15} {t_val:>9.3f} {v_val * 3.6:>13.1f}")
    else:
        print(f"  {name:<15} {'---':>9} {'---':>13}")

print()

velocity_targets = [
    ("0-60 mph",   60.0 * 1.60934 / 3.6),
    ("0-100 km/h", 100.0 / 3.6),
    ("0-200 km/h", 200.0 / 3.6),
]
for label, v_target in velocity_targets:
    t_val = interp_at_vel(v_target)
    if t_val is not None:
        print(f"  {label:<13}: {t_val:.2f} s")
    else:
        print(f"  {label:<13}: not reached within {max_dist:.0f} m")

print()

# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
fig.suptitle(f"Drag test  —  {series}", fontsize=13)

# Dashed vertical lines at each milestone distance
for name, d_m in DRAG_DISTANCES.items():
    for ax in axes:
        ax.axvline(d_m, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

# Distance labels on the top subplot
label_y = vel[-1] * 3.6 * 0.05   # 5 % from bottom
for name, d_m in DRAG_DISTANCES.items():
    axes[0].text(d_m + 6, label_y, name, fontsize=8, color="gray", va="bottom")

# Velocity
axes[0].plot(dist, vel * 3.6, color="tab:blue")
axes[0].set_ylabel("Velocity [km/h]")
axes[0].grid(True, alpha=0.4)

# Acceleration
axes[1].plot(dist, result["a_x"], color="tab:orange")
axes[1].set_ylabel("Acceleration [m/s²]")
axes[1].grid(True, alpha=0.4)

# Gear
n_gears = len(car.pars_gearbox["i_trans"])
axes[2].step(dist, result["gear"] + 1, color="tab:green", where="post")
axes[2].set_ylabel("Gear")
axes[2].set_yticks(range(1, n_gears + 1))
axes[2].set_xlabel("Distance [m]")
axes[2].grid(True, alpha=0.4)

plt.tight_layout()
plt.show()
