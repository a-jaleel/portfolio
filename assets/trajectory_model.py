"""
Ping Pong Ball Launcher — Trajectory Model
==========================================

Predicts launch distance for a single-flywheel + hood ball launcher as a
function of motor RPM, launch angle, and slip efficiency.

Physics included:
  - 2D point-mass projectile motion (gravity)
  - Aerodynamic drag (Reynolds-dependent C_D, smooth sphere)
  - Magnus lift (spin-parameter-dependent C_L)
  - Slip efficiency between flywheel surface and ball exit velocity

Sanity check: setting drag=0 and lift=0 matches closed-form vacuum
projectile range to <0.1 % at all tested points.

Hardware-specific defaults:
  - EMAX GTII 2212C 1000 KV motor
  - BaneBots T81 2 in flywheel (40A urethane)
  - ITTF regulation 40 mm / 2.7 g ball
  - 4S Li-Po nominal (14.8 V)

Author: SB MechE Take-Home, trajectory deliverable
"""

from dataclasses import dataclass, field
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
G        = 9.81          # gravity, m/s^2
RHO_AIR  = 1.225         # air density, kg/m^3 (sea level, 20 C)
NU_AIR   = 1.516e-5      # kinematic viscosity, m^2/s

# ---------------------------------------------------------------------------
# Ball (ITTF regulation "40+" plastic ball)
# ---------------------------------------------------------------------------
M_BALL   = 2.7e-3        # kg
D_BALL   = 40e-3         # m
R_BALL   = D_BALL / 2
A_BALL   = np.pi * R_BALL**2

# ---------------------------------------------------------------------------
# Launcher hardware
# ---------------------------------------------------------------------------
D_FLYWHEEL = 2.0 * 0.0254   # 2 in BaneBots T81
R_FLYWHEEL = D_FLYWHEEL / 2

KV_MOTOR     = 1000.0       # rpm/V (EMAX GTII 2212C)
V_BATT       = 14.8         # 4S Li-Po nominal
THROTTLE_MAX = 0.85         # leave headroom under load
RPM_NO_LOAD  = KV_MOTOR * V_BATT
RPM_DESIGN   = THROTTLE_MAX * RPM_NO_LOAD   # ~12,580 rpm

LAUNCH_HEIGHT = 0.7        # m (arm-mounted exit height, rough)

# ---------------------------------------------------------------------------
# Aerodynamic coefficients
# ---------------------------------------------------------------------------
def reynolds(v):
    return abs(v) * D_BALL / NU_AIR

def C_D(v):
    """Drag coefficient for a smooth sphere.

    Stokes regime    (Re < 1)         : 24/Re
    Schiller-Naumann (1 < Re < 1e3)   : (24/Re)(1 + 0.15 Re^0.687)
    Subcritical      (1e3 < Re < 2e5) : 0.47
    Drag crisis      (2e5 < Re < 4e5) : linear drop to 0.10
    Supercritical    (Re > 4e5)       : 0.10
    """
    Re = reynolds(v)
    if Re < 1e-6: return 0.0
    if Re < 1.0:  return 24.0 / Re
    if Re < 1e3:  return (24.0 / Re) * (1.0 + 0.15 * Re**0.687)
    if Re < 2e5:  return 0.47
    if Re < 4e5:  return 0.47 - (Re - 2e5) / 2e5 * 0.37
    return 0.10

def C_L(omega, v):
    """Magnus lift coefficient for a smooth spinning sphere.

    Spin parameter S = |omega| * r / |v|.  Empirical fits (Briggs 1959,
    Mehta 1985, Watts & Ferrer 1987) show C_L roughly linear in S at low
    spin, saturating around 0.40 at high spin for smooth spheres.
    """
    if abs(v) < 1e-6: return 0.0
    S = abs(omega) * R_BALL / abs(v)
    return min(0.5 * S, 0.40)

# ---------------------------------------------------------------------------
# Launch kinematics
# ---------------------------------------------------------------------------
def flywheel_surface_speed(rpm):
    return (rpm * 2.0 * np.pi / 60.0) * R_FLYWHEEL

def ball_exit_state(rpm_motor, slip_eff=0.55):
    """Ball exit speed and backspin from a single-flywheel + hood mechanism.

    Ideal no-slip kinematics (top of ball matches wheel surface, bottom
    of ball matches stationary hood):
        v_center = v_wheel / 2
        omega    = v_wheel / (2 * r_ball)   (backspin)

    Real launchers slip; slip_eff captures the combined effect of finite
    contact time and tangential micro-slip. Values from FRC/Nerf-style
    flywheel data with light-mass projectiles: 0.45 to 0.65.
    """
    v_w = flywheel_surface_speed(rpm_motor)
    return slip_eff * v_w / 2.0, slip_eff * v_w / (2.0 * R_BALL)

# ---------------------------------------------------------------------------
# Trajectory ODE
# ---------------------------------------------------------------------------
def rhs(t, state, omega_spin, drag_on=True, lift_on=True):
    x, y, vx, vy = state
    v = np.hypot(vx, vy)
    if v < 1e-6:
        return [vx, vy, 0.0, -G]

    cd = C_D(v) if drag_on else 0.0
    cl = C_L(omega_spin, v) if lift_on else 0.0

    q  = 0.5 * RHO_AIR * v * v
    Fd = q * cd * A_BALL                 # drag magnitude
    Fl = q * cl * A_BALL                 # lift magnitude

    # Drag opposes velocity
    ax = -Fd * vx / v / M_BALL
    ay = -Fd * vy / v / M_BALL - G
    # Backspin lift is +90 deg from velocity (perp, upward component
    # for forward-moving ball with topward-rolling rear surface)
    ax += -Fl * vy / v / M_BALL
    ay +=  Fl * vx / v / M_BALL
    return [vx, vy, ax, ay]

def simulate(v0, theta_deg, omega_spin,
             y0=LAUNCH_HEIGHT, t_max=6.0,
             drag_on=True, lift_on=True):
    th = np.deg2rad(theta_deg)
    state0 = [0.0, y0, v0 * np.cos(th), v0 * np.sin(th)]

    def hit_ground(t, s, *a): return s[1]
    hit_ground.terminal = True
    hit_ground.direction = -1

    return solve_ivp(rhs, (0, t_max), state0,
                     args=(omega_spin, drag_on, lift_on),
                     events=hit_ground,
                     max_step=0.005, rtol=1e-9, atol=1e-11)

def shot_range(*args, **kwargs):
    sol = simulate(*args, **kwargs)
    return sol.y[0, -1]

# ---------------------------------------------------------------------------
# Hand-calc reference
# ---------------------------------------------------------------------------
def vacuum_range(v0, theta_deg, y0=LAUNCH_HEIGHT):
    """Closed-form range from initial height with no air."""
    th = np.deg2rad(theta_deg)
    vx0, vy0 = v0 * np.cos(th), v0 * np.sin(th)
    t_f = (vy0 + np.sqrt(vy0**2 + 2 * G * y0)) / G
    return vx0 * t_f

# ===========================================================================
# ANALYSIS
# ===========================================================================
def sanity_check():
    """Compare simulated trajectory (drag+lift OFF) to closed-form vacuum."""
    print("=" * 65)
    print(" SANITY CHECK : drag+lift OFF vs closed-form vacuum range")
    print("=" * 65)
    print(f" {'v0 (m/s)':>10} {'theta':>7} {'vacuum (m)':>12} "
          f"{'sim (m)':>10} {'err %':>7}")
    cases = [(10, 30), (15, 45), (20, 30), (25, 15), (15, 60)]
    for v0, th in cases:
        r_hand = vacuum_range(v0, th)
        r_sim  = shot_range(v0, th, 0.0, drag_on=False, lift_on=False)
        err = 100 * (r_sim - r_hand) / r_hand
        print(f" {v0:>10.1f} {th:>6}° {r_hand:>12.3f} "
              f"{r_sim:>10.3f} {err:>+6.3f}%")
    print()

def baseline_summary():
    print("=" * 65)
    print(" BASELINE OPERATING POINT")
    print("=" * 65)
    v_w = flywheel_surface_speed(RPM_DESIGN)
    v_b, w_b = ball_exit_state(RPM_DESIGN, slip_eff=0.55)
    print(f" Motor                    EMAX GTII 2212C, KV = {KV_MOTOR:.0f}")
    print(f" Battery                  4S nominal ({V_BATT:.1f} V)")
    print(f" No-load RPM              {RPM_NO_LOAD:>8.0f} rpm")
    print(f" Design RPM (85% of NL)   {RPM_DESIGN:>8.0f} rpm")
    print(f" Flywheel surface speed   {v_w:>8.2f} m/s")
    print(f" Ball exit velocity (η=0.55)   {v_b:>5.2f} m/s")
    print(f" Ball backspin            {w_b:>8.0f} rad/s "
          f"({w_b * 60 / (2*np.pi):.0f} rpm)")
    print(f" Spin parameter S = ωr/v  {w_b * R_BALL / v_b:>8.3f}")
    Re = reynolds(v_b)
    print(f" Reynolds number at exit  {Re:>8.0f}")
    print(f" C_D at exit              {C_D(v_b):>8.3f}")
    print(f" C_L at exit              {C_L(w_b, v_b):>8.3f}")
    # range comparison
    for ang in (15, 25, 35, 45):
        r_full = shot_range(v_b, ang, w_b)
        r_nodrag = vacuum_range(v_b, ang)
        r_nolift = shot_range(v_b, ang, w_b, lift_on=False)
        print(f"   θ = {ang:>2}° : vacuum {r_nodrag:5.2f} m | "
              f"drag-only {r_nolift:5.2f} m | full {r_full:5.2f} m")
    print()

# ---------------------------------------------------------------------------
# Sweeps and plots
# ---------------------------------------------------------------------------
OUT = Path("/home/claude/launcher/figures")
OUT.mkdir(parents=True, exist_ok=True)

def plot_trajectories():
    """Show vacuum vs drag-only vs full physics for the design point."""
    v_b, w_b = ball_exit_state(RPM_DESIGN, slip_eff=0.55)
    theta = 25
    sol_full   = simulate(v_b, theta, w_b)
    sol_nolift = simulate(v_b, theta, w_b, lift_on=False)
    sol_vac    = simulate(v_b, theta, 0.0, drag_on=False, lift_on=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(sol_vac.y[0],    sol_vac.y[1],    '--', lw=2,
            label=f'vacuum (R = {sol_vac.y[0,-1]:.1f} m)')
    ax.plot(sol_nolift.y[0], sol_nolift.y[1], '-.', lw=2,
            label=f'drag, no spin (R = {sol_nolift.y[0,-1]:.1f} m)')
    ax.plot(sol_full.y[0],   sol_full.y[1],   '-',  lw=2.5,
            label=f'drag + Magnus (R = {sol_full.y[0,-1]:.1f} m)')
    ax.axvline(10*0.3048, color='r', ls=':', alpha=0.6, label='10 ft target')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(f'Trajectories at v0 = {v_b:.1f} m/s, θ = {theta}°, '
                 f'spin = {w_b*60/(2*np.pi):.0f} rpm')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    ax.set_ylim(0, max(2, ax.get_ylim()[1]))
    fig.tight_layout()
    fig.savefig(OUT / 'fig1_trajectories.png', dpi=140)
    plt.close(fig)

def sweep_rpm():
    """Range vs motor RPM, several launch angles, slip = 0.55."""
    rpms = np.linspace(3000, RPM_NO_LOAD, 30)
    angles = [15, 25, 35, 45]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for ang in angles:
        ranges = []
        for rpm in rpms:
            v_b, w_b = ball_exit_state(rpm, slip_eff=0.55)
            ranges.append(shot_range(v_b, ang, w_b))
        ax.plot(rpms, ranges, lw=2, label=f'θ = {ang}°')
    ax.axhline(10*0.3048, color='r', ls=':', alpha=0.7, label='10 ft target')
    ax.axvline(RPM_DESIGN, color='k', ls='--', alpha=0.5,
               label=f'design RPM = {RPM_DESIGN:.0f}')
    ax.set_xlabel('Motor RPM')
    ax.set_ylabel('Range (m)')
    ax.set_title('Range vs motor RPM  (slip η = 0.55, full aero)')
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / 'fig2_range_vs_rpm.png', dpi=140)
    plt.close(fig)

def sweep_angle():
    """Range vs launch angle at design RPM, varying slip."""
    angles = np.linspace(5, 75, 35)
    slips = [0.45, 0.55, 0.65]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for eta in slips:
        v_b, w_b = ball_exit_state(RPM_DESIGN, slip_eff=eta)
        ranges = [shot_range(v_b, a, w_b) for a in angles]
        a_opt = angles[int(np.argmax(ranges))]
        ax.plot(angles, ranges, lw=2,
                label=f'η = {eta:.2f}  (v0 = {v_b:.1f} m/s, θ_opt ≈ {a_opt:.0f}°)')
    ax.axhline(10*0.3048, color='r', ls=':', alpha=0.7, label='10 ft target')
    ax.set_xlabel('Launch angle (deg)')
    ax.set_ylabel('Range (m)')
    ax.set_title(f'Range vs launch angle  (RPM = {RPM_DESIGN:.0f})')
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / 'fig3_range_vs_angle.png', dpi=140)
    plt.close(fig)

def sweep_2d_contour():
    """Range as a function of (RPM, launch angle), heatmap."""
    rpms   = np.linspace(3000, RPM_NO_LOAD, 25)
    angles = np.linspace(5, 75, 25)
    R, A = np.meshgrid(rpms, angles)
    Z = np.zeros_like(R)
    for i, ang in enumerate(angles):
        for j, rpm in enumerate(rpms):
            v_b, w_b = ball_exit_state(rpm, slip_eff=0.55)
            Z[i, j] = shot_range(v_b, ang, w_b)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    cs = ax.contourf(R, A, Z, levels=20, cmap='viridis')
    cb = fig.colorbar(cs, ax=ax, label='Range (m)')
    cl = ax.contour(R, A, Z, levels=[10*0.3048], colors='red',
                    linewidths=2.0, linestyles='--')
    ax.clabel(cl, fmt='10 ft target', inline=True)
    ax.axvline(RPM_DESIGN, color='w', ls='-', alpha=0.6, lw=1)
    ax.set_xlabel('Motor RPM')
    ax.set_ylabel('Launch angle (deg)')
    ax.set_title('Range  R(RPM, θ)   slip η = 0.55')
    fig.tight_layout()
    fig.savefig(OUT / 'fig4_range_heatmap.png', dpi=140)
    plt.close(fig)

def sweep_slip():
    """How sensitive is range to the slip-efficiency assumption?"""
    etas = np.linspace(0.35, 0.75, 20)
    angles = [20, 30, 45]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for ang in angles:
        ranges = []
        for eta in etas:
            v_b, w_b = ball_exit_state(RPM_DESIGN, slip_eff=eta)
            ranges.append(shot_range(v_b, ang, w_b))
        ax.plot(etas, ranges, lw=2, label=f'θ = {ang}°')
    ax.axhline(10*0.3048, color='r', ls=':', alpha=0.7, label='10 ft target')
    ax.axvspan(0.45, 0.65, color='gray', alpha=0.18,
               label='expected η range')
    ax.set_xlabel('Slip efficiency η  (v_ball = η · v_wheel/2)')
    ax.set_ylabel('Range (m)')
    ax.set_title(f'Sensitivity to slip efficiency  (RPM = {RPM_DESIGN:.0f})')
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / 'fig5_range_vs_slip.png', dpi=140)
    plt.close(fig)

def find_min_rpm_for_target(target_m=10*0.3048, slip_eff=0.55):
    """Lowest RPM that still clears the 10-ft target at the optimal angle."""
    print("=" * 65)
    print(" MIN RPM TO CLEAR 10 FT TARGET (slip η = 0.55, optimal angle)")
    print("=" * 65)
    for rpm in np.arange(2000, RPM_NO_LOAD, 250):
        v_b, w_b = ball_exit_state(rpm, slip_eff=slip_eff)
        best = max(shot_range(v_b, a, w_b) for a in np.arange(10, 60, 5))
        if best > target_m:
            print(f" Min RPM that hits 10 ft : {rpm:.0f}  "
                  f"({rpm/RPM_NO_LOAD*100:.1f}% of no-load)")
            print(f" ➜ Throttle headroom      : "
                  f"{100 - rpm/RPM_NO_LOAD*100:.0f} %")
            print()
            return rpm
    print(" 10 ft NOT achievable in this RPM range.")
    return None

# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":
    sanity_check()
    baseline_summary()
    find_min_rpm_for_target()
    plot_trajectories()
    sweep_rpm()
    sweep_angle()
    sweep_2d_contour()
    sweep_slip()
    print(f"Figures written to: {OUT}")
