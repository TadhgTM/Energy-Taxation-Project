from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import replace, dataclass
from typing import Dict, Tuple, Optional
from scipy.optimize import minimize, root
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from copy import deepcopy

"""
DYNAMIC GENERAL EQUILIBRIUM MODEL WITH ENVIRONMENTAL EXTERNALITIES

This file contains a unified implementation of:
1. general equilibrium model
2. Dynamic Ramsey-Cass-Koopmans capital accumulation
3. Optimal carbon tax policy over time
4. Parallel computation for sensitivity analysis

STRUCTURE:
- 3-sector CES energy aggregate (oil, gas, clean)
- Cobb-Douglas final production with K, L, E
- Capital accumulation: K_{t+1} = (1-δ)K_t + I_t
- Euler equation for consumption smoothing
- Terminal capital constraint
- Parallel grid search for optimal policy

Notes:
- Hardware tested on: Macbook Pro (M5, 24GB RAM, 2025)
- Last updated: January 2026

=============================================================================
"""

# ----------------------------
# Constants
# ----------------------------

ENERGY_LABELS = ("oil", "gas", "clean")
J = 3
EPS = 1e-12

# Global counter for equilibrium solves
_SOLVE_COUNTER = 0


# ----------------------------
# Static Model Components (from AGEMACRO.py)
# ----------------------------

def _clamp_pos(x, eps: float = 1e-12):
    """Clamp to be strictly positive (works for scalars and arrays)"""
    if np.isscalar(x):
        return float(max(float(x), eps))
    x = np.asarray(x, dtype=float)
    return np.maximum(x, eps)


@dataclass(frozen=True)
class Params:
    """Parameters for static equilibrium model"""
    # Endowments
    L_bar: float
    K_bar: float

    # Final good technology (Cobb-Douglas)
    A_final: float
    a_c: float
    b_c: float
    e_c: float

    # Energy sector technologies
    A_j: np.ndarray      # shape (3,)
    a_j: np.ndarray      # capital share per energy sector
    
    # Energy aggregator (CES)
    eta: float           # elasticity of substitution
    omega_E: np.ndarray  # CES weights, shape (3,)

    # Emissions intensities
    phi_j: np.ndarray    # shape (3,)


@dataclass(frozen=True)
class Preferences:
    """Household preferences"""
    gamma_hh: float = 2.0  # CRRA in consumption
    sigma: float = 0.5     # Frisch exponent
    chi: float = 1.0       # scale on disutility


@dataclass(frozen=True)
class Policy:
    """Government policy"""
    tau: float = 0.0              # emissions tax rate
    omega_damage: float = 0.08    # marginal damage weight
    damage_type: str = "utility"  # "utility" or "productivity"


@dataclass(frozen=True)
class Equilibrium:
    """Static equilibrium solution"""
    success: bool
    message: str
    
    # prices (final numeraire pY=1)
    w: float
    r: float
    pE: float
    pj: np.ndarray
    
    # allocations
    L: float
    Kc: float
    Lc: float
    Kj: np.ndarray
    Lj: np.ndarray
    
    # quantities
    Ej: np.ndarray
    E: float
    Y: float
    C: float
    
    # policy objects
    Z: float
    T_rev: float
    
    # welfare
    W: float
    
    # residual norm
    max_abs_resid: float
    damage_factor: float = 1.0

# ----------------------------
# Production Functions
def energy_output(K: float, L: float, A: float, a: float) -> float:
    """Energy sector output: E_j = A_j * K_j^{a_j} * L_j^{1-a_j}"""
    K = _clamp_pos(K)
    L = _clamp_pos(L)
    A = _clamp_pos(A)
    return float(A * (K ** a) * (L ** (1.0 - a)))


def energy_marginal_products(K: float, L: float, A: float, a: float):
    """Returns (E_j, dE/dK, dE/dL)"""
    K = _clamp_pos(K, eps=1e-12)
    L = _clamp_pos(L, eps=1e-12)
    
    Ej = energy_output(K, L, A, a)
    dEdK = a * Ej / _clamp_pos(K, eps=1e-12)
    dEdL = (1.0 - a) * Ej / _clamp_pos(L, eps=1e-12)
    return float(Ej), float(dEdK), float(dEdL)


def ces_quantity(E: np.ndarray, eta: float, omega: np.ndarray) -> float:
    """CES energy aggregate: E = [sum omega_j * E_j^{(eta-1)/eta}]^{eta/(eta-1)}"""
    E = np.asarray(E, dtype=float)
    omega = np.asarray(omega, dtype=float)
    assert E.shape == (J,) and omega.shape == (J,)
    
    E = _clamp_pos(E)
    if abs(eta - 1.0) < 1e-10:
        return float(np.exp(np.sum(omega * np.log(E))))
    
    rho = (eta - 1.0) / eta
    inside = np.sum(omega * (E ** rho))
    return float(inside ** (1.0 / rho))


def ces_price(p: np.ndarray, eta: float, omega: np.ndarray) -> float:
    """Dual CES price index"""
    p = np.asarray(p, dtype=float)
    omega = np.asarray(omega, dtype=float)
    assert p.shape == (J,) and omega.shape == (J,)
    
    p = _clamp_pos(p)
    if abs(eta - 1.0) < 1e-10:
        return float(np.exp(np.sum(omega * np.log(p))))
    
    inside = np.sum(omega * (p ** (1.0 - eta)))
    return float(inside ** (1.0 / (1.0 - eta)))


def final_output(K: float, L: float, E: float, A: float, a: float, b: float, e: float, damage_factor: float = 1.0) -> float:
    """Final output: Y = A * damage_factor * K^a * L^b * E^e"""
    K = float(_clamp_pos(K))
    L = float(_clamp_pos(L))
    E = float(_clamp_pos(E))
    A = float(_clamp_pos(A))
    damage_factor = float(np.clip(damage_factor, 0.01, 1.0))
    return float(A * damage_factor * (K ** a) * (L ** b) * (E ** e))


# ----------------------------
# Utility and Emissions
def emissions(Ej: np.ndarray, phi_j: np.ndarray) -> float:
    """Total emissions: Z = sum phi_j * E_j"""
    Ej = np.asarray(Ej, dtype=float)
    return float(np.dot(phi_j, Ej))


def utility(C: float, L: float, Z: float, pref: Preferences, omega_damage: float) -> float:
    """Utility: U(C,L) - omega_damage * Z"""
    C = float(_clamp_pos(C))
    L = float(_clamp_pos(L))
    Z = float(max(Z, 0.0))
    
    g = pref.gamma_hh
    s = pref.sigma
    chi = pref.chi
    
    # CRRA
    if abs(g - 1.0) < 1e-10:
        uC = np.log(C)
    else:
        uC = (C ** (1.0 - g)) / (1.0 - g)
    
    uL = chi * (L ** (1.0 + s)) / (1.0 + s)
    return float(uC - uL - omega_damage * Z)


def household_budget(w: float, r: float, K_bar: float, L: float, T_lump: float) -> float:
    """Budget constraint: C = wL + rK + T"""
    return float(w * L + r * K_bar + T_lump)


def household_labour_foc(w: float, C: float, pref: Preferences, L: float) -> float:
    """Labour FOC residual: w - chi*C^gamma*L^sigma = 0"""
    w = float(_clamp_pos(w))
    C = float(_clamp_pos(C))
    L = float(_clamp_pos(L))
    return float(w - pref.chi * (C ** pref.gamma_hh) * (L ** pref.sigma))


# ----------------------------
# Equilibrium Solver
def _implied_component_prices_from_ces(Ej: np.ndarray, E: float, pE: float, eta: float, omega: np.ndarray) -> np.ndarray:
    """Back out component prices from CES structure"""
    Ej = np.asarray(Ej, dtype=float)
    omega = np.asarray(omega, dtype=float)
    Ej = _clamp_pos(Ej, eps=1e-12)
    E = _clamp_pos(E, eps=1e-12)
    Ej_over_E = _clamp_pos(Ej / E, eps=1e-12)
    pj = pE * (omega ** (1.0 / eta)) * (Ej_over_E ** (-1.0 / eta))
    return pj


def residuals_static_ge(x: np.ndarray, P, pref: Preferences, pol: Policy) -> np.ndarray:
    """12-equation system for static GE"""
    x = np.asarray(x, dtype=float)
    assert x.shape == (12,)
    
    # prices
    w = float(np.exp(np.clip(x[0], -30, 30)))
    r = float(np.exp(np.clip(x[1], -30, 30)))
    pE = float(np.exp(np.clip(x[2], -30, 30)))
    
    # aggregate labour
    L = float(_clamp_pos(x[3]))
    
    # allocations (logs)
    K1, L1 = float(np.exp(np.clip(x[4], -30, 30))), float(np.exp(np.clip(x[5], -30, 30)))
    K2, L2 = float(np.exp(np.clip(x[6], -30, 30))), float(np.exp(np.clip(x[7], -30, 30)))
    K3, L3 = float(np.exp(np.clip(x[8], -30, 30))), float(np.exp(np.clip(x[9], -30, 30)))
    Kc, Lc = float(np.exp(np.clip(x[10], -30, 30))), float(np.exp(np.clip(x[11], -30, 30)))
    
    Kj = np.array([K1, K2, K3], dtype=float)
    Lj = np.array([L1, L2, L3], dtype=float)
    
    # energy sector outputs
    Ej = np.zeros(J)
    dEdK = np.zeros(J)
    dEdL = np.zeros(J)
    for j in range(J):
        Ej[j], dEdK[j], dEdL[j] = energy_marginal_products(Kj[j], Lj[j], P.A_j[j], P.a_j[j])
    
    # energy aggregate
    E = float(ces_quantity(Ej, P.eta, P.omega_E))
    
    # emissions and damages
    Z_temp = emissions(Ej, P.phi_j)
    damage_factor = 1.0  # Always use utility damages for now
    
    # final output
    Y = float(final_output(Kc, Lc, E, P.A_final, P.a_c, P.b_c, P.e_c, damage_factor))
    
    # component prices and wedges
    pj = _implied_component_prices_from_ces(Ej, E, pE, P.eta, P.omega_E)
    net_pj = pj - pol.tau * P.phi_j
    
    # energy firm FOCs
    eq_energy = np.zeros(6)
    for j in range(J):
        eq_energy[2*j] = net_pj[j] * dEdK[j] - r
        eq_energy[2*j + 1] = net_pj[j] * dEdL[j] - w
    
    # final firm FOCs
    eq_final_r = P.a_c * Y / _clamp_pos(Kc) - r
    eq_final_w = P.b_c * Y / _clamp_pos(Lc) - w
    eq_final_pE = P.e_c * Y / _clamp_pos(E) - pE
    
    # factor market clearing
    eq_K = (Kc + Kj.sum()) - P.K_bar
    eq_L = (Lc + Lj.sum()) - L
    
    # household labour FOC
    Z = emissions(Ej, P.phi_j)
    T_rev = pol.tau * Z
    C = household_budget(w, r, P.K_bar, L, T_rev)
    eq_hh = household_labour_foc(w, C, pref, L)
    
    return np.concatenate([
        eq_energy,
        np.array([eq_final_r, eq_final_w, eq_final_pE, eq_K, eq_L, eq_hh], dtype=float)
    ])


def solve_equilibrium(P, pref: Preferences, pol: Policy, x0: Optional[np.ndarray] = None) -> Equilibrium:
    """Solve static general equilibrium"""
    if x0 is None:
        x0 = np.zeros(12)
        x0[0] = np.log(2.0)    # w
        x0[1] = np.log(0.2)    # r
        x0[2] = np.log(0.02)   # pE
        x0[3] = 0.5            # L (level)
        x0[4:] = np.log(0.3)   # allocations
    
    def F(x):
        return residuals_static_ge(x, P, pref, pol)
    
    sol = root(F, x0, method="hybr", tol=1e-10)
    x = np.asarray(sol.x, dtype=float)
    
    w = float(np.exp(np.clip(x[0], -30, 30)))
    r = float(np.exp(np.clip(x[1], -30, 30)))
    pE = float(np.exp(np.clip(x[2], -30, 30)))
    L = float(_clamp_pos(x[3]))
    
    Kj = np.array([np.exp(x[4]), np.exp(x[6]), np.exp(x[8])], dtype=float)
    Lj = np.array([np.exp(x[5]), np.exp(x[7]), np.exp(x[9])], dtype=float)
    Kc = float(np.exp(x[10]))
    Lc = float(np.exp(x[11]))
    
    Ej = np.zeros(J)
    for j in range(J):
        Ej[j] = energy_output(Kj[j], Lj[j], P.A_j[j], P.a_j[j])
    
    E = float(ces_quantity(Ej, P.eta, P.omega_E))
    Y = float(final_output(Kc, Lc, E, P.A_final, P.a_c, P.b_c, P.e_c))
    
    Z = emissions(Ej, P.phi_j)
    T_rev = pol.tau * Z
    C = household_budget(w, r, P.K_bar, L, T_rev)
    
    pj = _implied_component_prices_from_ces(Ej, E, pE, P.eta, P.omega_E)
    W = utility(C, L, Z, pref, pol.omega_damage)
    
    resid = F(x)
    max_abs = float(np.max(np.abs(resid)))
    
    return Equilibrium(
        success=bool(sol.success),
        message=str(sol.message),
        w=w, r=r, pE=pE, pj=pj,
        L=L, Kc=Kc, Lc=Lc, Kj=Kj, Lj=Lj,
        Ej=Ej, E=E, Y=Y, C=C,
        Z=Z, T_rev=T_rev,
        W=W,
        max_abs_resid=max_abs
    )


# ----------------------------
# Dynamic Model Components
@dataclass
class DynamicParams:
    """Parameters for dynamic model"""
    beta: float = 0.96     # discount factor
    delta: float = 0.08    # depreciation rate
    T: int = 10            # time horizon
    K_terminal: float = 2.51  # terminal capital target
    
    def validate(self) -> None:
        assert 0 < self.beta < 1, "beta must be in (0,1)"
        assert 0 < self.delta < 1, "delta must be in (0,1)"
        assert self.T > 0, "T must be positive"
        assert self.K_terminal > 0, "K_terminal must be positive"


@dataclass(frozen=True)
class DynamicEquilibrium:
    """Complete dynamic equilibrium path"""
    success: bool
    message: str
    
    # Paths (length T)
    K_vec: np.ndarray
    C_vec: np.ndarray
    Y_vec: np.ndarray
    E_vec: np.ndarray
    Z_vec: np.ndarray
    I_vec: np.ndarray
    
    # Welfare
    W_present_value: float
    
    # Policy
    tau: float
    omega_damage: float


# ----------------------------
# Dynamic Equilibrium Solver
def solve_dynamic_equilibrium_simple(
    K0: float,
    dyn_params: DynamicParams,
    static_params: Dict,
    policy_params: Dict,
    verbose: bool = False
) -> Optional[DynamicEquilibrium]:
    """
    Solve dynamic equilibrium with FULL energy structure from static model.
    
    Uses root finding to satisfy terminal capital constraint.
    """
    T = dyn_params.T
    beta = dyn_params.beta
    delta = dyn_params.delta
    K_terminal = dyn_params.K_terminal
    
    tau = policy_params['tau']
    omega = policy_params['omega_damage']
    
    # Build static model objects
    P_base = Params(
        L_bar=static_params['L_bar'],
        K_bar=K0,  # Will be updated each period
        A_final=static_params['A_final'],
        a_c=static_params['a_c'],
        b_c=static_params['b_c'],
        e_c=static_params['e_c'],
        A_j=np.array(static_params['A_j']),
        a_j=np.array(static_params['a_j']),
        eta=static_params['eta'],
        omega_E=np.array(static_params['omega_E']),
        phi_j=np.array(static_params['phi_j'])
    )
    
    pref = Preferences(
        gamma_hh=static_params['gamma'],
        sigma=static_params['sigma'],
        chi=static_params['chi']
    )
    
    def simulate_path_full(C0: float):
        """Simulate full path given C0, return terminal error"""
        global _SOLVE_COUNTER
        
        C_path = np.zeros(T)
        K_path = np.zeros(T + 1)
        equilibria = []
        
        K_path[0] = K0
        C_path[0] = C0
        
        for t in range(T):
            K_t = K_path[t]
            C_t = C_path[t]
            
            # Update capital endowment for period t
            P_t = replace(P_base, K_bar=K_t)
            
            # Policy for this period
            pol_t = Policy(tau=tau, omega_damage=omega, damage_type="utility")
            
            # Solve static equilibrium for period t
            try:
                _SOLVE_COUNTER += 1
                eq_t = solve_equilibrium(P_t, pref, pol_t)
                if not eq_t.success or eq_t.max_abs_resid > 1e-6:
                    if verbose:
                        print(f"Period {t}: Failed to solve equilibrium")
                    return C_path, K_path, equilibria, 1e9
            except Exception as e:
                if verbose:
                    print(f"Period {t}: Exception: {e}")
                return C_path, K_path, equilibria, 1e9
            
            equilibria.append(eq_t)
            
            # Extract period t outcomes
            Y_t = eq_t.Y
            
            # Investment (residual from resource constraint)
            I_t = Y_t - C_t
            if I_t < -1e-6:
                if verbose:
                    print(f"Period {t}: Negative investment I={I_t:.6f}")
                return C_path, K_path, equilibria, 1e9
            
            I_t = max(I_t, 0.0)
            
            # Capital accumulation
            K_path[t + 1] = (1 - delta) * K_t + I_t
            
            # Euler equation for next period's consumption (if t < T-1)
            if t < T - 1:
                K_next = K_path[t + 1]
                P_next = replace(P_base, K_bar=K_next)
                
                # Iterate to find consistent C_{t+1}
                C_next_guess = C_t
                
                for _ in range(5):  # Euler iterations
                    try:
                        _SOLVE_COUNTER += 1
                        eq_next = solve_equilibrium(P_next, pref, pol_t)
                        if eq_next.success:
                            r_next = eq_next.r
                            Y_next = eq_next.Y
                            
                            # Euler equation
                            gamma = pref.gamma_hh
                            growth_factor = (beta * (1 - delta + r_next)) ** (1.0 / gamma)
                            C_next_new = C_t * growth_factor
                            
                            # Make sure it's feasible
                            C_next_new = min(C_next_new, 0.95 * Y_next)
                            
                            if abs(C_next_new - C_next_guess) < 1e-5:
                                C_path[t + 1] = C_next_new
                                break
                            C_next_guess = 0.5 * C_next_guess + 0.5 * C_next_new
                        else:
                            C_path[t + 1] = C_t * 1.02
                            break
                    except:
                        C_path[t + 1] = C_t * 1.02
                        break
                else:
                    C_path[t + 1] = C_next_guess
        
        terminal_error = K_path[T] - K_terminal
        return C_path, K_path, equilibria, terminal_error
    
    # Root finding for C_0
    def objective(C0_log):
        C0 = np.exp(C0_log)
        _, _, _, term_err = simulate_path_full(C0)
        return term_err
    
    # Initial guess
    global _SOLVE_COUNTER
    P_0 = replace(P_base, K_bar=K0)
    pol_0 = Policy(tau=tau, omega_damage=omega, damage_type="utility")
    try:
        _SOLVE_COUNTER += 1
        eq_0 = solve_equilibrium(P_0, pref, pol_0)
        if eq_0.success:
            Y_approx = eq_0.Y
            I_ss = delta * K0
            C0_guess = max(Y_approx - I_ss, 0.1)
            C_lower = 0.01
            C_upper = 0.95 * Y_approx
        else:
            C0_guess = 0.3
            C_lower = 0.01
            C_upper = 0.8
    except:
        C0_guess = 0.3
        C_lower = 0.01
        C_upper = 0.8
    
    from scipy.optimize import brentq
    try:
        C0_log_opt = brentq(objective, np.log(C_lower), np.log(C_upper),
                           xtol=1e-6, maxiter=100)
        C0_opt = np.exp(C0_log_opt)
        success = True
        message = "Dynamic equilibrium solved"
    except Exception as e:
        if verbose:
            print(f"Root finding failed: {e}")
        return None
    
    # Simulate final path
    C_path, K_path, equilibria, term_err = simulate_path_full(C0_opt)
    
    if abs(term_err) > 1e-3:
        if verbose:
            print(f"Terminal error too large: {term_err}")
        return None
    
    # Extract paths
    Y_vec = np.array([eq.Y for eq in equilibria])
    E_vec = np.array([eq.E for eq in equilibria])
    Z_vec = np.array([eq.Z for eq in equilibria])
    I_vec = Y_vec - C_path
    
    # Welfare (present value)
    W_terms = np.array([utility(C_path[t], equilibria[t].L, Z_vec[t], pref, omega) 
                       for t in range(T)])
    discounts = np.array([beta**t for t in range(T)])
    W_PV = float(np.sum(discounts * W_terms))
    
    return DynamicEquilibrium(
        success=True,
        message=message,
        K_vec=K_path[:T],
        C_vec=C_path,
        Y_vec=Y_vec,
        E_vec=E_vec,
        Z_vec=Z_vec,
        I_vec=I_vec,
        W_present_value=W_PV,
        tau=tau,
        omega_damage=omega
    )


# ----------------------------
# Parallel Helper Functions
def _evaluate_tau_parallel(args):
    """Helper for parallel grid search"""
    tau, K0, dyn_params, static_params, omega_damage = args
    policy_params = {'tau': float(tau), 'omega_damage': float(omega_damage)}
    try:
        eq = solve_dynamic_equilibrium_simple(K0, dyn_params, static_params, policy_params, verbose=False)
        if eq.success:
            return (tau, eq.W_present_value)
    except:
        pass
    return None


def _solve_omega_parallel(args):
    """Helper for parallel omega sensitivity"""
    om, K0, dyn_params, static_params, tau_bounds, grid_n = args
    opt = find_optimal_tau_dynamic(K0, dyn_params, static_params, float(om),
                                   tau_bounds=tau_bounds, grid_n=grid_n, parallel=False)
    
    if opt['success']:
        eq = opt['equilibrium']
        row = summarize_dynamic_eq(eq, opt['tau_star'], float(om), 'omega', float(om))
        row['at_bound'] = opt['at_lower_bound'] or opt['at_upper_bound']
        status = "HIT UPPER BOUND!" if opt['at_upper_bound'] else (
                "HIT LOWER BOUND!" if opt['at_lower_bound'] else f"W = {opt['W_star']:.4f}")
        return (om, opt['tau_star'], status, row)
    else:
        row = summarize_dynamic_eq(None, np.nan, float(om), 'omega', float(om))
        row['at_bound'] = False
        return (om, np.nan, "FAILED", row)


def _solve_ec_parallel(args):
    """Helper for parallel energy share sensitivity"""
    ec, K0, dyn_params, base_static_params, omega_damage, tau_bounds, grid_n = args
    
    # Adjust shares
    b_c = base_static_params['b_c']
    new_a_c = 1.0 - b_c - ec
    
    if new_a_c <= 0:
        return (ec, np.nan, "INVALID (a_c <= 0)", None, new_a_c)
    
    # Update parameters
    static_params = base_static_params.copy()
    static_params['a_c'] = new_a_c
    static_params['e_c'] = ec
    
    opt = find_optimal_tau_dynamic(K0, dyn_params, static_params, omega_damage,
                                   tau_bounds=tau_bounds, grid_n=grid_n, parallel=False)
    
    if opt['success']:
        eq = opt['equilibrium']
        row = summarize_dynamic_eq(eq, opt['tau_star'], omega_damage, 'e_c', float(ec))
        row['at_bound'] = opt['at_lower_bound'] or opt['at_upper_bound']
        row['a_c'] = new_a_c
        status = "HIT UPPER BOUND!" if opt['at_upper_bound'] else (
                "HIT LOWER BOUND!" if opt['at_lower_bound'] else f"W = {opt['W_star']:.4f}")
        return (ec, opt['tau_star'], status, row, new_a_c)
    else:
        row = summarize_dynamic_eq(None, np.nan, omega_damage, 'e_c', float(ec))
        row['at_bound'] = False
        row['a_c'] = new_a_c
        return (ec, np.nan, "FAILED", row, new_a_c)


# ----------------------------
# Optimal Tax Finding
def find_optimal_tau_dynamic(
    K0: float,
    dyn_params: DynamicParams,
    static_params: Dict,
    omega_damage: float,
    tau_bounds: Tuple[float, float] = (0.0, 0.30),
    grid_n: int = 21,
    parallel: bool = False
) -> Dict:
    """Find optimal carbon tax that maximizes welfare"""
    tau_lo, tau_hi = tau_bounds
    grid = np.linspace(tau_lo, tau_hi, grid_n)
    
    # Grid search (parallel or serial)
    args_list = [(tau, K0, dyn_params, static_params, omega_damage) for tau in grid]
    
    if parallel:
        n_cores = min(cpu_count(), len(grid))
        with Pool(n_cores) as pool:
            results = pool.map(_evaluate_tau_parallel, args_list)
    else:
        results = [_evaluate_tau_parallel(args) for args in args_list]
    
    feasible = [r for r in results if r is not None]
    
    if len(feasible) == 0:
        return {
            'tau_star': np.nan,
            'W_star': -1e12,
            'success': False,
            'at_lower_bound': False,
            'at_upper_bound': False,
        }
    
    # Find best from grid
    tau0, W0 = max(feasible, key=lambda x: x[1])
    
    # Local refinement
    def objective(tau_arr):
        tau = float(tau_arr[0])
        policy_params = {'tau': tau, 'omega_damage': float(omega_damage)}
        try:
            eq = solve_dynamic_equilibrium_simple(K0, dyn_params, static_params, policy_params, verbose=False)
            if not eq.success:
                return 1e9
            return -eq.W_present_value
        except:
            return 1e9
    
    res = minimize(objective, x0=np.array([tau0]), bounds=[tau_bounds], method='L-BFGS-B', options={'ftol': 1e-8})
    tau_star = float(res.x[0])
    
    # Verify solution
    policy_params = {'tau': tau_star, 'omega_damage': float(omega_damage)}
    eq_star = solve_dynamic_equilibrium_simple(K0, dyn_params, static_params, policy_params, verbose=False)
    
    # Check boundary
    at_lower = abs(tau_star - tau_lo) < 1e-4
    at_upper = abs(tau_star - tau_hi) < 1e-4
    
    return {
        'tau_star': tau_star,
        'W_star': eq_star.W_present_value if eq_star else -1e12,
        'success': eq_star is not None and eq_star.success,
        'equilibrium': eq_star,
        'at_lower_bound': at_lower,
        'at_upper_bound': at_upper
    }


# ----------------------------
# Sensitivity Analysis
def summarize_dynamic_eq(eq: Optional[DynamicEquilibrium], tau: float, omega: float, 
                        param_type: str, param_value: float) -> Dict:
    """Summarize dynamic equilibrium for table output"""
    if eq is None:
        return {
            'Param': param_value,
            'tau': tau,
            'W_PV': np.nan,
            'Y_avg': np.nan,
            'C_avg': np.nan,
            'E_avg': np.nan,
            'Z_total': np.nan,
            'K_final': np.nan,
            'inv_rate_avg': np.nan,
            'success': False
        }
    
    Y_avg = float(np.mean(eq.Y_vec))
    C_avg = float(np.mean(eq.C_vec))
    E_avg = float(np.mean(eq.E_vec))
    Z_total = float(np.sum(eq.Z_vec))
    K_final = float(eq.K_vec[-1]) if len(eq.K_vec) > 0 else np.nan
    inv_rate = eq.I_vec / np.clip(eq.Y_vec, 1e-10, None)
    inv_rate_avg = float(np.mean(inv_rate))
    
    return {
        'Param': param_value,
        'tau': tau,
        'W_PV': eq.W_present_value,
        'Y_avg': Y_avg,
        'C_avg': C_avg,
        'E_avg': E_avg,
        'Z_total': Z_total,
        'K_final': K_final,
        'inv_rate_avg': inv_rate_avg,
        'success': eq.success
    }


def omega_sensitivity_dynamic(
    K0: float,
    dyn_params: DynamicParams,
    static_params: Dict,
    omega_grid: np.ndarray,
    tau_bounds: Tuple[float, float] = (0.0, 0.30),
    grid_n: int = 21
) -> pd.DataFrame:
    """Run sensitivity analysis over omega_damage parameter"""
    print("\n" + "="*70)
    print("PANEL A: OMEGA SENSITIVITY (DYNAMIC MODEL)")
    print("="*70)
    
    n_cores = min(cpu_count(), len(omega_grid))
    print(f"\nUsing {n_cores} CPU cores for parallel computation...")
    
    args_list = [(om, K0, dyn_params, static_params, tau_bounds, grid_n) 
                 for om in omega_grid]
    
    with Pool(n_cores) as pool:
        results = pool.map(_solve_omega_parallel, args_list)
    
    # Build DataFrame
    rows = []
    for om, tau_star, status, row in results:
        print(f"\nω = {om:.3f}: τ* = {tau_star:.4f}, {status}")
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def energy_share_sensitivity_dynamic(
    K0: float,
    dyn_params: DynamicParams,
    base_static_params: Dict,
    ec_grid: np.ndarray,
    omega_damage: float = 0.08,
    tau_bounds: Tuple[float, float] = (0.0, 0.30),
    grid_n: int = 21
) -> pd.DataFrame:
    """Run sensitivity analysis over energy share parameter"""
    print("\n" + "="*70)
    print("PANEL B: ENERGY SHARE SENSITIVITY (DYNAMIC MODEL)")
    print("="*70)
    
    n_cores = min(cpu_count(), len(ec_grid))
    print(f"\nUsing {n_cores} CPU cores for parallel computation...")
    
    args_list = [(ec, K0, dyn_params, base_static_params, omega_damage, tau_bounds, grid_n) 
                 for ec in ec_grid]
    
    with Pool(n_cores) as pool:
        results = pool.map(_solve_ec_parallel, args_list)
    
    # Build DataFrame
    rows = []
    for ec, tau_star, status, row, new_a_c in results:
        if row:
            print(f"\ne_c = {ec:.3f}: τ* = {tau_star:.4f}, {status}")
            rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


# ----------------------------
# Plotting
def plot_dynamic_sensitivity(df_omega: pd.DataFrame, df_ec: pd.DataFrame):
    """Plot sensitivity analysis results"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Dynamic Model: Sensitivity Analysis', fontsize=16, fontweight='bold')
    
    # Panel A: Omega sensitivity
    if 'Param' in df_omega.columns and len(df_omega) > 0:
        axes[0, 0].plot(df_omega['Param'], df_omega['tau'], 'o-', linewidth=2)
        axes[0, 0].set_xlabel('Damage weight ω')
        axes[0, 0].set_ylabel('Optimal tax τ*')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(df_omega['Param'], df_omega['W_PV'], 'o-', linewidth=2)
        axes[0, 1].set_xlabel('Damage weight ω')
        axes[0, 1].set_ylabel('Welfare (PV)')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[0, 2].plot(df_omega['Param'], df_omega['Z_total'], 'o-', linewidth=2)
        axes[0, 2].set_xlabel('Damage weight ω')
        axes[0, 2].set_ylabel('Total emissions (PV)')
        axes[0, 2].grid(True, alpha=0.3)
    
    # Panel B: Energy share sensitivity
    if 'Param' in df_ec.columns and len(df_ec) > 0:
        axes[1, 0].plot(df_ec['Param'], df_ec['tau'], 'o-', linewidth=2)
        axes[1, 0].set_xlabel('Energy share e_c')
        axes[1, 0].set_ylabel('Optimal tax τ*')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(df_ec['Param'], df_ec['W_PV'], 'o-', linewidth=2)
        axes[1, 1].set_xlabel('Energy share e_c')
        axes[1, 1].set_ylabel('Welfare (PV)')
        axes[1, 1].grid(True, alpha=0.3)
        
        axes[1, 2].plot(df_ec['Param'], df_ec['inv_rate_avg'], 'o-', linewidth=2)
        axes[1, 2].set_xlabel('Energy share e_c')
        axes[1, 2].set_ylabel('Avg investment rate')
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dynamic_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.show()


def print_dynamic_summary(df_omega: pd.DataFrame, df_ec: pd.DataFrame):
    """Print summary tables"""
    print("\n" + "="*80)
    print("DYNAMIC MODEL: SENSITIVITY ANALYSIS SUMMARY")
    print("="*80)
    
    print("\nPanel A: Damage Weight Sensitivity (ω)")
    print("-" * 80)
    cols_a = ['Param', 'tau', 'at_bound', 'W_PV', 'Y_avg', 'C_avg', 'Z_total', 'K_final', 'inv_rate_avg', 'success']
    print(df_omega[cols_a].to_string(index=False))
    
    print("\n" + "="*80)
    print("\nPanel B: Energy Share Sensitivity (e_c)")
    print("-" * 80)
    cols_b = ['Param', 'tau', 'at_bound', 'W_PV', 'Y_avg', 'C_avg', 'E_avg', 'inv_rate_avg', 'success']
    print(df_ec[cols_b].to_string(index=False))
    
    print("\n" + "="*80)


# ----------------------------
# Main Analysis
def run_dynamic_sensitivity_analysis():
    """Run complete sensitivity analysis"""
    global _SOLVE_COUNTER
    _SOLVE_COUNTER = 0
    
    print("\n" + "="*80)
    print("DYNAMIC GE: OPTIMAL POLICY (WITH FULL ENERGY STRUCTURE)")
    print("="*80)
    
    # Setup
    K0 = 2.51
    dyn_params = DynamicParams(beta=0.96, delta=0.08, T=10, K_terminal=2.51)
    
    base_static_params = {
        'L_bar': 1.0,
        'A_final': 1.0,
        'a_c': 0.247,
        'b_c': 0.655,
        'e_c': 0.098,
        'gamma': 2.0,
        'sigma': 0.5,
        'chi': 1.0,
        'A_j': [5.68, 5.68, 12.5],
        'a_j': [0.86, 0.86, 0.90],
        'eta': 2.0,
        'omega_E': [0.524/1.0, 0.396/1.0, 0.080/1.0],
        'phi_j': [1.0, 0.7741, 0.0]
    }
    
    # Panel A: Omega sensitivity
    omega_grid = np.array([0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.16, 0.20, 0.24, 0.30, 0.40])
    df_omega = omega_sensitivity_dynamic(
        K0, dyn_params, base_static_params, omega_grid,
        tau_bounds=(0.0, 0.30), grid_n=31
    )
    
    # Panel B: Energy share sensitivity
    ec_grid = np.array([0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16])
    df_ec = energy_share_sensitivity_dynamic(
        K0, dyn_params, base_static_params, ec_grid,
        omega_damage=0.08, tau_bounds=(0.0, 0.30), grid_n=31
    )
    
    # Print results
    print_dynamic_summary(df_omega, df_ec)
    
    # Plot
    print("\nGenerating plots...")
    plot_dynamic_sensitivity(df_omega, df_ec)
    
    # Key insights
    print("\n" + "="*80)
    print("KEY INSIGHTS FROM DYNAMIC MODEL")
    print("="*80)
    
    if df_omega['success'].all():
        tau_range = df_omega['tau'].max() / df_omega['tau'].min()
        print(f"\n1. DAMAGE SENSITIVITY:")
        print(f"   - When ω increases {omega_grid.max()/omega_grid.min():.1f}x (0.04→0.40):")
        print(f"   - Optimal tax increases {tau_range:.2f}x")
        print(f"   - Welfare (PV) falls from {df_omega['W_PV'].iloc[0]:.2f} to {df_omega['W_PV'].iloc[-1]:.2f}")
        print(f"   - Emissions (PV) fall {(1-df_omega['Z_total'].iloc[-1]/df_omega['Z_total'].iloc[0])*100:.1f}%")
        print(f"   - Average consumption falls {(1-df_omega['C_avg'].iloc[-1]/df_omega['C_avg'].iloc[0])*100:.1f}%")
    
    if df_ec['success'].all():
        print(f"\n2. ENERGY SHARE SENSITIVITY:")
        print(f"   - As e_c varies from {ec_grid.min():.2f} to {ec_grid.max():.2f}:")
        print(f"   - Optimal tax ranges from {df_ec['tau'].min():.4f} to {df_ec['tau'].max():.4f}")
        print(f"   - Investment rate ranges from {df_ec['inv_rate_avg'].min()*100:.1f}% to {df_ec['inv_rate_avg'].max()*100:.1f}%")
        print(f"   - More energy-intensive economies need stronger carbon pricing")
    
    print("\n3. DYNAMIC vs STATIC:")
    print("   - Dynamic model accounts for capital accumulation")
    print("   - Investment-consumption tradeoff is explicit")
    print("   - Welfare measured as present value over time")
    print("   - Policy affects not just current but all future periods")
    
    print("\n" + "="*80)
    
    # Computational statistics
    print("\n" + "="*80)
    print("COMPUTATIONAL STATISTICS")
    print("="*80)
    print(f"\nTotal equilibrium solves: {_SOLVE_COUNTER:,}")
    print(f"Parameters tested:")
    print(f"  Panel A: {len(omega_grid)} omega values")
    print(f"  Panel B: {len(ec_grid)} e_c values")
    print(f"  Grid search points: 31 per optimization")
    print(f"  Time horizon: {dyn_params.T} periods")
    print(f"\nAverage solves per parameter: {_SOLVE_COUNTER/(len(omega_grid)+len(ec_grid)):.0f}")
    print("="*80)
    
    return df_omega, df_ec


if __name__ == "__main__":
    df_omega, df_ec = run_dynamic_sensitivity_analysis()
