import numpy as np
import pandas as pd
from scipy.optimize import root, minimize
import matplotlib.pyplot as plt
# Our targets and parameters

L_bar = 1.0
GDP24 = 3.108551
K24   = 7.8 # 7.8
KY_target = K24 / GDP24     # =? 2.51
K_bar = KY_target * L_bar

# Emissions intensities
phi_j = np.array([1.0, 0.7741, 0.0])
omega = 0.08

# Capital shares 
a_j = np.array([0.86, 0.86, 0.90])

# Base TFP guesses
A_base = np.array([5.0, 5.0, 12.5])

# Final good cobb douglas shares
b_c = 0.6548
e_c = 0.098
a_c = 1 - b_c - e_c

# Preferences
tau = 0.00231  # initial tax guess
gamma_hh = 2.0
sigma = 0.5
chi_hh = 1.0 # initial guess

# Targets
raw_energy_shares = np.array([0.524, 0.396, 0.080])
TARGET_SHARES = raw_energy_shares / raw_energy_shares.sum()

EY_TARGET = 7.0
L_TARGET  = 0.5 * L_bar
Y_TARGET  = 1.0
TARGET_PRICE = e_c / EY_TARGET

# local solutions functions

def softmax(z):
    z = np.array(z)
    ez = np.exp(z - np.max(z))
    return ez / ez.sum()

def get_allocations(x, current_Ls):
    sK = softmax(x[:4])
    sL = softmax(x[4:8])
    return sK*K_bar, sL*current_Ls

# Production block - Energy and output. 

def energy_outputs(K, L, gammas, A_scaler):
    """E_j, dE/dK_j, dE/dL_j"""
    A_eff = A_base * A_scaler
    e  = np.zeros(3)
    dK = np.zeros(3)
    dL = np.zeros(3)

    for j in range(3):
        K_j = max(K[j], 1e-9)
        L_j = max(L[j], 1e-9)
        gamma = gammas[j]

        base = A_eff[j] * (K_j**a_j[j]) * (L_j**(1 - a_j[j]))
        e[j] = base**gamma

        dK[j] = gamma * a_j[j] * e[j] / K_j
        dL[j] = gamma * (1 - a_j[j]) * e[j] / L_j

    return e, e.sum(), dK, dL

def final_output(Kc, Lc, E, A_final):
    Kc = max(Kc, 1e-9)
    Lc = max(Lc, 1e-9)
    E  = max(E , 1e-9)
    return A_final * (Kc**a_c) * (Lc**b_c) * (E**e_c)

# HH Block

def household_values(w, r, L_s, e_vec, chi):
    T = tau * np.dot(phi_j, e_vec)
    income = w*L_s + r*K_bar + T
    C = max(income, 1e-9)
    Lrhs = w / (chi * max(C,1e-6)**gamma_hh)
    Lrhs = min(Lrhs, 1e6)    
    L_s_foc = max(Lrhs**(1.0/sigma), 1e-9)
    return C, L_s_foc

# Equalibrium System
def ge_system(x, params):
    A_energy = params[:3]
    A_final  = params[3]
    gammas   = params[4:7]
    chi      = params[7]

    r = np.exp(np.clip(x[8], -10, 10))
    w = np.exp(np.clip(x[9], -10, 10))    
    L_s = max(x[10], 0.01)

    K_alloc, L_alloc = get_allocations(x[:8], L_s)
    K_e, K_c = K_alloc[:3], K_alloc[3]
    L_e, L_c = L_alloc[:3], L_alloc[3]

    e_vec, E_total, dK_e, dL_e = energy_outputs(K_e, L_e, gammas, A_energy)
    Y = final_output(K_c, L_c, E_total, A_final)

    if E_total > 1e-9:
        P_E = e_c * Y / E_total
    else:
        P_E = 10.0

    C, L_s_foc = household_values(w, r, L_s, e_vec, chi)

    F = np.zeros(11)

    # Energy FOCs
    for j in range(3):
        p_net = P_E - tau * phi_j[j]
        F[j]     = p_net*dK_e[j] - r
        F[j + 3] = p_net*dL_e[j] - w

    # Final good FOCs
    F[6] = a_c*Y/max(K_c,1e-9) - r
    F[7] = b_c*Y/max(L_c,1e-9) - w

    # Household FOC
    F[10] = L_s - L_s_foc

    return F

def set_tau(new_tau):
    global tau
    tau = new_tau
    
def compute_welfare(sol, theta):
    x = sol.x
    L_s = max(x[10], 1e-9)
    r   = np.exp(x[8])
    w   = np.exp(x[9])

    # Unpack structural params
    A_energy = theta[:3]
    gammas   = theta[4:7]
    chi      = theta[7]

    # Recover allocations
    K_alloc, L_alloc = get_allocations(x[:8], L_s)
    K_e, K_c = K_alloc[:3], K_alloc[3]
    L_e, L_c = L_alloc[:3], L_alloc[3]

    # Recompute energy outputs to get e_vec
    e_vec, E_total, _, _ = energy_outputs(K_e, L_e, gammas, A_energy)
    Z = np.dot(phi_j, e_vec)  # total emissions

    # C_with_rebate is current consumption including tax rebate
    C_with_rebate, _ = household_values(w, r, L_s, e_vec, chi)

    # Subtract damages from consumption
    C_eff = C_with_rebate - omega * Z
    C_eff = max(C_eff, 1e-9)

    util = (C_eff**(1 - gamma_hh)) / (1 - gamma_hh) \
           - chi * (L_s**(1 + sigma)) / (1 + sigma)

    return float(util)

def welfare_at_tax(tau_value, theta):
    set_tau(tau_value)
    sol = solve_ge(theta)
    if not sol.success:
        return -1e12  # penalize non-convergence
    return compute_welfare(sol, theta)

def welfare_grid(theta, tau_min=0, tau_max=200, n=80):
    taus = np.linspace(tau_min, tau_max, n)
    welfare_vals = []
    for t in taus:
        W = welfare_at_tax(t, theta)
        welfare_vals.append(W)
        print(f"tau={t:.2f}, Welfare={W:.6f}")
    return taus, np.array(welfare_vals)

def solve_ge(theta):
    """Try LM first; if it fails, fall back to HYBR."""
    x0 = np.zeros(11)
    x0[8]  = np.log(0.2)
    x0[9]  = np.log(2.0)
    x0[10] = 0.3

    params = theta
    sol = root(ge_system, x0, args=(params,), method="lm", tol=1e-7)

    if not sol.success:
        sol = root(ge_system, x0, args=(params,), method="hybr", tol=1e-7)

    return sol

# Threaded calibration loss with jittered restart for new local loss solution.

Nfeval = 1

def calibration_loss(theta):
    global Nfeval

    A_energy = theta[:3]
    A_final  = theta[3]
    gammas   = theta[4:7]
    chi      = theta[7]

    sol = solve_ge(theta)
    if not sol.success:
        return 1e9

    L_s = max(sol.x[10], 1e-9)
    K_alloc, L_alloc = get_allocations(sol.x[:8], L_s)

    e_vec, E_total, _, _ = energy_outputs(K_alloc[:3], L_alloc[:3], gammas, A_energy)
    if E_total <= 1e-9:
        return 1e9

    Y = final_output(K_alloc[3], L_alloc[3], E_total, A_final)
    if Y <= 1e-9:
        return 1e9

    model_shares = e_vec / E_total
    price_model = e_c * Y / E_total
    EY_model = E_total / Y
    KY_model = K_bar / Y

    share_err = np.sum((model_shares - TARGET_SHARES)**2)
    price_err = (price_model - TARGET_PRICE)**2
    L_err     = (L_s - L_TARGET)**2
    Y_err     = (Y - Y_TARGET)**2
    KY_err    = (KY_model - KY_target)**2
    EY_err    = (EY_model - EY_TARGET)**2
    reg_err = 0.1 * np.sum((gammas - 0.9)**2)

    loss = ( #adjust weights as needed
        25*share_err +
        25*price_err +
        150*L_err +
        25*Y_err +
        25*EY_err +
        25*KY_err +
        25*reg_err
    )

    if Nfeval % 50 == 0:
        print(f"[{Nfeval}] Loss={loss:.4f}  Y={Y:.3f}  L={L_s:.3f}  P_E={price_model:.3f}")

    Nfeval += 1
    return loss

def run_calibration_loop(start_theta=None):
    global Nfeval
    
    # Use the passed starting point if available, otherwise use the hardcoded guess
    if start_theta is not None:
        theta_best = np.array(start_theta)
        print(">>> Starting calibration from previous best theta...")
    else:
        # Default starting guess
        theta_best = np.array([
            5.0, 5.0, 1.2,
            3.0,
            0.85, 0.83, 0.89,
            1.5
        ])

    best_loss = 1e9
    num_restarts = 1  # Reduce restarts for robustness speed
    
    # Check if our starting point is already good enough
    initial_loss = calibration_loss(theta_best)
    if initial_loss < 0.01:
        print(f"Initial theta is already good (Loss={initial_loss:.6f}). Skipping loop.")
        return theta_best

    bounds = [
        (1, 10), (1, 10), (1, 5),
        (0.5, 10.0),
        (0.7, 1.0), (0.7, 1.0), (0.7, 1.0),
        (0.1, 5.0)
    ]

    for attempt in range(1, num_restarts + 1):
        if attempt == 1:
            start = theta_best
        else:
            jitter = np.random.uniform(0.99, 1.01, size=len(theta_best)) # Smaller jitter
            start = theta_best * jitter

        res = minimize(
            calibration_loss,
            start,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-6} # Reduced tolerance for speed
        )

        loss = res.fun
        if loss < best_loss:
            best_loss = loss
            theta_best = res.x

    return theta_best

# Reporting function

def print_final_report(theta):
    """
    Print a structured calibration report and produce summary plots.
    """
    import matplotlib.pyplot as plt

    # Unpack parameters
    A_energy = theta[:3]
    A_final  = theta[3]
    gammas   = theta[4:7]
    chi      = theta[7]

    # Solve GE at a calibrated theta
    sol = solve_ge(theta)
    if not sol.success:
        print("\nWARNING: GE solver did NOT CONVERGE at these parameters.")
    x = sol.x

    L_s = max(x[10], 1e-9)
    r   = np.exp(x[8])
    w   = np.exp(x[9])

    # allocations
    K_alloc, L_alloc = get_allocations(x[:8], L_s)
    K_e, K_c = K_alloc[:3], K_alloc[3]
    L_e, L_c = L_alloc[:3], L_alloc[3]

    # energy and final output
    e_vec, E_total, dK_e, dL_e = energy_outputs(K_e, L_e, gammas, A_energy)
    Y = final_output(K_c, L_c, E_total, A_final)
    total_emissions = np.dot(phi_j, e_vec)
    emissions_intensity = total_emissions / Y

    # prices and household
    P_E = e_c * Y / E_total

    # compute household values WITH rebate
    C_with_rebate, _ = household_values(w, r, L_s, e_vec, chi)

    # Compute tax revenues
    tax_revenue = tau * np.dot(phi_j, e_vec)

    # Compute consumption WITHOUT tax rebate
    C_no_rebate = w * L_s + r * K_bar


    # key ratios
    KY_model = K_bar / Y
    EY_model = E_total / Y
    model_shares = e_vec / E_total
    energy_spend_share = (P_E * E_total) / Y   # should be close to e_c
    # Run welfare grid around the calibrated parameters
    
    # recompute objective for context
    current_loss = calibration_loss(theta)
    
    # Compute taxes
    tax_revenue = tau * np.dot(phi_j, e_vec)

    # Consumption WITHOUT rebate
    C_no_rebate = w * L_s + r * K_bar

    # Consumption WITH rebate
    C_with_rebate = C_no_rebate + tax_revenue
    # Welfare at the calibrated baseline tau

    # Continuous optimum BEFORE macro_data
    obj = lambda t: -welfare_at_tax(t[0], theta)
    taus = np.linspace(0.0, 0.05, 40)
    Wvals = np.array([welfare_at_tax(t, theta_ec) for t in taus])
    tau0 = taus[np.argmax(Wvals)]

    res = minimize(
        lambda t: -welfare_at_tax(t[0], theta_ec),
        x0=[tau0],
        bounds=[(0.0, 0.05)]
    )
    tau_star = res.x[0]    
    W_star = welfare_at_tax(tau_star_cont, theta)

    # Macro Table
    macro_data = [
        ("Taxes collected T",      tax_revenue,    None),
        ("Consumption (no rebate)", C_no_rebate,   None),
        ("Consumption (with rebate)", C_with_rebate, None),
        ("Output Y",            Y,          Y_TARGET),
        ("Labour L_s",          L_s,        L_TARGET),
        ("K/Y",                 KY_model,   KY_target),
        ("E/Y",                 EY_model,   EY_TARGET),
        ("Energy price P_E",    P_E,        TARGET_PRICE),
        ("Energy spend share",  energy_spend_share, e_c),
        ("Wage w",              w,          None),
        ("Rental rate r",       r,          None),
        ("Welfare at tau*",     W_star,              None),
        ("Net Emissions", total_emissions, None),
        ("Emissions intensity (per unit Y)", emissions_intensity, None),
    ]

    rows = []
    for name, model_val, target_val in macro_data:
        if target_val is None or target_val == 0:
            abs_err = None
            rel_err = None
        else:
            abs_err = model_val - target_val
            rel_err = 100 * abs_err / target_val
        rows.append({
            "Variable":   name,
            "Model":      model_val,
            "Target":     target_val,
            "Abs_Error":  abs_err,
            "Rel_Error_%": rel_err
        })

    df_macro = pd.DataFrame(rows)

    # Energy share Table
    share_rows = []
    labels_energy = ["Oil", "Gas", "Clean"]
    for j in range(3):
        m = model_shares[j]
        t = TARGET_SHARES[j]
        abs_err = m - t
        rel_err = 100 * abs_err / t if t != 0 else None
        share_rows.append({
            "Tech":        labels_energy[j],
            "Model_Share": m,
            "Target_Share": t,
            "Abs_Error":   abs_err,
            "Rel_Error_%": rel_err
        })
    df_shares = pd.DataFrame(share_rows)

    # Sector Table
    total_K = K_alloc.sum()
    total_L = L_alloc.sum()
    sector_labels = ["Oil", "Gas", "Clean", "Final"]

    sector_outputs = list(e_vec) + [Y]
    sector_rows = []
    for i in range(4):
        K_i = K_alloc[i]
        L_i = L_alloc[i]
        Y_i = sector_outputs[i]
        sector_rows.append({
            "Sector":        sector_labels[i],
            "K":             K_i,
            "L":             L_i,
            "Output":        Y_i,
            "K_Share":       K_i / total_K,
            "L_Share":       L_i / total_L
        })
    df_sector = pd.DataFrame(sector_rows)

    # Paramater table
    df_params = pd.DataFrame([
        {"Parameter":"A_oil",        "Value":A_energy[0], "Type":"TFP (energy)"},
        {"Parameter":"A_gas",        "Value":A_energy[1], "Type":"TFP (energy)"},
        {"Parameter":"A_clean",      "Value":A_energy[2], "Type":"TFP (energy)"},
        {"Parameter":"A_final",      "Value":A_final,     "Type":"TFP (final)"},
        {"Parameter":"gamma_oil",    "Value":gammas[0],   "Type":"returns to scale"},
        {"Parameter":"gamma_gas",    "Value":gammas[1],   "Type":"returns to scale"},
        {"Parameter":"gamma_clean",  "Value":gammas[2],   "Type":"returns to scale"},
        {"Parameter":"chi_hh",       "Value":chi,         "Type":"preference (labour disutility)"},
        {"Parameter":"a_c",          "Value":a_c,         "Type":"capital share final"},
        {"Parameter":"b_c",          "Value":b_c,         "Type":"labour share final"},
        {"Parameter":"e_c",          "Value":e_c,         "Type":"energy share final"},
        
    ])

    # prints report
    pd.set_option("display.float_format", lambda v: f"{v: .4f}")

    print("\n=================================================================")
    print(" FINAL CALIBRATION REPORT")
    print("=================================================================")
    print(f"Calibration loss at theta: {current_loss:.6f}")
    print("Solver success:", sol.success)

    print("\n--- Calibrated Parameters ---")
    print(df_params.to_string(index=False))

    print("\n--- Macro Outcomes vs Targets ---")
    print(df_macro.to_string(index=False))

    print("\n--- Energy Shares (Model vs Target) ---")
    print(df_shares.to_string(index=False))

    print("\n--- Sectoral Allocations ---")
    print(df_sector.to_string(index=False))

    # PLOTS

    # 1) Energy shares bar chart
    plt.figure(figsize=(6,4))
    x = np.arange(3)
    width = 0.35
    plt.bar(x - width/2, model_shares, width, label="Model")
    plt.bar(x + width/2, TARGET_SHARES, width, label="Target")
    plt.xticks(x, labels_energy)
    plt.ylabel("Share of total energy")
    plt.title("Energy Shares: Model vs Target")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2) Macro fit vs targets
    macro_names = ["L_s", "K/Y", "E/Y", "P_E", "Y"]
    macro_model_vals  = [L_s, KY_model, EY_model, P_E, Y]
    macro_target_vals = [L_TARGET, KY_target, EY_TARGET, TARGET_PRICE, Y_TARGET]

    plt.figure(figsize=(7,4))
    x2 = np.arange(len(macro_names))
    width2 = 0.35
    plt.bar(x2 - width2/2, macro_model_vals, width2, label="Model")
    plt.bar(x2 + width2/2, macro_target_vals, width2, label="Target")
    plt.xticks(x2, macro_names)
    plt.title("Macro Targets: Model vs Data")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 3) Sectoral outputs
    plt.figure(figsize=(6,4))
    plt.bar(sector_labels, sector_outputs)
    plt.ylabel("Output")
    plt.title("Sectoral Outputs (Energy and Final Good)")
    plt.tight_layout()
    plt.show()
    # 4) Welfare grid
    print("\nComputing welfare grid...")
    taus, Ws = welfare_grid(theta, tau_min=0.00221, tau_max=0.00241, n=50)
    tau_star = taus[np.argmax(Ws)]
    print(f"\nGrid-search optimal tau* =? {tau_star:.3f}")

    #   welfare sensuety around tau
    print("\nPlotting welfare sensitivity around tau* ...")

    # Define a symmetric grid around tau*
    span = 0.01      # 20% on either side
    tau_low  = max(0.0, tau_star_cont * (1 - span))
    tau_high = tau_star_cont * (1 + span)

    # Build grid
    taus = np.linspace(tau_low, tau_high, 10)
    welfare_vals = []

    for t in taus:
        W = welfare_at_tax(t, theta)
        welfare_vals.append(W)

    welfare_vals = np.array(welfare_vals)

    # Identify max on this local grid
    idx_star = np.argmax(welfare_vals)
    tau_star_grid = taus[idx_star]
    W_star_grid = welfare_vals[idx_star]

    # Plot
    plt.figure(figsize=(8,5))
    plt.plot(taus, welfare_vals, linewidth=3)
    plt.scatter([tau_star_grid], [W_star_grid], color='black', s=60)

    plt.axvline(tau_star_cont, linestyle='--', linewidth=2, label=f"Continuous optimum tau*={tau_star_cont:.5f}")

    plt.title("Welfare Sensitivity Around Optimal Tax Rate", fontsize=14)
    plt.xlabel("Energy Tax tau")
    plt.ylabel("Welfare (Utility)")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"\nLocal grid optimum tau =? {tau_star_grid:.6f}")
    print(f"Continuous optimum tau* =? {tau_star_cont:.6f}")
    print(f"Welfare difference W(tau_grid*) - W(tau*) = {W_star_grid - welfare_at_tax(tau_star_cont, theta):.6e}")

    # Continuous optimum
    print(f"\nContinuous optimum tau* = {tau_star_cont:.3f}")

def simple_tax_grid(theta):
    grid = np.linspace(0.0, 0.02, 50) 
    
    best_tau = 0.0
    best_w = -1e12
    
    for t in grid:
        w = welfare_at_tax(t, theta)
        if w > best_w and w != -1e12:
            best_w = w
            best_tau = t
            
    # Stage 2: refine very closely around the winner
    if best_tau > 0:
        span = 0.002 # +/- 0.2%
        fine_grid = np.linspace(max(0, best_tau - span), best_tau + span, 40)
        for t in fine_grid:
            w = welfare_at_tax(t, theta)
            if w > best_w:
                best_w = w
                best_tau = t
                
    return best_tau, best_w

def get_full_macro_state(theta, tau_val, scenario_name, param_val):
    """
    Solves the model at a specific tax and returns a dictionary of ALL variables.
    """
    # Set the tax and solve
    set_tau(tau_val)
    sol = solve_ge(theta)
    
    if not sol.success:
        return None # Skip failed runs

    # Extract Basic Variables
    x = sol.x
    r   = np.exp(x[8])
    w   = np.exp(x[9])
    L_s = max(x[10], 1e-9)
    
    # Allocations
    K_alloc, L_alloc = get_allocations(x[:8], L_s)
    K_e, K_c = K_alloc[:3], K_alloc[3]
    L_e, L_c = L_alloc[:3], L_alloc[3]

    # Production & Energy
    A_energy = theta[:3]
    gammas   = theta[4:7]
    A_final  = theta[3]
    chi_val  = theta[7]

    e_vec, E_total, _, _ = energy_outputs(K_e, L_e, gammas, A_energy)
    Y = final_output(K_c, L_c, E_total, A_final)
    
    # Emissions
    Z = np.dot(phi_j, e_vec)  # Total emissions
    
        # Consumption & Government
    # Tax Revenue = tau * Emissions (since tax is on emissions intensity phi * e)
    Tax_Rev = tau_val * Z
    
    # Household Income = wL + rK + T
    Income = w * L_s + r * K_bar + Tax_Rev
    C = Income # Market clearing C = Income
    
    # Prices
    if E_total > 1e-9:
        P_E = e_c * Y / E_total # Energy price from FOC
    else:
        P_E = 0.0

    # Welfare
    # Recalculate utility with the specific damage parameter (global omega)
    C_eff = max(C - omega * Z, 1e-9)
    Util  = (C_eff**(1 - gamma_hh)) / (1 - gamma_hh) - chi_val * (L_s**(1 + sigma)) / (1 + sigma)

    # Return EVERYTHING in a dictionary
    return {
        "Scenario": scenario_name,
        "Param_Value": param_val,
        "Opt_Tax_Rate_%": tau_val * 100,
        
        # Macro Aggregates
        "Output_Y": Y,
        "Consumption_C": C,
        "Labour_L": L_s,
        "Energy_E": E_total,
        "Emissions_Z": Z,
        "Tax_Revenue": Tax_Rev,
        
        # Prices
        "Wage_w": w,
        "Rental_r": r,
        "Energy_Price_P_E": P_E,
        
        # Ratios
        "E_over_Y": E_total / Y if Y > 0 else 0,
        "K_over_Y": K_bar / Y if Y > 0 else 0,
        "C_over_Y": C / Y if Y > 0 else 0,
        "TaxRev_over_Y_%": (Tax_Rev / Y)*100 if Y > 0 else 0,
        "Emissions_Intensity_Z_Y": Z / Y if Y > 0 else 0,
        
        # Welfare
        "Welfare_Util": Util,
        
        # Sectoral Shares (Energy Mix)
        "Oil_Share_%": (e_vec[0]/E_total)*100 if E_total > 0 else 0,
        "Gas_Share_%": (e_vec[1]/E_total)*100 if E_total > 0 else 0,
        "Clean_Share_%": (e_vec[2]/E_total)*100 if E_total > 0 else 0,
    }

def get_baseline_equilibrium(theta):
    sol = solve_ge(theta)
    if not sol.success:
        return None

    x = sol.x
    L_s = max(x[10], 1e-9)
    r   = np.exp(x[8])
    w   = np.exp(x[9])

    K_alloc, L_alloc = get_allocations(x[:8], L_s)
    K_e, K_c = K_alloc[:3], K_alloc[3]
    L_e, L_c = L_alloc[:3], L_alloc[3]

    e_vec, E_total, _, _ = energy_outputs(
        K_e, L_e,
        theta[4:7],
        theta[:3]
    )

    Y = final_output(K_c, L_c, E_total, theta[3])
    Z = np.dot(phi_j, e_vec)
    P_E = e_c * Y / E_total if E_total > 0 else np.nan
    tax_rev = tau * Z

    return {
        "omega": omega,
        "e_c": e_c,
        "tax": tau * 100,
        "Y": Y,
        "L": L_s,
        "E": E_total,
        "Emissions": Z,
        "Energy_Price": P_E,
        "Tax_Revenue": tax_rev
    }

def run_robustness_analysis(baseline_theta):
    global omega, e_c

    base_omega = omega
    base_e_c   = e_c

    omega_range = np.array([0.5, 1.0, 1.5, 2.0, 5.0]) * base_omega
    ec_range    = np.linspace(0.03, 0.15, 25)

    omega_rows = []
    ec_rows    = []

    # baseline equilibrium
    baseline_row = get_baseline_equilibrium(baseline_theta)

    # omega robustness
    for val in omega_range:
        omega = val
        e_c   = base_e_c

        tau_star, _ = simple_tax_grid(baseline_theta)
        set_tau(tau_star)
        sol = solve_ge(baseline_theta)

        if not sol.success:
            continue

        x = sol.x
        L_s = max(x[10], 1e-9)
        K_alloc, L_alloc = get_allocations(x[:8], L_s)

        e_vec, E_total, _, _ = energy_outputs(
            K_alloc[:3], L_alloc[:3],
            baseline_theta[4:7],
            baseline_theta[:3]
        )

        Y = final_output(
            K_alloc[3], L_alloc[3],
            E_total,
            baseline_theta[3]
        )

        omega_rows.append({
            "omega": val,
            "tax": tau_star * 100,
            "Y": Y
        })

    # energy share robustness
    for val in ec_range:
        e_c   = val
        omega = base_omega

        tau_star, _ = simple_tax_grid(baseline_theta)
        set_tau(tau_star)
        sol = solve_ge(baseline_theta)

        if not sol.success:
            continue

        x = sol.x
        L_s = max(x[10], 1e-9)
        K_alloc, L_alloc = get_allocations(x[:8], L_s)

        e_vec, E_total, _, _ = energy_outputs(
            K_alloc[:3], L_alloc[:3],
            baseline_theta[4:7],
            baseline_theta[:3]
        )

        Y = final_output(
            K_alloc[3], L_alloc[3],
            E_total,
            baseline_theta[3]
        )

        ec_rows.append({
            "e_c": val,
            "tax": tau_star * 100,
            "Y": Y
        })

    omega = base_omega
    e_c   = base_e_c

    df_baseline = pd.DataFrame([baseline_row])
    df_omega    = pd.DataFrame(omega_rows)
    df_ec       = pd.DataFrame(ec_rows)

    return df_baseline, df_omega, df_ec

# ROBUSTNESS MODULE

def welfare_at_tax_with_damage(tau_value, theta, omega):
    """
    Welfare with damage.
    """
    set_tau(tau_value)
    sol = solve_ge(theta)
    if not sol.success:
        return -1e12

    x = sol.x
    L_s = max(x[10], 1e-9)
    r   = np.exp(x[8])
    w   = np.exp(x[9])

    # allocations
    K_alloc, L_alloc = get_allocations(x[:8], L_s)
    K_e, K_c = K_alloc[:3], K_alloc[3]
    L_e, L_c = L_alloc[:3], L_alloc[3]

    # energy
    e_vec, E_total, _, _ = energy_outputs(K_e, L_e, theta[4:7], theta[:3])
    Z = np.dot(phi_j, e_vec)

    # household utility (with rebate)
    C, _ = household_values(w, r, L_s, e_vec, theta[7])

    utility = (C**(1 - gamma_hh)) / (1 - gamma_hh) \
              - theta[7] * (L_s**(1 + sigma)) / (1 + sigma) \
              - omega * Z

    return float(utility)


def optimal_tau_for_omega(theta, omega,
                          tau_bounds=(0.0, 0.05),
                          grid_n=40):
    """
    Finds tau* for a given omega using grid + local refinement.
    """
    taus = np.linspace(tau_bounds[0], tau_bounds[1], grid_n)
    Wvals = np.array([
        welfare_at_tax_with_damage(t, theta, omega) for t in taus
    ])

    tau0 = taus[np.argmax(Wvals)]

    obj = lambda t: -welfare_at_tax_with_damage(t[0], theta, omega)

    res = minimize(obj, x0=[tau0], bounds=[tau_bounds], tol=1e-8)
    return res.x[0], -res.fun


def run_omega_robustness(theta):
    omega_grid = [0.04, 0.08, 0.12, 0.16, 0.40]
    rows = []

    for omega in omega_grid:
        tau_star, W_star = optimal_tau_for_omega(theta, omega)
        set_tau(tau_star)
        sol = solve_ge(theta)

        if not sol.success:
            continue

        x = sol.x
        L_s = x[10]
        K_alloc, L_alloc = get_allocations(x[:8], L_s)
        e_vec, E, _, _ = energy_outputs(
            K_alloc[:3], L_alloc[:3], theta[4:7], theta[:3]
        )
        Y = final_output(K_alloc[3], L_alloc[3], E, theta[3])
        Z = np.dot(phi_j, e_vec)

        rows.append({
            "omega": omega,
            "tau_star": tau_star,
            "Y": Y,
            "E": E,
            "Z": Z,
            "W": W_star
        })

    return pd.DataFrame(rows)


def reset_model_state():
    global tau, Nfeval
    tau = 0.002
    Nfeval = 1

def run_ec_robustness(ec_values):
    """
    Full structural robustness to energy share.
    Recalibrates model for each e_c.
    """
    global e_c, a_c

    results = []

    for ec_val in ec_values:
        reset_model_state()
        e_c = ec_val
        a_c = 1 - b_c - e_c
        # Recalibrate
        theta_ec = run_calibration_loop()

        # Find optimal tax
        obj = lambda t: -welfare_at_tax(t[0], theta_ec)
        taus = np.linspace(0.0, 0.05, 40)
        Wvals = np.array([welfare_at_tax(t, theta_ec) for t in taus])
        tau0 = taus[np.argmax(Wvals)]

        res = minimize(
            lambda t: -welfare_at_tax(t[0], theta_ec),
            x0=[tau0],
            bounds=[(0.0, 0.05)]
        )
        tau_star = res.x[0]
        # Solve GE at tau*
        set_tau(tau_star)
        sol = solve_ge(theta_ec)
        if not sol.success:
            continue

        x = sol.x
        L_s = x[10]
        K_alloc, L_alloc = get_allocations(x[:8], L_s)
        e_vec, E, _, _ = energy_outputs(
            K_alloc[:3], L_alloc[:3], theta_ec[4:7], theta_ec[:3]
        )
        Y = final_output(K_alloc[3], L_alloc[3], E, theta_ec[3])
        Z = np.dot(phi_j, e_vec)

        results.append({
            "e_c": ec_val,
            "tau_star": tau_star,
            "Y": Y,
            "E": E,
            "Z": Z,
            "E_Y": E / Y,
            "Clean_share": e_vec[2] / E
        })

    return pd.DataFrame(results)


MASTER_COLS = [
    "Scenario", "Param", "Tax_%", "Y", "C", "L", "E", "Z",
    "w", "r", "P_E", "E/Y", "K/Y", "C/Y", "Z/Y",
    "Oil_%", "Gas_%", "Clean_%"
]

def build_master_row(theta, tau_star, scenario, param_value):
    set_tau(tau_star)
    sol = solve_ge(theta)
    if not sol.success:
        raise RuntimeError("GE failed")

    x = sol.x
    L_s = max(x[10], 1e-9)
    r = np.exp(x[8])
    w = np.exp(x[9])

    K_alloc, L_alloc = get_allocations(x[:8], L_s)
    K_e, K_c = K_alloc[:3], K_alloc[3]
    L_e, L_c = L_alloc[:3], L_alloc[3]

    e_vec, E, _, _ = energy_outputs(K_e, L_e, theta[4:7], theta[:3])
    Y = final_output(K_c, L_c, E, theta[3])
    Z = np.dot(phi_j, e_vec)

    P_E = e_c * Y / E
    C, _ = household_values(w, r, L_s, e_vec, theta[7])

    shares = e_vec / E

    return {
        "Scenario": scenario,
        "Param": param_value,
        "Tax_%": 100 * tau_star,
        "Y": Y,
        "C": C,
        "L": L_s,
        "E": E,
        "Z": Z,
        "w": w,
        "r": r,
        "P_E": P_E,
        "E/Y": E / Y,
        "K/Y": K_bar / Y,
        "C/Y": C / Y,
        "Z/Y": Z / Y,
        "Oil_%": 100 * shares[0],
        "Gas_%": 100 * shares[1],
        "Clean_%": 100 * shares[2],
    }


def print_master_robustness_table(omega_grid, ec_grid):
    global e_c, a_c

    rows_A = []
    rows_B = []

    # Omega robustness 
    e_c = 0.08
    a_c = 1 - b_c - e_c
    theta_base = run_calibration_loop()

    for omega in omega_grid:
        def welfare_omega(t):
            set_tau(t)
            sol = solve_ge(theta_base)
            if not sol.success:
                return -1e12
            x = sol.x
            L_s = max(x[10], 1e-9)
            r = np.exp(x[8])
            w = np.exp(x[9])
            K_alloc, L_alloc = get_allocations(x[:8], L_s)
            e_vec, _, _, _ = energy_outputs(
                K_alloc[:3], L_alloc[:3], theta_base[4:7], theta_base[:3]
            )
            Z = np.dot(phi_j, e_vec)
            C, _ = household_values(w, r, L_s, e_vec, theta_base[7])
            return (
                (C**(1 - gamma_hh)) / (1 - gamma_hh)
                - theta_base[7] * (L_s**(1 + sigma)) / (1 + sigma)
                - omega * Z
            )

        taus = np.linspace(0, 0.05, 40)
        tau_star = taus[np.argmax([welfare_omega(t) for t in taus])]

        rows_A.append(
            build_master_row(theta_base, tau_star, "omega-Robustness", omega)
        )

    #  e_c robustness
    for ec_val in ec_grid:
        e_c = ec_val
        a_c = 1 - b_c - e_c
        theta_ec = run_calibration_loop()

        taus = np.linspace(0, 0.05, 40)
        tau_star = taus[np.argmax([welfare_at_tax(t, theta_ec) for t in taus])]

        rows_B.append(
            build_master_row(theta_ec, tau_star, "e_c-Robustness", ec_val)
        )

    df_A = pd.DataFrame(rows_A)[MASTER_COLS]
    df_B = pd.DataFrame(rows_B)[MASTER_COLS]

    print("\n================ ROBUSTNESS MASTER TABLE ================\n")
    print("Panel A: Sensitivity to Climate Damages (omega)\n")
    print(df_A.to_string(index=False))

    print("\nPanel B: Sensitivity to Structural Energy Share (e_c)\n")
    print(df_B.to_string(index=False))

    return df_A, df_B

if __name__ == "__main__":

    print("\n================ ROBUSTNESS ANALYSIS =================")

    # Baseline calibrated parameters
    theta_base = run_calibration_loop()

    #  Omega robustness
    omega_df = run_omega_robustness(theta_base)
    print("\nOmega Robustness Results")
    print(omega_df)

    #  e_c robustness
    ec_grid = [0.04, 0.06, 0.08, 0.10, 0.12]
    ec_df = run_ec_robustness(ec_grid)
    print("\nEnergy Share Robustness Results")
    print(ec_df)

    print("\n================ END ROBUSTNESS =================")

    print_master_robustness_table(
        omega_grid=[0.04, 0.08, 0.12, 0.16, 0.40],
        ec_grid=[0.04, 0.06, 0.08, 0.10, 0.12]
    )

    print(f"Starting baseline calibration with omega = {omega}...")
    best_theta = run_calibration_loop()

    df_baseline, df_omega, df_ec = run_robustness_analysis(best_theta)

    print("\nBaseline equilibrium")
    print(df_baseline.to_string(index=False))
    
    # structural output response
    plt.figure(figsize=(8, 5))
    plt.plot(df_ec["e_c"], df_ec["Y"], marker="o", linewidth=2.5)
    plt.xlabel("Energy share e_c")
    plt.ylabel("Output Y")
    plt.tight_layout()
    plt.show()

    # robustness panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(df_omega["omega"], df_omega["tax"], marker="s", linewidth=3)
    ax1.set_xlabel("omega")
    ax1.set_ylabel("optimal tax (%)")
    ax1.grid(True, alpha=0.6)

    ax2.plot(df_ec["e_c"], df_ec["tax"], marker="o", linewidth=2)
    ax2.set_xlabel("energy share e_c")
    ax2.set_ylabel("optimal tax (%)")
    ax2.grid(True, alpha=0.6)

    plt.tight_layout()
    plt.show()