"""
Calculate number of equilibrium solves, and time in Model.py
"""

# Configuration from the code
T = 10 # Time periods
n_omega = 11  # Omega sensitivity points
n_ec = 8  # Energy share sensitivity points
grid_n = 40  # Grid search points per optimization

# Euler iterations per period
euler_iterations = 5

# Root finding iterations 
root_iterations_avg = 25  # Conservative estimate

print("="*80)
print("EQUILIBRIUM SOLVE COUNT CALCULATION")
print("="*80)

print("\nConfiguration:")
print(f"  Time periods (T): {T}")
print(f"  Omega sensitivity points: {n_omega}")
print(f"  Energy share sensitivity points: {n_ec}")
print(f"  Grid search points per optimization: {grid_n}")
print(f"  Euler iterations per period: {euler_iterations}")
print(f"  Root finding iterations (avg): {root_iterations_avg}")

print("\n" + "="*80)
print("BREAKDOWN PER TAU EVALUATION")
print("="*80)

# Per tau evaluation breakdown
initial_guess = 1
print(f"\n1. Initial guess (get Y_approx): {initial_guess:,} solve")

# Root finding
print(f"\n2. Root finding (brentq to find C0):")
print(f"   - Iterations: {root_iterations_avg}")
print(f"   - Per iteration (simulate_path_full):")
print(f"     * Period 0: 1 solve")
print(f"     * Periods 1-{T-1}: {euler_iterations} Euler iterations each")
print(f"     * Solves per period 1-{T-1}: {euler_iterations} solves")
print(f"     * Total per iteration: 1 + {T-1}×{euler_iterations} = {1 + (T-1)*euler_iterations} solves")

solves_per_root_iter = 1 + (T-1) * euler_iterations
root_finding_total = root_iterations_avg * solves_per_root_iter
print(f"   - Root finding total: {root_iterations_avg} × {solves_per_root_iter} = {root_finding_total:,} solves")

# Final verification
final_verify = solves_per_root_iter
print(f"\n3. Final verification: {final_verify:,} solves")

solves_per_tau = initial_guess + root_finding_total + final_verify
print(f"\n{'─'*80}")
print(f"TOTAL PER TAU EVALUATION: {solves_per_tau:,} solves")
print(f"{'─'*80}")

print("\n" + "="*80)
print("PANEL A: OMEGA SENSITIVITY")
print("="*80)

solves_per_omega = grid_n * solves_per_tau
panel_a_total = n_omega * solves_per_omega

print(f"\nPer omega value:")
print(f"  Grid points: {grid_n}")
print(f"  Solves: {grid_n} × {solves_per_tau:,} = {solves_per_omega:,}")

print(f"\nPanel A total:")
print(f"  Omega values: {n_omega}")
print(f"  Total solves: {n_omega} × {solves_per_omega:,} = {panel_a_total:,}")

print("\n" + "="*80)
print("PANEL B: ENERGY SHARE SENSITIVITY")
print("="*80)

solves_per_ec = grid_n * solves_per_tau
panel_b_total = n_ec * solves_per_ec

print(f"\nPer e_c value:")
print(f"  Grid points: {grid_n}")
print(f"  Solves: {grid_n} × {solves_per_tau:,} = {solves_per_ec:,}")

print(f"\nPanel B total:")
print(f"  Energy share values: {n_ec}")
print(f"  Total solves: {n_ec} × {solves_per_ec:,} = {panel_b_total:,}")

print("\n" + "="*80)
print("OPERATIONS HIERARCHY")
print("="*80)

total_optimizations = n_omega + n_ec
total_tau_evals = total_optimizations * grid_n

print("\nLevel 1: Parameter Optimizations (top-level operations)")
print(f"  Panel A: {n_omega} omega optimizations")
print(f"  Panel B: {n_ec} e_c optimizations")
print(f"  {'─'*76}")
print(f"  TOTAL: {total_optimizations} optimization runs")

print("\nLevel 2: Tau Evaluations (grid search points)")
print(f"  Per optimization: {grid_n} tau values tested")
print(f"  Total evaluations: {total_optimizations} × {grid_n} = {total_tau_evals} tau evaluations")

print("\nLevel 3: Equilibrium Solves (computational work)")
print(f"  Per tau evaluation: {solves_per_tau:,} equilibrium solves")
print(f"  Total solves: {total_tau_evals} × {solves_per_tau:,} = {panel_a_total + panel_b_total:,} solves")

print("\n" + "="*80)
print("WHY 19 OPERATIONS BECOMES 352,811 SOLVES")
print("="*80)

grand_total = panel_a_total + panel_b_total

print("\nThe nested structure multiplies computational cost:")
print(f"\n  Each parameter optimization:")
print(f"    ├─ Tests {grid_n} different tax policies (grid search)")
print(f"    └─ Each tax policy evaluation:")
print(f"         ├─ Solves for initial consumption C0 (root finding)")
print(f"         ├─ Simulates {T} periods forward")
print(f"         ├─ Each period: {euler_iterations} Euler iterations to converge")
print(f"         └─ Result: {solves_per_tau:,} equilibrium solves per tau")

print(f"\n  So each parameter optimization requires:")
print(f"    {grid_n} tau values × {solves_per_tau:,} solves = {solves_per_omega:,} solves")

print(f"\n  That's {solves_per_omega/1000:.1f}k solves per parameter!")

print("\n" + "─"*80)
print("COMPARISON TO SIMPLER MODELS:")
print("─"*80)

print("\nIf this were a STATIC model (no dynamics):")
simple_static = total_optimizations * grid_n * 1  # Just 1 solve per tau
print(f"  {total_optimizations} params × {grid_n} tau × 1 solve = {simple_static} solves")
print(f"  Reduction: {grand_total/simple_static:.0f}x fewer solves")

print("\nIf there were NO root finding (exogenous C0):")
no_root = total_optimizations * grid_n * solves_per_root_iter  # Just simulate once
print(f"  {total_optimizations} params × {grid_n} tau × {solves_per_root_iter} solves = {no_root:,} solves")
print(f"  Reduction: {grand_total/no_root:.1f}x fewer solves")

print("\nIf there were NO Euler iterations (simple growth rule):")
simple_euler = total_optimizations * grid_n * root_iterations_avg * T  # Just T solves per iteration
print(f"  {total_optimizations} params × {grid_n} tau × {root_iterations_avg} iter × {T} periods = {simple_euler:,} solves")
print(f"  Reduction: {grand_total/simple_euler:.1f}x fewer solves")

print("\n" + "─"*80)
print("GRANULARITY IMPACT:")
print("─"*80)

print(f"\nCurrent: {grid_n} grid points per optimization")
print(f"  → {grand_total:,} total solves")

for new_grid in [11, 21, 51, 101]:
    new_total = total_optimizations * new_grid * solves_per_tau
    print(f"\nIf grid_n = {new_grid}:")
    print(f"  → {new_total:,} total solves ({new_total/grand_total:.2f}x current)")

print("\n" + "="*80)
print("BOTTOM LINE")
print("="*80)
print(f"\n19 operations × nested computation = {grand_total:,} solves")
print(f"\nThis is a FULL dynamic general equilibrium model with:")
print(f"  • Capital accumulation over {T} periods")
print(f"  • Euler equation convergence each period")
print(f"  • Terminal capital constraint (requires root finding)")
print(f"  • 3-sector CES energy structure")
print(f"  • 12-variable general equilibrium system per period")
print(f"\nEach solve is expensive. Hence: {grand_total:,} total solves.")

print("\n" + "="*80)
print("GRAND TOTAL")
print("="*80)

print(f"\nPanel A: {panel_a_total:,} solves")
print(f"Panel B: {panel_b_total:,} solves")
print(f"{'─'*80}")
print(f"GRAND TOTAL: {grand_total:,} equilibrium solves")
print(f"{'─'*80}")

print(f"\nSummary:")
print(f"  • {total_optimizations} parameter optimizations")
print(f"  • {total_tau_evals} tau evaluations") 
print(f"  • {grand_total:,} equilibrium solves")

print("\n" + "="*80)
print("PERFORMANCE IMPLICATIONS")
print("="*80)

# Assume each solve takes some time
solve_time_ms = 50  # milliseconds per equilibrium solve (rough estimate)
total_time_serial_sec = (grand_total * solve_time_ms) / 1000
total_time_serial_min = total_time_serial_sec / 60

n_cores = 10  # M5 MacBook Pro
total_time_parallel_sec = total_time_serial_sec / n_cores
total_time_parallel_min = total_time_parallel_sec / 60

print(f"\nAssuming {solve_time_ms}ms per equilibrium solve:")
print(f"  Serial execution: {total_time_serial_min:.1f} minutes")
print(f"  Parallel ({n_cores} cores): {total_time_parallel_min:.1f} minutes")
print(f"  Speedup: {n_cores:.0f}x faster with parallelization")

print("\n" + "="*80)
