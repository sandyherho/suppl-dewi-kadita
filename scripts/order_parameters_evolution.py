#!/usr/bin/env python3
"""
Order Parameters Evolution Analysis

Creates a 2×2 figure showing time evolution of Polarization (P) and 
Rotation (M) order parameters for all four Couzin model scenarios.

Output:
- Figure: ../figs/order_parameters_evolution.{pdf,png,eps}
- Report: ../reports/order_parameters_evolution_report.txt

Author: Sandy H. S. Herho
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path("../data")
FIG_DIR = Path("../figs")
REPORT_DIR = Path("../reports")

FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

NC_FILES = {
    'Case 1 - Swarm': 'case1_swarm.nc',
    'Case 2 - Torus': 'case2_torus.nc',
    'Case 3 - Dynamic Parallel': 'case3_dynamic_parallel.nc',
    'Case 4 - Highly Parallel': 'case4_highly_parallel.nc'
}

CASE_COLORS = {
    'Case 1 - Swarm': '#E74C3C',
    'Case 2 - Torus': '#27AE60',
    'Case 3 - Dynamic Parallel': '#3498DB',
    'Case 4 - Highly Parallel': '#F39C12'
}

CASE_SHORT_NAMES = {
    'Case 1 - Swarm': 'Swarm',
    'Case 2 - Torus': 'Torus',
    'Case 3 - Dynamic Parallel': 'Dyn. Parallel',
    'Case 4 - Highly Parallel': 'High. Parallel'
}

FIG_WIDTH = 12
FIG_HEIGHT = 10
DPI = 500

# ============================================================================
# DATA LOADING
# ============================================================================

def load_netcdf_data(filepath):
    """Load order parameter data from NetCDF file."""
    with Dataset(filepath, 'r') as nc:
        data = {
            'time': nc.variables['time'][:],
            'polarization': nc.variables['polarization'][:],
            'rotation': nc.variables['rotation'][:],
            'n_fish': nc.n_fish,
            'box_size': nc.box_size,
            'speed': nc.speed,
            'r_repulsion': nc.r_repulsion,
            'r_orientation': nc.r_orientation,
            'r_attraction': nc.r_attraction,
            'max_turn': nc.max_turn,
            'noise': nc.noise,
            'blind_angle': nc.blind_angle,
            'scenario_name': nc.scenario_name,
            'final_polarization': nc.final_polarization,
            'final_rotation': nc.final_rotation,
            'mean_polarization': nc.mean_polarization,
            'mean_rotation': nc.mean_rotation
        }
    return data

# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def compute_order_statistics(data):
    """Compute comprehensive statistics for order parameters."""
    P = data['polarization']
    M = data['rotation']
    n = len(P)
    half = n // 2
    
    stats = {
        'P_initial': P[0],
        'P_final': P[-1],
        'P_mean': np.mean(P),
        'P_std': np.std(P),
        'P_mean_2nd_half': np.mean(P[half:]),
        'P_std_2nd_half': np.std(P[half:]),
        'P_min': np.min(P),
        'P_max': np.max(P),
        'P_range': np.max(P) - np.min(P),
        'M_initial': M[0],
        'M_final': M[-1],
        'M_mean': np.mean(M),
        'M_std': np.std(M),
        'M_mean_2nd_half': np.mean(M[half:]),
        'M_std_2nd_half': np.std(M[half:]),
        'M_min': np.min(M),
        'M_max': np.max(M),
        'M_range': np.max(M) - np.min(M),
        'dominant_order': 'Polarization' if np.mean(P[half:]) > np.mean(M[half:]) else 'Rotation',
        'order_index_final': max(P[-1], M[-1]),
        'order_index_mean': np.mean([max(P[i], M[i]) for i in range(half, n)])
    }
    
    window = min(50, n // 4)
    if window > 1:
        P_gradient = np.gradient(P[-window:])
        M_gradient = np.gradient(M[-window:])
        stats['P_converged'] = np.abs(np.mean(P_gradient)) < 0.001
        stats['M_converged'] = np.abs(np.mean(M_gradient)) < 0.001
    else:
        stats['P_converged'] = True
        stats['M_converged'] = True
    
    return stats

# ============================================================================
# VISUALIZATION
# ============================================================================

def setup_style():
    """Setup matplotlib style for publication."""
    plt.style.use('default')
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': '#333333',
        'axes.labelcolor': '#000000',
        'axes.linewidth': 1.0,
        'xtick.color': '#333333',
        'ytick.color': '#333333',
        'text.color': '#333333',
        'grid.color': '#CCCCCC',
        'grid.alpha': 0.5,
        'grid.linewidth': 0.5,
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'legend.fontsize': 10,
        'mathtext.fontset': 'dejavusans'
    })

def create_figure(all_data):
    """Create 2×2 order parameters evolution figure."""
    
    setup_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor='white')
    axes = axes.flatten()
    
    panel_labels = ['a', 'b', 'c', 'd']
    case_names = list(all_data.keys())
    
    legend_handles = []
    legend_labels = []
    
    for idx, case_name in enumerate(case_names):
        ax = axes[idx]
        data = all_data[case_name]
        color = CASE_COLORS[case_name]
        
        time = data['time']
        P = data['polarization']
        M = data['rotation']
        
        # Plot polarization (solid line)
        line_p, = ax.plot(time, P, color=color, linewidth=1.8, 
                          linestyle='-', label='Polarization (P)')
        
        # Plot rotation (dashed line)
        line_m, = ax.plot(time, M, color=color, linewidth=1.8, 
                          linestyle='--', label='Rotation (M)')
        
        # Mean lines (second half) - dotted
        n = len(time)
        half = n // 2
        P_mean = np.mean(P[half:])
        M_mean = np.mean(M[half:])
        
        ax.axhline(y=P_mean, color=color, linestyle=':', linewidth=1.0, alpha=0.6)
        ax.axhline(y=M_mean, color=color, linestyle=':', linewidth=1.0, alpha=0.6)
        
        # Formatting
        ax.set_xlim(time[0], time[-1])
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Order Parameter', fontsize=12)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        
        # Panel label
        ax.text(0.02, 0.98, f'({panel_labels[idx]})', transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top', ha='left')
        
        # Stats annotation
        stats_text = f'P={P[-1]:.3f}, M={M[-1]:.3f}'
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                fontsize=9, va='top', ha='right', family='monospace',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='#CCCCCC', alpha=0.9))
        
        # Collect legend handles (only from first panel)
        if idx == 0:
            legend_handles = [line_p, line_m]
            legend_labels = ['Polarization (P)', 'Rotation (M)']
    
    # Add unified legend at bottom
    fig.legend(legend_handles, legend_labels, loc='lower center', 
               ncol=2, bbox_to_anchor=(0.5, 0.02), frameon=True,
               fancybox=False, shadow=False, fontsize=11,
               edgecolor='#CCCCCC')
    
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    
    return fig

# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(all_data, all_stats):
    """Generate comprehensive statistical report."""
    
    lines = []
    lines.append("=" * 80)
    lines.append("ORDER PARAMETERS EVOLUTION ANALYSIS")
    lines.append("=" * 80)
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("")
    
    lines.append("-" * 80)
    lines.append("OVERVIEW")
    lines.append("-" * 80)
    lines.append("""
This analysis examines the temporal evolution of two fundamental order parameters
that characterize collective states in fish schools:

1. POLARIZATION (P): Measures alignment coherence
   P = (1/N)|Σᵢv̂ᵢ| ∈ [0, 1]
   - P → 0: Disordered, random orientations (paramagnetic-like)
   - P → 1: Perfectly aligned, parallel motion (ferromagnetic-like)

2. ROTATION (M): Measures milling/torus coherence  
   M = (1/N)|Σᵢ(v̂ᵢ × r̂ᵢᶜ)| ∈ [0, 1]
   - M → 0: No collective rotation
   - M → 1: Perfect milling around common center

The Order Index OI = max(P, M) indicates dominant ordering behavior.
""")
    
    lines.append("-" * 80)
    lines.append("STATISTICAL RESULTS BY CASE")
    lines.append("-" * 80)
    
    for case_name in all_data.keys():
        data = all_data[case_name]
        stats = all_stats[case_name]
        
        lines.append(f"\n{case_name}")
        lines.append("~" * 60)
        
        lines.append(f"  Simulation Parameters:")
        lines.append(f"    N = {data['n_fish']}, L = {data['box_size']:.1f}, v₀ = {data['speed']:.2f}")
        lines.append(f"    Zones: rᵣ={data['r_repulsion']:.1f}, rₒ={data['r_orientation']:.1f}, rₐ={data['r_attraction']:.1f}")
        lines.append(f"    θ_max = {data['max_turn']:.2f} rad, σ = {data['noise']:.2f} rad, α = {data['blind_angle']:.1f}°")
        lines.append(f"    Time points: {len(data['time'])}, T_final = {data['time'][-1]:.1f}")
        
        lines.append(f"\n  Polarization (P):")
        lines.append(f"    Initial:           {stats['P_initial']:.6f}")
        lines.append(f"    Final:             {stats['P_final']:.6f}")
        lines.append(f"    Mean (full):       {stats['P_mean']:.6f} ± {stats['P_std']:.6f}")
        lines.append(f"    Mean (2nd half):   {stats['P_mean_2nd_half']:.6f} ± {stats['P_std_2nd_half']:.6f}")
        lines.append(f"    Range:             [{stats['P_min']:.6f}, {stats['P_max']:.6f}]")
        lines.append(f"    Converged:         {'Yes' if stats['P_converged'] else 'No'}")
        
        lines.append(f"\n  Rotation (M):")
        lines.append(f"    Initial:           {stats['M_initial']:.6f}")
        lines.append(f"    Final:             {stats['M_final']:.6f}")
        lines.append(f"    Mean (full):       {stats['M_mean']:.6f} ± {stats['M_std']:.6f}")
        lines.append(f"    Mean (2nd half):   {stats['M_mean_2nd_half']:.6f} ± {stats['M_std_2nd_half']:.6f}")
        lines.append(f"    Range:             [{stats['M_min']:.6f}, {stats['M_max']:.6f}]")
        lines.append(f"    Converged:         {'Yes' if stats['M_converged'] else 'No'}")
        
        lines.append(f"\n  Classification:")
        lines.append(f"    Dominant order:    {stats['dominant_order']}")
        lines.append(f"    Order Index (OI):  {stats['order_index_final']:.6f} (final)")
        lines.append(f"                       {stats['order_index_mean']:.6f} (mean 2nd half)")
        
        P_eq = stats['P_mean_2nd_half']
        M_eq = stats['M_mean_2nd_half']
        if P_eq > 0.9:
            state = "Highly Parallel"
        elif P_eq > 0.7:
            state = "Dynamic Parallel"
        elif M_eq > 0.5:
            state = "Torus/Milling"
        elif P_eq > 0.4 or M_eq > 0.3:
            state = "Transitional"
        else:
            state = "Swarm"
        lines.append(f"    Collective state:  {state}")
    
    lines.append("\n" + "-" * 80)
    lines.append("COMPARATIVE SUMMARY")
    lines.append("-" * 80)
    
    lines.append("\n  Final Order Parameters:")
    lines.append(f"    {'Case':<30} {'P_final':>10} {'M_final':>10} {'OI':>10} {'State':<20}")
    lines.append("    " + "-" * 80)
    
    for case_name in all_data.keys():
        stats = all_stats[case_name]
        short_name = case_name.split(' - ')[1] if ' - ' in case_name else case_name
        P_eq = stats['P_mean_2nd_half']
        M_eq = stats['M_mean_2nd_half']
        if P_eq > 0.9:
            state = "Highly Parallel"
        elif P_eq > 0.7:
            state = "Dynamic Parallel"
        elif M_eq > 0.5:
            state = "Torus"
        else:
            state = "Swarm"
        lines.append(f"    {short_name:<30} {stats['P_final']:>10.4f} {stats['M_final']:>10.4f} "
                    f"{stats['order_index_final']:>10.4f} {state:<20}")
    
    lines.append("\n" + "-" * 80)
    lines.append("FIGURE SPECIFICATIONS")
    lines.append("-" * 80)
    lines.append(f"  Layout:      2×2 panels")
    lines.append(f"  Dimensions:  {FIG_WIDTH}\" × {FIG_HEIGHT}\"")
    lines.append(f"  Resolution:  {DPI} DPI")
    lines.append(f"  Formats:     PDF, PNG, EPS")
    lines.append(f"  Legend:      External, bottom center")
    lines.append("")
    lines.append("=" * 80)
    
    return "\n".join(lines)

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution function."""
    
    print("=" * 70)
    print("Order Parameters Evolution Analysis")
    print("=" * 70)
    
    print("\n[1/4] Loading NetCDF data...")
    all_data = {}
    for case_name, filename in NC_FILES.items():
        filepath = DATA_DIR / filename
        if filepath.exists():
            print(f"      Loading {filename}...")
            all_data[case_name] = load_netcdf_data(filepath)
        else:
            print(f"      WARNING: {filepath} not found!")
    
    if len(all_data) == 0:
        print("ERROR: No data files found!")
        return
    
    print("\n[2/4] Computing statistics...")
    all_stats = {}
    for case_name, data in all_data.items():
        all_stats[case_name] = compute_order_statistics(data)
    
    print("\n[3/4] Creating figure...")
    fig = create_figure(all_data)
    
    base_name = "order_parameters_evolution"
    for fmt in ['pdf', 'png', 'eps']:
        output_path = FIG_DIR / f"{base_name}.{fmt}"
        print(f"      Saving {output_path}...")
        fig.savefig(output_path, dpi=DPI, facecolor='white',
                   edgecolor='none', bbox_inches='tight', format=fmt)
    plt.close(fig)
    
    print("\n[4/4] Generating report...")
    report_text = generate_report(all_data, all_stats)
    report_path = REPORT_DIR / f"{base_name}_report.txt"
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f"      Saved {report_path}")
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print(f"  Figure: {FIG_DIR}/{base_name}.{{pdf,png,eps}}")
    print(f"  Report: {report_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
