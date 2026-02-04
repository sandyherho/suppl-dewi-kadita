#!/usr/bin/env python3
"""
Final State Entropy Profile Analysis

Creates a 2×2 figure showing final state entropy profiles comparing
all four Couzin model scenarios:
- (a) Primary entropy metrics (H_coh, H_pol, H_vel)
- (b) Secondary entropy metrics (H_depth, H_ang, H_NN, H_shape)
- (c) Composite indices (OSI, OI)
- (d) Order parameters (P, M)

Output:
- Figure: ../figs/final_state_entropy_profile.{pdf,png,eps}
- Report: ../reports/final_state_entropy_profile_report.txt

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
    'Case 3 - Dynamic Parallel': 'Dyn. Par.',
    'Case 4 - Highly Parallel': 'High. Par.'
}

FIG_WIDTH = 12
FIG_HEIGHT = 10
DPI = 500

# ============================================================================
# DATA LOADING
# ============================================================================

def load_netcdf_data(filepath):
    """Load all metrics from NetCDF file."""
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
            'scenario_name': nc.scenario_name
        }
        
        entropy_vars = [
            'school_cohesion_entropy',
            'polarization_entropy',
            'depth_stratification_entropy',
            'angular_momentum_entropy',
            'nearest_neighbor_entropy',
            'velocity_correlation_entropy',
            'school_shape_entropy',
            'oceanic_schooling_index',
            'order_index'
        ]
        
        for var in entropy_vars:
            try:
                data[var] = nc.variables[var][:]
            except:
                data[var] = None
                
    return data

# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def extract_final_values(all_data):
    """Extract final values for all metrics."""
    
    final_vals = {}
    
    for case_name, data in all_data.items():
        final_vals[case_name] = {
            'P': data['polarization'][-1],
            'M': data['rotation'][-1],
            'H_coh': data['school_cohesion_entropy'][-1] if data['school_cohesion_entropy'] is not None else np.nan,
            'H_pol': data['polarization_entropy'][-1] if data['polarization_entropy'] is not None else np.nan,
            'H_depth': data['depth_stratification_entropy'][-1] if data['depth_stratification_entropy'] is not None else np.nan,
            'H_ang': data['angular_momentum_entropy'][-1] if data['angular_momentum_entropy'] is not None else np.nan,
            'H_NN': data['nearest_neighbor_entropy'][-1] if data['nearest_neighbor_entropy'] is not None else np.nan,
            'H_vel': data['velocity_correlation_entropy'][-1] if data['velocity_correlation_entropy'] is not None else np.nan,
            'H_shape': data['school_shape_entropy'][-1] if data['school_shape_entropy'] is not None else np.nan,
            'OSI': data['oceanic_schooling_index'][-1] if data['oceanic_schooling_index'] is not None else np.nan,
            'OI': data['order_index'][-1] if data['order_index'] is not None else np.nan
        }
    
    return final_vals

def compute_statistics(all_data):
    """Compute comprehensive statistics for report."""
    
    stats = {}
    
    for case_name, data in all_data.items():
        n = len(data['time'])
        half = n // 2
        
        case_stats = {
            'n_fish': data['n_fish'],
            'box_size': data['box_size'],
            'speed': data['speed'],
            'n_timesteps': n,
            'T_final': data['time'][-1]
        }
        
        P = data['polarization']
        M = data['rotation']
        case_stats['P_final'] = P[-1]
        case_stats['P_mean_eq'] = np.mean(P[half:])
        case_stats['P_std_eq'] = np.std(P[half:])
        case_stats['M_final'] = M[-1]
        case_stats['M_mean_eq'] = np.mean(M[half:])
        case_stats['M_std_eq'] = np.std(M[half:])
        
        metrics = [
            ('school_cohesion_entropy', 'H_coh'),
            ('polarization_entropy', 'H_pol'),
            ('depth_stratification_entropy', 'H_depth'),
            ('angular_momentum_entropy', 'H_ang'),
            ('nearest_neighbor_entropy', 'H_NN'),
            ('velocity_correlation_entropy', 'H_vel'),
            ('school_shape_entropy', 'H_shape'),
            ('oceanic_schooling_index', 'OSI'),
            ('order_index', 'OI')
        ]
        
        for key, short in metrics:
            if data[key] is not None:
                vals = data[key]
                case_stats[f'{short}_final'] = vals[-1]
                case_stats[f'{short}_mean_eq'] = np.mean(vals[half:])
                case_stats[f'{short}_std_eq'] = np.std(vals[half:])
            else:
                case_stats[f'{short}_final'] = np.nan
                case_stats[f'{short}_mean_eq'] = np.nan
                case_stats[f'{short}_std_eq'] = np.nan
        
        stats[case_name] = case_stats
    
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
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.labelsize': 12,
        'legend.fontsize': 9,
        'mathtext.fontset': 'dejavusans'
    })

def create_figure(all_data, final_vals):
    """Create 2×2 bar chart comparison figure."""
    
    setup_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor='white')
    axes = axes.flatten()
    
    panel_labels = ['a', 'b', 'c', 'd']
    case_names = list(all_data.keys())
    n_cases = len(case_names)
    
    panels = [
        {
            'metrics': ['H_coh', 'H_pol', 'H_vel'],
            'labels': [r'$H_{coh}$', r'$H_{pol}$', r'$H_{vel}$'],
            'ylabel': 'Entropy'
        },
        {
            'metrics': ['H_depth', 'H_ang', 'H_NN', 'H_shape'],
            'labels': [r'$H_{depth}$', r'$H_{ang}$', r'$H_{NN}$', r'$H_{shape}$'],
            'ylabel': 'Entropy'
        },
        {
            'metrics': ['OSI', 'OI'],
            'labels': ['OSI', 'OI'],
            'ylabel': 'Index Value'
        },
        {
            'metrics': ['P', 'M'],
            'labels': ['P', 'M'],
            'ylabel': 'Order Parameter'
        }
    ]
    
    legend_handles = []
    legend_labels = []
    
    for panel_idx, panel_config in enumerate(panels):
        ax = axes[panel_idx]
        metrics = panel_config['metrics']
        labels = panel_config['labels']
        n_metrics = len(metrics)
        
        x = np.arange(n_metrics)
        width = 0.18
        offsets = np.linspace(-(n_cases-1)*width/2, (n_cases-1)*width/2, n_cases)
        
        for case_idx, case_name in enumerate(case_names):
            color = CASE_COLORS[case_name]
            short_name = CASE_SHORT_NAMES[case_name]
            
            values = [final_vals[case_name][m] for m in metrics]
            
            bars = ax.bar(x + offsets[case_idx], values, width, 
                         color=color, edgecolor='white', linewidth=0.5, 
                         label=short_name)
            
            if panel_idx == 0:
                legend_handles.append(bars[0])
                legend_labels.append(short_name)
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel(panel_config['ylabel'], fontsize=12)
        ax.set_ylim(-0.05, 1.15)
        ax.grid(True, axis='y', alpha=0.3, linewidth=0.5)
        
        ax.text(0.02, 0.98, f'({panel_labels[panel_idx]})', transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top', ha='left')
    
    fig.legend(legend_handles, legend_labels, loc='lower center',
               ncol=4, bbox_to_anchor=(0.5, 0.02), frameon=True,
               fancybox=False, shadow=False, fontsize=10,
               edgecolor='#CCCCCC')
    
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    
    return fig

# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(all_data, stats, final_vals):
    """Generate comprehensive statistical report."""
    
    lines = []
    lines.append("=" * 80)
    lines.append("FINAL STATE ENTROPY PROFILE ANALYSIS")
    lines.append("=" * 80)
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("")
    
    lines.append("-" * 80)
    lines.append("OVERVIEW")
    lines.append("-" * 80)
    lines.append("""
This analysis compares the final (equilibrium) state entropy profiles across
all four Couzin model scenarios. The metrics are organized into four categories:

Panel (a) - Primary Entropy Metrics:
  H_coh: School cohesion (NND distribution)
  H_pol: Polarization (heading distribution)  
  H_vel: Velocity correlation (pairwise alignment)

Panel (b) - Secondary Entropy Metrics:
  H_depth: Depth stratification (vertical distribution)
  H_ang:   Angular momentum (rotation contributions)
  H_NN:    Nearest neighbor (k-NN variability)
  H_shape: School shape (PCA eigenvalue ratios)

Panel (c) - Composite Indices:
  OSI: Oceanic Schooling Index (0=ordered, 1=disordered)
  OI:  Order Index = max(P, M)

Panel (d) - Classical Order Parameters:
  P: Polarization (alignment)
  M: Rotation (milling)
""")
    
    lines.append("-" * 80)
    lines.append("FINAL STATE VALUES BY CASE")
    lines.append("-" * 80)
    
    for case_name in all_data.keys():
        s = stats[case_name]
        fv = final_vals[case_name]
        
        lines.append(f"\n{case_name}")
        lines.append("~" * 60)
        
        lines.append(f"  System: N={s['n_fish']}, L={s['box_size']:.1f}, v₀={s['speed']:.2f}")
        lines.append(f"  Simulation: {s['n_timesteps']} steps, T={s['T_final']:.1f}")
        
        lines.append(f"\n  Order Parameters:")
        lines.append(f"    P (Polarization):  {fv['P']:.6f}  (eq: {s['P_mean_eq']:.4f} ± {s['P_std_eq']:.4f})")
        lines.append(f"    M (Rotation):      {fv['M']:.6f}  (eq: {s['M_mean_eq']:.4f} ± {s['M_std_eq']:.4f})")
        
        lines.append(f"\n  Primary Entropy Metrics:")
        lines.append(f"    H_coh:  {fv['H_coh']:.6f}  (eq: {s['H_coh_mean_eq']:.4f} ± {s['H_coh_std_eq']:.4f})")
        lines.append(f"    H_pol:  {fv['H_pol']:.6f}  (eq: {s['H_pol_mean_eq']:.4f} ± {s['H_pol_std_eq']:.4f})")
        lines.append(f"    H_vel:  {fv['H_vel']:.6f}  (eq: {s['H_vel_mean_eq']:.4f} ± {s['H_vel_std_eq']:.4f})")
        
        lines.append(f"\n  Secondary Entropy Metrics:")
        lines.append(f"    H_depth: {fv['H_depth']:.6f}  (eq: {s['H_depth_mean_eq']:.4f} ± {s['H_depth_std_eq']:.4f})")
        lines.append(f"    H_ang:   {fv['H_ang']:.6f}  (eq: {s['H_ang_mean_eq']:.4f} ± {s['H_ang_std_eq']:.4f})")
        lines.append(f"    H_NN:    {fv['H_NN']:.6f}  (eq: {s['H_NN_mean_eq']:.4f} ± {s['H_NN_std_eq']:.4f})")
        lines.append(f"    H_shape: {fv['H_shape']:.6f}  (eq: {s['H_shape_mean_eq']:.4f} ± {s['H_shape_std_eq']:.4f})")
        
        lines.append(f"\n  Composite Indices:")
        lines.append(f"    OSI: {fv['OSI']:.6f}  (eq: {s['OSI_mean_eq']:.4f} ± {s['OSI_std_eq']:.4f})")
        lines.append(f"    OI:  {fv['OI']:.6f}  (eq: {s['OI_mean_eq']:.4f} ± {s['OI_std_eq']:.4f})")
    
    lines.append("\n" + "-" * 80)
    lines.append("COMPARATIVE SUMMARY TABLE")
    lines.append("-" * 80)
    
    header = f"  {'Metric':<12}"
    for case_name in all_data.keys():
        short = case_name.split(' - ')[1][:8] if ' - ' in case_name else case_name[:8]
        header += f" {short:>10}"
    lines.append(header)
    lines.append("  " + "-" * (12 + 11 * len(all_data)))
    
    metrics_to_compare = [
        ('P', 'P'),
        ('M', 'M'),
        ('H_coh', 'H_coh'),
        ('H_pol', 'H_pol'),
        ('H_vel', 'H_vel'),
        ('H_depth', 'H_depth'),
        ('H_ang', 'H_ang'),
        ('H_NN', 'H_NN'),
        ('H_shape', 'H_shape'),
        ('OSI', 'OSI'),
        ('OI', 'OI')
    ]
    
    for metric_key, metric_label in metrics_to_compare:
        row = f"  {metric_label:<12}"
        for case_name in all_data.keys():
            val = final_vals[case_name][metric_key]
            row += f" {val:>10.4f}"
        lines.append(row)
    
    lines.append("\n" + "-" * 80)
    lines.append("KEY FINDINGS")
    lines.append("-" * 80)
    
    osi_ranking = sorted(all_data.keys(), key=lambda x: final_vals[x]['OSI'])
    oi_ranking = sorted(all_data.keys(), key=lambda x: final_vals[x]['OI'], reverse=True)
    
    lines.append(f"\n  OSI Ranking (low → high, ordered → disordered):")
    for i, case_name in enumerate(osi_ranking, 1):
        short = case_name.split(' - ')[1] if ' - ' in case_name else case_name
        lines.append(f"    {i}. {short}: OSI = {final_vals[case_name]['OSI']:.4f}")
    
    lines.append(f"\n  OI Ranking (high → low, most ordered first):")
    for i, case_name in enumerate(oi_ranking, 1):
        short = case_name.split(' - ')[1] if ' - ' in case_name else case_name
        lines.append(f"    {i}. {short}: OI = {final_vals[case_name]['OI']:.4f}")
    
    lines.append("\n" + "-" * 80)
    lines.append("FIGURE SPECIFICATIONS")
    lines.append("-" * 80)
    lines.append(f"  Layout:      2×2 bar chart panels")
    lines.append(f"  Dimensions:  {FIG_WIDTH}\" × {FIG_HEIGHT}\"")
    lines.append(f"  Resolution:  {DPI} DPI")
    lines.append(f"  Formats:     PDF, PNG, EPS")
    lines.append("")
    lines.append("=" * 80)
    
    return "\n".join(lines)

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution function."""
    
    print("=" * 70)
    print("Final State Entropy Profile Analysis")
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
    final_vals = extract_final_values(all_data)
    stats = compute_statistics(all_data)
    
    print("\n[3/4] Creating figure...")
    fig = create_figure(all_data, final_vals)
    
    base_name = "final_state_entropy_profile"
    for fmt in ['pdf', 'png', 'eps']:
        output_path = FIG_DIR / f"{base_name}.{fmt}"
        print(f"      Saving {output_path}...")
        fig.savefig(output_path, dpi=DPI, facecolor='white',
                   edgecolor='none', bbox_inches='tight', format=fmt)
    plt.close(fig)
    
    print("\n[4/4] Generating report...")
    report_text = generate_report(all_data, stats, final_vals)
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
