#!/usr/bin/env python3
"""
Entropy Metrics Evolution Analysis

Creates a 2×2 figure showing time evolution of key oceanic entropy metrics
for all four Couzin model scenarios:
- (a) School Cohesion Entropy
- (b) Polarization Entropy  
- (c) Velocity Correlation Entropy
- (d) Oceanic Schooling Index (OSI)

Output:
- Figure: ../figs/entropy_metrics_evolution.{pdf,png,eps}
- Report: ../reports/entropy_metrics_evolution_report.txt

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
    """Load entropy metrics from NetCDF file."""
    with Dataset(filepath, 'r') as nc:
        data = {
            'time': nc.variables['time'][:],
            'polarization': nc.variables['polarization'][:],
            'rotation': nc.variables['rotation'][:],
            'n_fish': nc.n_fish,
            'box_size': nc.box_size,
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

def compute_entropy_statistics(data):
    """Compute statistics for all entropy metrics."""
    
    metrics = [
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
    
    stats = {}
    n = len(data['time'])
    half = n // 2
    
    for metric in metrics:
        if data[metric] is not None:
            vals = data[metric]
            stats[metric] = {
                'initial': vals[0],
                'final': vals[-1],
                'mean': np.mean(vals),
                'std': np.std(vals),
                'mean_2nd_half': np.mean(vals[half:]),
                'std_2nd_half': np.std(vals[half:]),
                'min': np.min(vals),
                'max': np.max(vals),
                'range': np.max(vals) - np.min(vals)
            }
        else:
            stats[metric] = None
    
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
        'legend.fontsize': 9,
        'mathtext.fontset': 'dejavusans'
    })

def create_figure(all_data):
    """Create 2×2 entropy metrics evolution figure."""
    
    setup_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor='white')
    axes = axes.flatten()
    
    panel_labels = ['a', 'b', 'c', 'd']
    metric_keys = [
        'school_cohesion_entropy',
        'polarization_entropy',
        'velocity_correlation_entropy',
        'oceanic_schooling_index'
    ]
    y_labels = [
        r'$H_{coh}$',
        r'$H_{pol}$',
        r'$H_{vel}$',
        'OSI'
    ]
    
    legend_handles = []
    legend_labels = []
    
    for panel_idx, (metric, ylabel) in enumerate(zip(metric_keys, y_labels)):
        ax = axes[panel_idx]
        
        for case_name, data in all_data.items():
            if data[metric] is not None:
                time = data['time']
                vals = data[metric]
                color = CASE_COLORS[case_name]
                short_name = CASE_SHORT_NAMES[case_name]
                
                line, = ax.plot(time, vals, color=color, linewidth=1.5, label=short_name)
                
                # Collect legend handles from first panel only
                if panel_idx == 0:
                    legend_handles.append(line)
                    legend_labels.append(short_name)
        
        # Formatting
        ax.set_xlim(0, max(d['time'][-1] for d in all_data.values()))
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        
        # Panel label
        ax.text(0.02, 0.98, f'({panel_labels[panel_idx]})', transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top', ha='left')
    
    # Add unified legend at bottom
    fig.legend(legend_handles, legend_labels, loc='lower center',
               ncol=4, bbox_to_anchor=(0.5, 0.02), frameon=True,
               fancybox=False, shadow=False, fontsize=10,
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
    lines.append("ENTROPY METRICS EVOLUTION ANALYSIS")
    lines.append("=" * 80)
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("")
    
    lines.append("-" * 80)
    lines.append("ENTROPY METRICS DEFINITIONS")
    lines.append("-" * 80)
    lines.append("""
Seven oceanic entropy metrics characterize different aspects of school organization:

1. SCHOOL COHESION ENTROPY (H_coh):
   - Based on nearest-neighbor distance (NND) distribution
   - Low entropy: tight, uniform spacing (good predator defense)
   - High entropy: variable, dispersed spacing

2. POLARIZATION ENTROPY (H_pol):
   - Based on velocity heading distribution on unit sphere
   - Low entropy: aligned headings (coordinated migration)
   - High entropy: random orientations

3. DEPTH STRATIFICATION ENTROPY (H_depth):
   - Based on vertical (z) position distribution
   - Low entropy: concentrated at specific depth
   - High entropy: uniform vertical spread

4. ANGULAR MOMENTUM ENTROPY (H_ang):
   - Based on individual rotation contributions
   - Characterizes milling behavior uniformity

5. NEAREST NEIGHBOR ENTROPY (H_NN):
   - Based on k-NN distance variability (coefficient of variation)
   - Characterizes local density structure

6. VELOCITY CORRELATION ENTROPY (H_vel):
   - Based on pairwise velocity dot product distribution
   - Low entropy: all pairs aligned (high correlation)
   - High entropy: correlations spread from -1 to 1

7. SCHOOL SHAPE ENTROPY (H_shape):
   - Based on PCA eigenvalue ratios
   - Low entropy: elongated shape (λ₁ >> λ₂, λ₃)
   - High entropy: spherical shape (λ₁ ≈ λ₂ ≈ λ₃)

OCEANIC SCHOOLING INDEX (OSI):
   OSI = Σᵢ wᵢHᵢ ∈ [0, 1]
   Composite weighted metric: OSI → 0 (ordered), OSI → 1 (disordered)
   Weights: w_pol=0.28, w_coh=0.18, w_vel=0.18, w_NN=0.10, w_shape=0.10, 
            w_depth=0.08, w_ang=0.08
""")
    
    lines.append("-" * 80)
    lines.append("STATISTICAL RESULTS BY CASE")
    lines.append("-" * 80)
    
    key_metrics = [
        ('school_cohesion_entropy', 'Cohesion Entropy (H_coh)'),
        ('polarization_entropy', 'Polarization Entropy (H_pol)'),
        ('velocity_correlation_entropy', 'Velocity Correlation Entropy (H_vel)'),
        ('oceanic_schooling_index', 'Oceanic Schooling Index (OSI)'),
        ('order_index', 'Order Index (OI)')
    ]
    
    for case_name in all_data.keys():
        stats = all_stats[case_name]
        
        lines.append(f"\n{case_name}")
        lines.append("~" * 60)
        
        for metric_key, metric_name in key_metrics:
            if stats[metric_key] is not None:
                s = stats[metric_key]
                lines.append(f"\n  {metric_name}:")
                lines.append(f"    Initial:         {s['initial']:.6f}")
                lines.append(f"    Final:           {s['final']:.6f}")
                lines.append(f"    Mean (2nd half): {s['mean_2nd_half']:.6f} ± {s['std_2nd_half']:.6f}")
                lines.append(f"    Range:           [{s['min']:.6f}, {s['max']:.6f}]")
    
    lines.append("\n" + "-" * 80)
    lines.append("COMPARATIVE SUMMARY - KEY METRICS")
    lines.append("-" * 80)
    
    lines.append(f"\n  {'Case':<20} {'H_coh':>10} {'H_pol':>10} {'H_vel':>10} {'OSI':>10} {'OI':>10}")
    lines.append("  " + "-" * 70)
    
    for case_name in all_data.keys():
        stats = all_stats[case_name]
        short_name = case_name.split(' - ')[1] if ' - ' in case_name else case_name
        
        h_coh = stats['school_cohesion_entropy']['final'] if stats['school_cohesion_entropy'] else np.nan
        h_pol = stats['polarization_entropy']['final'] if stats['polarization_entropy'] else np.nan
        h_vel = stats['velocity_correlation_entropy']['final'] if stats['velocity_correlation_entropy'] else np.nan
        osi = stats['oceanic_schooling_index']['final'] if stats['oceanic_schooling_index'] else np.nan
        oi = stats['order_index']['final'] if stats['order_index'] else np.nan
        
        lines.append(f"  {short_name:<20} {h_coh:>10.4f} {h_pol:>10.4f} {h_vel:>10.4f} {osi:>10.4f} {oi:>10.4f}")
    
    lines.append("\n" + "-" * 80)
    lines.append("INTERPRETATION")
    lines.append("-" * 80)
    lines.append("""
Entropy-Order Parameter Relationship:

- SWARM: High OSI (disordered), low Order Index
  All entropy metrics elevated due to random organization
  
- TORUS: Medium OSI, medium Order Index (M > P)
  Angular entropy elevated (milling), moderate polarization entropy
  
- DYNAMIC PARALLEL: Low-Medium OSI, high P
  Low polarization entropy (aligned), moderate cohesion entropy
  
- HIGHLY PARALLEL: Low OSI (ordered), high Order Index (P >> M)
  All entropy metrics low, especially H_pol and H_vel
  Strong alignment creates ordered, low-entropy state
""")
    
    lines.append("-" * 80)
    lines.append("FIGURE SPECIFICATIONS")
    lines.append("-" * 80)
    lines.append(f"  Layout:      2×2 panels")
    lines.append(f"  Panel (a):   School Cohesion Entropy (H_coh)")
    lines.append(f"  Panel (b):   Polarization Entropy (H_pol)")
    lines.append(f"  Panel (c):   Velocity Correlation Entropy (H_vel)")
    lines.append(f"  Panel (d):   Oceanic Schooling Index (OSI)")
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
    print("Entropy Metrics Evolution Analysis")
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
    
    has_entropy = any(all_data[k].get('oceanic_schooling_index') is not None 
                      for k in all_data.keys())
    if not has_entropy:
        print("WARNING: No entropy metrics found in data files!")
        print("         Ensure compute_entropy=true in simulation config.")
    
    print("\n[2/4] Computing statistics...")
    all_stats = {}
    for case_name, data in all_data.items():
        all_stats[case_name] = compute_entropy_statistics(data)
    
    print("\n[3/4] Creating figure...")
    fig = create_figure(all_data)
    
    base_name = "entropy_metrics_evolution"
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
