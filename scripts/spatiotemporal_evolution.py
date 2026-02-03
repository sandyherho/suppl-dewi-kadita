#!/usr/bin/env python3
"""
Figure 1: Spatiotemporal Evolution of Fish Schooling Dynamics

Creates a 4×3 multipanel figure showing the evolution of collective states
across all four Couzin model scenarios at three time points (initial, middle, final).

Features:
- Clean white background for publication
- 3D fish positions with velocity vectors
- Bold X, Y, Z axis labels
- Panel labels (a)-(l) without titles
- Publication-quality output (PDF, PNG, EPS)
- Comprehensive statistical report

Author: Sandy H. S. Herho
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from netCDF4 import Dataset
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
DATA_DIR = Path("../data")
FIG_DIR = Path("../figs")
REPORT_DIR = Path("../reports")

# Create directories
FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Data files
NC_FILES = {
    'Case 1 - Swarm': 'case1_swarm.nc',
    'Case 2 - Torus': 'case2_torus.nc',
    'Case 3 - Dynamic Parallel': 'case3_dynamic_parallel.nc',
    'Case 4 - Highly Parallel': 'case4_highly_parallel.nc'
}

# Color palette for white background
COLORS = {
    'background': '#FFFFFF',
    'pane_xy': '#F5F8FA',
    'pane_z': '#EBF2F7',
    'grid': '#D0D8E0',
    'fish_primary': '#0077B6',
    'text': '#2C3E50',
    'title': '#1A252F',
    'stats': '#34495E',
    'label': '#000000',
    'axis_label': '#000000'
}

# Case-specific colors for fish visualization
CASE_FISH_COLORS = {
    'Case 1 - Swarm': '#E74C3C',
    'Case 2 - Torus': '#27AE60',
    'Case 3 - Dynamic Parallel': '#3498DB',
    'Case 4 - Highly Parallel': '#F39C12'
}

# Figure settings
FIG_WIDTH = 18
FIG_HEIGHT = 22
DPI = 300

# ============================================================================
# DATA LOADING
# ============================================================================

def load_netcdf_data(filepath):
    """Load all relevant data from NetCDF file."""
    with Dataset(filepath, 'r') as nc:
        data = {
            'time': nc.variables['time'][:],
            'positions': nc.variables['positions'][:],
            'velocities': nc.variables['velocities'][:],
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
        
        try:
            data['oceanic_schooling_index'] = nc.variables['oceanic_schooling_index'][:]
            data['order_index'] = nc.variables['order_index'][:]
        except:
            data['oceanic_schooling_index'] = None
            data['order_index'] = None
            
    return data

# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def compute_spatial_statistics(positions, box_size):
    """Compute spatial statistics for a single time snapshot."""
    from scipy.spatial import cKDTree
    
    n = len(positions)
    tree = cKDTree(positions, boxsize=box_size)
    distances, _ = tree.query(positions, k=2)
    nnd = distances[:, 1]
    
    centroid = np.mean(positions, axis=0)
    r_from_centroid = np.linalg.norm(positions - centroid, axis=1)
    spread = np.sqrt(np.mean(r_from_centroid**2))
    
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(positions)
        hull_volume = hull.volume
    except:
        hull_volume = np.nan
    
    return {
        'nnd_mean': np.mean(nnd),
        'nnd_std': np.std(nnd),
        'nnd_min': np.min(nnd),
        'nnd_max': np.max(nnd),
        'spread': spread,
        'centroid': centroid,
        'hull_volume': hull_volume,
        'density_local': n / (hull_volume + 1e-10)
    }

def compute_velocity_statistics(velocities):
    """Compute velocity-based statistics."""
    speeds = np.linalg.norm(velocities, axis=1)
    
    norms = np.linalg.norm(velocities, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    unit_vel = velocities / norms
    
    mean_dir = np.mean(unit_vel, axis=0)
    polarization = np.linalg.norm(mean_dir)
    
    dot_products = np.dot(unit_vel, mean_dir)
    angular_dispersion = np.std(np.arccos(np.clip(dot_products / (polarization + 1e-10), -1, 1)))
    
    return {
        'speed_mean': np.mean(speeds),
        'speed_std': np.std(speeds),
        'polarization': polarization,
        'angular_dispersion': angular_dispersion,
        'mean_direction': mean_dir
    }

# ============================================================================
# VISUALIZATION
# ============================================================================

def setup_white_style():
    """Setup matplotlib with clean white theme for publication."""
    plt.style.use('default')
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': '#333333',
        'axes.labelcolor': '#000000',
        'xtick.color': '#333333',
        'ytick.color': '#333333',
        'text.color': '#333333',
        'grid.color': '#CCCCCC',
        'grid.alpha': 0.5,
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.labelsize': 14,
        'axes.titlesize': 12,
        'mathtext.fontset': 'dejavusans'
    })

def plot_fish_school_3d(ax, positions, velocities, box_size, panel_label, 
                        time_val, polarization, rotation, case_name):
    """Plot a single 3D fish school visualization with white background."""
    
    fish_color = CASE_FISH_COLORS.get(case_name, COLORS['fish_primary'])
    
    depths = positions[:, 2]
    depth_norm = (depths - depths.min()) / (depths.max() - depths.min() + 1e-10)
    
    # Plot velocity vectors (quiver)
    ax.quiver(positions[:, 0], positions[:, 1], positions[:, 2],
              velocities[:, 0], velocities[:, 1], velocities[:, 2],
              length=0.8, normalize=True,
              color=fish_color, alpha=0.8,
              arrow_length_ratio=0.25, linewidth=0.6)
    
    # Scatter for fish positions
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
               c=[fish_color], s=12, alpha=0.6, edgecolors='white', linewidth=0.3)
    
    # Set limits
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    ax.set_zlim(0, box_size)
    
    # =========================================================================
    # BIG BOLD AXIS LABELS
    # =========================================================================
    ax.set_xlabel('X', fontsize=16, fontweight='bold', color=COLORS['axis_label'],
                  labelpad=8)
    ax.set_ylabel('Y', fontsize=16, fontweight='bold', color=COLORS['axis_label'],
                  labelpad=8)
    ax.set_zlabel('Z', fontsize=16, fontweight='bold', color=COLORS['axis_label'],
                  labelpad=8)
    
    # Remove tick labels but keep the axis labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    
    # Style 3D panes
    ax.xaxis.pane.fill = True
    ax.yaxis.pane.fill = True
    ax.zaxis.pane.fill = True
    ax.xaxis.pane.set_facecolor(COLORS['pane_xy'])
    ax.yaxis.pane.set_facecolor(COLORS['pane_xy'])
    ax.zaxis.pane.set_facecolor(COLORS['pane_z'])
    ax.xaxis.pane.set_alpha(1.0)
    ax.yaxis.pane.set_alpha(1.0)
    ax.zaxis.pane.set_alpha(1.0)
    
    # Edge colors
    ax.xaxis.pane.set_edgecolor(COLORS['grid'])
    ax.yaxis.pane.set_edgecolor(COLORS['grid'])
    ax.zaxis.pane.set_edgecolor(COLORS['grid'])
    
    # Grid styling
    ax.xaxis._axinfo['grid']['color'] = COLORS['grid']
    ax.yaxis._axinfo['grid']['color'] = COLORS['grid']
    ax.zaxis._axinfo['grid']['color'] = COLORS['grid']
    ax.xaxis._axinfo['grid']['linewidth'] = 0.5
    ax.yaxis._axinfo['grid']['linewidth'] = 0.5
    ax.zaxis._axinfo['grid']['linewidth'] = 0.5
    
    # Panel label (a), (b), etc.
    ax.text2D(0.02, 0.98, f'({panel_label})', transform=ax.transAxes,
              fontsize=14, fontweight='bold', color=COLORS['label'],
              ha='left', va='top')
    
    # Stats annotation (bottom)
    stats_text = f't={time_val:.1f}  P={polarization:.2f}  M={rotation:.2f}'
    ax.text2D(0.5, 0.02, stats_text, transform=ax.transAxes,
              fontsize=9, color=COLORS['stats'], ha='center', va='bottom',
              family='monospace')
    
    # Set view angle
    ax.view_init(elev=25, azim=45)

def create_multipanel_figure(all_data):
    """Create the 4×3 multipanel figure with white background."""
    
    setup_white_style()
    
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor='white')
    
    labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']
    time_labels = ['Initial', 'Middle', 'Final']
    case_names = list(all_data.keys())
    
    panel_idx = 0
    for row, case_name in enumerate(case_names):
        data = all_data[case_name]
        n_times = len(data['time'])
        time_indices = [0, n_times // 2, -1]
        
        for col, t_idx in enumerate(time_indices):
            ax = fig.add_subplot(4, 3, panel_idx + 1, projection='3d',
                                facecolor='white')
            
            plot_fish_school_3d(
                ax,
                data['positions'][t_idx],
                data['velocities'][t_idx],
                data['box_size'],
                labels[panel_idx],
                data['time'][t_idx],
                data['polarization'][t_idx],
                data['rotation'][t_idx],
                case_name
            )
            
            panel_idx += 1
    
    # Row labels (case names)
    for row, case_name in enumerate(case_names):
        short_name = case_name.split(' - ')[1] if ' - ' in case_name else case_name
        y_pos = 0.88 - row * 0.235
        fig.text(0.01, y_pos, short_name, fontsize=12, fontweight='bold',
                color=COLORS['title'], rotation=90, va='center', ha='left')
    
    # Column labels (time points)
    for col, time_label in enumerate(time_labels):
        x_pos = 0.22 + col * 0.28
        fig.text(x_pos, 0.98, time_label, fontsize=12, fontweight='bold',
                color=COLORS['title'], ha='center', va='top')
    
    plt.tight_layout(rect=[0.03, 0.01, 0.99, 0.96])
    
    return fig

# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(all_data, stats_summary):
    """Generate comprehensive statistical report."""
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("FIGURE 1: SPATIOTEMPORAL EVOLUTION OF FISH SCHOOLING DYNAMICS")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().isoformat()}")
    report_lines.append("")
    
    report_lines.append("-" * 80)
    report_lines.append("METHODS")
    report_lines.append("-" * 80)
    report_lines.append("""
This figure presents the spatiotemporal evolution of collective states across
four distinct Couzin model scenarios. The visualization employs a 4×3 grid
structure with rows representing different behavioral regimes (Swarm, Torus,
Dynamic Parallel, Highly Parallel) and columns representing temporal snapshots
(initial t=0, middle t=T/2, final t=T).

Model Description:
- Couzin zone-based model (2002) with three behavioral zones:
  * Zone of Repulsion (ZOR): Collision avoidance (highest priority)
  * Zone of Orientation (ZOO): Alignment with neighbors
  * Zone of Attraction (ZOA): Cohesion with distant neighbors

Order Parameters:
- Polarization (P): P = (1/N)|Σᵢv̂ᵢ|, measures alignment (P=1: perfect alignment)
- Rotation (M): M = (1/N)|Σᵢ(v̂ᵢ × r̂ᵢᶜ)|, measures milling (M=1: perfect torus)

Visualization:
- 3D quiver plots showing fish positions and velocity vectors
- Case-specific colors for each behavioral regime
- Clean white background for publication clarity
- Bold X, Y, Z axis labels for clear orientation
- Panel labels (a)-(l) following journal conventions
""")
    
    report_lines.append("-" * 80)
    report_lines.append("STATISTICAL SUMMARY BY SCENARIO")
    report_lines.append("-" * 80)
    
    for case_name, data in all_data.items():
        report_lines.append(f"\n{case_name}")
        report_lines.append("~" * 60)
        
        report_lines.append(f"  System Parameters:")
        report_lines.append(f"    N (fish count):        {data['n_fish']}")
        report_lines.append(f"    L (box size):          {data['box_size']:.1f}")
        report_lines.append(f"    v₀ (speed):            {data['speed']:.2f}")
        report_lines.append(f"    rᵣ (repulsion):        {data['r_repulsion']:.2f}")
        report_lines.append(f"    rₒ (orientation):      {data['r_orientation']:.2f}")
        report_lines.append(f"    rₐ (attraction):       {data['r_attraction']:.2f}")
        report_lines.append(f"    θ_max (max turn):      {data['max_turn']:.2f} rad")
        report_lines.append(f"    σ (noise):             {data['noise']:.2f} rad")
        report_lines.append(f"    α (blind angle):       {data['blind_angle']:.1f}°")
        
        n_times = len(data['time'])
        t_total = data['time'][-1]
        report_lines.append(f"  Simulation:")
        report_lines.append(f"    Time steps saved:      {n_times}")
        report_lines.append(f"    Total time:            {t_total:.1f}")
        
        P = data['polarization']
        M = data['rotation']
        report_lines.append(f"  Polarization (P):")
        report_lines.append(f"    Initial:               {P[0]:.4f}")
        report_lines.append(f"    Final:                 {P[-1]:.4f}")
        report_lines.append(f"    Mean (2nd half):       {np.mean(P[n_times//2:]):.4f}")
        report_lines.append(f"    Std (2nd half):        {np.std(P[n_times//2:]):.4f}")
        report_lines.append(f"    Min:                   {np.min(P):.4f}")
        report_lines.append(f"    Max:                   {np.max(P):.4f}")
        
        report_lines.append(f"  Rotation (M):")
        report_lines.append(f"    Initial:               {M[0]:.4f}")
        report_lines.append(f"    Final:                 {M[-1]:.4f}")
        report_lines.append(f"    Mean (2nd half):       {np.mean(M[n_times//2:]):.4f}")
        report_lines.append(f"    Std (2nd half):        {np.std(M[n_times//2:]):.4f}")
        report_lines.append(f"    Min:                   {np.min(M):.4f}")
        report_lines.append(f"    Max:                   {np.max(M):.4f}")
        
        for t_name, t_idx in [('Initial', 0), ('Middle', n_times//2), ('Final', -1)]:
            stats = compute_spatial_statistics(data['positions'][t_idx], data['box_size'])
            vel_stats = compute_velocity_statistics(data['velocities'][t_idx])
            
            report_lines.append(f"  Spatial Stats ({t_name}, t={data['time'][t_idx]:.1f}):")
            report_lines.append(f"    NND mean ± std:        {stats['nnd_mean']:.3f} ± {stats['nnd_std']:.3f}")
            report_lines.append(f"    NND range:             [{stats['nnd_min']:.3f}, {stats['nnd_max']:.3f}]")
            report_lines.append(f"    School spread (RMS):   {stats['spread']:.3f}")
            report_lines.append(f"    Hull volume:           {stats['hull_volume']:.2f}")
            report_lines.append(f"    Angular dispersion:    {vel_stats['angular_dispersion']:.4f} rad")
    
    report_lines.append("\n" + "-" * 80)
    report_lines.append("INTERPRETATION")
    report_lines.append("-" * 80)
    report_lines.append("""
Collective State Classification:

1. SWARM (Case 1): 
   - Characterized by low polarization (P < 0.4) and low rotation (M < 0.3)
   - Fish aggregate loosely without coherent directional motion
   - Large repulsion zone and high noise prevent alignment
   - Ecologically represents feeding aggregations or resting schools

2. TORUS (Case 2):
   - Characterized by medium polarization and high rotation (M > 0.5)
   - Fish form rotating mill with tangential velocities
   - Large blind angle (150°) promotes following behavior
   - Reduced orientation weight (0.1) prevents parallel alignment
   - Ecologically observed as defensive milling behavior

3. DYNAMIC PARALLEL (Case 3):
   - High polarization (0.7 < P < 0.9) with temporal fluctuations
   - School maintains alignment but changes direction
   - Moderate noise causes directional instability
   - Ecologically represents migrating schools under perturbation

4. HIGHLY PARALLEL (Case 4):
   - Very high, stable polarization (P > 0.9)
   - Minimal rotation - coherent linear motion
   - Large orientation zone and low noise maximize alignment
   - Ecologically observed in fast-moving pelagic schools

Temporal Evolution:
- Initial states show configuration effects (random vs torus initialization)
- Middle states reveal transition dynamics and stability
- Final states represent quasi-equilibrium collective patterns
""")
    
    report_lines.append("-" * 80)
    report_lines.append("FIGURE SPECIFICATIONS")
    report_lines.append("-" * 80)
    report_lines.append(f"  Dimensions:        {FIG_WIDTH}\" × {FIG_HEIGHT}\"")
    report_lines.append(f"  Resolution:        {DPI} DPI")
    report_lines.append(f"  Layout:            4 rows × 3 columns")
    report_lines.append(f"  Panel labels:      (a) through (l)")
    report_lines.append(f"  Axis labels:       Bold X, Y, Z (16pt)")
    report_lines.append(f"  Output formats:    PDF, PNG, EPS")
    report_lines.append(f"  Color scheme:      White background, case-specific fish colors")
    report_lines.append("")
    report_lines.append("=" * 80)
    
    return "\n".join(report_lines)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    
    print("=" * 70)
    print("Figure 1: Spatiotemporal Evolution of Fish Schooling Dynamics")
    print("         (White Background + Bold X/Y/Z Labels)")
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
    
    print("\n[2/4] Computing spatial statistics...")
    stats_summary = {}
    for case_name, data in all_data.items():
        n_times = len(data['time'])
        stats_summary[case_name] = {
            'initial': compute_spatial_statistics(data['positions'][0], data['box_size']),
            'middle': compute_spatial_statistics(data['positions'][n_times//2], data['box_size']),
            'final': compute_spatial_statistics(data['positions'][-1], data['box_size'])
        }
    
    print("\n[3/4] Creating multipanel figure...")
    fig = create_multipanel_figure(all_data)
    
    base_name = "spatiotemporal_evolution"
    
    for fmt in ['pdf', 'png', 'eps']:
        output_path = FIG_DIR / f"{base_name}.{fmt}"
        print(f"      Saving {output_path}...")
        fig.savefig(output_path, dpi=DPI, facecolor='white',
                   edgecolor='none', bbox_inches='tight', format=fmt)
    
    plt.close(fig)
    
    print("\n[4/4] Generating statistical report...")
    report_text = generate_report(all_data, stats_summary)
    report_path = REPORT_DIR / f"{base_name}_report.txt"
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f"      Saved {report_path}")
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print(f"  Figures: {FIG_DIR}/{base_name}.{{pdf,png,eps}}")
    print(f"  Report:  {report_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
