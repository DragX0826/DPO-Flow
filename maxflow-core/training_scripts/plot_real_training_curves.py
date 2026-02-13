#!/usr/bin/env python
"""
MaxFlow: HONEST Training Curve Visualization.
Generates figures from ACTUAL training data only. No hardcoded values.
Fixed: Non-breaking spaces removed. Paths adjusted for general use.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import argparse

# Set global style properties for publication-quality plots
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 18
})

def load_and_prepare(csv_path):
    """Load CSV and add a global step index."""
    try:
        df = pd.read_csv(csv_path)
        # Ensure required columns exist
        required_cols = ['epoch', 'step', 'loss', 'reward_win']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"CSV missing required column: {col}")
        
        df['global_step'] = range(len(df))
        # Fill NaN MaxRL values with 0 for active signal checking
        if 'MaxRL' in df.columns:
            df['MaxRL'] = df['MaxRL'].fillna(0)
        return df
    except FileNotFoundError:
        print(f"❌ Error: File not found at {csv_path}")
        exit(1)
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        exit(1)

def detect_phases(df):
    """
    Detect training restart boundaries (where epoch resets to 1).
    This ensures plots don't loop back in time.
    """
    boundaries = [0]
    for i in range(1, len(df)):
        # A restart is characterized by epoch resetting to 1 at step 1
        if df.iloc[i]['epoch'] == 1 and df.iloc[i]['step'] == 1:
            # Simple debounce to prevent false positives if logging is duplicate
            if i > boundaries[-1] + 5: 
                boundaries.append(i)
    boundaries.append(len(df))
    return boundaries

def plot_training_overview(df, boundaries, output_dir):
    """4-panel figure: Loss, Reward, MaxRL Signal, Component Losses."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=150)
    fig.suptitle('MaxFlow Training Dynamics (Empirical Data)', fontweight='bold')
    
    colors = ['#95a5a6', '#95a5a6', '#e67e22'] # Gray for early, Orange for MaxRL
    phase_labels = []
    for i in range(len(boundaries) - 1):
        if i == len(boundaries) - 2:
            phase_labels.append('MaxFlow (Flagship)')
        else:
            phase_labels.append(f'Pre-Align Phase {i+1}')
    
    # ── Panel 1: Training Loss ──
    ax = axes[0, 0]
    for i in range(len(boundaries) - 1):
        seg = df.iloc[boundaries[i]:boundaries[i+1]]
        # Use a smaller window for more granular view, larger for smoother
        rolling = seg['loss'].rolling(window=20, min_periods=1).mean()
        ax.plot(seg['global_step'], rolling, color=colors[i % len(colors)], 
                alpha=0.85, linewidth=1.5, label=phase_labels[i])
        
        # Add annotation for the "Jump" if more than one phase exists
        if len(boundaries) > 2 and i == len(boundaries) - 2:
            jump_x = df.iloc[boundaries[i]]['global_step']
            ax.axvline(x=jump_x, color='red', linestyle='--', alpha=0.5)

    ax.set_ylabel('Loss (20-step Moving Avg)')
    ax.set_xlabel('Global Training Step')
    ax.set_title('Total Training Loss')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # ── Panel 2: Reward Signal (The most important one) ──
    ax = axes[0, 1]
    for i in range(len(boundaries) - 1):
        seg = df.iloc[boundaries[i]:boundaries[i+1]]
        rolling = seg['reward_win'].rolling(window=20, min_periods=1).mean()
        ax.plot(seg['global_step'], rolling, color=colors[i % len(colors)], 
                alpha=0.85, linewidth=1.5, label=phase_labels[i])
        
        # Annotate the Vertical Jump (The "Honest" Annotation)
        if len(boundaries) > 2 and i == len(boundaries) - 2:
            jump_idx = boundaries[i]
            jump_x = df.iloc[jump_idx]['global_step']
            jump_y = df.iloc[jump_idx]['reward_win']
            
            # Place text above data to avoid horizontal scaling issues
            ax.annotate('Fine-Tuning Start\n(Checkpoint Load)', 
                        xy=(jump_x, jump_y),
                        xytext=(0, 40), # 40 points vertically above
                        textcoords='offset points',
                        arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=6, alpha=0.6),
                        fontsize=9, fontweight='bold', color='red', ha='center',
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='red', alpha=0.7))

    ax.set_ylabel('Composite Reward (20-step MA)')
    ax.set_xlabel('Global Training Step')
    ax.set_title('Reward Signal Evolution (r_win)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # ── Panel 3: MaxRL Loss Component ──
    ax = axes[1, 0]
    has_MaxRL_data = False
    for i in range(len(boundaries) - 1):
        seg = df.iloc[boundaries[i]:boundaries[i+1]]
        if 'MaxRL' in seg.columns:
            # Only plot non-zero MaxRL values (active signal)
            MaxRL_mask = seg['MaxRL'].abs() > 1e-6
            if MaxRL_mask.any():
                has_MaxRL_data = True
                active = seg[MaxRL_mask]
                rolling = active['MaxRL'].rolling(window=10, min_periods=1).mean()
                ax.plot(active['global_step'], rolling, color=colors[i % len(colors)], 
                        alpha=0.6, linewidth=1.0, label=f'{phase_labels[i]}')

    if has_MaxRL_data:
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylabel('L_MaxRL (10-step MA, active only)')
        ax.set_title('MaxRL Gradient Signal (Preference Strength)')
    else:
        ax.text(0.5, 0.5, 'No Active MaxRL Signal Detected', ha='center', va='center')

    ax.set_xlabel('Global Training Step')
    ax.grid(True, alpha=0.3)
    
    # ── Panel 4: Auxiliary Loss Components ──
    ax = axes[1, 1]
    # Use last (best) phase for component breakdown
    last_phase_idx = len(boundaries) - 2
    last_phase = df.iloc[boundaries[last_phase_idx]:boundaries[last_phase_idx+1]]
    
    components = ['anchor', 'clip', 'conf', 'vib']
    comp_colors = ['#2ecc71', '#3498db', '#e74c3c', '#8e44ad']
    found_components = False
    for comp, cc in zip(components, comp_colors):
        if comp in last_phase.columns:
            found_components = True
            rolling = last_phase[comp].rolling(window=30, min_periods=1).mean()
            ax.plot(last_phase['global_step'], rolling, color=cc, 
                    alpha=0.8, linewidth=1.2, label=f'L_{comp}')
            
    if found_components:
        ax.set_ylabel('Loss Value (30-step MA)')
        ax.set_title(f'Auxiliary Losses ({phase_labels[last_phase_idx]})')
        ax.legend()
    else:
         ax.text(0.5, 0.5, 'No Auxiliary Losses Logged', ha='center', va='center')

    ax.set_xlabel('Global Training Step')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'real_training_curves.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved Training Overview: {out_path}")

def plot_MaxRL_preference_landscape(df, output_dir):
    """Plot distribution of MaxRL winning vs losing rewards with linear scale for clarity."""
    if 'reward_lose' not in df.columns:
        print("⚠️ 'reward_lose' column missing. Skipping preference landscape plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    # Filter for active steps only
    active_df = df[df['reward_win'] != df['reward_lose']].copy()
    
    sns.histplot(active_df['reward_win'], color='#2ecc71', alpha=0.5, label='MaxRL Winner (Preferred)', 
                 kde=True, bins=40, ax=ax)
    
    sns.histplot(active_df['reward_lose'], color='#e74c3c', alpha=0.5, label='MaxRL Loser (Rejected)', 
                 kde=True, bins=40, ax=ax)
    
    ax.set_title('Separation of Preference Distributions in MaxFlow', fontsize=15, fontweight='bold')
    ax.set_xlabel('Composite Reward Score (Physics/Affinity)', fontsize=13)
    ax.set_ylabel('Density (Linear Scale)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, ls='--')
    
    out_path = os.path.join(output_dir, 'MaxRL_preference_landscape.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved MaxRL Landscape: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate honest training visualization from MaxRL logs.")
    parser.add_argument("csv_path", type=str, help="Path to the training log CSV file.")
    parser.add_argument("--output_dir", type=str, default="./figures", help="Directory to save plots.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"🔍 Loading training data from: {args.csv_path}")
    df = load_and_prepare(args.csv_path)
    boundaries = detect_phases(df)
    
    print(f"📌 Detected {len(boundaries)-1} training phase(s).")
    print(f"📊 Total data points: {len(df)}")
    
    print("\n🎨 Generating training overview curves...")
    plot_training_overview(df, boundaries, args.output_dir)
    
    print("🎨 Generating MaxRL preference landscape histogram...")
    plot_MaxRL_preference_landscape(df, args.output_dir)
    
    print(f"\n✨ Done. All figures saved to: {args.output_dir}")

if __name__ == "__main__":
    # Use seaborn style for better aesthetics if available
    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid")
    except ImportError:
        pass
    main()
