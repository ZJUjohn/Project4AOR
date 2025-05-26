import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from .config import (
    PLOTS_DIR, STD_FIG_SIZE, STD_LABEL_FONTSIZE, STD_TITLE_FONTSIZE,
    STD_TICK_FONTSIZE, STD_LEGEND_FONTSIZE, STD_LEGEND_LOC, STD_LEGEND_BBOX,
    LEGEND_ADJUST_RIGHT, FIGURE_TITLE_Y_ADJUST
)

def plot_am_mode_split_q3_academic(df_q3_direct, df_q3_coastal):
    print("Generating Chart 2: AM Mode Split (Q3)...")
    if (df_q3_direct is None or df_q3_direct.empty) and \
       (df_q3_coastal is None or df_q3_coastal.empty):
        print("Warning (Chart 2): Both direct and coastal Q3 data are empty or None. Skipping.")
        return

    # Ensure 'AM' and 'AnnualVolume' columns exist
    required_cols = ['AM', 'AnnualVolume']
    if df_q3_direct is not None and not df_q3_direct.empty:
        if not all(col in df_q3_direct.columns for col in required_cols):
            print(f"Warning (Chart 2): df_q3_direct missing one of {required_cols}. Treating as empty for this chart.")
            df_q3_direct = pd.DataFrame(columns=required_cols) # Make it empty but with correct columns
    else:
        df_q3_direct = pd.DataFrame(columns=required_cols)


    if df_q3_coastal is not None and not df_q3_coastal.empty:
        if not all(col in df_q3_coastal.columns for col in required_cols):
            print(f"Warning (Chart 2): df_q3_coastal missing one of {required_cols}. Treating as empty for this chart.")
            df_q3_coastal = pd.DataFrame(columns=required_cols)
    else:
        df_q3_coastal = pd.DataFrame(columns=required_cols)


    am_direct_vol = df_q3_direct.groupby('AM')['AnnualVolume'].sum().rename('Direct Volume') if not df_q3_direct.empty else pd.Series(name='Direct Volume', dtype='float64')
    am_coastal_vol = df_q3_coastal.groupby('AM')['AnnualVolume'].sum().rename('Coastal Volume') if not df_q3_coastal.empty else pd.Series(name='Coastal Volume', dtype='float64')

    df_am_plot = pd.concat([am_direct_vol, am_coastal_vol], axis=1).fillna(0)
    all_ams = ['AM1', 'AM2', 'AM3']
    df_am_plot = df_am_plot.reindex(index=all_ams, fill_value=0).sort_index()

    if df_am_plot.empty or df_am_plot.sum().sum() == 0:
        print("Warning (Chart 2): No data to plot after processing. Skipping.")
        return

    fig, ax = plt.subplots(figsize=STD_FIG_SIZE)
    colors = ['#4C72B0', '#55A868'] # Standard colors for direct/coastal
    df_am_plot.plot(kind='bar', stacked=True, ax=ax, color=colors, width=0.7)

    ax.set_xlabel('Manufacturing Cluster (AM)', fontsize=STD_LABEL_FONTSIZE)
    ax.set_ylabel('Total Annual Volume (Units x1000)', fontsize=STD_LABEL_FONTSIZE)
    ax.set_title('Q3 Volume by AM and Transport Mode', fontsize=STD_TITLE_FONTSIZE, fontweight='bold', y=FIGURE_TITLE_Y_ADJUST)
    ax.tick_params(axis='x', rotation=0, labelsize=STD_TICK_FONTSIZE)
    ax.tick_params(axis='y', labelsize=STD_TICK_FONTSIZE)
    ax.legend(title='Transport Mode', fontsize=STD_LEGEND_FONTSIZE, loc=STD_LEGEND_LOC, bbox_to_anchor=STD_LEGEND_BBOX, borderaxespad=0.)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x_val, _: f'{x_val/1e3:,.0f}')) # Original was K, but data is in units, so /1e3
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add labels to bars
    # Calculate max sum for percentage threshold dynamically if df_am_plot is not empty
    max_total_volume_per_am = 0
    if not df_am_plot.empty:
        max_total_volume_per_am = df_am_plot.sum(axis=1).max()

    for c_container in ax.containers:
        labels_to_add = []
        for v_bar in c_container:
            height = v_bar.get_height()
            # Show label if height is significant relative to max total volume for any AM
            if max_total_volume_per_am > 0 and height > 0.05 * max_total_volume_per_am:
                 labels_to_add.append(f'{height/1e3:,.0f}K')
            elif height > 0: # For very small plots, show if non-zero
                 labels_to_add.append(f'{height/1e3:,.1f}K')
            else:
                 labels_to_add.append('')
        ax.bar_label(c_container, labels=labels_to_add, label_type='center', fontsize=8, color='white', fontweight='bold')

    fig.subplots_adjust(right=LEGEND_ADJUST_RIGHT - 0.05, top=0.88) # Adjust slightly for this plot
    fig.savefig(os.path.join(PLOTS_DIR, "02_am_volume_split_q3_academic.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Chart 2 saved.")