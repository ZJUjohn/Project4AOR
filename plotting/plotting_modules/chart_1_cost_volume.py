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

def plot_cost_volume_structure_academic(df_q1_annualized_cost, df_q3_summary, df_q3_direct_vol_data, q1_direct_unit_costs_df_data):
    print("Generating Chart 1: Cost and Volume Structure Comparison...")

    if df_q3_summary.empty or 'TotalOptimalCost_Annual' not in df_q3_summary.columns:
        print("Error (Chart 1): Q3 summary data is missing 'TotalOptimalCost_Annual'. Skipping cost plot.")
        return

    q1_total_cost = df_q1_annualized_cost # This is a scalar value
    q3_total_optimal_cost = df_q3_summary['TotalOptimalCost_Annual'].iloc[0]

    q3_direct_cost_estimated = 0
    if df_q3_direct_vol_data is not None and not df_q3_direct_vol_data.empty and \
       q1_direct_unit_costs_df_data is not None and not q1_direct_unit_costs_df_data.empty:
        # Original logic from plotting.py for q3_direct_cost_estimated
        direct_costs_map = {}
        id_col_q1 = 'Cust_ID' if 'Cust_ID' in q1_direct_unit_costs_df_data.columns else 'Customer_Location'
        # Ensure Customer_Location exists if it's the fallback
        if id_col_q1 == 'Customer_Location' and 'Customer_Location' not in q1_direct_unit_costs_df_data.columns:
            print(f"Warning (Chart 1 Cost): Fallback ID column '{id_col_q1}' not found in q1_direct_unit_costs_df_data. Direct cost estimation may be inaccurate.")
        elif id_col_q1 not in q1_direct_unit_costs_df_data.columns:
             print(f"Warning (Chart 1 Cost): ID column '{id_col_q1}' not found in q1_direct_unit_costs_df_data. Direct cost estimation may be inaccurate.")

        for _, row in q1_direct_unit_costs_df_data.iterrows():
            if id_col_q1 not in row: continue # Skip if ID column is missing in the row
            cust_id_val = row[id_col_q1]
            for am in ['AM1', 'AM2', 'AM3']:
                cost_col = f'{am}_Direct_Cost_Per_Unit'
                if cost_col in row and pd.notna(row[cost_col]) and row[cost_col] != np.inf:
                    direct_costs_map[(am, cust_id_val)] = row[cost_col]
        
        if 'AM' in df_q3_direct_vol_data.columns and \
           'Customer' in df_q3_direct_vol_data.columns and \
           'AnnualVolume' in df_q3_direct_vol_data.columns:
            for _, row in df_q3_direct_vol_data.iterrows():
                unit_cost = direct_costs_map.get((row['AM'], row['Customer']), 0)
                q3_direct_cost_estimated += row['AnnualVolume'] * unit_cost
        else:
            print("Warning (Chart 1 Cost): df_q3_direct_vol_data is missing one or more required columns: 'AM', 'Customer', 'AnnualVolume'. Direct cost estimation may be inaccurate.")


    q3_coastal_total_cost = q3_total_optimal_cost - q3_direct_cost_estimated
    if q3_coastal_total_cost < 0: q3_coastal_total_cost = 0
    # q3_direct_cost_estimated = q3_total_optimal_cost - q3_coastal_total_cost # Re-affirm if coastal was capped

    cost_labels = ['Q1 Baseline\n(Annualized)', 'Q3 Optimized']
    direct_costs_values = [q1_total_cost, q3_direct_cost_estimated]
    coastal_costs_q3_values = [0, q3_coastal_total_cost]

    fig1, ax1 = plt.subplots(figsize=STD_FIG_SIZE)
    bar_width = 0.4
    index = np.arange(len(cost_labels))

    bar1_plot = ax1.bar(index, direct_costs_values, bar_width, label='Direct Transport Cost', color='#4C72B0')
    bar2_plot = ax1.bar(index, coastal_costs_q3_values, bar_width, bottom=direct_costs_values, label='Coastal Transport Cost', color='#55A868')

    ax1.set_ylabel('Total Annual Cost (USD Millions)', fontsize=STD_LABEL_FONTSIZE)
    ax1.set_title('Annual Transportation Cost Comparison', fontsize=STD_TITLE_FONTSIZE, fontweight='bold', y=FIGURE_TITLE_Y_ADJUST)
    ax1.set_xticks(index)
    ax1.set_xticklabels(cost_labels, fontsize=STD_TICK_FONTSIZE)
    ax1.legend(loc=STD_LEGEND_LOC, bbox_to_anchor=STD_LEGEND_BBOX, fontsize=STD_LEGEND_FONTSIZE, borderaxespad=0.)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x_val, _: f'{x_val/1e6:,.1f}M'))
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    for rect_group in [bar1_plot, bar2_plot]:
        for rect in rect_group:
            height = rect.get_height()
            # Determine bar_total_ref based on which bar it is (Q1 or Q3)
            bar_total_ref = q1_total_cost if rect.get_x() < bar_width/2 else q3_total_optimal_cost # Approximate check
            if height > 0.01 * bar_total_ref and height > 1e-6 : # Add small threshold for very small values
                y_pos = rect.get_y() + height / 2.
                ax1.annotate(f'{height/1e6:,.1f}M',
                             xy=(rect.get_x() + rect.get_width() / 2., y_pos),
                             xytext=(0,0), textcoords="offset points",
                             ha='center', va='center', fontsize=8, color='white', fontweight='bold')

    if pd.notna(q1_total_cost) and q1_total_cost > 0:
        ax1.annotate(f'Total:\n{q1_total_cost/1e6:,.2f}M', xy=(index[0], q1_total_cost), xytext=(0,5),
                     textcoords='offset points', ha='center', va='bottom', fontsize=8, fontweight='bold')
    if pd.notna(q3_total_optimal_cost) and q3_total_optimal_cost > 0 :
        ax1.annotate(f'Total:\n{q3_total_optimal_cost/1e6:,.2f}M', xy=(index[1], q3_total_optimal_cost), xytext=(0,5),
                     textcoords='offset points', ha='center', va='bottom', fontsize=8, fontweight='bold')

    fig1.subplots_adjust(right=LEGEND_ADJUST_RIGHT, top=0.88)
    fig1.savefig(os.path.join(PLOTS_DIR, "01_cost_structure_comparison_academic.png"), dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print("Chart 1 (Cost Structure) saved.")

    # --- Volume Plot ---
    if df_q3_summary.empty or not all(c in df_q3_summary.columns for c in ['TotalDirectVolume_Optimal', 'TotalCoastalVolume_Optimal']):
        print("Error (Chart 1 - Volume): Q3 summary data is missing volume columns. Skipping.")
        return

    # q1_total_volume is the sum of Q3 optimal volumes for baseline comparison, as per original plotting.py
    q1_total_volume = df_q3_summary['TotalDirectVolume_Optimal'].iloc[0] + df_q3_summary['TotalCoastalVolume_Optimal'].iloc[0]
    q3_direct_volume = df_q3_summary['TotalDirectVolume_Optimal'].iloc[0]
    q3_coastal_volume = df_q3_summary['TotalCoastalVolume_Optimal'].iloc[0]

    volume_labels = ['Q1 Baseline\n(Equivalent Vol.)', 'Q3 Optimized']
    direct_volumes_values = [q1_total_volume, q3_direct_volume] # Q1 baseline is all direct for this comparison
    coastal_volumes_q3_values = [0, q3_coastal_volume] # Q1 has no coastal volume in this model

    fig2, ax2 = plt.subplots(figsize=STD_FIG_SIZE)
    bar_v1_plot = ax2.bar(index, direct_volumes_values, bar_width, label='Direct Transport Volume', color='#6495ED')
    bar_v2_plot = ax2.bar(index, coastal_volumes_q3_values, bar_width, bottom=direct_volumes_values, label='Coastal Transport Volume', color='#FF7F50')

    ax2.set_ylabel('Total Annual Volume (Units x1000)', fontsize=STD_LABEL_FONTSIZE)
    ax2.set_title('Annual Transportation Volume Structure', fontsize=STD_TITLE_FONTSIZE, fontweight='bold', y=FIGURE_TITLE_Y_ADJUST)
    ax2.set_xticks(index)
    ax2.set_xticklabels(volume_labels, fontsize=STD_TICK_FONTSIZE)
    ax2.legend(loc=STD_LEGEND_LOC, bbox_to_anchor=STD_LEGEND_BBOX, fontsize=STD_LEGEND_FONTSIZE, borderaxespad=0.)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x_val, _: f'{x_val/1e3:,.0f}K'))
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    for rect_group_idx, rect_group in enumerate([bar_v1_plot, bar_v2_plot]):
        for rect_idx, rect in enumerate(rect_group):
            height = rect.get_height()
            # Determine bar_total_ref based on which bar it is (Q1 or Q3)
            current_bar_total_ref = q1_total_volume if rect.get_x() < bar_width/2 else (q3_direct_volume + q3_coastal_volume)
            if height > 0.01 * current_bar_total_ref and height > 1e-6: # Add small threshold
                y_pos_base = rect.get_y()
                y_pos = y_pos_base + height / 2.
                label_text = f'{height/1e3:,.0f}K'
                ax2.annotate(label_text,
                             xy=(rect.get_x() + rect.get_width() / 2., y_pos),
                             xytext=(0,0), textcoords="offset points",
                             ha='center', va='center', fontsize=8, color='black', fontweight='bold')

    if pd.notna(q1_total_volume) and q1_total_volume > 0:
        ax2.annotate(f'Total:\n{q1_total_volume/1e3:,.0f}K', xy=(index[0], q1_total_volume), xytext=(0,5),
                     textcoords='offset points', ha='center', va='bottom', fontsize=8, fontweight='bold')
    if pd.notna(q3_direct_volume + q3_coastal_volume) and (q3_direct_volume + q3_coastal_volume) > 0:
        ax2.annotate(f'Total:\n{(q3_direct_volume + q3_coastal_volume)/1e3:,.0f}K', xy=(index[1], q3_direct_volume + q3_coastal_volume),
                     xytext=(0,5), textcoords='offset points', ha='center', va='bottom', fontsize=8, fontweight='bold')

    fig2.subplots_adjust(right=LEGEND_ADJUST_RIGHT, top=0.88)
    fig2.savefig(os.path.join(PLOTS_DIR, "01_volume_structure_comparison_academic.png"), dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print("Chart 1 (Volume Structure) saved.")