import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .config import (
    PLOTS_DIR, STD_TITLE_FONTSIZE, STD_TICK_FONTSIZE
    # STD_FIG_SIZE is not used, figsize is hardcoded (7, 4.5)
)

def plot_kpi_summary_academic(df_q3_summary_data):
    print("Generating Chart 5: KPI Summary...")

    if df_q3_summary_data is None or df_q3_summary_data.empty or 'Status' not in df_q3_summary_data.columns:
        print("Error (Chart 5): Q3 summary data is missing, None, or malformed. Skipping KPI plot.")
        # Create a dummy plot indicating data is missing
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.axis('off')
        fig.suptitle("Optimization Results: Key Performance Indicators", fontsize=STD_TITLE_FONTSIZE, fontweight='bold', y=0.97)
        ax.text(0.5, 0.5, "Q3 summary data unavailable for KPI plot.", 
                ha='center', va='center', fontsize=STD_TICK_FONTSIZE, color='red')
        fig.savefig(os.path.join(PLOTS_DIR, "05_kpi_summary_academic_error.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)
        return

    # Ensure all required columns exist, providing defaults if not
    required_cols = {
        'Status': 'Data Unavailable',
        'TotalOptimalCost_Annual': 0.0,
        'Annualized_DirectCost_Q1_Reference': 0.0,
        'TotalDirectVolume_Optimal': 0,
        'TotalCoastalVolume_Optimal': 0,
        'ShipsUsed_Optimal': 'N/A'
    }
    summary_row = df_q3_summary_data.iloc[0].copy() # Work with a copy of the first row
    for col, default_val in required_cols.items():
        if col not in summary_row or pd.isna(summary_row[col]):
            summary_row[col] = default_val
            print(f"Info (Chart 5): Using default value for missing/NaN KPI '{col}': {default_val}")


    status = summary_row['Status']
    q3_cost = summary_row['TotalOptimalCost_Annual']
    q1_ref_cost = summary_row['Annualized_DirectCost_Q1_Reference']

    savings_amount = 0
    savings_percent = 0
    if pd.notna(q3_cost) and pd.notna(q1_ref_cost) and q1_ref_cost > 1e-6: # Avoid division by zero or near-zero
        savings_amount = q1_ref_cost - q3_cost
        savings_percent = (savings_amount / q1_ref_cost) * 100
    elif pd.notna(q3_cost) and pd.notna(q1_ref_cost): # If q1_ref_cost is zero
        savings_amount = -q3_cost # Savings is negative of q3_cost
        savings_percent = -np.inf if q3_cost > 0 else (0 if q3_cost == 0 else np.inf)


    total_direct_vol_q3 = summary_row['TotalDirectVolume_Optimal']
    total_coastal_vol_q3 = summary_row['TotalCoastalVolume_Optimal']
    ships_used_str = summary_row['ShipsUsed_Optimal']
    if pd.isna(ships_used_str) or not ships_used_str: ships_used_str = "None"


    fig, ax = plt.subplots(figsize=(7, 4.5)) # Original figsize
    ax.axis('off')

    fig.suptitle("Optimization Results: Key Performance Indicators", fontsize=STD_TITLE_FONTSIZE, fontweight='bold', y=0.97)

    kpi_details = [
        ("Solver Status:", str(status)),
        ("Q1 Annualized Direct Cost (Baseline):", f"${q1_ref_cost:,.2f}"),
        ("Q3 Optimized Total Annual Cost:", f"${q3_cost:,.2f}"),
        ("Annual Cost Savings:", f"${savings_amount:,.2f} ({savings_percent:.2f}%)" if pd.notna(savings_percent) and savings_percent != -np.inf and savings_percent != np.inf else f"${savings_amount:,.2f} (N/A %)"),
        ("Q3 Total Direct Volume (Annual):", f"{total_direct_vol_q3:,.0f} units"),
        ("Q3 Total Coastal Volume (Annual):", f"{total_coastal_vol_q3:,.0f} units"),
        ("Q3 Ships Utilized:", str(ships_used_str))
    ]

    y_pos_start = 0.85
    x_label_coord = 0.05
    x_value_coord = 0.55 # Start of value text
    line_spacing = 0.11

    for i, (label_text, value_text) in enumerate(kpi_details):
        ax.text(x_label_coord, y_pos_start - i * line_spacing, label_text, 
                fontsize=STD_TICK_FONTSIZE, fontweight='normal', verticalalignment='top', wrap=True)
        
        color_text = 'black'
        current_fontweight_text = 'normal'
        if "Savings" in label_text:
            if savings_amount < -1e-6: color_text = '#D62728' # Red for negative savings (loss)
            elif savings_amount > 1e-6: color_text = '#2CA02C' # Green for positive savings
        elif "Optimized Total Annual Cost" in label_text and pd.notna(q3_cost) and pd.notna(q1_ref_cost) and q3_cost < q1_ref_cost:
            color_text = '#2CA02C' # Green if Q3 cost is less than Q1
        
        if "Cost" in label_text or "Savings" in label_text or "Status" in label_text:
             current_fontweight_text ='bold'

        ax.text(x_value_coord, y_pos_start - i * line_spacing, value_text, 
                fontsize=STD_TICK_FONTSIZE, color=color_text, fontweight=current_fontweight_text, 
                verticalalignment='top', wrap=True)

    plt.tight_layout(rect=[0, 0, 1, 0.90]) # Adjust rect to ensure suptitle is visible
    fig.savefig(os.path.join(PLOTS_DIR, "05_kpi_summary_academic.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Chart 5 saved.")