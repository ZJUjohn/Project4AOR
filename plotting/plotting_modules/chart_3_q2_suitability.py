import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # Seaborn is used here
from .config import (
    PLOTS_DIR, STD_LABEL_FONTSIZE, STD_TITLE_FONTSIZE,
    STD_TICK_FONTSIZE, STD_LEGEND_FONTSIZE, STD_LEGEND_LOC, STD_LEGEND_BBOX,
    LEGEND_ADJUST_RIGHT, FIGURE_TITLE_Y_ADJUST
    # STD_FIG_SIZE is not explicitly used here, figsize is hardcoded (9.5, 6.5)
)

def plot_q2_suitability_bubble_academic(df_q2_summary, q1_data_for_demand):
    print("Generating Chart 3: Q2 Customer Suitability (Bubble Chart)...")

    if df_q2_summary is None or df_q2_summary.empty :
        print("Warning (Chart 3): Q2 summary data is missing or None. Skipping bubble chart.")
        return

    # Create a copy to avoid SettingWithCopyWarning
    df_q2_summary_copy = df_q2_summary.copy()

    if q1_data_for_demand is None or q1_data_for_demand.empty:
        print("Warning (Chart 3): Q1 data for demand is missing or None for bubble sizes. Using default size.")
        df_q2_summary_copy['AnnualDemand_Total'] = 1000 # Default demand
    else:
        q1_temp_demand_df = q1_data_for_demand.copy()
        id_col_q1 = 'Cust_ID' if 'Cust_ID' in q1_temp_demand_df.columns else 'Customer_Location'
        
        if id_col_q1 not in q1_temp_demand_df.columns:
            print(f"Warning (Chart 3): ID column '{id_col_q1}' not found in q1_data_for_demand. Using default demand.")
            df_q2_summary_copy['AnnualDemand_Total'] = 1000
        elif 'Customer_Location' not in df_q2_summary_copy.columns:
            print(f"Warning (Chart 3): 'Customer_Location' not found in df_q2_summary. Cannot map demand. Using default demand.")
            df_q2_summary_copy['AnnualDemand_Total'] = 1000
        else:
            demand_sum_cols = []
            for am_id_loop in ['AM1', 'AM2', 'AM3']:
                for m_col_suffix in ['_M1', '_M2', '_M3']: # Assuming these are month suffixes or similar
                    col_name = f'{am_id_loop}{m_col_suffix}'
                    if col_name not in q1_temp_demand_df.columns: q1_temp_demand_df[col_name] = 0
                    q1_temp_demand_df[col_name] = pd.to_numeric(q1_temp_demand_df[col_name], errors='coerce').fillna(0)
                    demand_sum_cols.append(col_name)
            
            if not demand_sum_cols: # If no demand columns were found
                 print(f"Warning (Chart 3): No demand columns (e.g., AM1_M1) found in q1_data_for_demand. Using default demand.")
                 df_q2_summary_copy['AnnualDemand_Total'] = 1000
            else:
                q1_temp_demand_df['Total_3Mo_Demand_All_AM'] = q1_temp_demand_df[demand_sum_cols].sum(axis=1)
                # Assuming demand is quarterly, so multiply by 4 for annual
                customer_demand_map = q1_temp_demand_df.groupby(id_col_q1)['Total_3Mo_Demand_All_AM'].sum() * 4
                df_q2_summary_copy['AnnualDemand_Total'] = df_q2_summary_copy['Customer_Location'].map(customer_demand_map).fillna(100) # Default if no map
                df_q2_summary_copy['AnnualDemand_Total'] = np.maximum(df_q2_summary_copy['AnnualDemand_Total'], 1) # Ensure positive demand

    am_focus = 'AM1' # As per original
    direct_cost_col = f'{am_focus}_Direct_Cost_PU'
    mm_cost_a_col = f'{am_focus}_Min_MM_Cost_PU_ShipA'
    mm_cost_b_col = f'{am_focus}_Min_MM_Cost_PU_ShipB'
    suitable_a_col = f'{am_focus}_Suitable_ShipA'
    suitable_b_col = f'{am_focus}_Suitable_ShipB'

    required_cols_q2 = [direct_cost_col, mm_cost_a_col, mm_cost_b_col, 
                        suitable_a_col, suitable_b_col, 
                        'Customer_Location', 'AnnualDemand_Total']
    
    # Check if all required columns are in df_q2_summary_copy
    if not all(col in df_q2_summary_copy.columns for col in required_cols_q2):
        missing = [col for col in required_cols_q2 if col not in df_q2_summary_copy.columns]
        print(f"Error (Chart 3): Necessary columns for {am_focus} or demand are missing from Q2 summary ({missing}). Skipping.")
        return

    df_plot_q2 = df_q2_summary_copy.copy() # Work with this copy

    for col in [direct_cost_col, mm_cost_a_col, mm_cost_b_col]:
        df_plot_q2[col] = pd.to_numeric(df_plot_q2[col], errors='coerce')
    df_plot_q2.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_plot_q2.dropna(subset=[direct_cost_col, 'AnnualDemand_Total'], inplace=True)

    def determine_suitability_category_eng(row):
        s_a_text = str(row.get(suitable_a_col, "N/A")).strip() # .get for safety
        s_b_text = str(row.get(suitable_b_col, "N/A")).strip()
        cost_direct = row.get(direct_cost_col, np.inf)
        cost_mm_a = row.get(mm_cost_a_col, np.inf)
        cost_mm_b = row.get(mm_cost_b_col, np.inf)

        mm_a_is_cheaper = pd.notna(cost_mm_a) and cost_mm_a < cost_direct
        mm_b_is_cheaper = pd.notna(cost_mm_b) and cost_mm_b < cost_direct

        if mm_a_is_cheaper and mm_b_is_cheaper: return 'Favorable (Ship A or B)'
        if mm_a_is_cheaper: return 'Favorable (Ship A)'
        if mm_b_is_cheaper: return 'Favorable (Ship B)'
        return 'Not Favorable for Coastal'

    df_plot_q2['Suitability_Category'] = df_plot_q2.apply(determine_suitability_category_eng, axis=1)
    df_plot_q2['Min_MM_Cost_PU'] = df_plot_q2[[mm_cost_a_col, mm_cost_b_col]].min(axis=1, skipna=True)
    df_plot_q2.dropna(subset=['Min_MM_Cost_PU'], inplace=True)

    if df_plot_q2.empty:
        print("Warning (Chart 3): No valid data points for bubble chart after all filtering. Skipping.")
        return

    fig, ax = plt.subplots(figsize=(9.5, 6.5)) # Original figsize
    palette = {
        'Favorable (Ship A or B)': '#2ca02c', 'Favorable (Ship A)': '#98df8a',
        'Favorable (Ship B)': '#aec7e8', 'Not Favorable for Coastal': '#ff7f0e'
    }

    min_demand_log = np.log1p(df_plot_q2['AnnualDemand_Total'].min())
    max_demand_log = np.log1p(df_plot_q2['AnnualDemand_Total'].max())

    if max_demand_log == min_demand_log or pd.isna(min_demand_log) or pd.isna(max_demand_log) :
        df_plot_q2['BubbleSizeScaled'] = 40.0 # Default size
    else:
        df_plot_q2['BubbleSizeScaled'] = 40 + 760 * (np.log1p(df_plot_q2['AnnualDemand_Total']) - min_demand_log) / (max_demand_log - min_demand_log)
    df_plot_q2['BubbleSizeScaled'] = df_plot_q2['BubbleSizeScaled'].fillna(40.0) # Fill NA sizes

    sns.scatterplot(
        data=df_plot_q2, x=direct_cost_col, y='Min_MM_Cost_PU',
        size='BubbleSizeScaled', hue='Suitability_Category', palette=palette,
        alpha=0.65, ax=ax, edgecolor='k', linewidth=0.3, legend='auto' # legend='auto' is fine
    )

    x_data_min_val = df_plot_q2[direct_cost_col].min(skipna=True)
    x_data_max_val = df_plot_q2[direct_cost_col].max(skipna=True)
    y_data_min_val = df_plot_q2['Min_MM_Cost_PU'].min(skipna=True)
    y_data_max_val = df_plot_q2['Min_MM_Cost_PU'].max(skipna=True)

    plot_min_val, plot_max_val = 0, 100 # Default axis limits
    if not (pd.isna(x_data_min_val) or pd.isna(y_data_min_val) or pd.isna(x_data_max_val) or pd.isna(y_data_max_val)):
        plot_min_val = max(0, min(x_data_min_val, y_data_min_val) * 0.95) # Ensure non-negative
        plot_max_val = max(x_data_max_val, y_data_max_val) * 1.05
        if plot_max_val <= plot_min_val: # Handle edge case where max <= min after calculation
            plot_max_val = plot_min_val + 10 # Add a small range

    ax.plot([plot_min_val, plot_max_val], [plot_min_val, plot_max_val], 'k--', alpha=0.6, zorder=0, linewidth=1, label='Cost Equivalence Line')
    ax.set_xlim(left=plot_min_val)
    ax.set_ylim(bottom=plot_min_val)
    if plot_max_val > plot_min_val: # Only set if valid range
        ax.set_xlim(right=plot_max_val)
        ax.set_ylim(top=plot_max_val)

    ax.set_xlabel(f'{am_focus} Direct Cost per Unit (USD)', fontsize=STD_LABEL_FONTSIZE)
    ax.set_ylabel(f'{am_focus} Min. Multimodal Cost per Unit (USD)', fontsize=STD_LABEL_FONTSIZE)
    ax.set_title(f'Q2: Customer Suitability for Coastal Shipping ({am_focus})', fontsize=STD_TITLE_FONTSIZE, fontweight='bold', y=FIGURE_TITLE_Y_ADJUST)

    handles, labels = ax.get_legend_handles_labels()
    # Reorder legend to put "Cost Equivalence Line" first
    try:
        eq_line_idx = labels.index('Cost Equivalence Line')
    except ValueError: # If label is slightly different (e.g. with details)
        try:
            eq_line_idx = labels.index('Cost Equivalence Line (Direct = Multimodal)')
        except ValueError:
            eq_line_idx = -1 # Not found

    if eq_line_idx != -1:
        handles = [handles[eq_line_idx]] + [h for i, h in enumerate(handles) if i != eq_line_idx]
        labels = [labels[eq_line_idx]] + [l for i, l in enumerate(labels) if i != eq_line_idx]

    ax.legend(handles, labels, title='Suitability & Demand Volume', 
              fontsize=STD_LEGEND_FONTSIZE-1, # Slightly smaller as per original
              loc=STD_LEGEND_LOC, bbox_to_anchor=STD_LEGEND_BBOX, borderaxespad=0.)

    ax.annotate('Bubble size ~ Customer Annual Demand',
                xy=(0.98, 0.01), xycoords='axes fraction',
                ha='right', va='bottom', fontsize=7, style='italic', # Original fontsize 7
                bbox=dict(boxstyle='round,pad=0.3', fc='lightgray', alpha=0.5))

    ax.grid(True, linestyle=':', alpha=0.5) # Original style
    ax.tick_params(axis='both', labelsize=STD_TICK_FONTSIZE)

    fig.subplots_adjust(right=LEGEND_ADJUST_RIGHT - 0.1, top=0.88) # Original adjustment
    fig.savefig(os.path.join(PLOTS_DIR, "03_q2_suitability_bubble_academic.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Chart 3 saved.")