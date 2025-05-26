# Standard libraries
import os
import pandas as pd
import numpy as np

# Visualization libraries - Matplotlib & Seaborn
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# Plotly libraries for interactive Sankey diagrams
import plotly.graph_objects as go

# --- Font Configuration ---
PREFERRED_FONT_LIST = [
    'Microsoft YaHei', 'SimHei', 'Songti SC', 'STKaiti', # Chinese fonts
    'Latin Modern Roman',
    'CMU Serif',
    'Times New Roman',
    'DejaVu Serif',
    'Georgia',
    'DejaVu Sans',
    'Calibri',
    'Arial',
    'Helvetica'
]
print(f"Attempting to use font list, prioritizing: {PREFERRED_FONT_LIST[0]}")

FONT_SETTINGS_SUCCESS = False
PLOTLY_FONT_FAMILY = "Songti SC"
try:
    first_font_is_chinese = PREFERRED_FONT_LIST[0] in ['Microsoft YaHei', 'SimHei', 'Songti SC', 'STKaiti']

    if first_font_is_chinese:
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = PREFERRED_FONT_LIST
        print(f"Configured for Chinese font priority. Sans-serif list: {plt.rcParams['font.sans-serif']}")
    else:
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = PREFERRED_FONT_LIST
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Calibri', 'Arial', 'Helvetica'] # Ensure sans-serif is also available
        print(f"Configured for English serif font priority. Serif list: {plt.rcParams['font.serif']}")

    plt.rcParams['axes.unicode_minus'] = False

    sns_rc_params = {
        "font.family": plt.rcParams['font.family'],
        "axes.unicode_minus": False,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.titlesize": 16,
        "axes.titlesize": 14
    }
    if plt.rcParams['font.family'] == 'serif':
        sns_rc_params["font.serif"] = plt.rcParams['font.serif']
    if plt.rcParams['font.family'] == 'sans-serif':
        sns_rc_params["font.sans-serif"] = plt.rcParams['font.sans-serif']
    # Ensure both are passed if defined for robustness with seaborn themes
    if 'font.serif' in plt.rcParams:
         sns_rc_params["font.serif"] = plt.rcParams['font.serif']
    if 'font.sans-serif' in plt.rcParams:
         sns_rc_params["font.sans-serif"] = plt.rcParams['font.sans-serif']


    sns.set_theme(style="whitegrid", rc=sns_rc_params)
    FONT_SETTINGS_SUCCESS = True
except Exception as e:
    print(f"Error setting font: {e}. Graphs may not use the preferred font.")
    print(f"Please ensure '{PREFERRED_FONT_LIST[0]}' or an alternative is installed and Matplotlib's cache is updated if necessary.")

# Define colors for consistency
AM_COLORS = {'AM1': 'rgba(31, 119, 180, 0.9)', 'AM2': 'rgba(255, 127, 14, 0.9)', 'AM3': 'rgba(44, 160, 44, 0.9)'}
SHIP_COLORS = {'ShipA': 'rgba(214, 39, 40, 0.8)', 'ShipB': 'rgba(148, 103, 189, 0.8)'}
PORT_NODE_COLOR = 'rgba(127, 127, 127, 0.6)'
DIRECT_TRUCK_COLOR = 'rgba(255, 187, 120, 0.7)'  # Light Orange for direct trucking links
COASTAL_TRUCK_COLOR = 'rgba(188, 189, 34, 0.7)' # Olive for trucking to/from port
SEA_LINK_COLOR = 'rgba(23, 190, 207, 0.7)'     # Cyan for sea links
MODE_SPLIT_NODE_COLOR = 'rgba(150, 150, 150, 0.6)'

# --- Global Settings ---
RESULTS_DIR = "results"
PLOTS_DIR = "plots"

if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)
    print(f"Created directory for plots: {PLOTS_DIR}")

STD_FIG_SIZE = (10, 7) # Made default a bit larger for Sankey potential
STD_TITLE_FONTSIZE = 16
STD_LABEL_FONTSIZE = 13
STD_TICK_FONTSIZE = 11
STD_LEGEND_FONTSIZE = 11
STD_LEGEND_LOC = 'upper left'
STD_LEGEND_BBOX = (1.02, 1)
LEGEND_ADJUST_RIGHT = 0.75
FIGURE_TITLE_Y_ADJUST = 1.03

# --- Chart 1: Cost and Volume Structure Comparison ---
def plot_cost_volume_structure(df_q1_annualized_cost, df_q3_summary, df_q3_direct_vol_data, q1_direct_unit_costs_df_data):
    print("Generating Chart 1: Cost and Volume Structure Comparison...")

    if df_q3_summary.empty or 'TotalOptimalCost_Annual' not in df_q3_summary.columns:
        print("Error (Chart 1): Q3 summary data is missing 'TotalOptimalCost_Annual'. Skipping cost plot.")
        return

    q1_total_cost = df_q1_annualized_cost
    q3_total_optimal_cost = df_q3_summary['TotalOptimalCost_Annual'].iloc[0]

    q3_direct_cost_estimated = 0
    if df_q3_direct_vol_data is not None and not df_q3_direct_vol_data.empty and \
       q1_direct_unit_costs_df_data is not None and not q1_direct_unit_costs_df_data.empty:
        direct_costs_map = {}
        id_col_q1 = 'Cust_ID' if 'Cust_ID' in q1_direct_unit_costs_df_data.columns else 'Customer_Location'
        for _, row in q1_direct_unit_costs_df_data.iterrows():
            cust_id_val = row[id_col_q1]
            for am in ['AM1', 'AM2', 'AM3']:
                cost_col = f'{am}_Direct_Cost_Per_Unit'
                if cost_col in row and pd.notna(row[cost_col]) and row[cost_col] != np.inf:
                    direct_costs_map[(am, cust_id_val)] = row[cost_col]
        for _, row in df_q3_direct_vol_data.iterrows():
            unit_cost = direct_costs_map.get((row['AM'], row['Customer']), 0)
            q3_direct_cost_estimated += row['AnnualVolume'] * unit_cost

    q3_coastal_total_cost = q3_total_optimal_cost - q3_direct_cost_estimated
    if q3_coastal_total_cost < 0: q3_coastal_total_cost = 0
    q3_direct_cost_estimated = q3_total_optimal_cost - q3_coastal_total_cost

    cost_labels = ['Q1 Baseline\n(Annualized)', 'Q3 Optimized']
    direct_costs_values = [q1_total_cost, q3_direct_cost_estimated]
    coastal_costs_q3_values = [0, q3_coastal_total_cost]

    fig1, ax1 = plt.subplots(figsize=STD_FIG_SIZE) # Use standard size
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
            bar_total_ref = q1_total_cost if rect.get_x() < bar_width else q3_total_optimal_cost
            if height > 0.01 * bar_total_ref :
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
    fig1.savefig(os.path.join(PLOTS_DIR, "01_cost_structure_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print("Chart 1 (Cost Structure) saved.")

    if df_q3_summary.empty or not all(c in df_q3_summary.columns for c in ['TotalDirectVolume_Optimal', 'TotalCoastalVolume_Optimal']):
        print("Error (Chart 1 - Volume): Q3 summary data is missing volume columns. Skipping.")
        return

    q1_total_volume = df_q3_summary['TotalDirectVolume_Optimal'].iloc[0] + df_q3_summary['TotalCoastalVolume_Optimal'].iloc[0]
    q3_direct_volume = df_q3_summary['TotalDirectVolume_Optimal'].iloc[0]
    q3_coastal_volume = df_q3_summary['TotalCoastalVolume_Optimal'].iloc[0]

    volume_labels = ['Q1 Baseline\n(Equivalent Vol.)', 'Q3 Optimized']
    direct_volumes_values = [q1_total_volume, q3_direct_volume]
    coastal_volumes_q3_values = [0, q3_coastal_volume]

    fig2, ax2 = plt.subplots(figsize=STD_FIG_SIZE) # Use standard size
    bar_v1_plot = ax2.bar(index, direct_volumes_values, bar_width, label='Direct Transport Volume', color='#6495ED')
    bar_v2_plot = ax2.bar(index, coastal_volumes_q3_values, bar_width, bottom=direct_volumes_values, label='Coastal Transport Volume', color='#FF7F50')

    ax2.set_ylabel('Total Annual Volume (Units x1000)', fontsize=STD_LABEL_FONTSIZE)
    ax2.set_title('Annual Transportation Volume Structure', fontsize=STD_TITLE_FONTSIZE, fontweight='bold', y=FIGURE_TITLE_Y_ADJUST)
    ax2.set_xticks(index)
    ax2.set_xticklabels(volume_labels, fontsize=STD_TICK_FONTSIZE)
    ax2.legend(loc=STD_LEGEND_LOC, bbox_to_anchor=STD_LEGEND_BBOX, fontsize=STD_LEGEND_FONTSIZE, borderaxespad=0.)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x_val, _: f'{x_val/1e3:,.0f}K'))
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    for rect_group in [bar_v1_plot, bar_v2_plot]:
        for rect in rect_group:
            height = rect.get_height()
            bar_total_ref = q1_total_volume if rect.get_x() < bar_width else (q3_direct_volume + q3_coastal_volume)
            if height > 0.01 * bar_total_ref :
                y_pos = rect.get_y() + height / 2.
                ax2.annotate(f'{height/1e3:,.0f}K',
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
    fig2.savefig(os.path.join(PLOTS_DIR, "01_volume_structure_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print("Chart 1 (Volume Structure) saved.")

# --- Chart 2: AM Perspective - Mode Split (Q3) ---
def plot_am_mode_split_q3(df_q3_direct, df_q3_coastal):
    print("Generating Chart 2: AM Mode Split (Q3)...")
    if df_q3_direct.empty and df_q3_coastal.empty:
        print("Warning (Chart 2): Both direct and coastal Q3 data are empty. Skipping.")
        return

    am_direct_vol = df_q3_direct.groupby('AM')['AnnualVolume'].sum().rename('Direct Volume') if not df_q3_direct.empty else pd.Series(name='Direct Volume', dtype='float64')
    am_coastal_vol = df_q3_coastal.groupby('AM')['AnnualVolume'].sum().rename('Coastal Volume') if not df_q3_coastal.empty else pd.Series(name='Coastal Volume', dtype='float64')

    df_am_plot = pd.concat([am_direct_vol, am_coastal_vol], axis=1).fillna(0)
    all_ams = ['AM1', 'AM2', 'AM3']
    df_am_plot = df_am_plot.reindex(index=all_ams, fill_value=0).sort_index()

    fig, ax = plt.subplots(figsize=STD_FIG_SIZE) # Use standard size
    colors = ['#4C72B0', '#55A868']
    df_am_plot.plot(kind='bar', stacked=True, ax=ax, color=colors, width=0.7)

    ax.set_xlabel('Manufacturing Cluster (AM)', fontsize=STD_LABEL_FONTSIZE)
    ax.set_ylabel('Total Annual Volume (Units x1000)', fontsize=STD_LABEL_FONTSIZE)
    ax.set_title('Q3 Volume by AM and Transport Mode', fontsize=STD_TITLE_FONTSIZE, fontweight='bold', y=FIGURE_TITLE_Y_ADJUST)
    ax.tick_params(axis='x', rotation=0, labelsize=STD_TICK_FONTSIZE)
    ax.tick_params(axis='y', labelsize=STD_TICK_FONTSIZE)
    ax.legend(title='Transport Mode', fontsize=STD_LEGEND_FONTSIZE, loc=STD_LEGEND_LOC, bbox_to_anchor=STD_LEGEND_BBOX, borderaxespad=0.)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x_val, _: f'{x_val/1e3:,.0f}'))
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    for c_container in ax.containers:
        labels = [f'{v.get_height()/1e3:,.0f}K' if v.get_height() > 0.05 * df_am_plot.sum(axis=1).max() else '' for v in c_container]
        ax.bar_label(c_container, labels=labels, label_type='center', fontsize=8, color='white', fontweight='bold')

    fig.subplots_adjust(right=LEGEND_ADJUST_RIGHT - 0.05, top=0.88)
    fig.savefig(os.path.join(PLOTS_DIR, "02_am_volume_split_q3.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Chart 2 saved.")


# --- Chart 3: Q2 Results - Customer Suitability (Bubble Chart) ---
def plot_suitability_bubble(df_q2_summary, q1_data_for_demand, df_q3_coastal_vol):
    print("Generating Chart 3: Q2 Customer Suitability (Bubble Chart)...")

    if df_q2_summary.empty:
        print("Warning (Chart 3): Q2 summary data is missing. Skipping bubble chart.")
        return

    if q1_data_for_demand.empty:
        print("Warning (Chart 3): Q1 data for demand is missing for bubble sizes. Using default placeholder demand.")
        df_q2_summary['AnnualDemand_Total'] = 100  # Placeholder demand
    else:
        id_col_q1 = 'Cust_ID' if 'Cust_ID' in q1_data_for_demand.columns else 'Customer_Location'
        q1_temp_demand_df = q1_data_for_demand.copy()
        demand_sum_cols = []
        for am_id_loop in ['AM1', 'AM2', 'AM3']:
            for m_col_suffix in ['_M1', '_M2', '_M3']:
                col_name = f'{am_id_loop}{m_col_suffix}'
                if col_name not in q1_temp_demand_df.columns: q1_temp_demand_df[col_name] = 0
                q1_temp_demand_df[col_name] = pd.to_numeric(q1_temp_demand_df[col_name], errors='coerce').fillna(0)
                demand_sum_cols.append(col_name)
        q1_temp_demand_df['Total_3Mo_Demand_All_AM'] = q1_temp_demand_df[demand_sum_cols].sum(axis=1)
        customer_demand_map = q1_temp_demand_df.groupby(id_col_q1)['Total_3Mo_Demand_All_AM'].sum() * 4
        df_q2_summary['AnnualDemand_Total'] = df_q2_summary['Customer_Location'].map(customer_demand_map).fillna(100)
        df_q2_summary['AnnualDemand_Total'] = np.maximum(df_q2_summary['AnnualDemand_Total'], 1) # Ensure min demand is 1

    # Determine am_focus (analysis machine focus)
    for am_focus in ['AM1','AM2', 'AM3']: # Defaulting to AM1 as per user image example for now

        print(f"Info (Chart 3): Using am_focus = '{am_focus}' for plotting.")

        direct_cost_col = f'{am_focus}_Direct_Cost_PU'
        mm_cost_a_col = f'{am_focus}_Min_MM_Cost_PU_ShipA'
        mm_cost_b_col = f'{am_focus}_Min_MM_Cost_PU_ShipB'
        
        required_cols_q2 = [direct_cost_col, mm_cost_a_col, mm_cost_b_col, 'Customer_Location', 'AnnualDemand_Total']
        if not all(col in df_q2_summary.columns for col in required_cols_q2):
            missing = [col for col in required_cols_q2 if col not in df_q2_summary.columns]
            print(f"Error (Chart 3): Necessary columns for {am_focus} or demand are missing ({missing}). Skipping.")
            if not os.path.exists(PLOTS_DIR): os.makedirs(PLOTS_DIR, exist_ok=True)
            return

        df_plot_q2 = df_q2_summary.copy()
        if 'Customer_Location' not in df_plot_q2.columns and 'Cust_ID' in df_plot_q2.columns:
            df_plot_q2['Customer_Location'] = df_plot_q2['Cust_ID']

        for col in [direct_cost_col, mm_cost_a_col, mm_cost_b_col, 'AnnualDemand_Total']:
            df_plot_q2[col] = pd.to_numeric(df_plot_q2[col], errors='coerce')
        df_plot_q2.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Ensure AnnualDemand_Total is present before this dropna for it
        df_plot_q2.dropna(subset=[direct_cost_col, 'AnnualDemand_Total'], inplace=True)


        def determine_suitability_category(row):
            cost_direct = row.get(direct_cost_col, np.inf)
            cost_mm_a = row.get(mm_cost_a_col, np.inf)
            cost_mm_b = row.get(mm_cost_b_col, np.inf)
            
            # Check if multimodal costs are actual numbers (not NaN or Inf)
            valid_mm_a = pd.notna(cost_mm_a) and np.isfinite(cost_mm_a)
            valid_mm_b = pd.notna(cost_mm_b) and np.isfinite(cost_mm_b)

            mm_a_is_cheaper = valid_mm_a and cost_mm_a < cost_direct
            mm_b_is_cheaper = valid_mm_b and cost_mm_b < cost_direct
            
            if mm_a_is_cheaper and mm_b_is_cheaper: return 'Favorable (Ship A or B)'
            if mm_a_is_cheaper: return 'Favorable (Ship A)'
            if mm_b_is_cheaper: return 'Favorable (Ship B)'
            return 'Not Favorable for Coastal'

        df_plot_q2['Suitability_Category'] = df_plot_q2.apply(determine_suitability_category, axis=1)
        
        if df_q3_coastal_vol is not None and not df_q3_coastal_vol.empty and \
        'AM' in df_q3_coastal_vol.columns and 'Customer' in df_q3_coastal_vol.columns:
            df_filtered_coastal_am = df_q3_coastal_vol[df_q3_coastal_vol['AM'] == am_focus]
            if not df_filtered_coastal_am.empty:
                coastal_customers_list_for_am_focus = df_filtered_coastal_am['Customer'].unique()
                if 'Customer_Location' in df_plot_q2.columns:
                    condition = df_plot_q2['Customer_Location'].isin(coastal_customers_list_for_am_focus)
                    df_plot_q2.loc[condition, 'Suitability_Category'] = "Coastal"
                else:
                    print("Warning (Chart 3): 'Customer_Location' not in df_plot_q2 for 'Coastal' category update.")
        else:
            print("Warning (Chart 3): df_q3_coastal_vol is missing, empty, or does not contain 'AM'/'Customer' columns. 'Coastal' category from Q3 results may not be applied.")

        df_plot_q2['Min_MM_Cost_PU'] = df_plot_q2[[mm_cost_a_col, mm_cost_b_col]].min(axis=1, skipna=True)
        df_plot_q2.dropna(subset=['Min_MM_Cost_PU'], inplace=True) # Also drop if Min_MM_Cost_PU could not be determined

        if df_plot_q2.empty or df_plot_q2['AnnualDemand_Total'].isnull().all(): # check AnnualDemand as well
            print("Warning (Chart 3): No valid data points for bubble chart after all filtering. Skipping.")
            return

        fig, ax = plt.subplots(figsize=(9.5, 6.5))
        palette = {
            'Favorable (Ship A or B)': '#2ca02c', # Dark green
            'Favorable (Ship A)': '#98df8a',    # Light green (placeholder, adjust if needed)
            'Favorable (Ship B)': '#aec7e8',    # Light blue (placeholder, adjust if needed)
            'Coastal': "#1f53b4",               # Darker blue
            'Not Favorable for Coastal': '#ff7f0e' # Orange
        }

        # --- Bubble Size Parameters (Continuous Scaling) ---
        # Define the range of bubble areas for the plot (min_area, max_area)
        # These values are in points^2. Adjust for desired visual effect.
        plot_bubble_area_range = (10, 1000) 
        
        # Representative demand values for the legend and their labels
        legend_size_demand_values = [1, 10, 100, 1000, 10000]
        legend_size_labels = ['1', '10', '100', '1,000', '10,000']

        # --- Main Scatter Plot ---
        # Ensure 'AnnualDemand_Total' is positive for sensible sizes
        df_plot_q2['AnnualDemand_Total_Plot'] = np.maximum(df_plot_q2['AnnualDemand_Total'], 1)

        scatter_plot = sns.scatterplot(
            data=df_plot_q2, 
            x=direct_cost_col, 
            y='Min_MM_Cost_PU',
            size='AnnualDemand_Total_Plot',  # Use actual demand for continuous sizing
            sizes=plot_bubble_area_range, # Control min/max bubble area
            hue='Suitability_Category', 
            palette=palette,
            alpha=0.7, 
            ax=ax, 
            edgecolor='k', 
            linewidth=0.3,
            legend=False # Manual legend creation will follow
        )

        # --- Cost Equivalence Line ---
        all_costs = pd.concat([
            df_plot_q2[direct_cost_col], 
            df_plot_q2['Min_MM_Cost_PU']
        ]).dropna()

        if not all_costs.empty:
            plot_min_val = max(0, all_costs.min() * 0.95)
            plot_max_val = all_costs.max() * 1.05
            if plot_max_val <= plot_min_val : plot_max_val = plot_min_val + 100 # Ensure max > min
        else: # Fallback if all cost data is missing after filtering
            plot_min_val = 0
            plot_max_val = 1200 
                
        line_eq, = ax.plot([plot_min_val, plot_max_val], [plot_min_val, plot_max_val], 
                        'k--', alpha=0.6, zorder=0, linewidth=1, label='Cost Equivalence Line')

        ax.set_xlim(left=plot_min_val, right=plot_max_val)
        ax.set_ylim(bottom=plot_min_val, top=plot_max_val)
        
        ax.set_xlabel(f'{am_focus} Direct Cost per Unit (USD)', fontsize=STD_LABEL_FONTSIZE)
        ax.set_ylabel(f'{am_focus} Min. Multimodal Cost per Unit (USD)', fontsize=STD_LABEL_FONTSIZE)
        ax.set_title(f'Q2: Customer Suitability for Coastal Shipping ({am_focus})', 
                    fontsize=STD_TITLE_FONTSIZE, fontweight='bold', y=FIGURE_TITLE_Y_ADJUST)

        # --- Manual Legend Creation ---
        final_legend_handles = []
        final_legend_labels = []

        # a. Cost Equivalence Line
        final_legend_handles.append(line_eq)
        final_legend_labels.append(line_eq.get_label())

        # b. Hue (Suitability_Category)
        plotted_suitability_categories = df_plot_q2['Suitability_Category'].unique()
        # Sort to have a consistent order, e.g., Favorable ones first
        sorted_palette_keys = sorted(palette.keys(), key=lambda k: (not k.startswith("Favorable"), k)) 

        for category_label in sorted_palette_keys: 
            if category_label in plotted_suitability_categories:
                final_legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', # 'w' for no visible center dot
                                                    markerfacecolor=palette[category_label], 
                                                    linestyle='None', markersize=8, label=category_label)) # Adjust markersize for legend
                final_legend_labels.append(category_label)
        
        # c. Bubble Size Legend
        # Get min/max of actual demand values used for sizing in the plot for normalization
        # Use 'AnnualDemand_Total_Plot' as this is what was passed to `size`
        min_data_demand = df_plot_q2['AnnualDemand_Total_Plot'].min()
        max_data_demand = df_plot_q2['AnnualDemand_Total_Plot'].max()

        s_min_plot, s_max_plot = plot_bubble_area_range

        for i in range(len(legend_size_demand_values)):
            demand_val = legend_size_demand_values[i]
            label = legend_size_labels[i]
            
            scaled_area = s_min_plot # Default to min size
            if max_data_demand > min_data_demand:
                # Normalize the legend demand value based on the actual data range
                norm_demand = (demand_val - min_data_demand) / (max_data_demand - min_data_demand)
                # Clamp normalized value to [0, 1] in case legend values are outside actual data range
                norm_demand_clamped = np.clip(norm_demand, 0, 1)
                scaled_area = s_min_plot + norm_demand_clamped * (s_max_plot - s_min_plot)
            elif demand_val <= min_data_demand : # If all data demands are same, or legend demand is less/equal
                scaled_area = s_min_plot
            elif demand_val >= max_data_demand: # if legend demand is greater or equal
                scaled_area = s_max_plot


            final_legend_handles.append(
                ax.scatter([], [], s=scaled_area, color='grey', 
                        alpha=0.7, edgecolor='k', linewidth=0.3) 
            )
            final_legend_labels.append(label)

        ax.legend(final_legend_handles, final_legend_labels, 
                title='Suitability & Demand Volume', 
                fontsize=STD_LEGEND_FONTSIZE -1 , 
                loc=STD_LEGEND_LOC, 
                bbox_to_anchor=STD_LEGEND_BBOX,
                borderaxespad=0.,
                labelspacing=0.8, # Adjusted spacing
                title_fontsize=STD_LEGEND_FONTSIZE) 
        
        ax.annotate('Bubble size ~ Customer Annual Demand', # Updated annotation
                    xy=(0.98, 0.01), xycoords='axes fraction',
                    ha='right', va='bottom', fontsize=7, style='italic',
                    bbox=dict(boxstyle='round,pad=0.3', fc='lightgray', alpha=0.5))

        ax.grid(True, linestyle=':', alpha=0.5)
        ax.tick_params(axis='both', labelsize=STD_TICK_FONTSIZE)
        
        try: 
            fig.subplots_adjust(right=LEGEND_ADJUST_RIGHT, top=0.90, bottom=0.1) 
        except Exception as e:
            print(f"Note (Chart 3): Using default tight_layout due to error in subplots_adjust or missing constants: {e}")
            fig.tight_layout(rect=[0, 0, LEGEND_ADJUST_RIGHT if LEGEND_ADJUST_RIGHT else 0.85, 0.95])


        try:
            if not os.path.exists(PLOTS_DIR):
                os.makedirs(PLOTS_DIR, exist_ok=True)
            save_path = os.path.join(PLOTS_DIR, f"03_suitability_bubble_{am_focus}.png")
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Chart 3 with continuous sizes saved to {save_path}")
        except Exception as e: 
            save_path_fallback = f"03_suitability_bubble_{am_focus}.png"
            print(f"Error saving Chart 3 to PLOTS_DIR. Attempting to save to current directory: {save_path_fallback}. Error: {e}")
            try:
                fig.savefig(save_path_fallback, dpi=300, bbox_inches='tight')
            except Exception as e_fallback:
                print(f"Failed to save chart to current directory as well. Error: {e_fallback}")
            
        plt.close(fig)

# --- Chart 4: Q3 Optimized Coastal Shipment Flows (Plotly Sankey) ---
def plot_q3_sankey_flows_plotly(df_q3_direct_vol, df_q3_coastal_vol, output_dir=PLOTS_DIR):
    """
    Generates two Sankey diagrams using Plotly for Q3 shipment flows.
    Chart 4a: AM -> Mode Split (Direct/Coastal) -> Aggregated Destinations.
    Chart 4b: AM -> Mode Split -> Detailed Path (Coastal: AM->Origin->Ship->Dest) -> Individual Customers.
    Saves both PNG and HTML versions.
    """
    print("Generating Chart 4 (Plotly Sankey): Q3 Optimized Shipment Flows...")

    if (df_q3_direct_vol is None or df_q3_direct_vol.empty) and \
       (df_q3_coastal_vol is None or df_q3_coastal_vol.empty):
        print("Warning (Chart 4 - Plotly): Both direct and coastal Q3 shipment data are empty. Skipping Sankey diagrams.")
        return

    # Ensure 'ShipType' column exists in coastal data
    if df_q3_coastal_vol is not None and not df_q3_coastal_vol.empty:
        if 'ShipType' not in df_q3_coastal_vol.columns:
            if 'Ship' in df_q3_coastal_vol.columns:
                df_q3_coastal_vol = df_q3_coastal_vol.rename(columns={'Ship': 'ShipType'})
            else:
                print("Warning (Chart 4 - Plotly): 'ShipType' or 'Ship' column not found in coastal data. Using default 'UnknownShip'.")
                df_q3_coastal_vol['ShipType'] = 'UnknownShip'
    else:
        df_q3_coastal_vol = pd.DataFrame(columns=['AM', 'OriginPort', 'ShipType', 'DestinationPort', 'Customer', 'AnnualVolume'])

    if df_q3_direct_vol is None or df_q3_direct_vol.empty:
        df_q3_direct_vol = pd.DataFrame(columns=['AM', 'Customer', 'AnnualVolume'])

    # --- General Font settings for Plotly ---
    plotly_font_dict = dict(
        family=PLOTLY_FONT_FAMILY, # Defined at the top of the script
        size=22, # Base font size, can be overridden for specific elements
        color="black"
    )
    plotly_title_font_dict = dict(
        family=PLOTLY_FONT_FAMILY,
        size=28, # Larger title font
        color="black"
    )
    plotly_node_label_font_dict = dict(
        family=PLOTLY_FONT_FAMILY,
        size=20, # Font size for node labels
        color="black"
    )
    plotly_link_label_font_dict = dict(
        family=PLOTLY_FONT_FAMILY,
        size=12, # Font size for link labels
        color="black"
    )


    # --- Chart 4a: AM -> Mode Split -> Aggregated Destinations ---
    print("  Generating Chart 4a: AM -> Mode Split -> Aggregated Destinations...")
    try:
        nodes_4a_labels, nodes_4a_colors, nodes_4a_x, nodes_4a_y = [], [], [], []
        links_4a_source, links_4a_target, links_4a_value, links_4a_color, links_4a_hoverlabel, links_4a_textlabel = [], [], [], [], [], []
        label_to_idx_4a = {}
        
        y_pos_total_nodes_in_x = {} # x_coord -> total nodes in this x-layer

        def add_node_4a(label, x_coord, color):
            if label not in label_to_idx_4a:
                label_to_idx_4a[label] = len(nodes_4a_labels)
                nodes_4a_labels.append(label)
                nodes_4a_colors.append(color)
                nodes_4a_x.append(x_coord)
                y_pos_total_nodes_in_x[x_coord] = y_pos_total_nodes_in_x.get(x_coord, 0) + 1
        
        # Define nodes first to count nodes per x-layer for y-positioning
        all_ams_4a = sorted(pd.concat([df_q3_direct_vol['AM'], df_q3_coastal_vol['AM']]).unique())
        for am in all_ams_4a: add_node_4a(f"AM: {am}", 0.05, AM_COLORS.get(am, 'grey'))
        
        direct_mode_label = "Direct (Road)"
        coastal_mode_label = "Coastal (Mutimodal)"
        add_node_4a(direct_mode_label, 0.35, MODE_SPLIT_NODE_COLOR)
        add_node_4a(coastal_mode_label, 0.35, MODE_SPLIT_NODE_COLOR)

        unique_origin_ports_4a = sorted(df_q3_coastal_vol['OriginPort'].unique())
        for op in unique_origin_ports_4a: add_node_4a(f"Origin Port: {op.replace(' Port','')}", 0.65, PORT_NODE_COLOR)

        dest_direct_label = "Customers (via Direct)"
        dest_coastal_label = "Customers (via Coastal)"
        add_node_4a(dest_direct_label, 0.95, DIRECT_TRUCK_COLOR.replace('0.7','0.9'))
        add_node_4a(dest_coastal_label, 0.95, SEA_LINK_COLOR.replace('0.7','0.9'))

        # Assign Y positions based on counts per layer
        for x_coord_key in y_pos_total_nodes_in_x: # Iterate in defined x_coord order if necessary
            count_in_layer = y_pos_total_nodes_in_x[x_coord_key]
            current_y_idx_in_layer = 0
            for i_node, node_x_val in enumerate(nodes_4a_x):
                if node_x_val == x_coord_key:
                    nodes_4a_y.append((current_y_idx_in_layer + 0.5) / count_in_layer if count_in_layer > 0 else 0.5)
                    current_y_idx_in_layer +=1
        if len(nodes_4a_y) < len(nodes_4a_labels): # Fallback if any layer was empty
             nodes_4a_y.extend([0.5] * (len(nodes_4a_labels) - len(nodes_4a_y)))


        # Links for 4a
        direct_grouped = df_q3_direct_vol.groupby('AM')['AnnualVolume'].sum()
        for am, volume in direct_grouped.items():
            if volume > 0.1:
                s_idx, t_idx = label_to_idx_4a.get(f"AM: {am}"), label_to_idx_4a.get(direct_mode_label)
                if s_idx is not None and t_idx is not None:
                    links_4a_source.append(s_idx); links_4a_target.append(t_idx); links_4a_value.append(volume)
                    links_4a_color.append(AM_COLORS.get(am, 'grey').replace('0.9', '0.6'))
                    links_4a_hoverlabel.append(f"{am} → {direct_mode_label}: {volume:,.0f} units"); links_4a_textlabel.append(f"{volume/1000:,.1f}K")

        coastal_grouped_am = df_q3_coastal_vol.groupby('AM')['AnnualVolume'].sum()
        for am, volume in coastal_grouped_am.items():
            if volume > 0.1:
                s_idx, t_idx = label_to_idx_4a.get(f"AM: {am}"), label_to_idx_4a.get(coastal_mode_label)
                if s_idx is not None and t_idx is not None:
                    links_4a_source.append(s_idx); links_4a_target.append(t_idx); links_4a_value.append(volume)
                    links_4a_color.append(AM_COLORS.get(am, 'grey').replace('0.9', '0.6'))
                    links_4a_hoverlabel.append(f"{am} → {coastal_mode_label}: {volume:,.0f} units"); links_4a_textlabel.append(f"{volume/1000:,.1f}K")

        total_direct_volume = df_q3_direct_vol['AnnualVolume'].sum()
        if total_direct_volume > 0.1:
            s_idx, t_idx = label_to_idx_4a.get(direct_mode_label), label_to_idx_4a.get(dest_direct_label)
            if s_idx is not None and t_idx is not None:
                links_4a_source.append(s_idx); links_4a_target.append(t_idx); links_4a_value.append(total_direct_volume)
                links_4a_color.append(DIRECT_TRUCK_COLOR);
                links_4a_hoverlabel.append(f"{direct_mode_label} → {dest_direct_label}: {total_direct_volume:,.0f} units"); links_4a_textlabel.append(f"{total_direct_volume/1000:,.1f}K")

        coastal_am_op_grouped = df_q3_coastal_vol.groupby(['AM', 'OriginPort'])['AnnualVolume'].sum().reset_index()
        for _, row in coastal_am_op_grouped.iterrows():
            if row['AnnualVolume'] > 0.1:
                s_idx, t_idx = label_to_idx_4a.get(coastal_mode_label), label_to_idx_4a.get(f"Origin Port: {row['OriginPort'].replace(' Port','')}")
                if s_idx is not None and t_idx is not None:
                    links_4a_source.append(s_idx); links_4a_target.append(t_idx); links_4a_value.append(row['AnnualVolume'])
                    links_4a_color.append(COASTAL_TRUCK_COLOR)
                    links_4a_hoverlabel.append(f"{coastal_mode_label} ( {row['AM']}) → {row['OriginPort']}: {row['AnnualVolume']:,.0f} units"); links_4a_textlabel.append(f"{row['AnnualVolume']/1000:,.1f}K")
        
        coastal_op_total_dest = df_q3_coastal_vol.groupby('OriginPort')['AnnualVolume'].sum().reset_index()
        for _, row in coastal_op_total_dest.iterrows():
            if row['AnnualVolume'] > 0.1:
                s_idx, t_idx = label_to_idx_4a.get(f"Origin Port: {row['OriginPort'].replace(' Port','')}"), label_to_idx_4a.get(dest_coastal_label)
                if s_idx is not None and t_idx is not None:
                    links_4a_source.append(s_idx); links_4a_target.append(t_idx); links_4a_value.append(row['AnnualVolume'])
                    links_4a_color.append(SEA_LINK_COLOR)
                    links_4a_hoverlabel.append(f"{row['OriginPort']} → {dest_coastal_label}: {row['AnnualVolume']:,.0f} units"); links_4a_textlabel.append(f"{row['AnnualVolume']/1000:,.1f}K")

        fig4a = go.Figure(data=[go.Sankey(
            arrangement="snap",
            node=dict(
                pad=30, thickness=25, line=dict(color="black", width=0.5),
                label=nodes_4a_labels, color=nodes_4a_colors, x=nodes_4a_x, y=nodes_4a_y,
                hovertemplate='%{label}<extra></extra>'
            ),
            link=dict(
                source=links_4a_source, target=links_4a_target, value=links_4a_value,
                color=links_4a_color,
                label=links_4a_textlabel, # Display volume on link
                hovertemplate='%{customdata}<extra></extra>', customdata=links_4a_hoverlabel
            ),
            textfont=plotly_node_label_font_dict # Apply to node labels
        )])
        fig4a.update_layout(
            title_text="Chart 4a: Aggregated Transport Flows (AM → Mode → Destination Type)",
            font=plotly_font_dict, # Overall font for the figure
            title_font=plotly_title_font_dict,
            height=800, width=1400, # Larger figure
            margin=dict(t=80, b=40, l=40, r=40)
        )
        fig4a.write_html(os.path.join(output_dir, "04a_aggregated_flows_sankey.html"))
        fig4a.write_image(os.path.join(output_dir, "04a_aggregated_flows_sankey.png"), scale=2)
        print(f"  Chart 4a (Plotly Sankey) saved to {os.path.join(output_dir, '04a_aggregated_flows_sankey.png')} and .html")

    except Exception as e:
        print(f"Error generating Chart 4a (Aggregated Plotly Sankey): {e}")
        import traceback
        traceback.print_exc()

    # --- Chart 4b: Detailed AM -> Mode Split -> Path -> Individual Customers ---
    print("  Generating Chart 4b: Detailed End-to-End Coastal and Direct Flows...")
    try:
        node_labels_4b, node_colors_4b, node_x_4b, node_y_4b = [], [], [], []
        link_source_4b, link_target_4b, link_value_4b, link_color_4b, link_hover_4b = [], [], [], [], []
        label_to_idx_4b = {}
        
        y_pos_map = {} # For trying to align nodes vertically better

        def add_node_4b(label, x_coord, color, layer_nodes_list_for_y_calc):
            if label not in label_to_idx_4b:
                label_to_idx_4b[label] = len(node_labels_4b)
                node_labels_4b.append(label)
                node_colors_4b.append(color)
                node_x_4b.append(x_coord)
                layer_nodes_list_for_y_calc.append(label)

        # Define layers for X positioning and for Y distribution
        layer_am = []
        layer_origin_port = []
        layer_ship_type = []
        layer_dest_port = []
        layer_customer_coastal = []

        # Layer 1: AMs
        all_ams_b = sorted(pd.concat([df_q3_direct_vol['AM'], df_q3_coastal_vol['AM']]).unique())
        for am in all_ams_b: add_node_4b(f"AM: {am}", 0.01, AM_COLORS.get(am, 'grey'), layer_am)

        # Layer 2: Origin Ports (Coastal Only)
        unique_ops_b = sorted(df_q3_coastal_vol['OriginPort'].unique())
        for op in unique_ops_b: add_node_4b(f"Origin: {op.replace(' Port','')}", 0.25, PORT_NODE_COLOR, layer_origin_port)

        # Layer 3: Ship Types (Coastal Only)
        unique_ships_b = sorted(df_q3_coastal_vol['ShipType'].unique())
        for st in unique_ships_b: add_node_4b(f"Ship: {st}", 0.50, SHIP_COLORS.get(st, 'grey'), layer_ship_type)
        
        # Layer 4: Destination Ports (Coastal Only)
        unique_dps_b = sorted(df_q3_coastal_vol['DestinationPort'].unique())
        for dp in unique_dps_b: add_node_4b(f"Dest Port: {dp.replace(' Port','')}", 0.75, PORT_NODE_COLOR, layer_dest_port)

        # Layer 5: Customers (Separate for Coastal and Direct for clarity if needed, or combine if too many)
        # For now, all customers go to the far right.
        all_customers = pd.concat([
            df_q3_direct_vol[['Customer', 'AM']],
            df_q3_coastal_vol[['Customer', 'AM']]
        ]).drop_duplicates(subset=['Customer'])
        
        # To avoid extreme density, limit displayed customers or aggregate
        MAX_CUSTOMERS_DISPLAYED = 50 # Example limit
        if len(all_customers) > MAX_CUSTOMERS_DISPLAYED:
            print(f"  Warning (Chart 4b): Number of unique customers ({len(all_customers)}) is large. Aggregating less significant customer flows.")
            # Logic to aggregate smaller customer flows would go here. For now, we proceed with all.
            # For a production chart, aggregation (e.g., by state) or showing top N would be critical.
        
        for _, cust_row in all_customers.iterrows():
            cust_label = f"Cust: {cust_row['Customer']}"
            # Assign color based on source AM for consistency if possible, or a neutral customer color
            cust_color = AM_COLORS.get(cust_row['AM'], 'rgba(180,180,180,0.7)') 
            add_node_4b(cust_label, 0.99, cust_color, layer_customer_coastal) # Add to one list for y-pos

        # Calculate Y positions for nodes in each layer
        layers_for_y_calc = [layer_am, layer_origin_port, layer_ship_type, layer_dest_port, layer_customer_coastal] # layer_customer_coastal now holds all customer nodes
        
        current_max_y = 0
        for layer_nodes in layers_for_y_calc:
            num_in_layer = len(layer_nodes)
            if num_in_layer == 0: continue
            # Distribute Y positions for nodes within this layer
            for i, node_label in enumerate(layer_nodes):
                idx = label_to_idx_4b[node_label]
                node_y_pos = (i + 0.5) / num_in_layer # Spread 0 to 1 within layer block
                node_y_4b.append(node_y_pos)
        
        # Ensure node_y_4b has entry for every node
        if len(node_y_4b) < len(node_labels_4b):
             node_y_4b.extend([0.5] * (len(node_labels_4b) - len(node_y_4b)))


        # Create Links for 4b
        # 1. Direct AM -> Customer
        for _, row in df_q3_direct_vol.iterrows():
            if row['AnnualVolume'] > 0:
                s_idx = label_to_idx_4b.get(f"AM: {row['AM']}")
                t_idx = label_to_idx_4b.get(f"Cust: {row['Customer']}")
                if s_idx is not None and t_idx is not None:
                    link_source_4b.append(s_idx)
                    link_target_4b.append(t_idx)
                    link_value_4b.append(row['AnnualVolume'])
                    link_color_4b.append(DIRECT_TRUCK_COLOR)
                    link_hover_4b.append(f"Direct Truck from {row['AM']} to {row['Customer']}: {row['AnnualVolume']:,.0f} units")

        # 2. Coastal: AM -> Origin Port (Trucking)
        am_op_agg_b = df_q3_coastal_vol.groupby(['AM', 'OriginPort'])['AnnualVolume'].sum().reset_index()
        for _, row in am_op_agg_b.iterrows():
            if row['AnnualVolume'] > 0:
                s_idx = label_to_idx_4b.get(f"AM: {row['AM']}")
                t_idx = label_to_idx_4b.get(f"Origin: {row['OriginPort'].replace(' Port','')}")
                if s_idx is not None and t_idx is not None:
                    link_source_4b.append(s_idx)
                    link_target_4b.append(t_idx)
                    link_value_4b.append(row['AnnualVolume'])
                    link_color_4b.append(COASTAL_TRUCK_COLOR) # Trucking to port
                    link_hover_4b.append(f"Truck from {row['AM']} to {row['OriginPort']}: {row['AnnualVolume']:,.0f} units")
        
        # 3. Coastal: Origin Port -> ShipType
        op_ship_agg_b = df_q3_coastal_vol.groupby(['OriginPort', 'ShipType'])['AnnualVolume'].sum().reset_index()
        for _, row in op_ship_agg_b.iterrows():
            if row['AnnualVolume'] > 0:
                s_idx = label_to_idx_4b.get(f"Origin: {row['OriginPort'].replace(' Port','')}")
                t_idx = label_to_idx_4b.get(f"Ship: {row['ShipType']}")
                if s_idx is not None and t_idx is not None:
                    link_source_4b.append(s_idx)
                    link_target_4b.append(t_idx)
                    link_value_4b.append(row['AnnualVolume'])
                    link_color_4b.append(SHIP_COLORS.get(row['ShipType'], 'grey').replace('0.8','0.6')) # Ship color, slightly transparent
                    link_hover_4b.append(f"Loading on {row['ShipType']} at {row['OriginPort']}: {row['AnnualVolume']:,.0f} units")

        # 4. Coastal: ShipType -> Destination Port (Sea voyage)
        ship_dp_agg_b = df_q3_coastal_vol.groupby(['ShipType', 'DestinationPort'])['AnnualVolume'].sum().reset_index()
        for _, row in ship_dp_agg_b.iterrows():
            if row['AnnualVolume'] > 0:
                s_idx = label_to_idx_4b.get(f"Ship: {row['ShipType']}")
                t_idx = label_to_idx_4b.get(f"Dest Port: {row['DestinationPort'].replace(' Port','')}")
                if s_idx is not None and t_idx is not None:
                    link_source_4b.append(s_idx)
                    link_target_4b.append(t_idx)
                    link_value_4b.append(row['AnnualVolume'])
                    link_color_4b.append(SHIP_COLORS.get(row['ShipType'], 'grey')) # Ship color
                    link_hover_4b.append(f"{row['ShipType']} to {row['DestinationPort']}: {row['AnnualVolume']:,.0f} units (Sea)")

        # 5. Coastal: Destination Port -> Customer (Trucking)
        dp_cust_agg_b = df_q3_coastal_vol.groupby(['DestinationPort', 'Customer'])['AnnualVolume'].sum().reset_index()
        for _, row in dp_cust_agg_b.iterrows():
            if row['AnnualVolume'] > 0:
                s_idx = label_to_idx_4b.get(f"Dest Port: {row['DestinationPort'].replace(' Port','')}")
                t_idx = label_to_idx_4b.get(f"Cust: {row['Customer']}")
                if s_idx is not None and t_idx is not None:
                    link_source_4b.append(s_idx)
                    link_target_4b.append(t_idx)
                    link_value_4b.append(row['AnnualVolume'])
                    link_color_4b.append(COASTAL_TRUCK_COLOR) # Trucking from port
                    link_hover_4b.append(f"Truck from {row['DestinationPort']} to {row['Customer']}: {row['AnnualVolume']:,.0f} units")

        fig4b = go.Figure(data=[go.Sankey(
            arrangement="snap", # Perpendicular or Freeform might also work for complex ones
            node=dict(
                pad=15, thickness=20, line=dict(color="black", width=0.5),
                label=node_labels_4b, color=node_colors_4b, x=node_x_4b, y=node_y_4b,
                hovertemplate='%{label}<extra></extra>'
            ),
            link=dict(
                source=link_source_4b, target=link_target_4b, value=link_value_4b,
                color=link_color_4b,
                label=[f"{v/1000:.1f}K" if v > 100 else "" for v in link_value_4b], # Show values on links, hide if too small
                hovertemplate='%{customdata}<extra></extra>', customdata=link_hover_4b
            ),
            textfont=plotly_node_label_font_dict # Apply to node labels
        )])
        fig4b.update_layout(
            title_text="Chart 4b: Detailed End-to-End Shipment Flows (with Customers)",
            font_family=PLOTLY_FONT_FAMILY, font_size=10, # Smaller font for dense diagram
            font=plotly_font_dict, # Overall font for the figure
            title_font=plotly_title_font_dict,
            height=max(1000, len(all_customers) * 20), # Adjust height based on number of customers
            width=1800, # Wider for more stages
            margin=dict(t=70, b=50, l=50, r=50)
        )
        fig4b.write_image(os.path.join(output_dir, "04b_detailed_end_to_end_sankey.png"), scale=1.5) # Adjust scale for resolution
        print(f"  Chart 4b saved to {os.path.join(output_dir, '04b_detailed_end_to_end_sankey.png')}")
    except Exception as e:
        print(f"Error generating Chart 4b (Detailed Plotly Sankey): {e}")
        import traceback
        traceback.print_exc()

# --- Chart 5: KPI Summary (Text Plot) ---
def plot_kpi_summary(df_q3_summary_data):
    print("Generating Chart 5: KPI Summary...")

    if df_q3_summary_data.empty or 'Status' not in df_q3_summary_data.columns:
        print("Error (Chart 5): Q3 summary data is missing or malformed. Skipping KPI plot.")
        return

    status = df_q3_summary_data['Status'].iloc[0]
    q3_cost = df_q3_summary_data['TotalOptimalCost_Annual'].iloc[0]
    q1_ref_cost = df_q3_summary_data['Annualized_DirectCost_Q1_Reference'].iloc[0]

    savings_amount = 0
    savings_percent = 0
    if pd.notna(q3_cost) and pd.notna(q1_ref_cost) and q1_ref_cost > 0:
        savings_amount = q1_ref_cost - q3_cost
        savings_percent = (savings_amount / q1_ref_cost) * 100

    total_direct_vol_q3 = df_q3_summary_data['TotalDirectVolume_Optimal'].iloc[0]
    total_coastal_vol_q3 = df_q3_summary_data['TotalCoastalVolume_Optimal'].iloc[0]
    ships_used_str = df_q3_summary_data['ShipsUsed_Optimal'].iloc[0]

    fig, ax = plt.subplots(figsize=(7, 4.5)) # Standard size for this one
    ax.axis('off')

    fig.suptitle("Optimization Results: Key Performance Indicators", fontsize=STD_TITLE_FONTSIZE, fontweight='bold', y=0.97)

    kpi_details = [
        ("Solver Status:", status),
        ("Q1 Annualized Direct Cost (Baseline):", f"${q1_ref_cost:,.2f}"),
        ("Q3 Optimized Total Annual Cost:", f"${q3_cost:,.2f}"),
        ("Annual Cost Savings:", f"${savings_amount:,.2f} ({savings_percent:.2f}%)"),
        ("Q3 Total Direct Volume (Annual):", f"{total_direct_vol_q3:,.0f} units"),
        ("Q3 Total Coastal Volume (Annual):", f"{total_coastal_vol_q3:,.0f} units"),
        ("Q3 Ships Utilized:", ships_used_str if pd.notna(ships_used_str) and ships_used_str else "None")
    ]

    y_pos_start = 0.85
    x_label_coord = 0.05
    x_value_coord = 0.55
    line_spacing = 0.11

    for i, (label_text, value_text) in enumerate(kpi_details):
        ax.text(x_label_coord, y_pos_start - i * line_spacing, label_text, fontsize=STD_TICK_FONTSIZE, fontweight='normal', verticalalignment='top', wrap=True)
        color_text = 'black'
        current_fontweight_text = 'normal'
        if "Savings" in label_text and savings_amount < 0: color_text = '#D62728'
        elif "Savings" in label_text and savings_amount > 0: color_text = '#2CA02C'
        elif "Optimized Total Annual Cost" in label_text and pd.notna(q3_cost) and pd.notna(q1_ref_cost) and q3_cost < q1_ref_cost : color_text = '#2CA02C'
        if "Cost" in label_text or "Savings" in label_text or "Status" in label_text:
             current_fontweight_text ='bold'

        ax.text(x_value_coord, y_pos_start - i * line_spacing, value_text, fontsize=STD_TICK_FONTSIZE, color=color_text, fontweight=current_fontweight_text, verticalalignment='top', wrap=True)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(os.path.join(PLOTS_DIR, "05_kpi_summary.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Chart 5 saved.")

# --- Main execution block for standalone testing ---
if __name__ == '__main__':
    print(f"Generating academic style plots. Output directory: '{PLOTS_DIR}'")
    if FONT_SETTINGS_SUCCESS:
        current_family_main = plt.rcParams['font.family']
        if isinstance(current_family_main, list): current_family_main = current_family_main[0]

        font_list_to_check_main = []
        if current_family_main == 'serif' and 'font.serif' in plt.rcParams:
             font_list_to_check_main = plt.rcParams['font.serif']
        elif current_family_main == 'sans-serif' and 'font.sans-serif' in plt.rcParams:
             font_list_to_check_main = plt.rcParams['font.sans-serif']

        if font_list_to_check_main:
            print(f"Plots are configured to use font family: '{current_family_main}'. Preferred font from list: {font_list_to_check_main[0]}")
        else:
            print(f"Plots are configured to use font family: '{current_family_main}', but specific font list for it is empty.")
    else:
        print("Font settings may not have been fully successful. Using Matplotlib defaults.")


    def load_csv_safely(filename, results_dir_main=RESULTS_DIR):
        path = os.path.join(results_dir_main, filename)
        if os.path.exists(path):
            try: return pd.read_csv(path)
            except Exception as e_load: print(f"Error reading {filename}: {e_load}")
        else: print(f"Warning: File not found - {path}")
        return pd.DataFrame()

    # Load data for all charts for a full run
    df_q1_details = load_csv_safely("q1_direct_transport_costs_and_details.csv")
    df_q3_summary = load_csv_safely("q3_optimization_results_summary.csv")
    df_q3_direct_vol = load_csv_safely("q3_optimization_results_direct_shipments.csv")
    df_q3_coastal_vol = load_csv_safely("q3_optimization_results_coastal_shipments.csv")
    df_q2_summary = load_csv_safely("q2_multimodal_summary_by_customer.csv")

    q1_annual_cost_val = 0
    if not df_q3_summary.empty and 'Annualized_DirectCost_Q1_Reference' in df_q3_summary.columns and pd.notna(df_q3_summary.iloc[0]['Annualized_DirectCost_Q1_Reference']):
        q1_annual_cost_val = df_q3_summary['Annualized_DirectCost_Q1_Reference'].iloc[0]
    elif not df_q1_details.empty:
        q1_total_3mo_cost = 0
        for am_id_val_main in ['AM1', 'AM2', 'AM3']:
            cost_col_val_main = f'{am_id_val_main}_Direct_Cost'
            if cost_col_val_main in df_q1_details.columns:
                q1_total_3mo_cost += df_q1_details[cost_col_val_main].sum(skipna=True)
        q1_annual_cost_val = q1_total_3mo_cost * 4
        if df_q3_summary.empty:
             df_q3_summary = pd.DataFrame({'Annualized_DirectCost_Q1_Reference': [q1_annual_cost_val]})
        elif 'Annualized_DirectCost_Q1_Reference' not in df_q3_summary.columns :
             df_q3_summary['Annualized_DirectCost_Q1_Reference'] = pd.NA
             if not df_q3_summary.empty:
                 df_q3_summary.loc[0, 'Annualized_DirectCost_Q1_Reference'] = q1_annual_cost_val
        elif pd.isna(df_q3_summary.loc[0, 'Annualized_DirectCost_Q1_Reference']):
             df_q3_summary.loc[0, 'Annualized_DirectCost_Q1_Reference'] = q1_annual_cost_val
    else:
        print("Warning: Q1 annualized cost could not be determined reliably. Chart 1 may be affected.")
        if df_q3_summary.empty:
            df_q3_summary = pd.DataFrame({'Annualized_DirectCost_Q1_Reference': [0.0]}) # Ensure float for consistency
            # Make sure other required columns also exist if creating df_q3_summary from scratch
            df_q3_summary['TotalOptimalCost_Annual'] = 0.0
            df_q3_summary['TotalDirectVolume_Optimal'] = 0
            df_q3_summary['TotalCoastalVolume_Optimal'] = 0
            df_q3_summary['ShipsUsed_Optimal'] = "N/A"
            df_q3_summary['Status'] = "Data Unavailable"

        elif 'Annualized_DirectCost_Q1_Reference' not in df_q3_summary.columns:
            df_q3_summary['Annualized_DirectCost_Q1_Reference'] = 0.0
            if not df_q3_summary.empty:
                 df_q3_summary.loc[0, 'Annualized_DirectCost_Q1_Reference'] = 0.0
        elif pd.isna(df_q3_summary.loc[0, 'Annualized_DirectCost_Q1_Reference']):
            df_q3_summary.loc[0, 'Annualized_DirectCost_Q1_Reference'] = 0.0


    required_q3_cols = {
        'TotalOptimalCost_Annual': 0.0, 'Annualized_DirectCost_Q1_Reference': q1_annual_cost_val if pd.notna(q1_annual_cost_val) else 0.0,
        'TotalDirectVolume_Optimal': 0, 'TotalCoastalVolume_Optimal': 0,
        'ShipsUsed_Optimal': 'N/A', 'Status': 'Data Unavailable'
    }
    if df_q3_summary.empty:
        print("Info: Q3 Summary DataFrame was empty after loading attempts, initializing with default structure.")
        df_q3_summary = pd.DataFrame([required_q3_cols])
    else:
        for col_name_check_main, default_scalar_val_main in required_q3_cols.items():
            if col_name_check_main not in df_q3_summary.columns:
                print(f"Info: Adding missing column '{col_name_check_main}' to Q3 Summary DataFrame with default.")
                if len(df_q3_summary) == 0:
                    df_q3_summary = pd.DataFrame(columns=list(df_q3_summary.columns) + [col_name_check_main])
                    df_q3_summary.loc[0, col_name_check_main] = default_scalar_val_main
                else:
                    df_q3_summary[col_name_check_main] = default_scalar_val_main
                if not df_q3_summary.empty and pd.isna(df_q3_summary.loc[0, col_name_check_main]):
                     df_q3_summary.loc[0, col_name_check_main] = default_scalar_val_main


    if not df_q3_summary.empty and pd.notna(df_q3_summary['TotalOptimalCost_Annual'].iloc[0]):
        plot_cost_volume_structure(q1_annual_cost_val, df_q3_summary, df_q3_direct_vol, df_q1_details)
    else: print("Skipping Chart 1 due to missing or invalid 'TotalOptimalCost_Annual' in Q3 summary.")

    if not df_q3_direct_vol.empty or not df_q3_coastal_vol.empty:
        plot_am_mode_split_q3(df_q3_direct_vol, df_q3_coastal_vol)
    else: print("Skipping Chart 2 due to missing Q3 shipment data.")

    if not df_q2_summary.empty :
        plot_suitability_bubble(df_q2_summary, df_q1_details, df_q3_coastal_vol)
    else: print("Skipping Chart 3 due to missing Q2 summary data.")

    if not df_q3_coastal_vol.empty:
        plot_q3_sankey_flows_plotly(df_q3_direct_vol, df_q3_coastal_vol)
    else: print("Skipping Chart 4 (Matplotlib Sankey) due to missing Q3 coastal shipment data.")

    if not df_q3_summary.empty:
        plot_kpi_summary(df_q3_summary)
    else: print("Skipping Chart 5 due to missing Q3 summary data.")

    print(f"\nPlot generation attempt finished. Please check the '{PLOTS_DIR}' directory.")
    print("Note for font rendering: Ensure your system has one of the preferred fonts installed and Matplotlib's cache is updated if needed.")