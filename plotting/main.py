# This file is now the main entry point for plotting.
# It handles data loading/preparation and then calls the orchestrator.

import os
import pandas as pd
import numpy as np

# Import the main plotting orchestrator and config elements
from plotting_modules.generate_all_plots_logic import run_all_plots
from plotting_modules.config import PLOTS_DIR, RESULTS_DIR, FONT_SETTINGS_SUCCESS # Import necessary configs

def load_csv_safely(filename, base_dir=RESULTS_DIR):
    """Loads a CSV file safely, returning an empty DataFrame on error or if not found."""
    path = os.path.join(base_dir, filename)
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception as e_load:
            print(f"Error reading {filename} from {path}: {e_load}")
    else:
        print(f"Warning: File not found - {path}")
    return pd.DataFrame()

def prepare_all_data():
    """
    Loads and prepares all necessary data for the plots.
    This function replicates the data loading logic from the original
    plotting.py's __main__ block.
    """
    print("--- Loading Data for Plots ---")
    
    # Load data as in the original plotting.py's main block
    df_q1_details = load_csv_safely("q1_direct_transport_costs_and_details.csv")
    df_q3_summary = load_csv_safely("q3_optimization_results_summary.csv")
    df_q3_direct_vol = load_csv_safely("q3_optimization_results_direct_shipments.csv")
    df_q3_coastal_vol = load_csv_safely("q3_optimization_results_coastal_shipments.csv")
    df_q2_summary = load_csv_safely("q2_multimodal_summary_by_customer.csv")

    # --- q1_annual_cost_scalar calculation (from original plotting.py) ---
    q1_annual_cost_scalar = 0.0 # Default
    if not df_q3_summary.empty and \
       'Annualized_DirectCost_Q1_Reference' in df_q3_summary.columns and \
       pd.notna(df_q3_summary.iloc[0]['Annualized_DirectCost_Q1_Reference']):
        q1_annual_cost_scalar = df_q3_summary.iloc[0]['Annualized_DirectCost_Q1_Reference']
    elif not df_q1_details.empty:
        q1_total_3mo_cost = 0
        for am_id_val_main in ['AM1', 'AM2', 'AM3']:
            cost_col_val_main = f'{am_id_val_main}_Direct_Cost' # As per original logic
            if cost_col_val_main in df_q1_details.columns:
                q1_total_3mo_cost += pd.to_numeric(df_q1_details[cost_col_val_main], errors='coerce').sum(skipna=True)
        q1_annual_cost_scalar = q1_total_3mo_cost * 4
        # If df_q3_summary was empty or didn't have the Q1 ref cost, try to add it
        if df_q3_summary.empty:
            df_q3_summary = pd.DataFrame({'Annualized_DirectCost_Q1_Reference': [q1_annual_cost_scalar]})
        elif 'Annualized_DirectCost_Q1_Reference' not in df_q3_summary.columns:
            df_q3_summary['Annualized_DirectCost_Q1_Reference'] = q1_annual_cost_scalar
        elif pd.isna(df_q3_summary.loc[0, 'Annualized_DirectCost_Q1_Reference']):
            df_q3_summary.loc[0, 'Annualized_DirectCost_Q1_Reference'] = q1_annual_cost_scalar
    else:
        print("Warning (plotting.py): Q1 annualized cost could not be determined. Using 0.0. Chart 1 may be affected.")
        if df_q3_summary.empty: # If still empty, create with default Q1 ref
             df_q3_summary = pd.DataFrame({'Annualized_DirectCost_Q1_Reference': [0.0]})

    # --- Ensure df_q3_summary has all required columns (from original plotting.py) ---
    required_q3_cols = {
        'TotalOptimalCost_Annual': 0.0,
        'Annualized_DirectCost_Q1_Reference': q1_annual_cost_scalar if pd.notna(q1_annual_cost_scalar) else 0.0,
        'TotalDirectVolume_Optimal': 0,
        'TotalCoastalVolume_Optimal': 0,
        'ShipsUsed_Optimal': 'N/A',
        'Status': 'Data Unavailable'
    }
    if df_q3_summary.empty:
        print("Info (plotting.py): Q3 Summary DataFrame was empty, initializing with default structure.")
        df_q3_summary = pd.DataFrame([required_q3_cols])
    else:
        # Ensure the first row exists if columns are added to a non-empty but potentially 0-row df
        if len(df_q3_summary) == 0:
            df_q3_summary = pd.DataFrame(columns=list(df_q3_summary.columns)) # Keep existing cols
            # Add a row to allow .loc[0, col] assignment
            new_row = pd.Series({col: default_val for col, default_val in required_q3_cols.items()})
            df_q3_summary = pd.concat([df_q3_summary, new_row.to_frame().T], ignore_index=True)


        for col_name, default_val in required_q3_cols.items():
            if col_name not in df_q3_summary.columns:
                print(f"Info (plotting.py): Adding missing column '{col_name}' to Q3 Summary with default.")
                df_q3_summary[col_name] = default_val
            # Fill NaN in the first row if it exists for required columns
            if not df_q3_summary.empty and pd.isna(df_q3_summary.loc[0, col_name]):
                df_q3_summary.loc[0, col_name] = default_val
    
    print("--- Data Loading and Preparation Complete ---")
    
    return {
        "q1_annual_cost_scalar": q1_annual_cost_scalar,
        "df_q3_summary_for_charts_1_5": df_q3_summary,
        "df_q3_direct_vol_for_charts_1_2_4": df_q3_direct_vol,
        "df_q1_details_for_charts_1_3": df_q1_details,
        "df_q3_coastal_vol_for_charts_2_4": df_q3_coastal_vol,
        "df_q2_summary_for_chart_3": df_q2_summary
    }

if __name__ == '__main__':
    print(f"--- Executing plots generation script: {__file__} ---")
    if not FONT_SETTINGS_SUCCESS:
        print("Warning: Font settings from config.py may not have been applied correctly. Check console output from config.py.")

    # 1. Load and prepare all necessary data
    plot_data = prepare_all_data()

    # 2. Call the main plotting function from the new module
    # Ensure the arguments match the definition in run_all_plots
    # Check if essential dataframes for specific plots are not empty before calling run_all_plots,
    # or let individual plot functions handle their specific data checks.
    # The current run_all_plots calls all plot functions; individual functions then check their data.

    print("\n--- Initiating Plot Generation via run_all_plots ---")
    run_all_plots(
        q1_annual_cost_scalar=plot_data["q1_annual_cost_scalar"],
        df_q3_summary_for_charts_1_5=plot_data["df_q3_summary_for_charts_1_5"],
        df_q3_direct_vol_for_charts_1_2_4=plot_data["df_q3_direct_vol_for_charts_1_2_4"],
        df_q1_details_for_charts_1_3=plot_data["df_q1_details_for_charts_1_3"],
        df_q3_coastal_vol_for_charts_2_4=plot_data["df_q3_coastal_vol_for_charts_2_4"],
        df_q2_summary_for_chart_3=plot_data["df_q2_summary_for_chart_3"]
    )

    print(f"\n--- Plot generation process initiated by {__file__} has finished. ---")
    print(f"--- Please check the '{os.path.abspath(PLOTS_DIR)}' directory for output images. ---")
    print("--- Note on font rendering: Ensure your system has one of the preferred fonts installed and Matplotlib's cache is up-to-date if issues persist (see config.py output). ---")