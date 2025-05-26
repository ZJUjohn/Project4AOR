import os
from .config import PLOTS_DIR # PLOTS_DIR is defined in config
from .chart_1_cost_volume import plot_cost_volume_structure_academic
from .chart_2_am_split import plot_am_mode_split_q3_academic
from .chart_3_q2_suitability import plot_q2_suitability_bubble_academic
from .chart_4_sankey_flows import plot_q3_sankey_flows_plotly
from .chart_5_kpi_summary import plot_kpi_summary_academic

def run_all_plots(
    # Data for Chart 1
    q1_annual_cost_scalar, # Renamed from df_q1_annualized_cost to reflect it's a scalar
    df_q3_summary_for_charts_1_5,
    df_q3_direct_vol_for_charts_1_2_4,
    df_q1_details_for_charts_1_3, # Used for unit costs in chart 1, and demand in chart 3
    # Data for Chart 2 (also uses df_q3_direct_vol_for_charts_1_2_4)
    df_q3_coastal_vol_for_charts_2_4,
    # Data for Chart 3 (also uses df_q1_details_for_charts_1_3)
    df_q2_summary_for_chart_3
    # Chart 4 uses df_q3_direct_vol_for_charts_1_2_4 and df_q3_coastal_vol_for_charts_2_4
    # Chart 5 uses df_q3_summary_for_charts_1_5
):
    """
    Orchestrates the generation of all plots.
    Data for each plot must be passed as arguments.
    """
    # PLOTS_DIR existence is checked in config.py, but an extra check here doesn't hurt.
    if not os.path.exists(PLOTS_DIR):
        print(f"CRITICAL WARNING: Plots directory '{PLOTS_DIR}' does not exist and could not be created via config. Plots will likely fail to save.")
        # os.makedirs(PLOTS_DIR) # Attempt creation again if really needed, but should be handled by config
        # print(f"Attempted to create directory for plots: {PLOTS_DIR}")


    print(f"\n--- Starting All Plot Generation (Output to: {os.path.abspath(PLOTS_DIR)}) ---")

    # Chart 1
    plot_cost_volume_structure_academic(
        df_q1_annualized_cost=q1_annual_cost_scalar,
        df_q3_summary=df_q3_summary_for_charts_1_5,
        df_q3_direct_vol_data=df_q3_direct_vol_for_charts_1_2_4,
        q1_direct_unit_costs_df_data=df_q1_details_for_charts_1_3
    )
    # Chart 2
    plot_am_mode_split_q3_academic(
        df_q3_direct=df_q3_direct_vol_for_charts_1_2_4,
        df_q3_coastal=df_q3_coastal_vol_for_charts_2_4
    )
    # Chart 3
    plot_q2_suitability_bubble_academic(
        df_q2_summary=df_q2_summary_for_chart_3,
        q1_data_for_demand=df_q1_details_for_charts_1_3 # df_q1_details contains demand info
    )
    # Chart 4
    plot_q3_sankey_flows_plotly( # Uses PLOTS_DIR from config by default
        df_q3_direct_vol=df_q3_direct_vol_for_charts_1_2_4,
        df_q3_coastal_vol=df_q3_coastal_vol_for_charts_2_4
    )
    # Chart 5
    plot_kpi_summary_academic(
        df_q3_summary_data=df_q3_summary_for_charts_1_5
    )

    print("--- All Plot Generation Attempt Complete ---")