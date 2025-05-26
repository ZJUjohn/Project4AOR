import pandas as pd
import numpy as np
import os
from solve.data_loader import DataLoader
from solve.q1_solver import Q1Solver
from solve.q2_solver import Q2Solver
from solve.q3_optimizer import Q3Optimizer
from solve.config import (
    AM_CAPACITY_ANNUAL, SHIPS_DATA, PORT_HANDLING_CHARGE_USD_UNIT
)

def main():
    # Create results directory
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created '{results_dir}' directory")

    # --- Float formatting string for amounts ---
    float_format_csv = '%.2f' # Float format for CSV files (amounts with two decimal places)
    # For command line printing, f-string with :.2f already handles two decimal places

    # Load and preprocess data
    print("--- Starting data loading and preprocessing ---")
    data_loader = DataLoader("data/DATA_original.xlsx") # Assuming data file is in a 'data' subdirectory
    df_locations_raw, df_port_cust_dist_raw = data_loader.load_data()
    df_locations_processed = data_loader.preprocess_data(df_locations_raw, df_port_cust_dist_raw)
    print("--- Data loading and preprocessing complete ---")

    # --- Solve Question 1 ---
    print("\n--- Starting Question 1 ---")
    q1_solver = Q1Solver(df_locations_processed)
    total_direct_cost_q1, df_with_q1_costs = q1_solver.solve()
    print(f"Question 1 estimated current total direct road transport cost (3 months): ${total_direct_cost_q1:,.2f}") # Command line output already two decimal places
    q1_output_path = os.path.join(results_dir, "q1_direct_transport_costs_and_details.csv")
    df_with_q1_costs.to_csv(q1_output_path, index=False, float_format=float_format_csv) # CSV output two decimal places
    print(f"Question 1 detailed costs and calculations saved to {q1_output_path}")
    print("--- Question 1 complete ---")

    # --- Solve Question 2 ---
    print("\n--- Starting Question 2 ---")
    q2_solver = Q2Solver(df_with_q1_costs)
    df_final_q2, df_q2_summary = q2_solver.solve()
    q2_output_path = os.path.join(results_dir, "q2_multimodal_analysis_details.csv")
    df_final_q2.to_csv(q2_output_path, index=False, float_format=float_format_csv) # CSV output two decimal places
    print(f"Question 2 multimodal detailed analysis saved to {q2_output_path}")
    q2_summary_output_path = os.path.join(results_dir, "q2_multimodal_summary_by_customer.csv")
    df_q2_summary.to_csv(q2_summary_output_path, index=False, float_format=float_format_csv) # CSV output two decimal places
    print(f"Question 2 multimodal summary by customer saved to {q2_summary_output_path}")
    print("--- Question 2 complete ---")

    # --- Solve Question 3 ---
    print("\n--- Starting Question 3 (COPT Optimizer) ---")
    cost_params_q3 = {'port_handling_charge_usd_unit': PORT_HANDLING_CHARGE_USD_UNIT}
    q3_optimizer = Q3Optimizer(df_with_q1_costs, AM_CAPACITY_ANNUAL, SHIPS_DATA, cost_params_q3)
    results_q3_optimization = q3_optimizer.optimize()
    print("--- Question 3 complete ---")

    # --- Save Question 3 results to CSV (ensure amounts are two decimal places) ---
    q3_base_path = os.path.join(results_dir, "q3_optimization_results")
    print(f"\nSaving Question 3 optimization results to CSV files (prefix: {q3_base_path})...")

    # 1. Overall Summary (status and total cost)
    total_optimal_cost_annual = results_q3_optimization.get('total_optimal_cost')
    annualized_q1_cost = total_direct_cost_q1 * 4 if total_direct_cost_q1 is not None else None

    summary_q3_data = {
        'Status': [results_q3_optimization['status']],
        'TotalOptimalCost_Annual': [total_optimal_cost_annual], # Amount
        'Annualized_DirectCost_Q1_Reference': [annualized_q1_cost], # Amount
        'TotalDirectVolume_Optimal': [results_q3_optimization.get('total_direct_volume')],
        'TotalCoastalVolume_Optimal': [results_q3_optimization.get('total_coastal_volume')],
        'ShipsUsed_Optimal': [', '.join(results_q3_optimization.get('ships_used', []))]
    }
    df_summary_q3 = pd.DataFrame(summary_q3_data)
    df_summary_q3.to_csv(f"{q3_base_path}_summary.csv", index=False, float_format=float_format_csv) # CSV output two decimal places
    print(f"  - Optimization summary saved to: {q3_base_path}_summary.csv")

    # 2. Direct Shipments (Optimized) - Volume is not an amount
    if results_q3_optimization.get('direct_shipments'):
        direct_shipments_data = []
        for (am, cust), volume in results_q3_optimization['direct_shipments'].items():
            direct_shipments_data.append({'AM': am, 'Customer': cust, 'AnnualVolume': volume})
        df_direct_q3 = pd.DataFrame(direct_shipments_data)
        df_direct_q3.to_csv(f"{q3_base_path}_direct_shipments.csv", index=False, float_format='%.0f') # Volume usually integer
        print(f"  - Optimized direct shipments saved to: {q3_base_path}_direct_shipments.csv")

    # 3. Coastal Shipments (Optimized) - Volume is not an amount
    if results_q3_optimization.get('coastal_shipments'):
        coastal_shipments_data = []
        for (am, cust, p_orig, q_dest, ship), volume in results_q3_optimization['coastal_shipments'].items():
            coastal_shipments_data.append({
                'AM': am, 'Customer': cust,
                'OriginPort': p_orig, 'DestinationPort': q_dest,
                'ShipType': ship, 'AnnualVolume': volume
            })
        df_coastal_q3 = pd.DataFrame(coastal_shipments_data)
        df_coastal_q3.to_csv(f"{q3_base_path}_coastal_shipments.csv", index=False, float_format='%.0f') # Volume usually integer
        print(f"  - Optimized coastal shipments saved to: {q3_base_path}_coastal_shipments.csv")

    # 4. Annual Voyages per Route (Optimized) - Voyages usually 1 or 0 decimal places
    if results_q3_optimization.get('voyages_per_route_ship'):
        voyages_data = []
        for (ship, p_orig, q_dest), num_voyages in results_q3_optimization['voyages_per_route_ship'].items():
            voyages_data.append({
                'ShipType': ship, 'OriginPort': p_orig,
                'DestinationPort': q_dest, 'AnnualVoyages': num_voyages
            })
        df_voyages_q3 = pd.DataFrame(voyages_data)
        df_voyages_q3.to_csv(f"{q3_base_path}_annual_voyages.csv", index=False, float_format='%.1f') # Voyages with 1 decimal place
        print(f"  - Optimized annual voyages saved to: {q3_base_path}_annual_voyages.csv")

    # --- New: Save Question 3 optimization results to TXT file (ensure amounts are two decimal places) ---
    q3_summary_txt_path = os.path.join(results_dir, "q3_optimization_summary.txt")
    print(f"\nSaving Question 3 optimization summary to TXT file: {q3_summary_txt_path}...")

    with open(q3_summary_txt_path, 'w') as f:
        f.write(f"Status: {results_q3_optimization['status']}\n")
        
        total_optimal_cost_annual_q3 = results_q3_optimization.get('total_optimal_cost')
        if total_optimal_cost_annual_q3 is not None:
            # Format total cost to two decimal places
            formatted_total_cost_txt = "{:.2f}".format(total_optimal_cost_annual_q3)
            f.write(f"Total Optimal Cost: {formatted_total_cost_txt}\n")
        else:
            f.write("Total Optimal Cost: Not available\n")
        
        # Optional: Include other summary information in the TXT file (formatted)
        if annualized_q1_cost is not None:
             f.write(f"Annualized_DirectCost_Q1_Reference: {annualized_q1_cost:.2f}\n")
        if results_q3_optimization.get('total_direct_volume') is not None:
             f.write(f"TotalDirectVolume_Optimal: {results_q3_optimization.get('total_direct_volume'):.0f}\n")
        if results_q3_optimization.get('total_coastal_volume') is not None:
             f.write(f"TotalCoastalVolume_Optimal: {results_q3_optimization.get('total_coastal_volume'):.0f}\n")
        if results_q3_optimization.get('ships_used'):
             f.write(f"ShipsUsed_Optimal: {', '.join(results_q3_optimization.get('ships_used', []))}\n")
            
    print(f"  - Question 3 optimization summary saved to: {q3_summary_txt_path}")
    # --- End New Code ---

    # --- Console print output (ensure Q3 amounts are two decimal places) ---
    output_q1_str = f"\nQuestion 1: Estimated current total direct road transport cost (3 months): ${total_direct_cost_q1:,.2f}\n"
    output_q1_str += f"Question 1 detailed results saved to {q1_output_path}\n"

    output_q2_str = f"\n\nQuestion 2: Identification of customer locations suitable for switching to coastal transport\n"
    output_q2_str += f"Question 2 detailed results saved to {q2_output_path} and {q2_summary_output_path}\n"

    output_q3_str = f"\n\nQuestion 3: Optimized coastal shipping system design results (using COPT modeling):\n"
    output_q3_str += f"Solver Status: {results_q3_optimization['status']}\n"
    
    if total_optimal_cost_annual is not None:
        output_q3_str += f"Optimal/Feasible Annual Total Transport Cost: ${total_optimal_cost_annual:,.2f}\n" # Ensure two decimal places
        if annualized_q1_cost is not None:
            output_q3_str += f"  Compared to annualized pure road cost (Q1 estimate): ${annualized_q1_cost:,.2f}\n" # Ensure two decimal places
            if total_optimal_cost_annual < annualized_q1_cost:
                savings = annualized_q1_cost - total_optimal_cost_annual
                savings_percent = (savings / annualized_q1_cost) * 100 if annualized_q1_cost > 0 else 0
                output_q3_str += f"  Cost savings with optimized plan: ${savings:,.2f} (approx. {savings_percent:.2f}%)\n" # Ensure two decimal places
    else:
        output_q3_str += "Could not calculate total cost for the optimized plan.\n"
        
    output_q3_str += f"Question 3 optimization detailed results saved to multiple CSV files with prefix: {q3_base_path}\n"
    output_q3_str += f"Question 3 optimization summary also saved to TXT file: {q3_summary_txt_path}\n" # Also mention TXT file in console output
    # Optionally print other Q3 summary info to console, e.g.:
    output_q3_str += f"  Ship types used in optimized plan: {', '.join(results_q3_optimization.get('ships_used', ['None']))}\n"
    output_q3_str += f"  Total direct transport volume in optimized plan: {results_q3_optimization.get('total_direct_volume', 0):,.0f} units/year\n"
    output_q3_str += f"  Total coastal transport volume in optimized plan: {results_q3_optimization.get('total_coastal_volume', 0):,.0f} units/year\n"


    final_print_output = output_q1_str + "\n" + output_q2_str + "\n" + output_q3_str
    print(final_print_output)

if __name__ == "__main__":
    main()