import pandas as pd
import numpy as np
from solve.config import (
    AM_TO_PORT_DIST, PORT_HANDLING_CHARGE_USD_UNIT,
    SEA_COST_SHIP_A, SEA_COST_SHIP_B,
    calculate_land_leg_cost_per_unit
)

class Q2Solver:
    def __init__(self, df_with_q1_costs):
        self.df_q2 = df_with_q1_costs.copy()

    def solve(self):
        """
        Identifies which customer locations are suitable for switching from
        pure road transport to coastal multimodal transport.
        Returns:
            tuple: (DataFrame updated with multimodal costs and suitability assessment,
                    summary DataFrame for printing)
        """
        results_q2_summary_list = []

        for index, row in self.df_q2.iterrows():
            cust_loc = row['Customer_Location']
            summary_row = {'Customer_Location': cust_loc}

            for am_id in ['AM1', 'AM2', 'AM3']:
                direct_cost_pu = row.get(f'{am_id}_Direct_Cost_Per_Unit', np.inf)
                if direct_cost_pu == 0 and row.get(f'{am_id}_Total_Demand_3Mo', 0) == 0:
                    direct_cost_pu = np.inf

                demand_for_am = row.get(f'{am_id}_Total_Demand_3Mo', 0)
                summary_row[f'{am_id}_Direct_Cost_PU'] = direct_cost_pu

                if demand_for_am == 0:
                    for ship_letter in ['A', 'B']:
                        self.df_q2.loc[index, f'{am_id}_MM_Cost_PU_Ship{ship_letter}'] = np.inf
                        self.df_q2.loc[index, f'{am_id}_Suitable_Ship{ship_letter}'] = "N/A (No Demand)"
                        summary_row[f'{am_id}_Min_MM_Cost_PU_Ship{ship_letter}'] = np.inf
                        summary_row[f'{am_id}_Suitable_Ship{ship_letter}'] = "N/A (No Demand)"
                    continue

                min_mm_cost_ship_a_for_am_cust = np.inf
                min_mm_cost_ship_b_for_am_cust = np.inf

                for dest_port_option in ['Chennai Port', 'Pipavav Port']:
                    origin_port_am = ''
                    fml_dist_am = np.nan

                    if am_id == 'AM1':
                        origin_port_am = 'Chennai Port'
                    elif am_id == 'AM2':
                        origin_port_am = 'Pipavav Port' if AM_TO_PORT_DIST['AM2']['Pipavav Port'] < AM_TO_PORT_DIST['AM2']['Chennai Port'] else 'Chennai Port'
                    elif am_id == 'AM3':
                        origin_port_am = 'Pipavav Port'

                    fml_dist_am = AM_TO_PORT_DIST[am_id][origin_port_am]
                    fml_cost_pu = calculate_land_leg_cost_per_unit(fml_dist_am, demand_for_am)

                    lml_dist_port_cust = np.nan
                    if dest_port_option == 'Chennai Port':
                        lml_dist_port_cust = row.get('Dist_to_Chennai_Port', np.inf)
                    else:
                        lml_dist_port_cust = row.get('Dist_to_Pipavav_Port', np.inf)

                    lml_cost_pu = calculate_land_leg_cost_per_unit(lml_dist_port_cust, demand_for_am)

                    current_mm_cost_ship_a_path = np.inf
                    current_mm_cost_ship_b_path = np.inf

                    if origin_port_am != dest_port_option:
                        sea_cost_ship_a_leg = SEA_COST_SHIP_A
                        sea_cost_ship_b_leg = SEA_COST_SHIP_B

                        current_mm_cost_ship_a_path = (fml_cost_pu +
                                                     PORT_HANDLING_CHARGE_USD_UNIT +
                                                     sea_cost_ship_a_leg +
                                                     PORT_HANDLING_CHARGE_USD_UNIT +
                                                     lml_cost_pu)
                        current_mm_cost_ship_b_path = (fml_cost_pu +
                                                     PORT_HANDLING_CHARGE_USD_UNIT +
                                                     sea_cost_ship_b_leg +
                                                     PORT_HANDLING_CHARGE_USD_UNIT +
                                                     lml_cost_pu)

                    min_mm_cost_ship_a_for_am_cust = min(min_mm_cost_ship_a_for_am_cust, current_mm_cost_ship_a_path)
                    min_mm_cost_ship_b_for_am_cust = min(min_mm_cost_ship_b_for_am_cust, current_mm_cost_ship_b_path)

                self.df_q2.loc[index, f'{am_id}_MM_Cost_PU_ShipA'] = min_mm_cost_ship_a_for_am_cust
                self.df_q2.loc[index, f'{am_id}_Suitable_ShipA'] = "Suitable" if min_mm_cost_ship_a_for_am_cust < direct_cost_pu else "Not Suitable"
                if min_mm_cost_ship_a_for_am_cust == np.inf :
                     self.df_q2.loc[index, f'{am_id}_Suitable_ShipA'] = "N/A (No Valid Sea Route)"

                self.df_q2.loc[index, f'{am_id}_MM_Cost_PU_ShipB'] = min_mm_cost_ship_b_for_am_cust
                self.df_q2.loc[index, f'{am_id}_Suitable_ShipB'] = "Suitable" if min_mm_cost_ship_b_for_am_cust < direct_cost_pu else "Not Suitable"
                if min_mm_cost_ship_b_for_am_cust == np.inf :
                     self.df_q2.loc[index, f'{am_id}_Suitable_ShipB'] = "N/A (No Valid Sea Route)"


                summary_row[f'{am_id}_Min_MM_Cost_PU_ShipA'] = min_mm_cost_ship_a_for_am_cust
                summary_row[f'{am_id}_Suitable_ShipA'] = self.df_q2.loc[index, f'{am_id}_Suitable_ShipA']
                summary_row[f'{am_id}_Min_MM_Cost_PU_ShipB'] = min_mm_cost_ship_b_for_am_cust
                summary_row[f'{am_id}_Suitable_ShipB'] = self.df_q2.loc[index, f'{am_id}_Suitable_ShipB']

            results_q2_summary_list.append(summary_row)
        return self.df_q2, pd.DataFrame(results_q2_summary_list)