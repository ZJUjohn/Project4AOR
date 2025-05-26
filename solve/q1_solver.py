import pandas as pd
import numpy as np
from solve.config import TRUCK_CAPACITY, TRUCK_FIXED_COST, TRUCK_VARIABLE_COST_KM

class Q1Solver:
    def __init__(self, df_locations):
        self.df_locations = df_locations.copy()

    def solve(self):
        """
        Calculates the total outbound automotive distribution cost for transporting cars directly
        from factories to different customer locations via road trailers under the current scenario.
        Returns:
            tuple: (Total direct transport cost, DataFrame updated with cost details)
        """
        df_q1 = self.df_locations.copy()
        for am_id in ['AM1', 'AM2', 'AM3']:
            demand_cols = [f'{am_id}_M1', f'{am_id}_M2', f'{am_id}_M3']
            for d_col in demand_cols:
                if d_col in df_q1.columns:
                     df_q1[d_col] = pd.to_numeric(df_q1[d_col], errors='coerce').fillna(0)
                else:
                    df_q1[d_col] = 0.0

            df_q1[f'{am_id}_Total_Demand_3Mo'] = df_q1[demand_cols].sum(axis=1)
            mask_valid = (df_q1[f'{am_id}_Total_Demand_3Mo'] > 0) & (~df_q1[f'{am_id}_Dist'].isna())

            for col_suffix in ['_Direct_Num_Trips', '_Direct_Cost', '_Direct_Cost_Per_Unit']:
                df_q1[f'{am_id}{col_suffix}'] = 0.0

            if TRUCK_CAPACITY > 0 : # Prevent division by zero
                df_q1.loc[mask_valid, f'{am_id}_Direct_Num_Trips'] = np.ceil(
                    df_q1.loc[mask_valid, f'{am_id}_Total_Demand_3Mo'] / TRUCK_CAPACITY
                )
            else:
                df_q1.loc[mask_valid, f'{am_id}_Direct_Num_Trips'] = np.inf


            df_q1.loc[mask_valid, f'{am_id}_Direct_Cost'] = df_q1.loc[mask_valid, f'{am_id}_Direct_Num_Trips'] * \
                (TRUCK_FIXED_COST + (TRUCK_VARIABLE_COST_KM * df_q1.loc[mask_valid, f'{am_id}_Dist']))

            valid_demand_mask_for_pu = mask_valid & (df_q1.loc[mask_valid, f'{am_id}_Total_Demand_3Mo'] > 0)
            df_q1.loc[valid_demand_mask_for_pu, f'{am_id}_Direct_Cost_Per_Unit'] = \
                df_q1.loc[valid_demand_mask_for_pu, f'{am_id}_Direct_Cost'] / df_q1.loc[valid_demand_mask_for_pu, f'{am_id}_Total_Demand_3Mo']
            # Handle cases where demand is zero for Direct_Cost_Per_Unit
            df_q1.loc[mask_valid & (df_q1[f'{am_id}_Total_Demand_3Mo'] == 0), f'{am_id}_Direct_Cost_Per_Unit'] = np.inf


        total_direct_trucking_cost = df_q1[[f'AM1_Direct_Cost', f'AM2_Direct_Cost', f'AM3_Direct_Cost']].sum().sum()
        return total_direct_trucking_cost, df_q1