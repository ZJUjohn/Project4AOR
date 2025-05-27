import pandas as pd
import numpy as np
import coptpy as cp
from coptpy import COPT
import itertools # Added for voyage balancing constraint

# Assuming solve.config contains the necessary constants and functions
from solve.config import (
    AM_TO_PORT_DIST, PORT_STAY_SHIP_DAYS_COSTING,
    SEA_DISTANCE_NM_ONE_WAY,
    get_optimized_land_unit_cost,
    TIME_PER_ONE_WAY_VOYAGE_LEG_DAYS,
    ANNUAL_OPERATING_DAYS,
    CUSTOMER_SHIP_ELIGIBILITY
)

# Placeholder for config items if solve.config is not available
# Replace these with your actual config values or imports
AM_TO_PORT_DIST = {'AM1': {'Chennai Port': 100, 'Pipavav Port': 200}, 'AM2': {'Chennai Port': 150, 'Pipavav Port': 250}, 'AM3': {'Chennai Port': 200, 'Pipavav Port': 100}}
PORT_STAY_SHIP_DAYS_COSTING = 1.0
SEA_DISTANCE_NM_ONE_WAY = 500
def get_optimized_land_unit_cost(distance):
    if distance is None or pd.isna(distance): return 1e9 # High cost for missing distance
    return distance * 0.1 # Example cost
TIME_PER_ONE_WAY_VOYAGE_LEG_DAYS = {'ShipA': 6.46, 'ShipB': 5.41}
ANNUAL_OPERATING_DAYS = 365
CUSTOMER_SHIP_ELIGIBILITY = { # Example: {'AM1': {'Customer1': {'ShipA': True, 'ShipB': False}}}
    'AM1': {
        'CUST001': {'ShipA': True, 'ShipB': True}
    }
}


class Q3Optimizer:
    def __init__(self, df_with_q1_costs_input, am_annual_capacity_map, ships_data_map_config, cost_params_map_config):
        self.df_model_input = df_with_q1_costs_input.copy()
        self.am_annual_capacity_map = am_annual_capacity_map
        self.ships_data_map = ships_data_map_config
        self.cost_params_map = cost_params_map_config
        self.customer_ship_eligibility = CUSTOMER_SHIP_ELIGIBILITY # Loaded from config

    def optimize(self):
        """
        Solves the coastal shipping system design and shipment planning problem using COPT optimizer.
        Returns:
            dict: Optimization results including solution status, optimal cost, and decision variable values.
        """
        print("\n--- Starting Question 3: COPT Optimization Model ---")

        # Define sets
        AMS = ['AM1', 'AM2', 'AM3'] # Asset Managers
        if 'Cust_ID' not in self.df_model_input.columns:
            # If Cust_ID is not present, use Customer_Location as Cust_ID
            self.df_model_input['Cust_ID'] = self.df_model_input['Customer_Location']
        CUSTOMERS = self.df_model_input['Cust_ID'].dropna().unique().tolist()
        PORTS = ['Chennai Port', 'Pipavav Port'] # Available Ports
        SHIPS = list(self.ships_data_map.keys()) # Available Ship Types
        SEA_ROUTES_TUPLES = [(p_orig, p_dest) for p_orig in PORTS for p_dest in PORTS if p_orig != p_dest]

        # --- Pre-calculate parameters ---

        # Calculate annual demands for each (AM, Customer) pair
        annual_demands = {}
        for idx, row in self.df_model_input.iterrows():
            cust_id = row['Cust_ID']
            if pd.isna(cust_id): continue
            for am in AMS:
                demand_3mo = 0
                # Check if individual month columns exist for the AM
                if f'{am}_M1' in row and f'{am}_M2' in row and f'{am}_M3' in row:
                    for m_col in [f'{am}_M1', f'{am}_M2', f'{am}_M3']:
                        demand_3mo += pd.to_numeric(row.get(m_col, 0), errors='coerce')
                else:
                    # Fallback to total demand if individual months are not present
                    demand_3mo = pd.to_numeric(row.get(f'{am}_Total_Demand_3Mo', 0), errors='coerce')
                annual_demands[(am, cust_id)] = demand_3mo * 4 # Convert 3-month demand to annual

        # Calculate direct trucking unit costs
        direct_truck_unit_costs = {}
        for idx, row in self.df_model_input.iterrows():
            cust_id = row['Cust_ID']
            if pd.isna(cust_id): continue
            for am in AMS:
                cost = row.get(f'{am}_Direct_Cost_Per_Unit', np.inf)
                is_demand_positive = annual_demands.get((am, cust_id), 0) > 0
                # Handle problematic costs (NA, inf, 0 for positive demand, negative)
                if pd.isna(cost) or cost == np.inf or (cost == 0 and is_demand_positive) or cost < 0 :
                    direct_truck_unit_costs[(am, cust_id)] = 1e9 if is_demand_positive else 0 # High cost or 0 if no demand
                else:
                    direct_truck_unit_costs[(am, cust_id)] = cost

        # Calculate First-Mile Leg (FML) unit costs (AM to Port)
        fml_unit_costs = {}
        for am in AMS:
            for port in PORTS:
                fml_unit_costs[(am, port)] = get_optimized_land_unit_cost(AM_TO_PORT_DIST[am][port])

        # Calculate Last-Mile Leg (LML) unit costs (Port to Customer)
        lml_unit_costs = {}
        for idx, row in self.df_model_input.iterrows():
            cust_id = row['Cust_ID']
            if pd.isna(cust_id): continue
            for port in PORTS:
                dist_col = 'Dist_to_Chennai_Port' if port == 'Chennai Port' else 'Dist_to_Pipavav_Port'
                lml_unit_costs[(port, cust_id)] = get_optimized_land_unit_cost(row.get(dist_col))

        # Port handling cost per unit
        port_handling_cost = self.cost_params_map['port_handling_charge_usd_unit']

        # Sea voyage variable costs, ship fixed costs, and capacities
        sea_voyage_var_cost_per_trip = {}
        ship_annual_fixed_cost_map = {}
        ship_capacity_map = {}
        valid_voyage_triplets = set() # To store (ship, origin_port, dest_port)

        for s_name in SHIPS:
            ship_annual_fixed_cost_map[s_name] = self.ships_data_map[s_name]['fixed_cost_3months_usd'] * 4
            ship_capacity_map[s_name] = self.ships_data_map[s_name]['capacity']
            s_data = self.ships_data_map[s_name]

            # Calculate one-way voyage time based on sea distance and ship speed
            if s_data.get('speed_avg_knots') is None or s_data['speed_avg_knots'] == 0:
                voyage_time_days_one_way_transit = np.inf
            else:
                voyage_time_days_one_way_transit = SEA_DISTANCE_NM_ONE_WAY / s_data['speed_avg_knots'] / 24 # days

            # Variable cost for one leg (transit + port stay for that leg)
            vcs_one_leg_transit = voyage_time_days_one_way_transit * s_data['vcs_usd_day'] # Variable cost at sea
            vcp_at_ports_for_leg = (PORT_STAY_SHIP_DAYS_COSTING * 2) * s_data['vcp_usd_day'] # Variable cost at port (assuming 2 port calls per leg for costing)
            var_cost_per_one_way_trip = vcs_one_leg_transit + vcp_at_ports_for_leg

            for r_orig, r_dest in SEA_ROUTES_TUPLES:
                sea_voyage_var_cost_per_trip[(s_name, r_orig, r_dest)] = var_cost_per_one_way_trip
                valid_voyage_triplets.add((s_name, r_orig, r_dest))

        # --- Create COPT model ---
        env = cp.Envr()
        model = env.createModel("CoastalShippingOptimization_COPT_NoOrderSplit_EnglishComments")

        # --- Define decision variables ---

        # X_direct[am, cust]: Quantity shipped directly from AM to Customer
        x_direct = {}
        for i in AMS:
            for j in CUSTOMERS:
                if annual_demands.get((i, j), 0) > 0: # Only create var if demand exists
                     x_direct[i,j] = model.addVar(lb=0, name=f"X_direct_{i}_{j.replace(' ','_')}")

        # X_coastal[am, cust, orig_port, dest_port, ship]: Quantity shipped via coastal mode
        x_coastal = {}
        for i in AMS:
            for j in CUSTOMERS:
                if annual_demands.get((i, j), 0) > 0:
                    for p_orig in PORTS:
                        for q_dest in PORTS:
                            if p_orig != q_dest: # Origin and destination ports must be different
                                for s in SHIPS:
                                    # Check customer-ship eligibility
                                    is_eligible = True
                                    if self.customer_ship_eligibility:
                                        am_eligibility = self.customer_ship_eligibility.get(i, {})
                                        cust_eligibility = am_eligibility.get(j, {})
                                        if not cust_eligibility.get(s, True): # Default to True if not specified
                                            is_eligible = False
                                    if is_eligible:
                                        x_coastal[i,j,p_orig,q_dest,s] = model.addVar(
                                            lb=0, name=f"X_coastal_{i}_{j.replace(' ','_')}_{p_orig.replace(' ','_')}_{q_dest.replace(' ','_')}_{s}"
                                        )
        
        # Binary variables for choosing transport mode (no order splitting for an AM-Customer pair)
        # Y_direct_chosen[am, cust]: 1 if demand for (AM, Cust) is met by direct road transport
        # Y_coastal_ShipA_chosen[am, cust]: 1 if demand for (AM, Cust) is met by coastal transport using ShipA
        # Y_coastal_ShipB_chosen[am, cust]: 1 if demand for (AM, Cust) is met by coastal transport using ShipB
        y_direct_chosen = {}
        y_coastal_shipA_chosen = {} # Specific to 'ShipA' if it exists
        y_coastal_shipB_chosen = {} # Specific to 'ShipB' if it exists

        for i in AMS:
            for j in CUSTOMERS:
                if annual_demands.get((i, j), 0) > 0:
                    y_direct_chosen[i,j] = model.addVar(vtype=COPT.BINARY, name=f"Y_direct_{i}_{j.replace(' ','_')}")
                    if 'ShipA' in SHIPS: # Check if 'ShipA' is an available ship type
                        y_coastal_shipA_chosen[i,j] = model.addVar(vtype=COPT.BINARY, name=f"Y_coastal_ShipA_{i}_{j.replace(' ','_')}")
                    if 'ShipB' in SHIPS: # Check if 'ShipB' is an available ship type
                        y_coastal_shipB_chosen[i,j] = model.addVar(vtype=COPT.BINARY, name=f"Y_coastal_ShipB_{i}_{j.replace(' ','_')}")
        
        # V_annual_voyages[ship, orig_port, dest_port]: Number of annual voyages for a ship on a route
        v_annual_voyages = {}
        for s_trip in SHIPS:
            for p_orig_trip in PORTS:
                for q_dest_trip in PORTS:
                    if p_orig_trip != q_dest_trip:
                        if (s_trip, p_orig_trip, q_dest_trip) in valid_voyage_triplets: # Ensure it's a valid pre-calculated route
                            v_annual_voyages[s_trip,p_orig_trip,q_dest_trip] = model.addVar(
                                lb=0, name=f"V_annual_voyages_{s_trip}_{p_orig_trip.replace(' ','_')}_{q_dest_trip.replace(' ','_')}"
                            )
        
        # U_use_ship_type[ship]: Binary variable, 1 if ship type is used (to incur fixed costs)
        u_use_ship_type = {}
        for s_use in SHIPS:
            u_use_ship_type[s_use] = model.addVar(vtype=COPT.BINARY, name=f"U_use_ship_{s_use}")


        # --- NEW CONSTRAINT BLOCK 1: Apply "IF direct_cost < estimated_coastal_cost THEN forbid_coastal" ---
        # This is a hard-coded business rule based on pre-calculated unit costs.
        # Note: This might lead to suboptimal solutions as it doesn't consider global cost effects like fixed cost sharing.
        print("\n--- Applying explicit 'IF direct_cost < estimated_coastal_cost THEN forbid_coastal' constraints ---")
        for i_am in AMS:
            for j_cust in CUSTOMERS:
                demand_val_for_rule = annual_demands.get((i_am, j_cust), 0)
                if demand_val_for_rule <= 0:
                    continue

                C_direct_unit = direct_truck_unit_costs.get((i_am, j_cust), 1e9)
                is_direct_cost_problematic = pd.isna(C_direct_unit) or C_direct_unit == np.inf or \
                                             (C_direct_unit == 0 and demand_val_for_rule > 0) or C_direct_unit < 0
                if is_direct_cost_problematic:
                    C_direct_unit = 1e9

                for s_ship_rule in SHIPS:
                    y_coastal_var_to_constrain_rule = None
                    if s_ship_rule == 'ShipA' and (i_am, j_cust) in y_coastal_shipA_chosen:
                        y_coastal_var_to_constrain_rule = y_coastal_shipA_chosen[(i_am, j_cust)]
                    elif s_ship_rule == 'ShipB' and (i_am, j_cust) in y_coastal_shipB_chosen:
                        y_coastal_var_to_constrain_rule = y_coastal_shipB_chosen[(i_am, j_cust)]

                    if y_coastal_var_to_constrain_rule is None:
                        continue

                    min_estimated_coastal_unit_cost_for_ship = float('inf')
                    base_is_eligible_for_ship_for_cust_rule = True
                    if self.customer_ship_eligibility:
                        am_eligibility_rule = self.customer_ship_eligibility.get(i_am, {})
                        cust_eligibility_rule = am_eligibility_rule.get(j_cust, {})
                        if not cust_eligibility_rule.get(s_ship_rule, True):
                            base_is_eligible_for_ship_for_cust_rule = False
                    
                    if base_is_eligible_for_ship_for_cust_rule:
                        for p_orig_port_rule in PORTS:
                            for q_dest_port_rule in PORTS:
                                if p_orig_port_rule == q_dest_port_rule:
                                    continue

                                fml_c_rule = fml_unit_costs.get((i_am, p_orig_port_rule), float('inf'))
                                lml_c_rule = lml_unit_costs.get((q_dest_port_rule, j_cust), float('inf'))
                                current_ph_cost_rule = 2 * port_handling_cost

                                sea_leg_var_unit_c_rule = float('inf')
                                ship_cap_rule = ship_capacity_map.get(s_ship_rule)
                                sea_trip_var_cost_rule = sea_voyage_var_cost_per_trip.get((s_ship_rule, p_orig_port_rule, q_dest_port_rule))

                                if ship_cap_rule is not None and ship_cap_rule > 0 and \
                                   sea_trip_var_cost_rule is not None and sea_trip_var_cost_rule != float('inf'):
                                    sea_leg_var_unit_c_rule = sea_trip_var_cost_rule / ship_cap_rule
                                else:
                                    continue
                                
                                current_path_total_unit_cost_rule = fml_c_rule + current_ph_cost_rule + lml_c_rule + sea_leg_var_unit_c_rule
                                min_estimated_coastal_unit_cost_for_ship = min(min_estimated_coastal_unit_cost_for_ship, current_path_total_unit_cost_rule)
                    
                    if C_direct_unit < min_estimated_coastal_unit_cost_for_ship and min_estimated_coastal_unit_cost_for_ship != float('inf'):
                        model.addConstr(y_coastal_var_to_constrain_rule == 0,
                                        name=f"ExplicitForbid_Coastal_{s_ship_rule}_{i_am}_{j_cust.replace(' ','_')}")
                        print(f"  CONSTRAINT ADDED (Explicit Forbid): For ({i_am}, {j_cust}), Direct Unit Cost ({C_direct_unit:.2f}) < Min Estimated Coastal Unit Cost for {s_ship_rule} ({min_estimated_coastal_unit_cost_for_ship:.2f}). Forcing y_coastal_{s_ship_rule}_chosen == 0.")
        # --- End of Explicit Forbid Constraint Block ---


        # --- Define objective function (Minimize total cost) ---
        obj = cp.LinExpr()
        # Cost for direct shipments
        for (i_obj_d, j_obj_d), var in x_direct.items():
            cost = direct_truck_unit_costs.get((i_obj_d,j_obj_d), 1e9) # Use high cost if not found
            obj += float(cost) * var
        # Cost for coastal shipments (FML, LML, Port Handling)
        for (i_obj_c,j_obj_c,p_orig_obj_c,q_dest_obj_c,s_obj_c), var in x_coastal.items():
            fml_c = fml_unit_costs.get((i_obj_c,p_orig_obj_c), 1e9)
            lml_c = lml_unit_costs.get((q_dest_obj_c,j_obj_c), 1e9)
            leg_cost = fml_c + (2 * port_handling_cost) + lml_c # Cost per unit for land + port
            obj += float(leg_cost) * var
        # Variable cost for sea voyages
        for (s_obj_v, p_orig_obj_v, q_dest_obj_v), var in v_annual_voyages.items():
            var_c_sea_trip = sea_voyage_var_cost_per_trip.get((s_obj_v, p_orig_obj_v, q_dest_obj_v), 1e9) # Cost per trip
            obj += float(var_c_sea_trip) * var
        # Fixed cost for using ship types
        for s_obj_fc, var in u_use_ship_type.items():
            fixed_c = ship_annual_fixed_cost_map.get(s_obj_fc, 0) # Annual fixed cost
            obj += float(fixed_c) * var
        model.setObjective(obj, sense=COPT.MINIMIZE)

        # --- Define constraints ---

        # Constraints related to mode choice and demand satisfaction (No order splitting for an AM-Customer pair)
        for i_dem in AMS:
            for j_dem in CUSTOMERS:
                demand_val = annual_demands.get((i_dem,j_dem), 0)
                if demand_val > 0:
                    # Constraint 1.1: Each (AM, Cust) must choose one primary transport mode
                    mode_selection_lhs = cp.LinExpr()
                    mode_selection_lhs += y_direct_chosen[i_dem, j_dem]
                    if 'ShipA' in SHIPS and (i_dem, j_dem) in y_coastal_shipA_chosen:
                        mode_selection_lhs += y_coastal_shipA_chosen[i_dem, j_dem]
                    if 'ShipB' in SHIPS and (i_dem, j_dem) in y_coastal_shipB_chosen:
                        mode_selection_lhs += y_coastal_shipB_chosen[i_dem, j_dem]
                    model.addConstr(mode_selection_lhs == 1, name=f"ChooseOneMode_{i_dem}_{j_dem.replace(' ','_')}")

                    # Constraint 1.2: Link direct volume to direct mode choice
                    if (i_dem,j_dem) in x_direct: # Ensure x_direct variable exists
                        model.addConstr(x_direct[i_dem,j_dem] == demand_val * y_direct_chosen[i_dem,j_dem],
                                        name=f"Link_DirectVolume_{i_dem}_{j_dem.replace(' ','_')}")

                    # Constraint 1.3: Link coastal volume for ShipA to ShipA mode choice
                    if 'ShipA' in SHIPS and (i_dem, j_dem) in y_coastal_shipA_chosen:
                        total_coastal_shipA_for_am_cust = cp.LinExpr()
                        for p_orig_dem in PORTS:
                            for q_dest_dem in PORTS:
                                if p_orig_dem != q_dest_dem:
                                    if (i_dem,j_dem,p_orig_dem,q_dest_dem,'ShipA') in x_coastal: # Ensure x_coastal var exists
                                        total_coastal_shipA_for_am_cust += x_coastal[i_dem,j_dem,p_orig_dem,q_dest_dem,'ShipA']
                        model.addConstr(total_coastal_shipA_for_am_cust == demand_val * y_coastal_shipA_chosen[i_dem,j_dem],
                                        name=f"Link_CoastalShipAVolume_{i_dem}_{j_dem.replace(' ','_')}")

                    # Constraint 1.4: Link coastal volume for ShipB to ShipB mode choice
                    if 'ShipB' in SHIPS and (i_dem, j_dem) in y_coastal_shipB_chosen:
                        total_coastal_shipB_for_am_cust = cp.LinExpr()
                        for p_orig_dem in PORTS:
                            for q_dest_dem in PORTS:
                                if p_orig_dem != q_dest_dem:
                                    if (i_dem,j_dem,p_orig_dem,q_dest_dem,'ShipB') in x_coastal: # Ensure x_coastal var exists
                                        total_coastal_shipB_for_am_cust += x_coastal[i_dem,j_dem,p_orig_dem,q_dest_dem,'ShipB']
                        model.addConstr(total_coastal_shipB_for_am_cust == demand_val * y_coastal_shipB_chosen[i_dem,j_dem],
                                        name=f"Link_CoastalShipBVolume_{i_dem}_{j_dem.replace(' ','_')}")
                    
        # AM Supply Capacity Constraint
        for i_cap in AMS: # For each Asset Manager
            lhs_supply = cp.LinExpr()
            # Sum of direct shipments from this AM
            for j_cap_d in CUSTOMERS:
                if (i_cap,j_cap_d) in x_direct:
                    lhs_supply += x_direct[i_cap,j_cap_d]
            # Sum of coastal shipments originating from this AM
            for j_cap_c in CUSTOMERS:
                for p_orig_cap in PORTS:
                    for q_dest_cap in PORTS:
                        if p_orig_cap != q_dest_cap:
                            for s_cap in SHIPS:
                                if (i_cap,j_cap_c,p_orig_cap,q_dest_cap,s_cap) in x_coastal:
                                    lhs_supply += x_coastal[i_cap,j_cap_c,p_orig_cap,q_dest_cap,s_cap]
            model.addConstr(lhs_supply <= self.am_annual_capacity_map[i_cap], name=f"Supply_Capacity_{i_cap}")

        # Sea Route Capacity Constraint (total volume on ship-route <= voyages * ship_capacity)
        for s_sea_cap in SHIPS:
            for p_orig_sea_cap in PORTS:
                for q_dest_sea_cap in PORTS:
                    if p_orig_sea_cap == q_dest_sea_cap: continue # Skip if origin and destination are same
                    
                    voyage_key_constr = (s_sea_cap, p_orig_sea_cap, q_dest_sea_cap)
                    if voyage_key_constr in v_annual_voyages: # Check if voyage variable exists for this route
                        lhs_sea_volume_on_route_ship = cp.LinExpr()
                        # Sum all coastal flows using this ship (s_sea_cap) on this specific route (p_orig_sea_cap -> q_dest_sea_cap)
                        for i_sea_vol in AMS:
                            for j_sea_vol in CUSTOMERS:
                                if (i_sea_vol,j_sea_vol,p_orig_sea_cap,q_dest_sea_cap,s_sea_cap) in x_coastal:
                                    lhs_sea_volume_on_route_ship += x_coastal[i_sea_vol,j_sea_vol,p_orig_sea_cap,q_dest_sea_cap,s_sea_cap]
                        
                        # RHS: Number of voyages on this route * capacity of the ship
                        rhs_capacity = v_annual_voyages[voyage_key_constr] * ship_capacity_map[s_sea_cap]
                        model.addConstr(lhs_sea_volume_on_route_ship <= rhs_capacity,
                                        name=f"Sea_Route_Capacity_{s_sea_cap}_{p_orig_sea_cap.replace(' ','_')}_{q_dest_sea_cap.replace(' ','_')}")

        # Link total voyages of a ship type to its usage (U_use_ship_type) - for fixed costs
        M_large_number_voyages = sum(annual_demands.values()) * 2 if annual_demands else 1e7 # A sufficiently large number
        for s_link_v in SHIPS:
            if s_link_v in u_use_ship_type: # Check if U_use_ship_type variable exists
                total_voyages_for_ship_s = cp.LinExpr()
                # Sum all voyages made by this ship type across all routes
                for p_orig_link in PORTS:
                    for q_dest_link in PORTS:
                        if (s_link_v, p_orig_link, q_dest_link) in v_annual_voyages:
                            total_voyages_for_ship_s += v_annual_voyages[s_link_v, p_orig_link, q_dest_link]
                # If ship is used (U_use_ship_type=1), voyages can be positive; otherwise (U=0), voyages must be 0.
                model.addConstr(total_voyages_for_ship_s <= M_large_number_voyages * u_use_ship_type[s_link_v],
                                name=f"Link_TotalVoyages_To_UseShip_{s_link_v}")
        
        # Max Annual Operating Time Constraint for each ship type
        for s_max_voy_time in SHIPS:
            if s_max_voy_time in u_use_ship_type: # Check if U_use_ship_type variable exists
                total_operational_time_for_ship_s = cp.LinExpr()
                time_per_leg_s = TIME_PER_ONE_WAY_VOYAGE_LEG_DAYS.get(s_max_voy_time)
                
                if time_per_leg_s is None or time_per_leg_s <= 0 or time_per_leg_s == np.inf:
                    print(f"  Warning: Ship type {s_max_voy_time} has invalid one-way voyage time: {time_per_leg_s}. Skipping max voyage time constraint for it.")
                    continue # Skip this ship if time per leg is invalid
                
                # Sum (voyages_on_route * time_per_leg_for_this_ship) across all routes for this ship
                for p_orig_mv_time in PORTS:
                    for q_dest_mv_time in PORTS:
                        if (s_max_voy_time, p_orig_mv_time, q_dest_mv_time) in v_annual_voyages:
                            total_operational_time_for_ship_s += v_annual_voyages[s_max_voy_time, p_orig_mv_time, q_dest_mv_time] * time_per_leg_s
                
                # Total operational time <= ANNUAL_OPERATING_DAYS (if ship is used)
                model.addConstr(total_operational_time_for_ship_s <= ANNUAL_OPERATING_DAYS * u_use_ship_type[s_max_voy_time],
                                name=f"Max_Annual_OpTime_{s_max_voy_time}")

        # --- NEW CONSTRAINT BLOCK 2: Balance round-trip voyages (difference <= 1) ---
        print("\n--- Adding voyage balancing constraints (difference <= 1) ---")
        for s_bal_ship in SHIPS:
            if len(PORTS) >= 2: # Constraint is only relevant if there are at least two ports
                for p1_bal_port, p2_bal_port in itertools.combinations(PORTS, 2): # Iterate over unique pairs of ports
                    key_p1_to_p2 = (s_bal_ship, p1_bal_port, p2_bal_port)
                    key_p2_to_p1 = (s_bal_ship, p2_bal_port, p1_bal_port)

                    # Ensure voyage variables exist for both directions
                    if key_p1_to_p2 in v_annual_voyages and key_p2_to_p1 in v_annual_voyages:
                        voyages_p1_to_p2 = v_annual_voyages[key_p1_to_p2]
                        voyages_p2_to_p1 = v_annual_voyages[key_p2_to_p1]

                        p1_name_safe_bal = p1_bal_port.replace(' ','_')
                        p2_name_safe_bal = p2_bal_port.replace(' ','_')
                        
                        # Create a canonical name for the pair for consistent constraint naming
                        pair_sorted_names_bal = sorted([p1_name_safe_bal, p2_name_safe_bal])
                        constraint_base_name_bal = f"BalanceVoyages_{s_bal_ship}_{pair_sorted_names_bal[0]}_vs_{pair_sorted_names_bal[1]}"

                        # V(P1->P2) - V(P2->P1) == 0
                        model.addConstr(voyages_p1_to_p2 - voyages_p2_to_p1 == 0,
                                        name=f"{constraint_base_name_bal}_Diff1")
                    
                        # print(f"  CONSTRAINT ADDED (Voyage Balance): For {s_bal_ship} between {p1_bal_port} and {p2_bal_port}")

                        # 假设 x_direct 和 x_coastal 已经被正确定义
        # min_am3_shipment = 100000 # 举例，设置一个最小运输量
        # model.addConstr(
        #     cp.quicksum(x_direct[i,j] for (i,j) in x_direct if i == 'AM3') + \
        #     cp.quicksum(x_coastal[i,j,p,q,s] for (i,j,p,q,s) in x_coastal if i == 'AM3') >= min_am3_shipment
        # )
        # --- End of Voyage Balancing Constraint Block ---

        # --- Solve the model ---
        model.setParam("Logging", 1) # Enable COPT solver logging
        model.setParam("TimeLimit", 300) # Set a time limit for the solver (e.g., 300 seconds)

        status_code = -1000 # Default status code
        status_str = "Solver not started"
        try:
            print("\nStarting COPT solver...")
            model.solve()
            status_code = model.status # Get solver status code
            # Interpret status code
            if status_code == COPT.OPTIMAL: status_str = "Optimal (Optimal solution found)"
            elif status_code == COPT.INFEASIBLE: status_str = "Infeasible (Problem has no solution)"
            elif status_code == COPT.UNBOUNDED: status_str = "Unbounded (Problem is unbounded)"
            elif status_code == COPT.TIMEOUT: status_str = "Timeout (Time limit reached, solution might be suboptimal or none)"
            elif status_code == COPT.INTERRUPTED: status_str = "Interrupted (Solving was interrupted)"
            elif status_code == COPT.SUBOPTIMAL: status_str = "Suboptimal (Feasible solution found, but may not be optimal within optimality gap)"
            else: status_str = f"Not Solved or Other Status (COPT status code: {status_code})"
            print(f"COPT solving complete. Status: {status_str}")
        except Exception as e:
            status_str = f"Python level error during COPT solving: {e}"
            status_code = -1001 # Custom code for Python error
            print(f"ERROR during COPT solving: {e}")


        # --- Process and return results ---
        results_q3 = {
            "status": status_str,
            "total_optimal_cost": None,
            "direct_shipments": {},        # {(am, cust): volume}
            "coastal_shipments": {},       # {(am, cust, p_orig, q_dest, ship): volume}
            "voyages_per_route_ship": {},  # {(ship, p_orig, q_dest): num_voyages}
            "ships_used": [],              # [ship_name]
            "total_coastal_volume": 0,
            "total_direct_volume":0,
            "model_solve_status_code": status_code
        }

        # Populate results if a feasible solution was found
        if status_code == COPT.OPTIMAL or status_code == COPT.SUBOPTIMAL or status_code == COPT.TIMEOUT : # Also consider TIMEOUT if solution is available
            if hasattr(model, 'objval') and model.objval is not None: # Check if objval exists and is not None
                results_q3["total_optimal_cost"] = model.objval

            # Get values for direct shipments
            for (i_res_d,j_res_d), var_d in x_direct.items():
                if hasattr(var_d, 'x') and var_d.x > 1e-4: # Check if solution value var_d.x exists and is positive
                    results_q3["direct_shipments"][(i_res_d,j_res_d)] = var_d.x
                    results_q3["total_direct_volume"] += var_d.x
            
            # Get values for coastal shipments
            for (i_res_c,j_res_c,p_orig_c,q_dest_c,s_c), var_c in x_coastal.items():
                if hasattr(var_c, 'x') and var_c.x > 1e-4: # Check if solution value var_c.x exists and is positive
                    key_c = (i_res_c,j_res_c,p_orig_c,q_dest_c,s_c)
                    results_q3["coastal_shipments"][key_c] = var_c.x
                    results_q3["total_coastal_volume"] += var_c.x
            
            # Get list of used ships
            for s_res_u, var_u in u_use_ship_type.items():
                if hasattr(var_u, 'x') and var_u.x > 0.5: # Binary variable, check if close to 1
                    results_q3["ships_used"].append(s_res_u)
            
            # Get number of voyages per route
            for (s_res_v, p_res_v, q_res_v), var_v in v_annual_voyages.items():
                 if hasattr(var_v, 'x') and var_v.x > 1e-4: # Check if solution value var_v.x exists and is positive
                    results_q3["voyages_per_route_ship"][(s_res_v,p_res_v,q_res_v)] = var_v.x
        
        return results_q3