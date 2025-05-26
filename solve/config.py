import numpy as np
import pandas as pd

# --- Fixed Parameters (based on tables/project description in the project PDF) ---
# Exhibit 8: Trucking Costs
TRUCK_FIXED_COST = 185.69      # USD/trip (fixed cost per trip)
TRUCK_VARIABLE_COST_KM = 1.46  # USD/km (variable cost per kilometer)
TRUCK_CAPACITY = 8             # Standard Vehicle Units (capacity per truck)

# Exhibit 6: Road Distance from Automotive Manufacturing Clusters to Ports (km)
AM_TO_PORT_DIST = {
    'AM1': {'Chennai Port': 38.0, 'Pipavav Port': 2043.0},   # AM1 (Chennai) to various ports
    'AM2': {'Chennai Port': 2169.0, 'Pipavav Port': 1198.0}, # AM2 (NCR) to various ports
    'AM3': {'Chennai Port': 1869.0, 'Pipavav Port': 292.0}   # AM3 (Sanand) to various ports
}

# Exhibit 7: RoRo Ship Characteristics
SHIPS_DATA = {
    'ShipA': { # Corresponds to Ship 1 / Type A Ship
        'capacity': 800,                   # Standard Vehicle Units (ship capacity)
        'speed_avg_knots': 13,             # Knots (nautical miles/hour) (average speed)
        'vcp_usd_day': 3467,               # Variable Cost at Port (USD/day)
        'vcs_usd_day': 3218,               # Variable Cost at Sea (USD/day)
        'fixed_cost_3months_usd': 268366.0 # Fixed Cost (USD/3 months)
    },
    'ShipB': { # Corresponds to Ship 2 / Type B Ship
        'capacity': 3518,                  # Standard Vehicle Units (ship capacity)
        'speed_avg_knots': 17,             # Knots (average speed)
        'vcp_usd_day': 15925,              # Variable Cost at Port (USD/day)
        'vcs_usd_day': 6568,               # Variable Cost at Sea (USD/day)
        'fixed_cost_3months_usd': 536778.0 # Fixed Cost (USD/3 months)
    }
}

# Exhibit 5: Port-related Data
PORT_HANDLING_CHARGE_USD_UNIT = 2.0    # USD/vehicle (port handling charge per unit vehicle)
PORT_STAY_SHIP_DAYS_COSTING = 1.0      # Days/port call (ship stay days at each port for loading or unloading)

# Exhibit 4: Sea Distance
SEA_DISTANCE_NM_ONE_WAY = 1394.0  # Nautical miles one way between Chennai Port and Pipavav Port
NM_TO_KM = 1.852 # Nautical mile to kilometer conversion factor

# Exhibit 3: Plant Annual Capacity (Assuming Capacity is annual supply upper limit)
AM_CAPACITY_ANNUAL = {
    'AM1': 1240000, # AM1 annual capacity
    'AM2': 1830000, # AM2 annual capacity
    'AM3': 1300000  # AM3 annual capacity
}

ANNUAL_OPERATING_DAYS = 365 # Total annual operating days (based on actual days)

# --- Dynamically Calculated Parameters ---

# Calculate the time (days) required for a one-way voyage for each ship type
# Includes: loading time at origin port + sea transit time + unloading time at destination port
TIME_PER_ONE_WAY_VOYAGE_LEG_DAYS = {}
print("--- Calculating Time Required for One-Way Voyage (TIME_PER_ONE_WAY_VOYAGE_LEG_DAYS) ---")
for ship_name, data in SHIPS_DATA.items():
    if data['speed_avg_knots'] is None or data['speed_avg_knots'] == 0:
        sea_transit_time_days = np.inf
        print(f"  Warning: Ship type {ship_name} speed is 0 or invalid, sea transit time set to infinity.")
    else:
        # One-way sea transit time (days) = Distance / Speed (knots/hour) / 24 hours
        sea_transit_time_days = SEA_DISTANCE_NM_ONE_WAY / data['speed_avg_knots'] / 24
    
    # Total time for one-way voyage = Loading time at origin port + Sea transit time + Unloading time at destination port
    # Assume PORT_STAY_SHIP_DAYS_COSTING (1.0 day) is the time required for operations (loading or unloading) at each port call
    loading_time_days = PORT_STAY_SHIP_DAYS_COSTING
    unloading_time_days = PORT_STAY_SHIP_DAYS_COSTING
    
    total_leg_time = loading_time_days + sea_transit_time_days + unloading_time_days
    TIME_PER_ONE_WAY_VOYAGE_LEG_DAYS[ship_name] = total_leg_time
    print(f"  Ship Type {ship_name}:")
    print(f"    One-way sea transit time: {sea_transit_time_days:.2f} days")
    print(f"    Loading time: {loading_time_days:.2f} days")
    print(f"    Unloading time: {unloading_time_days:.2f} days")
    print(f"    Total time for one-way voyage: {total_leg_time:.2f} days")

# Placeholder for Customer Ship Eligibility (Ai, Bi, Ci like logic)
# Structure: {'AM_ID': {'Customer_ID': {'Ship_ID': True/False}}}
# True means eligible, False means ineligible.
# If a customer or AM is not listed, or a ship type is not listed for them, it defaults to eligible.
# Currently an empty dictionary, meaning all combinations are eligible by default.
CUSTOMER_SHIP_ELIGIBILITY = {}

# --- Helper Functions ---
def get_optimized_land_unit_cost(distance_km):
    """
    Calculates the unit cost for a land transport leg (simplified for the optimization model, based on average cost of a fully loaded truck).
    """
    if pd.isna(distance_km) or distance_km < 0: return 1e9 # Invalid distance set to a very high cost
    # Check if TRUCK_CAPACITY is zero or NaN to avoid division by zero error
    if TRUCK_CAPACITY is None or TRUCK_CAPACITY == 0 or pd.isna(TRUCK_CAPACITY):
        return 1e9
    return (TRUCK_FIXED_COST + TRUCK_VARIABLE_COST_KM * distance_km) / TRUCK_CAPACITY

def calculate_land_leg_cost_per_unit(distance_km, total_demand_for_leg):
    """
    Calculates the unit vehicle transport cost for a given land distance and demand volume. (This function is mainly used by Q2)
    Parameters:
        distance_km (float): Land distance (kilometers).
        total_demand_for_leg (float): Total demand for this transport leg (number of vehicles).
    Returns:
        float: Unit vehicle land transport cost (USD/vehicle), or infinity if not calculable.
    """
    if pd.isna(total_demand_for_leg) or total_demand_for_leg <= 0 or pd.isna(distance_km) or distance_km < 0 :
        return np.inf
    if TRUCK_CAPACITY == 0: # Avoid division by zero error
        return np.inf
    num_trips = np.ceil(total_demand_for_leg / TRUCK_CAPACITY)
    total_cost_for_leg = num_trips * (TRUCK_FIXED_COST + (TRUCK_VARIABLE_COST_KM * distance_km))
    return total_cost_for_leg / total_demand_for_leg

def calculate_sea_leg_cost_per_unit(ship_name, sea_dist_nm, port_stay_for_vcp_calc_days):
    """
    Calculates the one-way sea transport unit cost for a given ship type and sea distance. (This function is mainly used by Q2 for heuristic cost estimation)
    The Q3 model handles variable and fixed sea costs internally in a more detailed manner.
    Parameters:
        ship_name (str): Ship type name ('ShipA' or 'ShipB').
        sea_dist_nm (float): One-way sea distance (nautical miles).
        port_stay_for_vcp_calc_days (float): Port stay days used for VCP calculation (for Q2 estimation, this usually refers to total port stay affecting one-way cost).
    Returns:
        float: Unit vehicle sea transport cost (USD/vehicle), or infinity if not calculable.
    """
    ship = SHIPS_DATA[ship_name]
    if ship['speed_avg_knots'] == 0 or ship['capacity'] == 0:
        return np.inf

    voyage_time_days = sea_dist_nm / ship['speed_avg_knots'] / 24 # Sea transit time
    vcs_total_one_leg = voyage_time_days * ship['vcs_usd_day'] # Variable cost at sea
    # For Q2 estimation, port_stay_for_vcp_calc_days is usually the total port stay affecting this leg's cost (e.g., origin port + destination port)
    vcp_total_one_leg_ports = port_stay_for_vcp_calc_days * ship['vcp_usd_day'] # Variable cost at port
    total_variable_op_cost_one_leg = vcs_total_one_leg + vcp_total_one_leg_ports

    # The following fixed cost allocation logic is specific to Q2 heuristic estimation
    # The Q3 model directly adds annual fixed costs to the objective function via u_use_ship_type
    # A complete round trip cycle time = 2 * sea transit time + 2 * port stay time (assuming PORT_STAY_SHIP_DAYS_COSTING per port call)
    round_trip_cycle_time_days = (voyage_time_days * 2) + (PORT_STAY_SHIP_DAYS_COSTING * 2)
    if round_trip_cycle_time_days == 0: return np.inf

    # Assume for Q2 estimation, fixed costs are amortized over a 90-day (3-month) period
    num_round_trips_3_months = 90 / round_trip_cycle_time_days
    if num_round_trips_3_months == 0: return np.inf

    fixed_cost_per_round_trip = ship['fixed_cost_3months_usd'] / num_round_trips_3_months
    fixed_cost_per_one_way_leg = fixed_cost_per_round_trip / 2
    fixed_cost_per_unit_one_leg = fixed_cost_per_one_way_leg / ship['capacity']

    variable_cost_per_unit_one_leg = total_variable_op_cost_one_leg / ship['capacity']

    return variable_cost_per_unit_one_leg + fixed_cost_per_unit_one_leg

# SEA_COST_SHIP_A and SEA_COST_SHIP_B are primarily used by Q2Solver. Q3 calculates costs internally within the model.
# Here, it's assumed that the port_stay_for_vcp_calc_days parameter in the Q2 context represents the total port stay days affecting one-way cost (e.g., 1 day at origin + 1 day at destination = 2 days)
SEA_COST_SHIP_A = calculate_sea_leg_cost_per_unit('ShipA', SEA_DISTANCE_NM_ONE_WAY, PORT_STAY_SHIP_DAYS_COSTING * 2)
SEA_COST_SHIP_B = calculate_sea_leg_cost_per_unit('ShipB', SEA_DISTANCE_NM_ONE_WAY, PORT_STAY_SHIP_DAYS_COSTING * 2)