import numpy as np
import pandas as pd

# --- 固定参数 (根据项目PDF中的表格/项目描述) ---
# Exhibit 8: 汽车拖车费用
TRUCK_FIXED_COST = 185.69      # 美元/次运输 (每次行程的固定成本)
TRUCK_VARIABLE_COST_KM = 1.46  # 美元/公里 (每公里可变成本)
TRUCK_CAPACITY = 8             # 标准车辆数 (每辆拖车的运载能力)

# Exhibit 6: 汽车制造集群到港口的公路距离 (公里)
AM_TO_PORT_DIST = {
    'AM1': {'Chennai Port': 38.0, 'Pipavav Port': 2043.0},   # AM1 (金奈) 到各港口的距离
    'AM2': {'Chennai Port': 2169.0, 'Pipavav Port': 1198.0}, # AM2 (NCR) 到各港口的距离
    'AM3': {'Chennai Port': 1869.0, 'Pipavav Port': 292.0}   # AM3 (萨纳恩德) 到各港口的距离
}

# Exhibit 7: 滚装船特性
SHIPS_DATA = {
    'ShipA': { # 对应 Ship 1 / A型船
        'capacity': 800,                   # 标准车辆数 (船舶容量)
        'speed_avg_knots': 13,             # 节 (海里/小时) (平均航速)
        'vcp_usd_day': 3467,               # 在港运营成本(美元/天)
        'vcs_usd_day': 3218,               # 在海运营成本(美元/天)
        'fixed_cost_3months_usd': 268366.0 # 固定成本(美元/3个月)
    },
    'ShipB': { # 对应 Ship 2 / B型船
        'capacity': 3518,                  # 标准车辆数 (船舶容量)
        'speed_avg_knots': 17,             # 节 (平均航速)
        'vcp_usd_day': 15925,              # 在港运营成本(美元/天)
        'vcs_usd_day': 6568,               # 在海运营成本(美元/天)
        'fixed_cost_3months_usd': 536778.0 # 固定成本(美元/3个月)
    }
}

# Exhibit 5: 港口相关数据
PORT_HANDLING_CHARGE_USD_UNIT = 2.0    # 美元/车 (单位车辆的港口货物处理费)
PORT_STAY_SHIP_DAYS_COSTING = 1.0      # 天/次挂靠 (船舶在每个停靠港口的停留天数，用于装货或卸货)

# Exhibit 4: 海运距离
SEA_DISTANCE_NM_ONE_WAY = 1394.0  # 金奈港与皮帕瓦沃港之间的单程海里数
NM_TO_KM = 1.852 # 海里转换为公里的系数

# Exhibit 3: 工厂年产能 (假设 Capacity 是年供应上限)
AM_CAPACITY_ANNUAL = {
    'AM1': 1240000, # AM1 年产能
    'AM2': 1830000, # AM2 年产能
    'AM3': 1300000  # AM3 年产能
}

ANNUAL_OPERATING_DAYS = 365 # 年总运营天数 (按实际天数)

# --- 动态计算的参数 ---

# 计算每个船型单向航程所需的时间（天）
# 包括：始发港装货时间 + 海上运输时间 + 目的港卸货时间
TIME_PER_ONE_WAY_VOYAGE_LEG_DAYS = {}
print("--- 计算单向航程所需时间 (TIME_PER_ONE_WAY_VOYAGE_LEG_DAYS) ---")
for ship_name, data in SHIPS_DATA.items():
    if data['speed_avg_knots'] is None or data['speed_avg_knots'] == 0:
        sea_transit_time_days = np.inf
        print(f"  警告: 船型 {ship_name} 速度为0或无效，海上运输时间设为无穷大。")
    else:
        # 海上单向运输时间（天）= 距离 / 速度(海里/小时) / 24小时
        sea_transit_time_days = SEA_DISTANCE_NM_ONE_WAY / data['speed_avg_knots'] / 24
    
    # 单向航程的总时间 = 在始发港装货时间 + 海上运输时间 + 在目的港卸货时间
    # 假设 PORT_STAY_SHIP_DAYS_COSTING (1.0天) 是每次停靠港口进行作业（装货或卸货）所需的时间
    loading_time_days = PORT_STAY_SHIP_DAYS_COSTING
    unloading_time_days = PORT_STAY_SHIP_DAYS_COSTING
    
    total_leg_time = loading_time_days + sea_transit_time_days + unloading_time_days
    TIME_PER_ONE_WAY_VOYAGE_LEG_DAYS[ship_name] = total_leg_time
    print(f"  船型 {ship_name}:")
    print(f"    海上单向运输时间: {sea_transit_time_days:.2f} 天")
    print(f"    装货时间: {loading_time_days:.2f} 天")
    print(f"    卸货时间: {unloading_time_days:.2f} 天")
    print(f"    单向航程总耗时: {total_leg_time:.2f} 天")

# Placeholder for Customer Ship Eligibility (Ai, Bi, Ci like logic)
# 结构: {'AM_ID': {'Customer_ID': {'Ship_ID': True/False}}}
# True 表示合格, False 表示不合格。
# 如果客户或AM未列出，或船型未为其列出，则默认为合格。
# 目前为空字典，表示所有组合默认合格。
CUSTOMER_SHIP_ELIGIBILITY = {}

# --- 辅助函数 ---
def get_optimized_land_unit_cost(distance_km):
    """
    计算陆路运输段的单位成本 (为优化模型简化，基于单车满载的平均成本)。
    """
    if pd.isna(distance_km) or distance_km < 0: return 1e9 # 无效距离设为极高成本
    # 检查 TRUCK_CAPACITY 是否为零或NaN以避免除零错误
    if TRUCK_CAPACITY is None or TRUCK_CAPACITY == 0 or pd.isna(TRUCK_CAPACITY):
        return 1e9
    return (TRUCK_FIXED_COST + TRUCK_VARIABLE_COST_KM * distance_km) / TRUCK_CAPACITY

def calculate_land_leg_cost_per_unit(distance_km, total_demand_for_leg):
    """
    计算给定陆路距离和需求量下的单位车辆运输成本。 (此函数主要被Q2使用)
    参数:
        distance_km (float): 陆路距离 (公里)。
        total_demand_for_leg (float): 该运输段的总需求量 (车辆数)。
    返回:
        float: 单位车辆的陆路运输成本 (美元/车)，若无法计算则为无穷大。
    """
    if pd.isna(total_demand_for_leg) or total_demand_for_leg <= 0 or pd.isna(distance_km) or distance_km < 0 :
        return np.inf
    if TRUCK_CAPACITY == 0: # 避免除零错误
        return np.inf
    num_trips = np.ceil(total_demand_for_leg / TRUCK_CAPACITY)
    total_cost_for_leg = num_trips * (TRUCK_FIXED_COST + (TRUCK_VARIABLE_COST_KM * distance_km))
    return total_cost_for_leg / total_demand_for_leg

def calculate_sea_leg_cost_per_unit(ship_name, sea_dist_nm, port_stay_for_vcp_calc_days):
    """
    计算给定船型在特定海运距离下的单程海运单位成本。(此函数主要被Q2用于启发式成本估算)
    Q3模型在内部以更细致的方式处理可变和固定海运成本。
    参数:
        ship_name (str): 船型名称 ('ShipA' 或 'ShipB')。
        sea_dist_nm (float): 单程海运距离 (海里)。
        port_stay_for_vcp_calc_days (float): 用于计算VCP的港口停留天数 (Q2估算时，这通常指影响单程成本的港口总停留)。
    返回:
        float: 单位车辆的海运成本 (美元/车)，若无法计算则为无穷大。
    """
    ship = SHIPS_DATA[ship_name]
    if ship['speed_avg_knots'] == 0 or ship['capacity'] == 0:
        return np.inf

    voyage_time_days = sea_dist_nm / ship['speed_avg_knots'] / 24 # 海上运输时间
    vcs_total_one_leg = voyage_time_days * ship['vcs_usd_day'] # 海上可变成本
    # 对于Q2的估算，port_stay_for_vcp_calc_days 通常是影响该段成本的总港口停留（如始发港+目的港）
    vcp_total_one_leg_ports = port_stay_for_vcp_calc_days * ship['vcp_usd_day'] # 港口可变成本
    total_variable_op_cost_one_leg = vcs_total_one_leg + vcp_total_one_leg_ports

    # 以下固定成本摊销逻辑是Q2启发式估算特有的
    # Q3模型直接将年度固定成本通过 u_use_ship_type 加入目标函数
    # 一个完整的往返航程周期 = 2 * 海上运输时间 + 2 * 港口停留时间 (假设每次挂靠都停留 PORT_STAY_SHIP_DAYS_COSTING)
    round_trip_cycle_time_days = (voyage_time_days * 2) + (PORT_STAY_SHIP_DAYS_COSTING * 2)
    if round_trip_cycle_time_days == 0: return np.inf

    # 假设Q2估算时，固定成本按90天（3个月）周期摊销
    num_round_trips_3_months = 90 / round_trip_cycle_time_days
    if num_round_trips_3_months == 0: return np.inf

    fixed_cost_per_round_trip = ship['fixed_cost_3months_usd'] / num_round_trips_3_months
    fixed_cost_per_one_way_leg = fixed_cost_per_round_trip / 2
    fixed_cost_per_unit_one_leg = fixed_cost_per_one_way_leg / ship['capacity']

    variable_cost_per_unit_one_leg = total_variable_op_cost_one_leg / ship['capacity']

    return variable_cost_per_unit_one_leg + fixed_cost_per_unit_one_leg

# SEA_COST_SHIP_A 和 SEA_COST_SHIP_B 主要由 Q2Solver 使用，Q3 在模型内部计算成本。
# 这里假设 port_stay_for_vcp_calc_days 参数在 Q2 上下文中代表影响单程成本的总港口停留天数（例如，始发港1天 + 目的港1天 = 2天）
SEA_COST_SHIP_A = calculate_sea_leg_cost_per_unit('ShipA', SEA_DISTANCE_NM_ONE_WAY, PORT_STAY_SHIP_DAYS_COSTING * 2)
SEA_COST_SHIP_B = calculate_sea_leg_cost_per_unit('ShipB', SEA_DISTANCE_NM_ONE_WAY, PORT_STAY_SHIP_DAYS_COSTING * 2)