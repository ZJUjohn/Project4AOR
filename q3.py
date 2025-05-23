import pandas as pd
import numpy as np
import coptpy as cp
from coptpy import COPT

# --- 固定参数 (根据项目PDF中的表格/项目描述) ---
# Exhibit 8: 汽车拖车费用
TRUCK_FIXED_COST = 185.69  # 美元/次运输 (每次行程的固定成本)
TRUCK_VARIABLE_COST_KM = 1.46  # 美元/公里 (每公里可变成本)
TRUCK_CAPACITY = 8  # 标准车辆数 (每辆拖车的运载能力)

# Exhibit 6: 汽车制造集群到港口的公路距离 (公里)
AM_TO_PORT_DIST = {
    'AM1': {'Chennai Port': 38.0, 'Pipavav Port': 2043.0},  # AM1 (金奈) 到各港口的距离
    'AM2': {'Chennai Port': 2169.0, 'Pipavav Port': 1198.0}, # AM2 (NCR) 到各港口的距离
    'AM3': {'Chennai Port': 1869.0, 'Pipavav Port': 292.0}   # AM3 (萨纳恩德) 到各港口的距离
}

# Exhibit 7: 滚装船特性
SHIPS_DATA = {
    'ShipA': { # 对应 Ship 1 / A型船
        'capacity': 800,  # 标准车辆数 (船舶容量)
        'speed_avg_knots': 13,  # 节 (海里/小时) (平均航速)
        'vcp_usd_day': 3467,  # 在港运营成本(美元/天)
        'vcs_usd_day': 3218,  # 在海运营成本(美元/天)
        'fixed_cost_3months_usd': 268366.0 # 固定成本(美元/3个月)
    },
    'ShipB': { # 对应 Ship 2 / B型船
        'capacity': 3518, # 标准车辆数 (船舶容量)
        'speed_avg_knots': 17,  # 节 (平均航速)
        'vcp_usd_day': 15925, # 在港运营成本(美元/天)
        'vcs_usd_day': 6568,  # 在海运营成本(美元/天)
        'fixed_cost_3months_usd': 536778.0 # 固定成本(美元/3个月)
    }
}

# Exhibit 5: 港口相关数据
PORT_HANDLING_CHARGE_USD_UNIT = 2.0  # 美元/车 (单位车辆的港口货物处理费)
PORT_STAY_SHIP_DAYS_COSTING = 1.0      # 天/次挂靠 (船舶在港停留时间，用于计算VCP船舶在港成本)

# Exhibit 4: 海运距离
SEA_DISTANCE_NM_ONE_WAY = 1394.0  # 金奈港与皮帕瓦沃港之间的单程海里数
NM_TO_KM = 1.852 # 海里转换为公里的系数

# Exhibit 3: 工厂年产能 (假设 Capacity 是年供应上限)
AM_CAPACITY_ANNUAL = {
    'AM1': 1240000, # AM1 年产能
    'AM2': 1830000, # AM2 年产能
    'AM3': 1300000  # AM3 年产能
}

# --- 数据加载与预处理 ---
# 尝试从Excel文件读取数据，如果文件不存在则打印错误信息并退出
try:
    # header=2 表示将Excel文件的第3行（0-indexed）作为表头
    df_locations_input = pd.read_excel("DATA_original.xlsx", sheet_name="Customer Locations", header=2)
    # header=1 表示将Excel文件的第2行作为表头
    df_port_cust_dist_input = pd.read_excel("DATA_original.xlsx",sheet_name="PortCustDist", header=1)
except FileNotFoundError as e:
    print(f"错误：Excel文件 'DATA_original.xlsx' 未找到。请确保文件位于脚本执行目录中：{e}")
    raise # 抛出异常，中断程序执行

# 清洗客户位置数据 (df_locations)
# 重命名列以方便后续代码调用
df_locations_input.columns = ['Cust_Location_Num', 'Customer_Location', 'State_UT',
                        'AM1_Dist', 'AM1_M1', 'AM1_M2', 'AM1_M3',
                        'AM2_Dist', 'AM2_M1', 'AM2_M2', 'AM2_M3',
                        'AM3_Dist', 'AM3_M1', 'AM3_M2', 'AM3_M3']
df_locations_input.dropna(how='all', inplace=True) # 删除所有值为NaN的行

# 将距离和月度需求列转换为数值型，无法转换的则设为NaN
num_cols_loc = ['AM1_Dist', 'AM1_M1', 'AM1_M2', 'AM1_M3',
                'AM2_Dist', 'AM2_M1', 'AM2_M2', 'AM2_M3',
                'AM3_Dist', 'AM3_M1', 'AM3_M2', 'AM3_M3']
for col in num_cols_loc:
    df_locations_input[col] = pd.to_numeric(df_locations_input[col], errors='coerce')

# 清洗港口到客户距离数据 (df_port_cust_dist)
df_port_cust_dist_input.rename(columns={
    'Distance (km)':'Customer_Location_PortFile', # 临时列名，避免在合并前与主表列名冲突
    'Chennai':'Dist_to_Chennai_Port',
    'Pipavav':'Dist_to_Pipavav_Port'
    }, inplace=True)
# 仅保留需要的列，并基于 Customer_Location_PortFile 列删除空值行
df_port_cust_dist_input = df_port_cust_dist_input[['Customer_Location_PortFile', 'Dist_to_Chennai_Port', 'Dist_to_Pipavav_Port']].copy()
df_port_cust_dist_input.dropna(subset=['Customer_Location_PortFile'], inplace=True)
df_port_cust_dist_input['Customer_Location'] = df_port_cust_dist_input['Customer_Location_PortFile'] # 创建用于合并的共同键
# 将距离列转换为数值型
df_port_cust_dist_input['Dist_to_Chennai_Port'] = pd.to_numeric(df_port_cust_dist_input['Dist_to_Chennai_Port'], errors='coerce')
df_port_cust_dist_input['Dist_to_Pipavav_Port'] = pd.to_numeric(df_port_cust_dist_input['Dist_to_Pipavav_Port'], errors='coerce')

# 合并港口距离数据到主位置数据表 (df_locations)
df_locations = pd.merge(df_locations_input, df_port_cust_dist_input[['Customer_Location', 'Dist_to_Chennai_Port', 'Dist_to_Pipavav_Port']],
                        on="Customer_Location", how="left") # 使用左连接保留所有客户地点信息
df_locations['Cust_ID'] = df_locations['Customer_Location'] # 使用 Customer_Location 作为后续模型中的唯一客户ID

# --- 问题1：计算纯公路直接运输成本 ---
def solve_question1(df):
    """
    计算在当前情景下，汽车从工厂通过公路拖车直接运往不同客户地点的出站汽车分销总成本。
    参数:
        df (pd.DataFrame): 包含客户需求和距离等信息的DataFrame。
    返回:
        tuple: (总直接运输成本, 更新了成本详情的DataFrame)
    """
    df_q1 = df.copy() # 创建DataFrame副本进行操作
    for am_id in ['AM1', 'AM2', 'AM3']: # 遍历三个汽车制造集群
        demand_cols = [f'{am_id}_M1', f'{am_id}_M2', f'{am_id}_M3'] # 定义月度需求列名
        # 确保需求列存在且为数值型，缺失值(NaN)填充为0，以便后续加总
        for d_col in demand_cols:
            if d_col in df_q1.columns:
                 df_q1[d_col] = pd.to_numeric(df_q1[d_col], errors='coerce').fillna(0)
            else: # 如果某些AM对应的M1/M2/M3列在输入数据中缺失（理论上不应发生）
                df_q1[d_col] = 0.0

        # 计算3个月的总需求量
        df_q1[f'{am_id}_Total_Demand_3Mo'] = df_q1[demand_cols].sum(axis=1)

        # 筛选出有有效需求量 (>0) 且公路距离有效的客户进行成本计算
        mask_valid = (df_q1[f'{am_id}_Total_Demand_3Mo'] > 0) & (~df_q1[f'{am_id}_Dist'].isna())

        # 初始化将要计算的成本相关列为0.0
        for col_suffix in ['_Direct_Num_Trips', '_Direct_Cost', '_Direct_Cost_Per_Unit']:
            df_q1[f'{am_id}{col_suffix}'] = 0.0

        # 计算运输次数 (向上取整)
        df_q1.loc[mask_valid, f'{am_id}_Direct_Num_Trips'] = np.ceil(
            df_q1.loc[mask_valid, f'{am_id}_Total_Demand_3Mo'] / TRUCK_CAPACITY
        )
        # 计算总直接运输成本 (次数 * (单次固定成本 + 单次可变成本))
        df_q1.loc[mask_valid, f'{am_id}_Direct_Cost'] = df_q1.loc[mask_valid, f'{am_id}_Direct_Num_Trips'] * \
            (TRUCK_FIXED_COST + (TRUCK_VARIABLE_COST_KM * df_q1.loc[mask_valid, f'{am_id}_Dist']))

        # 计算单位车辆直接运输成本 (总成本 / 总需求)，确保需求大于0以避免除零错误
        valid_demand_mask_for_pu = mask_valid & (df_q1.loc[mask_valid, f'{am_id}_Total_Demand_3Mo'] > 0)
        df_q1.loc[valid_demand_mask_for_pu, f'{am_id}_Direct_Cost_Per_Unit'] = \
            df_q1.loc[valid_demand_mask_for_pu, f'{am_id}_Direct_Cost'] / df_q1.loc[valid_demand_mask_for_pu, f'{am_id}_Total_Demand_3Mo']

    # 加总所有AM到所有客户的直接运输成本，得到总成本
    total_direct_trucking_cost = df_q1[[f'AM1_Direct_Cost', f'AM2_Direct_Cost', f'AM3_Direct_Cost']].sum().sum()
    return total_direct_trucking_cost, df_q1

# --- 问题2：识别适合转向沿海运输的客户地点 ---
# 辅助函数：计算单程海运的单位成本（包含可变和分摊的固定成本）
def calculate_sea_leg_cost_per_unit(ship_name, sea_dist_nm, port_stay_for_vcp_calc_days):
    """
    计算给定船型在特定海运距离下的单程海运单位成本。
    参数:
        ship_name (str): 船型名称 ('ShipA' 或 'ShipB')。
        sea_dist_nm (float): 单程海运距离 (海里)。
        port_stay_for_vcp_calc_days (float): 用于计算VCP的港口停留天数 (例如，始发港1天+目的港1天=2天)。
    返回:
        float: 单位车辆的海运成本 (美元/车)，若无法计算则为无穷大。
    """
    ship = SHIPS_DATA[ship_name] # 获取船舶参数
    # 单程航行时间（天）
    voyage_time_days = sea_dist_nm / ship['speed_avg_knots'] / 24

    # 单程可变运营成本：海上VCS成本 + 此航程涉及的港口停留VCP成本
    vcs_total_one_leg = voyage_time_days * ship['vcs_usd_day']
    vcp_total_one_leg_ports = port_stay_for_vcp_calc_days * ship['vcp_usd_day']
    total_variable_op_cost_one_leg = vcs_total_one_leg + vcp_total_one_leg_ports

    # 固定成本分摊：计算3个月内可执行的往返航次数
    # 一次完整往返周期时间 = 2 * 单程航行时间 + 2 * 核心港口停留天数 (例如，金奈和皮帕瓦沃各停1天)
    round_trip_cycle_time_days = (voyage_time_days * 2) + (PORT_STAY_SHIP_DAYS_COSTING * 2) # PORT_STAY_SHIP_DAYS_COSTING 是每次挂靠的停留天数
    if round_trip_cycle_time_days == 0: return np.inf # 避免除以零

    num_round_trips_3_months = 90 / round_trip_cycle_time_days # 假设3个月为90天
    if num_round_trips_3_months == 0: return np.inf

    # 计算分摊到单程航线的固定成本
    fixed_cost_per_round_trip = ship['fixed_cost_3months_usd'] / num_round_trips_3_months
    fixed_cost_per_one_way_leg = fixed_cost_per_round_trip / 2 # 假设固定成本在往返程中平均分摊

    # 单位车辆成本（假设船舶满载以获得最佳单位成本）
    fixed_cost_per_unit_one_leg = fixed_cost_per_one_way_leg / ship['capacity']
    variable_cost_per_unit_one_leg = total_variable_op_cost_one_leg / ship['capacity']

    return variable_cost_per_unit_one_leg + fixed_cost_per_unit_one_leg

# 预先计算好两种船型的单位海运成本（金奈港-皮帕瓦沃港航线）
# 假设一次单程海运，船舶会在始发港和目的港各产生1天的VCP（共2天VCP影响此航程的船公司可变成本）
SEA_COST_SHIP_A = calculate_sea_leg_cost_per_unit('ShipA', SEA_DISTANCE_NM_ONE_WAY, PORT_STAY_SHIP_DAYS_COSTING * 2)
SEA_COST_SHIP_B = calculate_sea_leg_cost_per_unit('ShipB', SEA_DISTANCE_NM_ONE_WAY, PORT_STAY_SHIP_DAYS_COSTING * 2)

# 辅助函数：计算陆路运输段的单位成本
def calculate_land_leg_cost_per_unit(distance_km, total_demand_for_leg):
    """
    计算给定陆路距离和需求量下的单位车辆运输成本。
    参数:
        distance_km (float): 陆路距离 (公里)。
        total_demand_for_leg (float): 该运输段的总需求量 (车辆数)。
    返回:
        float: 单位车辆的陆路运输成本 (美元/车)，若无法计算则为无穷大。
    """
    # 如果需求为0或距离无效/为负，则成本为无穷大（表示此路径不可行或成本极高）
    if pd.isna(total_demand_for_leg) or total_demand_for_leg <= 0 or pd.isna(distance_km) or distance_km < 0 :
        return np.inf
    num_trips = np.ceil(total_demand_for_leg / TRUCK_CAPACITY)
    total_cost_for_leg = num_trips * (TRUCK_FIXED_COST + (TRUCK_VARIABLE_COST_KM * distance_km))
    return total_cost_for_leg / total_demand_for_leg

# 主要逻辑函数：解决问题2
def solve_question2(df):
    """
    识别哪些客户地点适合从纯公路运输转向沿海多式联运。
    参数:
        df (pd.DataFrame): 已包含问题1计算结果（如单位直接运输成本）的DataFrame。
    返回:
        tuple: (更新了多式联运成本和适用性判断的DataFrame, 用于打印的摘要DataFrame)
    """
    df_q2 = df.copy() # 使用已包含Q1结果的DataFrame副本
    results_q2_summary_list = [] # 用于存储供打印的摘要信息

    for index, row in df_q2.iterrows(): # 遍历每个客户地点
        cust_loc = row['Customer_Location']
        summary_row = {'Customer_Location': cust_loc} # 当前客户的摘要行

        for am_id in ['AM1', 'AM2', 'AM3']: # 遍历每个制造集群
            # 获取该AM到此客户的纯公路运输单位成本 (从Q1结果中获取)
            direct_cost_pu = row.get(f'{am_id}_Direct_Cost_Per_Unit', np.inf)
            # 如果单位成本为0但需求也为0，则视为无穷大，避免错误判断
            if direct_cost_pu == 0 and row.get(f'{am_id}_Total_Demand_3Mo', 0) == 0:
                direct_cost_pu = np.inf

            demand_for_am = row.get(f'{am_id}_Total_Demand_3Mo', 0) # 获取3个月的需求量
            summary_row[f'{am_id}_Direct_Cost_PU'] = direct_cost_pu

            # 如果该AM对此客户无需求，则海运不适用
            if demand_for_am == 0:
                for ship_letter in ['A', 'B']: # 为两种船型记录不适用状态
                    df_q2.loc[index, f'{am_id}_MM_Cost_PU_Ship{ship_letter}'] = np.inf
                    df_q2.loc[index, f'{am_id}_Suitable_Ship{ship_letter}'] = "N/A (无需求)"
                    summary_row[f'{am_id}_Min_MM_Cost_PU_Ship{ship_letter}'] = np.inf
                    summary_row[f'{am_id}_Suitable_Ship{ship_letter}'] = "N/A (无需求)"
                continue # 跳到下一个AM

            # 初始化此AM到此客户的最低多式联运成本 (分别为船A和船B)
            min_mm_cost_ship_a_for_am_cust = np.inf
            min_mm_cost_ship_b_for_am_cust = np.inf

            # 考虑两种主要的海运目的港选项：金奈港 或 皮帕瓦沃港
            for dest_port_option in ['Chennai Port', 'Pipavav Port']:
                origin_port_am = '' # 该AM的始发港
                fml_dist_am = np.nan # 第一英里距离 (工厂到始发港)

                # 根据AM确定其最合理的始发港
                if am_id == 'AM1': # AM1在金奈，始发港通常是金奈港
                    origin_port_am = 'Chennai Port'
                elif am_id == 'AM2': # AM2在NCR，判断离哪个主要港口更近
                    origin_port_am = 'Pipavav Port' if AM_TO_PORT_DIST['AM2']['Pipavav Port'] < AM_TO_PORT_DIST['AM2']['Chennai Port'] else 'Chennai Port'
                elif am_id == 'AM3': # AM3在Sanand，非常靠近皮帕瓦沃港
                    origin_port_am = 'Pipavav Port'

                fml_dist_am = AM_TO_PORT_DIST[am_id][origin_port_am] # 获取第一英里距离
                fml_cost_pu = calculate_land_leg_cost_per_unit(fml_dist_am, demand_for_am) # 计算第一英里单位成本

                # 获取从当前考虑的目的港到客户的最后一英里距离
                lml_dist_port_cust = np.nan
                if dest_port_option == 'Chennai Port':
                    lml_dist_port_cust = row.get('Dist_to_Chennai_Port', np.inf) # np.inf作为无效距离的默认值
                else: # dest_port_option == 'Pipavav Port'
                    lml_dist_port_cust = row.get('Dist_to_Pipavav_Port', np.inf)

                lml_cost_pu = calculate_land_leg_cost_per_unit(lml_dist_port_cust, demand_for_am) # 计算最后一英里单位成本

                # 初始化当前路径的多式联运成本
                current_mm_cost_ship_a_path = np.inf
                current_mm_cost_ship_b_path = np.inf

                # 只有当始发港和目的港不同时，才存在海运段，才计算海运成本
                if origin_port_am != dest_port_option:
                    # 获取预计算的单位海运成本 (金奈-皮帕瓦沃航线，假设成本对称)
                    sea_cost_ship_a_leg = SEA_COST_SHIP_A
                    sea_cost_ship_b_leg = SEA_COST_SHIP_B

                    # 计算此完整路径的总多式联运单位成本
                    current_mm_cost_ship_a_path = (fml_cost_pu +
                                                 PORT_HANDLING_CHARGE_USD_UNIT + # 始发港操作费
                                                 sea_cost_ship_a_leg +          # 海运费
                                                 PORT_HANDLING_CHARGE_USD_UNIT + # 目的港操作费
                                                 lml_cost_pu)                   # 最后一英里运输费
                    current_mm_cost_ship_b_path = (fml_cost_pu +
                                                 PORT_HANDLING_CHARGE_USD_UNIT +
                                                 sea_cost_ship_b_leg +
                                                 PORT_HANDLING_CHARGE_USD_UNIT +
                                                 lml_cost_pu)

                # 更新此AM到此客户通过不同路径的最低多式联运成本
                min_mm_cost_ship_a_for_am_cust = min(min_mm_cost_ship_a_for_am_cust, current_mm_cost_ship_a_path)
                min_mm_cost_ship_b_for_am_cust = min(min_mm_cost_ship_b_for_am_cust, current_mm_cost_ship_b_path)

            # 将找到的最低多式联运成本和适用性判断结果存入主DataFrame
            df_q2.loc[index, f'{am_id}_MM_Cost_PU_ShipA'] = min_mm_cost_ship_a_for_am_cust
            df_q2.loc[index, f'{am_id}_Suitable_ShipA'] = "适合" if min_mm_cost_ship_a_for_am_cust < direct_cost_pu else "不适合"
            if min_mm_cost_ship_a_for_am_cust == np.inf: # 如果没有找到有效的海运路径
                 df_q2.loc[index, f'{am_id}_Suitable_ShipA'] = "N/A (无有效海运路径)"


            df_q2.loc[index, f'{am_id}_MM_Cost_PU_ShipB'] = min_mm_cost_ship_b_for_am_cust
            df_q2.loc[index, f'{am_id}_Suitable_ShipB'] = "适合" if min_mm_cost_ship_b_for_am_cust < direct_cost_pu else "不适合"
            if min_mm_cost_ship_b_for_am_cust == np.inf: # 如果没有找到有效的海运路径
                 df_q2.loc[index, f'{am_id}_Suitable_ShipB'] = "N/A (无有效海运路径)"

            # 填充用于打印的摘要信息
            summary_row[f'{am_id}_Min_MM_Cost_PU_ShipA'] = min_mm_cost_ship_a_for_am_cust
            summary_row[f'{am_id}_Suitable_ShipA'] = df_q2.loc[index, f'{am_id}_Suitable_ShipA']
            summary_row[f'{am_id}_Min_MM_Cost_PU_ShipB'] = min_mm_cost_ship_b_for_am_cust
            summary_row[f'{am_id}_Suitable_ShipB'] = df_q2.loc[index, f'{am_id}_Suitable_ShipB']

        results_q2_summary_list.append(summary_row) # 添加此客户的完整AM分析摘要

    return df_q2, pd.DataFrame(results_q2_summary_list) # 返回完整DataFrame和用于打印的摘要DataFrame

# --- 问题3：优化沿海运输系统设计 (使用 COPT) ---
def solve_question3_copt(df_with_q1_costs_input, am_annual_capacity_map, ships_data_map, cost_params_map):
    """
    使用COPT优化器解决沿海运输系统设计和装运计划问题。
    参数:
        df_with_q1_costs_input (pd.DataFrame): 包含客户、需求和Q1计算的直接运输成本的DataFrame。
        am_annual_capacity_map (dict): AM的年供应能力。
        ships_data_map (dict): 船舶参数。
        cost_params_map (dict): 其他成本参数，如港口操作费。
    返回:
        dict: 包含求解状态、最优成本和决策变量值的优化结果。
    """
    print("\n--- 开始执行问题3：COPT优化模型 ---")

    # 1. 准备模型所需参数
    AMS = ['AM1', 'AM2', 'AM3'] # 汽车制造集群列表
    df_model_input = df_with_q1_costs_input.copy() # 复制输入数据以避免修改原始数据

    # 确保 Cust_ID 列存在且唯一，作为客户的唯一标识符
    if 'Cust_ID' not in df_model_input.columns:
        df_model_input['Cust_ID'] = df_model_input['Customer_Location']
    CUSTOMERS = df_model_input['Cust_ID'].dropna().unique().tolist() # 获取客户列表，去除NaN

    PORTS = ['Chennai Port', 'Pipavav Port'] # 主要港口列表
    SHIPS = list(ships_data_map.keys()) # 船型列表, e.g., ['ShipA', 'ShipB']
    # 定义有效的海运航线 (双向)
    SEA_ROUTES_TUPLES = [(p_orig, p_dest) for p_orig in PORTS for p_dest in PORTS if p_orig != p_dest]

    # 年化需求 D[(am, cust)]
    annual_demands = {}
    for idx, row in df_model_input.iterrows():
        cust_id = row['Cust_ID']
        if pd.isna(cust_id): continue # 跳过没有客户ID的行
        for am in AMS:
            # 3个月需求 * 4 = 年需求
            annual_demands[(am, cust_id)] = row.get(f'{am}_Total_Demand_3Mo', 0) * 4

    # 单位运输成本字典初始化
    direct_truck_unit_costs = {} # C_direct[(am, cust)]：直接卡车运输单位成本
    fml_unit_costs = {}          # C_fml[(am, port)]：第一英里单位成本 (工厂到始发港)
    lml_unit_costs = {}          # C_lml[(port, cust)]：最后一英里单位成本 (目的港到客户)

    # 陆路运输单位成本计算函数 (为优化模型简化，基于单车满载的平均成本)
    def get_optimized_land_unit_cost(distance_km):
        if pd.isna(distance_km) or distance_km < 0: return 1e9 # 无效距离设为极高成本
        # 检查 TRUCK_CAPACITY 是否为零或NaN以避免除零错误
        if TRUCK_CAPACITY is None or TRUCK_CAPACITY == 0 or pd.isna(TRUCK_CAPACITY):
            return 1e9
        return (TRUCK_FIXED_COST + TRUCK_VARIABLE_COST_KM * distance_km) / TRUCK_CAPACITY

    # 计算各陆路段的单位成本
    for idx, row in df_model_input.iterrows():
        cust_id = row['Cust_ID']
        if pd.isna(cust_id): continue
        for am in AMS:
            cost = row.get(f'{am}_Direct_Cost_Per_Unit', np.inf) # 从Q1结果获取
            # 如果单位成本为0但实际有需求，说明Q1计算可能有误，调整为极高成本
            direct_truck_unit_costs[(am, cust_id)] = cost if (cost > 0 and cost != np.inf) or annual_demands.get((am,cust_id),0) == 0 else 1e9

    for am in AMS:
        for port in PORTS:
            fml_unit_costs[(am, port)] = get_optimized_land_unit_cost(AM_TO_PORT_DIST[am][port])

    for idx, row in df_model_input.iterrows():
        cust_id = row['Cust_ID']
        if pd.isna(cust_id): continue
        for port in PORTS:
            dist_col = 'Dist_to_Chennai_Port' if port == 'Chennai Port' else 'Dist_to_Pipavav_Port'
            lml_unit_costs[(port, cust_id)] = get_optimized_land_unit_cost(row.get(dist_col))

    port_handling_cost = cost_params_map['port_handling_charge_usd_unit'] # 港口操作费

    # 海运相关参数计算 (单程可变成本/航次, 年固定成本/船型, 容量/航次)
    sea_voyage_var_cost_per_trip = {} # VC_sea[(ship, orig_port, dest_port)] -> 单次航行的可变成本
    ship_annual_fixed_cost_map = {} # FC_ship_annual[ship]
    ship_capacity_map = {}          # CapShip[ship]

    # 构建 valid_voyage_triplets 集合，包含 (ship, orig_port, dest_port)
    # 这个集合用于检查一个 (船, 始发港, 目的港) 组合是否是一个已定义的、我们关心的航线。
    # 在此模型中，所有 SHIPS 在所有 SEA_ROUTES_TUPLES 上原则上都可以航行，所以这个集合会包含所有这些组合。
    valid_voyage_triplets = set()
    for s_name in SHIPS:
        ship_annual_fixed_cost_map[s_name] = ships_data_map[s_name]['fixed_cost_3months_usd'] * 4
        ship_capacity_map[s_name] = ships_data_map[s_name]['capacity']
        s_data = ships_data_map[s_name]
        voyage_time_days_one_way = SEA_DISTANCE_NM_ONE_WAY / s_data['speed_avg_knots'] / 24
        vcs_one_leg = voyage_time_days_one_way * s_data['vcs_usd_day']
        vcp_one_leg_ports = (PORT_STAY_SHIP_DAYS_COSTING * 2) * s_data['vcp_usd_day']
        var_cost_per_one_way_trip = vcs_one_leg + vcp_one_leg_ports

        for r_orig, r_dest in SEA_ROUTES_TUPLES:
            sea_voyage_var_cost_per_trip[(s_name, r_orig, r_dest)] = var_cost_per_one_way_trip
            valid_voyage_triplets.add((s_name, r_orig, r_dest)) # 添加到有效航次集合

    # 2. 创建 COPT 模型实例
    env = cp.Envr() # 创建COPT环境
    model = env.createModel("CoastalShippingOptimization_COPT") # 创建模型对象，命名

    # 3. 定义决策变量
    # X_direct[(am, cust)]: 从AM am 到客户 cust 的年直接卡车运输量 (连续变量)
    x_direct = {}
    for i in AMS:
        for j in CUSTOMERS:
            # 仅为需求 > 0 的(am, cust)对创建变量
            if annual_demands.get((i, j), 0) > 0:
                 x_direct[i,j] = model.addVar(lb=0, name=f"X_direct_{i}_{j.replace(' ','_')}")

    # X_coastal[(am, cust, orig_port, dest_port, ship_type)]: 通过沿海运输从AM am经orig_port到dest_port使用ship_type运往客户cust的年运输量(连续变量)
    x_coastal = {}
    for i in AMS:
        for j in CUSTOMERS:
            # 仅为需求 > 0 的(am, cust)对创建变量
            if annual_demands.get((i, j), 0) > 0:
                for p_orig in PORTS:
                    for q_dest in PORTS:
                        if p_orig != q_dest: # 始发港和目的港不能相同
                            for s in SHIPS:
                                x_coastal[i,j,p_orig,q_dest,s] = model.addVar(lb=0, name=f"X_coastal_{i}_{j.replace(' ','_')}_{p_orig}_{q_dest}_{s}")

    # V_voyages[(ship_type, orig_port, dest_port)]: 船型ship_type在航线(orig_port, dest_port)上的年航行次数 (连续变量，但通常希望是整数，可后续处理或设为整数变量)
    # 在许多运输模型中，航次数可以是连续的，表示平均航次数或总服务能力。如果严格要求整数，则vtype=COPT.INTEGER
    v_annual_voyages = {} # 更改变量名以避免与之前的 v_voyages 列表混淆
    for s_trip in SHIPS:
        for p_orig_trip in PORTS:
            for q_dest_trip in PORTS:
                if p_orig_trip != q_dest_trip:
                    # 检查是否是有效的三元组，尽管在此定义下它们都应该是
                    if (s_trip, p_orig_trip, q_dest_trip) in valid_voyage_triplets:
                        v_annual_voyages[s_trip,p_orig_trip,q_dest_trip] = model.addVar(lb=0, name=f"V_annual_voyages_{s_trip}_{p_orig_trip}_{q_dest_trip}")

    # U_use_ship_type[ship_type]: 是否启用某种船型 (二元变量)
    u_use_ship_type = {}
    for s_use in SHIPS:
        u_use_ship_type[s_use] = model.addVar(vtype=COPT.BINARY, name=f"U_use_ship_{s_use}")

    # 4. 定义目标函数: 最小化年总运输成本
    obj = cp.LinExpr() # 创建COPT线性表达式对象

    #  直接运输成本部分
    for i_obj_d in AMS:
        for j_obj_d in CUSTOMERS:
            if (i_obj_d, j_obj_d) in x_direct: # 确保变量已创建 (即有需求)
                cost = direct_truck_unit_costs.get((i_obj_d,j_obj_d), 1e9)
                if cost < 1e8: # 仅添加有效的成本项
                    obj += float(direct_truck_unit_costs.get((i_obj_d,j_obj_d), 1e9)) * x_direct[i_obj_d,j_obj_d]
    #  沿海运输的陆路和港口操作成本部分
    for i_obj_c in AMS:
        for j_obj_c in CUSTOMERS:
            if annual_demands.get((i_obj_c, j_obj_c), 0) > 0: # 仅对有需求的组合处理
                for p_orig_obj_c in PORTS:
                    for q_dest_obj_c in PORTS:
                        if p_orig_obj_c != q_dest_obj_c:
                            for s_obj_c in SHIPS:
                                if (i_obj_c,j_obj_c,p_orig_obj_c,q_dest_obj_c,s_obj_c) in x_coastal: # 确保变量存在
                                    fml_c = fml_unit_costs.get((i_obj_c,p_orig_obj_c), 1e9)
                                    lml_c = lml_unit_costs.get((q_dest_obj_c,j_obj_c), 1e9)
                                    if fml_c < 1e8 and lml_c < 1e8:
                                        leg_cost = fml_c + (2 * port_handling_cost) + lml_c
                                        obj += float(leg_cost) * x_coastal[i_obj_c,j_obj_c,p_orig_obj_c,q_dest_obj_c,s_obj_c]

    #  海运可变成本部分 (基于航次数)
    for s_obj_v in SHIPS:
        for p_orig_obj_v in PORTS:
            for q_dest_obj_v in PORTS:
                current_voyage_key = (s_obj_v, p_orig_obj_v, q_dest_obj_v)
                if current_voyage_key in v_annual_voyages : # 确保航次变量已定义
                    var_c_sea_trip = sea_voyage_var_cost_per_trip.get(current_voyage_key, 1e9)
                    if var_c_sea_trip < 1e8:
                        obj += float(sea_voyage_var_cost_per_trip.get(current_voyage_key, 1e9)) * v_annual_voyages[current_voyage_key]

    #  船舶年固定成本部分 (基于是否启用船型)
    for s_obj_fc in SHIPS:
        if s_obj_fc in u_use_ship_type and s_obj_fc in ship_annual_fixed_cost_map: # 确保变量和成本都存在
            obj += float(ship_annual_fixed_cost_map[s_obj_fc]) * u_use_ship_type[s_obj_fc]

    model.setObjective(obj, sense=COPT.MINIMIZE)

    # 5. 定义约束条件
    #  需求满足约束：每个AM对每个客户的年需求必须被满足
    for i_dem in AMS:
        for j_dem in CUSTOMERS:
            demand_val = annual_demands.get((i_dem,j_dem), 0)
            if demand_val > 0:
                lhs_demand = cp.LinExpr()
                if (i_dem,j_dem) in x_direct:
                    lhs_demand += 1.0 * x_direct[i_dem,j_dem]

                for p_orig_dem in PORTS:
                    for q_dest_dem in PORTS:
                        if p_orig_dem != q_dest_dem:
                            for s_dem in SHIPS:
                                if (i_dem,j_dem,p_orig_dem,q_dest_dem,s_dem) in x_coastal:
                                    lhs_demand += 1.0 * x_coastal[i_dem,j_dem,p_orig_dem,q_dest_dem,s_dem]
                model.addConstr(lhs_demand == demand_val, name=f"Demand_Satisfy_{i_dem}_{j_dem.replace(' ','_').replace('(','').replace(')','')}")


    #  AM供应能力约束：每个AM的总出货量不能超过其年供应能力
    for i_cap in AMS:
        lhs_supply = cp.LinExpr()
        for j_cap in CUSTOMERS:
            if (i_cap,j_cap) in x_direct:
                lhs_supply += 1.0 * x_direct[i_cap,j_cap]
        for j_cap_c in CUSTOMERS:
            if annual_demands.get((i_cap, j_cap_c),0) > 0: # 优化：仅当客户对该AM有需求时才考虑沿海
                for p_orig_cap in PORTS:
                    for q_dest_cap in PORTS:
                        if p_orig_cap != q_dest_cap:
                            for s_cap in SHIPS:
                                if (i_cap,j_cap_c,p_orig_cap,q_dest_cap,s_cap) in x_coastal:
                                    lhs_supply += 1.0 * x_coastal[i_cap,j_cap_c,p_orig_cap,q_dest_cap,s_cap]
        model.addConstr(lhs_supply <= am_annual_capacity_map[i_cap], name=f"Supply_Capacity_{i_cap}")

    #  海运流量与航次及船舶容量关联
    for s_sea_cap in SHIPS:
        for p_orig_sea_cap in PORTS:
            for q_dest_sea_cap in PORTS:
                voyage_key_constr = (s_sea_cap, p_orig_sea_cap, q_dest_sea_cap)
                if voyage_key_constr in v_annual_voyages: # 确保航次变量存在
                    lhs_sea_volume_on_route_ship = cp.LinExpr()
                    for i_sea_vol in AMS:
                        for j_sea_vol in CUSTOMERS:
                             # 确保沿海运输变量存在
                            if (i_sea_vol,j_sea_vol,p_orig_sea_cap,q_dest_sea_cap,s_sea_cap) in x_coastal:
                                lhs_sea_volume_on_route_ship += 1.0 * x_coastal[i_sea_vol,j_sea_vol,p_orig_sea_cap,q_dest_sea_cap,s_sea_cap]

                    model.addConstr(lhs_sea_volume_on_route_ship <= v_annual_voyages[voyage_key_constr] * ship_capacity_map[s_sea_cap],
                                    name=f"Sea_Capacity_{s_sea_cap}_{p_orig_sea_cap}_{q_dest_sea_cap}")

    #  船舶使用与航次关联 (用于激活固定成本)
    M_large_number = sum(annual_demands.values()) if annual_demands else 1000000
    if M_large_number == 0 : M_large_number = 1000000

    for s_link in SHIPS:
        if s_link in u_use_ship_type: # 确保启用船型变量存在
            lhs_total_voyages_for_ship_type = cp.LinExpr()
            for p_orig_link in PORTS:
                for q_dest_link in PORTS:
                    voyage_key_link = (s_link, p_orig_link, q_dest_link)
                    if voyage_key_link in v_annual_voyages: # 确保航次变量存在
                        lhs_total_voyages_for_ship_type += 1.0 * v_annual_voyages[voyage_key_link]

            model.addConstr(lhs_total_voyages_for_ship_type <= M_large_number * u_use_ship_type[s_link],
                            name=f"Link_Voyage_Total_To_Use_Ship_{s_link}")


    # 6. 求解模型
    model.setParam("Logging", 1) # 打开COPT求解日志
    model.setParam("TimeLimit", 300) # 设置求解时间限制为300秒

    status_code = -1000
    status_str = "未开始求解"
    try:
        print("开始调用COPT求解器...")
        model.solve()
        status_code = model.status
        if status_code == COPT.OPTIMAL: status_str = "Optimal (已找到最优解)"
        elif status_code == COPT.INFEASIBLE: status_str = "Infeasible (问题无解)"
        elif status_code == COPT.UNBOUNDED: status_str = "Unbounded (问题无界)"
        elif status_code == COPT.TIMEOUT: status_str = "Timeout (已超时)"
        elif status_code == COPT.INTERRUPTED: status_str = "Interrupted (求解被中断)"
        elif status_code == COPT.SUBOPTIMAL: status_str = "Suboptimal (找到可行解但可能非最优)"
        else: status_str = f"Not Solved or Other Status (COPT状态码: {status_code})"
        print(f"COPT求解完成，状态: {status_str}")
    except Exception as e:
        status_str = f"COPT求解过程中发生Python级别错误: {e}"
        status_code = -1001
        print(status_str)


    # 7. 分析和返回结果
    results_q3 = {
        "status": status_str,
        "total_optimal_cost": None, "direct_shipments": {}, "coastal_shipments": {},
        "voyages_per_route_ship": {}, "ships_used": [], "total_coastal_volume": 0, "total_direct_volume":0
    }

    if status_code == COPT.OPTIMAL or status_code == COPT.SUBOPTIMAL:
        if hasattr(model, 'objval') and model.objval is not None:
            results_q3["total_optimal_cost"] = model.objval

        for (i_res_d,j_res_d), var in x_direct.items():
            if var.x > 0.01:
                results_q3["direct_shipments"][(i_res_d,j_res_d)] = var.x
                results_q3["total_direct_volume"] += var.x

        for (i_res_c,j_res_c,p_orig_c,q_dest_c,s_c), var in x_coastal.items():
            if var.x > 0.01:
                key_c = (i_res_c,j_res_c,p_orig_c,q_dest_c,s_c)
                results_q3["coastal_shipments"][key_c] = var.x
                results_q3["total_coastal_volume"] += var.x

        for s_res_u, var in u_use_ship_type.items():
            if var.x > 0.5:
                results_q3["ships_used"].append(s_res_u)

        for (s_res_v, p_res_v, q_res_v), var in v_annual_voyages.items():
            if var.x > 0.01:
                 results_q3["voyages_per_route_ship"][(s_res_v,p_res_v,q_res_v)] = round(var.x) # 航次数取整

    return results_q3


# --- 主执行流程 ---
# 解决问题1
print("--- 开始执行问题1 ---")
total_direct_cost_q1, df_with_q1_costs = solve_question1(df_locations.copy())
print("--- 问题1 执行完毕 ---")

# 解决问题2
print("\n--- 开始执行问题2 ---")
df_final_q2, df_q2_summary_for_print = solve_question2(df_with_q1_costs.copy())
print("--- 问题2 执行完毕 ---")

# 为问题3准备参数
cost_params_q3 = {'port_handling_charge_usd_unit': PORT_HANDLING_CHARGE_USD_UNIT}
# 调用问题3的COPT求解函数
results_q3_optimization = solve_question3_copt(df_with_q1_costs, AM_CAPACITY_ANNUAL, SHIPS_DATA, cost_params_q3)


# --- 准备打印输出 ---
# 问题1 和 问题2 的输出逻辑 (与您之前提供的版本相同)
output_q1_str = f"问题1：估算的当前纯公路直接运输总成本 (3个月): ${total_direct_cost_q1:,.2f}\n"
output_q1_str += "各AM到各客户的直接运输成本详情 (为简洁起见，展示前5个客户):\n"
q1_df_for_output = df_final_q2
if not q1_df_for_output.empty:
    first_cust_q1 = q1_df_for_output.iloc[0]
    output_q1_str += f"\n客户地点: {first_cust_q1['Customer_Location']}\n"
    for am in ['AM1', 'AM2', 'AM3']:
        demand = first_cust_q1.get(f'{am}_Total_Demand_3Mo', 0); trips = first_cust_q1.get(f'{am}_Direct_Num_Trips', 0); cost = first_cust_q1.get(f'{am}_Direct_Cost', 0); cost_pu = first_cust_q1.get(f'{am}_Direct_Cost_Per_Unit', np.nan)
        demand_str = f"{demand:,.0f}" if pd.notnull(demand) else "N/A"; trips_str = f"{trips:,.0f}" if pd.notnull(trips) else "N/A"
        cost_str = f"${cost:,.2f}" if pd.notnull(cost) and cost != np.inf and not (cost == 0 and pd.isna(demand) or demand == 0) else ("$0.00" if cost==0 else "N/A")
        cost_pu_str = f"${cost_pu:,.2f}" if pd.notnull(cost_pu) and cost_pu != np.inf and not (cost_pu == 0 and pd.isna(demand) or demand == 0) else ("$0.00" if cost_pu==0 else "N/A")
        output_q1_str += (f"  {am}:\n    总需求 (3个月): {demand_str}\n    直接运输次数: {trips_str}\n    直接运输成本: {cost_str}\n    单位直接成本: {cost_pu_str}\n")
if len(q1_df_for_output) > 1:
    output_q1_str += "\n后续客户 (纯公路运输成本 - 简略格式):\n"
    q1_subsequent_rows_to_print = q1_df_for_output.iloc[1:min(5, len(q1_df_for_output))]
    max_loc_len_q1 = q1_subsequent_rows_to_print['Customer_Location'].astype(str).map(len).max() if not q1_subsequent_rows_to_print.empty else 15; max_loc_len_q1 = max(max_loc_len_q1, 15)
    for index, row in q1_subsequent_rows_to_print.iterrows():
        location_name_padded = f"{str(row['Customer_Location'])[:max_loc_len_q1]:<{max_loc_len_q1}}"
        am_details_lines = []
        for am_idx, am in enumerate(['AM1', 'AM2', 'AM3']):
            demand = row.get(f'{am}_Total_Demand_3Mo', 0); trips = row.get(f'{am}_Direct_Num_Trips', 0); cost_pu = row.get(f'{am}_Direct_Cost_Per_Unit', np.nan)
            demand_str = f"{demand:,.0f}" if pd.notnull(demand) else "N/A"; trips_str = f"{trips:,.0f}" if pd.notnull(trips) else "N/A"
            cost_pu_str = f"${cost_pu:,.2f}" if pd.notnull(cost_pu) and cost_pu != np.inf and not (cost_pu == 0 and pd.isna(demand) or demand == 0) else ("$0.00" if cost_pu==0 else "N/A")
            am_detail_str = f"{am}[需求:{demand_str},次数:{trips_str},单位成本:{cost_pu_str}]"
            if am_idx == 0: am_details_lines.append(f"{location_name_padded} {am_detail_str}")
            else: am_details_lines.append(f"{' ' * max_loc_len_q1} {am_detail_str}")
        output_q1_str += "\n".join(am_details_lines) + "\n"
else: output_q1_str += "\n后续客户 (纯公路运输成本 - 简略格式): 无更多数据可展示。\n"

output_q2_str = "\n\n问题2：适合转向沿海运输的客户地点识别 (为简洁起见，展示前5个客户):\n"
output_q2_str += "判断标准：如果任何一种多式联运单位成本 < 该AM的纯公路运输单位成本，则为“适合”。\n"
output_q2_str += "MM_Cost列显示为该AM和船舶组合找到的最小多式联运单位成本 (后续行仅展示船A信息)。\n"
if not df_q2_summary_for_print.empty:
    first_cust_q2 = df_q2_summary_for_print.iloc[0]
    output_q2_str += f"\n客户地点: {first_cust_q2['Customer_Location']}\n"
    for am in ['AM1', 'AM2', 'AM3']:
        direct_pu = first_cust_q2.get(f'{am}_Direct_Cost_PU', np.inf); mm_cost_a = first_cust_q2.get(f'{am}_Min_MM_Cost_PU_ShipA', np.inf)
        suitable_a = first_cust_q2.get(f'{am}_Suitable_ShipA', "N/A (无有效海运路径)"); mm_cost_b = first_cust_q2.get(f'{am}_Min_MM_Cost_PU_ShipB', np.inf)
        suitable_b = first_cust_q2.get(f'{am}_Suitable_ShipB', "N/A (无有效海运路径)")
        direct_pu_str = f"${direct_pu:,.2f}" if pd.notnull(direct_pu) and direct_pu != np.inf and direct_pu !=0 else ("$0.00" if direct_pu==0 else "N/A")
        mm_cost_a_str = f"${mm_cost_a:,.2f}" if pd.notnull(mm_cost_a) and mm_cost_a != np.inf and mm_cost_a !=0 else ("$0.00" if mm_cost_a==0 else "N/A")
        mm_cost_b_str = f"${mm_cost_b:,.2f}" if pd.notnull(mm_cost_b) and mm_cost_b != np.inf and mm_cost_b !=0 else ("$0.00" if mm_cost_b==0 else "N/A")
        output_q2_str += (f"  {am}:\n"
                          f"    单位直接成本: {direct_pu_str}\n"
                          f"    船A - 最小多式联运单位成本: {mm_cost_a_str}, 适用性: {suitable_a}\n"
                          f"    船B - 最小多式联运单位成本: {mm_cost_b_str}, 适用性: {suitable_b}\n")
if len(df_q2_summary_for_print) > 1:
    output_q2_str += "\n后续客户 (海运适用性 - 简略格式，仅船A):\n"
    q2_subsequent_rows_to_print = df_q2_summary_for_print.iloc[1:min(5, len(df_q2_summary_for_print))]
    max_loc_len_q2 = q2_subsequent_rows_to_print['Customer_Location'].astype(str).map(len).max() if not q2_subsequent_rows_to_print.empty else 15; max_loc_len_q2 = max(max_loc_len_q2, 15)
    for index, row in q2_subsequent_rows_to_print.iterrows():
        location_name_padded = f"{str(row['Customer_Location'])[:max_loc_len_q2]:<{max_loc_len_q2}}"
        am_details_lines_q2 = []
        for am_idx, am in enumerate(['AM1', 'AM2', 'AM3']):
            direct_pu = row.get(f'{am}_Direct_Cost_PU', np.inf); mm_cost_a = row.get(f'{am}_Min_MM_Cost_PU_ShipA', np.inf)
            suitable_a = row.get(f'{am}_Suitable_ShipA', "N/A (无有效海运路径)")
            direct_pu_str = f"${direct_pu:,.2f}" if pd.notnull(direct_pu) and direct_pu != np.inf and direct_pu !=0 else ("$0.00" if direct_pu==0 else "N/A")
            mm_cost_a_str = f"${mm_cost_a:,.2f}" if pd.notnull(mm_cost_a) and mm_cost_a != np.inf and mm_cost_a !=0 else ("$0.00" if mm_cost_a==0 else "N/A")
            am_detail_str = f"{am}[直运单位成本:{direct_pu_str},船A海运单位成本:{mm_cost_a_str},船A适用:{suitable_a}]"
            if am_idx == 0: am_details_lines_q2.append(f"{location_name_padded} {am_detail_str}")
            else: am_details_lines_q2.append(f"{' ' * max_loc_len_q2} {am_detail_str}")
        output_q2_str += "\n".join(am_details_lines_q2) + "\n"
else: output_q2_str += "\n后续客户 (海运适用性 - 简略格式，仅船A): 无更多数据可展示。\n"


# 问题3 输出
output_q3_str = f"\n\n问题3：优化沿海运输系统设计结果 (使用 COPT 建模):\n"
output_q3_str += f"求解状态: {results_q3_optimization['status']}\n"

if results_q3_optimization['total_optimal_cost'] is not None and \
   (results_q3_optimization['status'].startswith("Optimal") or results_q3_optimization['status'].startswith("Suboptimal")):
    optimal_cost = results_q3_optimization['total_optimal_cost']
    output_q3_str += f"最优/可行年总运输成本: ${optimal_cost:,.2f}\n"

    annualized_direct_cost_q1 = total_direct_cost_q1 * 4 # Q1的成本是3个月的，年化处理
    output_q3_str += f"年化纯公路运输总成本 (对比参考): ${annualized_direct_cost_q1:,.2f}\n"
    if optimal_cost < annualized_direct_cost_q1:
        savings = annualized_direct_cost_q1 - optimal_cost
        savings_percent = (savings / annualized_direct_cost_q1) * 100 if annualized_direct_cost_q1 > 0 else 0
        output_q3_str += f"通过优化方案可节省成本: ${savings:,.2f} (约 {savings_percent:.2f}%)\n"
    else:
        output_q3_str += "优化方案成本不低于或等于纯公路运输成本。\n"

    output_q3_str += "\n选用的船型 (用于海运):\n"
    if results_q3_optimization['ships_used']:
        for ship in results_q3_optimization['ships_used']:
            output_q3_str += f"- {ship}\n"
    else:
        output_q3_str += "- 未选择任何船型进行海运 (可能纯公路运输是当前模型下的最优选择，或没有可行的海运方案能降低总成本)。\n"

    output_q3_str += "\n各航线及船型的年航次数 (为简洁，仅显示 > 0 的航次):\n"
    if results_q3_optimization['voyages_per_route_ship']:
        for (s, p_orig, q_dest), voyages in results_q3_optimization['voyages_per_route_ship'].items():
            output_q3_str += f"- 船型 {s} 从 {p_orig} 到 {q_dest}: {voyages:,.1f} 次/年\n"
    else:
        output_q3_str += "- 无海运航次。\n"

    output_q3_str += "\n沿海运输模式运输的总货运量: {:,.0f} 辆/年\n".format(results_q3_optimization.get('total_coastal_volume',0))
    output_q3_str += "纯公路运输模式运输的总货运量 (在优化方案中): {:,.0f} 辆/年\n".format(results_q3_optimization.get('total_direct_volume',0))

    output_q3_str += "\n部分沿海运输路径示例 (按AM-客户-始发港-目的港-船型, 最多显示5条):\n"
    count_coastal_examples = 0
    if results_q3_optimization['coastal_shipments']:
        for (i,j,p,q,s), vol in results_q3_optimization['coastal_shipments'].items():
            if count_coastal_examples < 5:
                 output_q3_str += f"- 从 {i} 经 {p}->{q} (船{s}) 到 {j}: {vol:,.0f}辆/年\n"
                 count_coastal_examples +=1
        if len(results_q3_optimization['coastal_shipments']) > count_coastal_examples: # 检查是否有更多未显示的路径
            output_q3_str += "- ... (更多沿海运输路径未显示)\n"
    else:
        output_q3_str += "- 无通过沿海模式运输的货物。\n"
elif "COPT库未加载" in results_q3_optimization['status'] or "Python级别错误" in results_q3_optimization['status']: # 更明确地捕获之前定义的错误类型
    output_q3_str += f"错误: {results_q3_optimization['status']}\n"
    output_q3_str += "请检查COPT的安装和配置，或查看详细错误信息。\n"
else: # 其他非最优或非已知错误状态，例如 Infeasible, Unbounded, Timeout
    output_q3_str += "未能找到最优解或求解过程中出现问题。\n"
    if "错误" not in results_q3_optimization['status'] and "Error" not in results_q3_optimization['status'] : #避免重复打印错误信息
         output_q3_str += f"详细状态信息: {results_q3_optimization['status']}\n"
    output_q3_str += "请检查COPT求解日志（如果已开启Logging=1），或检查模型约束/数据。\n"


# 最终合并输出
final_print_output = output_q1_str + "\n" + output_q2_str + "\n" + output_q3_str
print(final_print_output)