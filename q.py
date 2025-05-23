import pandas as pd
import numpy as np

# --- 固定参数 ---
# Exhibit 8: 汽车拖车费用 [cite: 2, 161]
TRUCK_FIXED_COST = 185.69  #美元/次运输 [cite: 2, 161]
TRUCK_VARIABLE_COST_KM = 1.46  #美元/公里 [cite: 2, 161]
TRUCK_CAPACITY = 8  #标准车辆数 [cite: 2, 161]

# Exhibit 6: 汽车制造集群到港口的公路距离 (公里) [cite: 2, 157]
AM_TO_PORT_DIST = {
    'AM1': {'Chennai Port': 38.0, 'Pipavav Port': 2043.0},  # [cite: 2, 157]
    'AM2': {'Chennai Port': 2169.0, 'Pipavav Port': 1198.0}, # [cite: 2, 157]
    'AM3': {'Chennai Port': 1869.0, 'Pipavav Port': 292.0}   # [cite: 2, 157]
}

# Exhibit 7: 滚装船特性 [cite: 2, 159]
SHIPS_DATA = {
    'ShipA': { # 对应 Ship 1 / A型船 [cite: 2, 159]
        'capacity': 800,  #标准车辆数 [cite: 2, 159]
        'speed_avg_knots': 13,  #节 (海里/小时) [cite: 2, 159]
        'vcp_usd_day': 3467,  # 在港运营成本(美元/天) [cite: 2, 159]
        'vcs_usd_day': 3218,  # 在海运营成本(美元/天) [cite: 2, 159]
        'fixed_cost_3months_usd': 268366.0 #固定成本(美元/3个月) [cite: 2, 159]
    },
    'ShipB': { # 对应 Ship 2 / B型船 [cite: 2, 159]
        'capacity': 3518, #标准车辆数 [cite: 2, 159]
        'speed_avg_knots': 17,  #节 [cite: 2, 159]
        'vcp_usd_day': 15925, # 在港运营成本(美元/天) [cite: 2, 159]
        'vcs_usd_day': 6568,  # 在海运营成本(美元/天) [cite: 2, 159]
        'fixed_cost_3months_usd': 536778.0 #固定成本(美元/3个月) [cite: 2, 159]
    }
}

# Exhibit 5: 港口相关数据 [cite: 2, 155]
PORT_HANDLING_CHARGE_USD_UNIT = 2.0  # 美元/车 (港口货物处理费) [cite: 2, 155]
PORT_STAY_SHIP_DAYS_COSTING = 1.0      # 天/次挂靠 (船舶在港停留时间，用于计算VCP) [cite: 2, 155]

# Exhibit 4: 海运距离 [cite: 2, 151]
SEA_DISTANCE_NM_ONE_WAY = 1394.0  # 金奈港与皮帕瓦沃港之间的海里数 [cite: 2, 151]
NM_TO_KM = 1.852 # 海里转公里换算系数 [cite: 2, 152]

# --- 使用 pandas 读取大型数据文件 ---
try:
    # header=2 表示将CSV文件的第3行作为表头
    df_locations = pd.read_excel("DATA_original.xlsx", sheet_name="Customer Locations", header=2)
    # header=1 表示将CSV文件的第2行作为表头
    df_port_cust_dist = pd.read_excel("DATA_original.xlsx",sheet_name="PortCustDist", header=1)
except FileNotFoundError as e:
    print(f"错误：一个或多个CSV文件未找到。请确保文件位于正确的目录中：{e}")
    raise # 在实际环境中，如果文件不存在则抛出异常

# --- 数据清洗与合并 ---

# 清洗 df_locations
df_locations.columns = ['Cust_Location_Num', 'Customer_Location', 'State_UT',
                        'AM1_Dist', 'AM1_M1', 'AM1_M2', 'AM1_M3',
                        'AM2_Dist', 'AM2_M1', 'AM2_M2', 'AM2_M3',
                        'AM3_Dist', 'AM3_M1', 'AM3_M2', 'AM3_M3']

df_locations.dropna(how='all', inplace=True) # 删除全为空白的行

# 将相关列转换为数值型，无法转换的变为NaN
num_cols_loc = ['AM1_Dist', 'AM1_M1', 'AM1_M2', 'AM1_M3',
                'AM2_Dist', 'AM2_M1', 'AM2_M2', 'AM2_M3',
                'AM3_Dist', 'AM3_M1', 'AM3_M2', 'AM3_M3']

for col in num_cols_loc:
    df_locations[col] = pd.to_numeric(df_locations[col], errors='coerce')

# 清洗 df_port_cust_dist
df_port_cust_dist.rename(columns={
    'Distance (km)':'Customer_Location_PortFile', # 临时列名，避免与主表列名冲突
    'Chennai':'Dist_to_Chennai_Port',
    'Pipavav':'Dist_to_Pipavav_Port'
    }, inplace=True)

# 保留需要的列，并基于 Customer_Location_PortFile 列删除空值行
df_port_cust_dist = df_port_cust_dist[['Customer_Location_PortFile', 'Dist_to_Chennai_Port', 'Dist_to_Pipavav_Port']].copy()
df_port_cust_dist.dropna(subset=['Customer_Location_PortFile'], inplace=True)
df_port_cust_dist['Customer_Location'] = df_port_cust_dist['Customer_Location_PortFile'] # 创建用于合并的共同键

# 转换距离列为数值型
df_port_cust_dist['Dist_to_Chennai_Port'] = pd.to_numeric(df_port_cust_dist['Dist_to_Chennai_Port'], errors='coerce')
df_port_cust_dist['Dist_to_Pipavav_Port'] = pd.to_numeric(df_port_cust_dist['Dist_to_Pipavav_Port'], errors='coerce')

# 合并港口距离数据到主位置数据表
df_locations = pd.merge(df_locations, df_port_cust_dist[['Customer_Location', 'Dist_to_Chennai_Port', 'Dist_to_Pipavav_Port']],
                        on="Customer_Location", how="left") # 左连接保留所有客户地点

# --- 问题1：计算纯公路直接运输成本 ---
def solve_question1(df):
    df_q1 = df.copy() # 创建副本进行操作
    for am_id in ['AM1', 'AM2', 'AM3']:
        demand_cols = [f'{am_id}_M1', f'{am_id}_M2', f'{am_id}_M3']
        # 确保需求列存在且为数值，缺失值填充为0
        for d_col in demand_cols:
            if d_col in df_q1.columns:
                 df_q1[d_col] = pd.to_numeric(df_q1[d_col], errors='coerce').fillna(0)
            else: # 如果CSV中某些AM没有M1/M2/M3列（不太可能，但做防御）
                df_q1[d_col] = 0.0

        df_q1[f'{am_id}_Total_Demand_3Mo'] = df_q1[demand_cols].sum(axis=1)
        
        # 筛选出有有效需求和距离的数据行进行计算
        mask_valid = (df_q1[f'{am_id}_Total_Demand_3Mo'] > 0) & (~df_q1[f'{am_id}_Dist'].isna())
        
        # 初始化结果列为0.0，避免部分行为NaN导致后续计算问题
        for col_suffix in ['_Direct_Num_Trips', '_Direct_Cost', '_Direct_Cost_Per_Unit']:
            df_q1[f'{am_id}{col_suffix}'] = 0.0

        df_q1.loc[mask_valid, f'{am_id}_Direct_Num_Trips'] = np.ceil(
            df_q1.loc[mask_valid, f'{am_id}_Total_Demand_3Mo'] / TRUCK_CAPACITY
        )
        df_q1.loc[mask_valid, f'{am_id}_Direct_Cost'] = df_q1.loc[mask_valid, f'{am_id}_Direct_Num_Trips'] * \
            (TRUCK_FIXED_COST + (TRUCK_VARIABLE_COST_KM * df_q1.loc[mask_valid, f'{am_id}_Dist']))
        
        # 计算单位成本，避免除以0
        df_q1.loc[mask_valid, f'{am_id}_Direct_Cost_Per_Unit'] = \
            df_q1.loc[mask_valid, f'{am_id}_Direct_Cost'] / df_q1.loc[mask_valid, f'{am_id}_Total_Demand_3Mo']
            
    # 计算总直接运输成本
    total_direct_trucking_cost = df_q1[[f'AM1_Direct_Cost', f'AM2_Direct_Cost', f'AM3_Direct_Cost']].sum().sum()
    return total_direct_trucking_cost, df_q1


# --- 问题2：识别适合转向沿海运输的客户地点 ---
# 辅助函数：计算单程海运的单位成本（包含可变和分摊的固定成本）
def calculate_sea_leg_cost_per_unit(ship_name, sea_dist_nm, port_stay_for_vcp_calc_days):
    ship = SHIPS_DATA[ship_name]
    # 单程航行时间（天）
    voyage_time_days = sea_dist_nm / ship['speed_avg_knots'] / 24
    
    # 单程可变运营成本：海上VCS + 此航程涉及的港口停留VCP
    # 假设一次单程海运，船舶会在始发港作业1天，目的港作业1天，产生VCP成本
    vcs_total_one_leg = voyage_time_days * ship['vcs_usd_day']
    vcp_total_one_leg_ports = port_stay_for_vcp_calc_days * ship['vcp_usd_day'] # port_stay_for_vcp_calc_days 通常是2（始发港1天+目的港1天）
    total_variable_op_cost_one_leg = vcs_total_one_leg + vcp_total_one_leg_ports
    
    # 固定成本分摊：计算3个月内可执行的往返航次数
    # 一次完整往返周期时间 = 2 * 单程航行时间 + 2 * 核心港口停留天数（例如，在两个主要港口各停留1天用于周转）
    round_trip_cycle_time_days = (voyage_time_days * 2) + (PORT_STAY_SHIP_DAYS_COSTING * 2)
    if round_trip_cycle_time_days == 0: return np.inf # 避免除以零
    
    num_round_trips_3_months = 90 / round_trip_cycle_time_days # 假设3个月为90天
    if num_round_trips_3_months == 0: return np.inf

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
    # 如果需求为0或距离无效，则成本为无穷大（不选择此路径）
    if pd.isna(total_demand_for_leg) or total_demand_for_leg == 0 or pd.isna(distance_km):
        return np.inf
    num_trips = np.ceil(total_demand_for_leg / TRUCK_CAPACITY)
    total_cost_for_leg = num_trips * (TRUCK_FIXED_COST + (TRUCK_VARIABLE_COST_KM * distance_km))
    return total_cost_for_leg / total_demand_for_leg

# 主要逻辑函数：解决问题2
def solve_question2(df):
    df_q2 = df.copy() # 使用已包含Q1结果的DataFrame副本
    # 用于生成最终打印摘要的列表
    results_q2_summary_list = [] 

    for index, row in df_q2.iterrows():
        cust_loc = row['Customer_Location']
        # 当前行客户的摘要信息
        summary_row = {'Customer_Location': cust_loc} 

        for am_id in ['AM1', 'AM2', 'AM3']:
            # 获取纯公路运输的单位成本，如果为0（因无需求导致），视为无穷大以便比较
            direct_cost_pu = row.get(f'{am_id}_Direct_Cost_Per_Unit', np.inf)
            if direct_cost_pu == 0 and row.get(f'{am_id}_Total_Demand_3Mo', 0) == 0:
                direct_cost_pu = np.inf # 确保无需求时，不会错误地判断海运“适合”
            
            demand_for_am = row.get(f'{am_id}_Total_Demand_3Mo', 0)
            
            summary_row[f'{am_id}_Direct_Cost_PU'] = direct_cost_pu

            # 如果该AM对此客户无需求，则海运不适用
            if demand_for_am == 0:
                for ship_letter in ['A', 'B']:
                    df_q2.loc[index, f'{am_id}_MM_Cost_PU_Ship{ship_letter}'] = np.inf
                    df_q2.loc[index, f'{am_id}_Suitable_Ship{ship_letter}'] = "N/A (无需求)"
                    summary_row[f'{am_id}_Min_MM_Cost_PU_Ship{ship_letter}'] = np.inf
                    summary_row[f'{am_id}_Suitable_Ship{ship_letter}'] = "N/A (无需求)"
                continue # 处理下一个AM

            # 初始化当前AM到此客户的最低多式联运成本
            min_mm_cost_ship_a_for_am_cust = np.inf
            min_mm_cost_ship_b_for_am_cust = np.inf

            # 考虑两种主要的海运目的港选项：金奈港 或 皮帕瓦沃港
            for dest_port_option in ['Chennai Port', 'Pipavav Port']:
                origin_port_am = '' # 该AM的始发港
                fml_dist_am = np.nan # 第一英里距离

                # 根据AM确定其最可能的始发港
                if am_id == 'AM1': # AM1在金奈
                    origin_port_am = 'Chennai Port'
                elif am_id == 'AM2': # AM2在NCR
                    # 选择距离AM2更近的港口作为始发港
                    origin_port_am = 'Pipavav Port' if AM_TO_PORT_DIST['AM2']['Pipavav Port'] < AM_TO_PORT_DIST['AM2']['Chennai Port'] else 'Chennai Port'
                elif am_id == 'AM3': # AM3在Sanand
                    origin_port_am = 'Pipavav Port' # 非常靠近皮帕瓦沃

                fml_dist_am = AM_TO_PORT_DIST[am_id][origin_port_am]
                fml_cost_pu = calculate_land_leg_cost_per_unit(fml_dist_am, demand_for_am)
                
                # 获取从当前考虑的目的港到客户的最后一英里距离
                lml_dist_port_cust = np.nan
                if dest_port_option == 'Chennai Port':
                    lml_dist_port_cust = row.get('Dist_to_Chennai_Port', np.inf)
                else: # dest_port_option == 'Pipavav Port'
                    lml_dist_port_cust = row.get('Dist_to_Pipavav_Port', np.inf)
                
                lml_cost_pu = calculate_land_leg_cost_per_unit(lml_dist_port_cust, demand_for_am)

                current_mm_cost_ship_a_path = np.inf
                current_mm_cost_ship_b_path = np.inf

                # 只有当始发港和目的港不同时，才存在海运段
                if origin_port_am != dest_port_option:
                    # 海运成本（假设金奈-皮帕瓦沃航线，成本对称）
                    # 如果未来有其他航线，这里的海运成本需要动态获取
                    sea_cost_ship_a_leg = SEA_COST_SHIP_A
                    sea_cost_ship_b_leg = SEA_COST_SHIP_B
                    
                    # 计算此路径的总多式联运单位成本
                    current_mm_cost_ship_a_path = (fml_cost_pu + 
                                                 PORT_HANDLING_CHARGE_USD_UNIT + # 始发港操作
                                                 sea_cost_ship_a_leg + 
                                                 PORT_HANDLING_CHARGE_USD_UNIT + # 目的港操作
                                                 lml_cost_pu)
                    current_mm_cost_ship_b_path = (fml_cost_pu +
                                                 PORT_HANDLING_CHARGE_USD_UNIT +
                                                 sea_cost_ship_b_leg +
                                                 PORT_HANDLING_CHARGE_USD_UNIT +
                                                 lml_cost_pu)
                
                # 更新此AM到此客户的最低多式联运成本
                min_mm_cost_ship_a_for_am_cust = min(min_mm_cost_ship_a_for_am_cust, current_mm_cost_ship_a_path)
                min_mm_cost_ship_b_for_am_cust = min(min_mm_cost_ship_b_for_am_cust, current_mm_cost_ship_b_path)

            # 将找到的最低多式联运成本和适用性判断结果存入DataFrame
            df_q2.loc[index, f'{am_id}_MM_Cost_PU_ShipA'] = min_mm_cost_ship_a_for_am_cust
            df_q2.loc[index, f'{am_id}_Suitable_ShipA'] = "适合" if min_mm_cost_ship_a_for_am_cust < direct_cost_pu else "不适合"
            if min_mm_cost_ship_a_for_am_cust == np.inf: # 如果没有找到有效的海运路径
                 df_q2.loc[index, f'{am_id}_Suitable_ShipA'] = "N/A (无有效海运路径)"


            df_q2.loc[index, f'{am_id}_MM_Cost_PU_ShipB'] = min_mm_cost_ship_b_for_am_cust
            df_q2.loc[index, f'{am_id}_Suitable_ShipB'] = "适合" if min_mm_cost_ship_b_for_am_cust < direct_cost_pu else "不适合"
            if min_mm_cost_ship_b_for_am_cust == np.inf:
                 df_q2.loc[index, f'{am_id}_Suitable_ShipB'] = "N/A (无有效海运路径)"
            
            # 填充用于打印的摘要信息
            summary_row[f'{am_id}_Min_MM_Cost_PU_ShipA'] = min_mm_cost_ship_a_for_am_cust
            summary_row[f'{am_id}_Suitable_ShipA'] = df_q2.loc[index, f'{am_id}_Suitable_ShipA']
            summary_row[f'{am_id}_Min_MM_Cost_PU_ShipB'] = min_mm_cost_ship_b_for_am_cust
            summary_row[f'{am_id}_Suitable_ShipB'] = df_q2.loc[index, f'{am_id}_Suitable_ShipB']
        
        results_q2_summary_list.append(summary_row) # 添加此客户的完整AM分析摘要
        
    return df_q2, pd.DataFrame(results_q2_summary_list) # 返回完整DataFrame和用于打印的摘要DataFrame




# --- 主执行流程 ---
# 解决问题1，获取总直接运输成本和更新后的DataFrame
total_direct_cost_q1, df_with_q1_costs = solve_question1(df_locations)

# 解决问题2，使用问题1的结果进行比较，并得到最终的分析DataFrame和打印摘要
df_final_q2, df_q2_summary_for_print = solve_question2(df_with_q1_costs)


# --- 准备打印输出 (修改后的输出逻辑) ---

# --- 问题1 输出 ---
output_q1_str = f"问题1：估算的当前纯公路直接运输总成本 (3个月): ${total_direct_cost_q1:,.2f}\n"
output_q1_str += "各AM到各客户的直接运输成本详情 (为简洁起见，展示前5个客户):\n"

q1_df_for_output = df_final_q2 # df_final_q2 包含了Q1的所有计算结果列

# 详细打印第一个客户 (第一行)
if not q1_df_for_output.empty:
    first_cust_q1 = q1_df_for_output.iloc[0]
    output_q1_str += f"\n客户地点: {first_cust_q1['Customer_Location']}\n"
    for am in ['AM1', 'AM2', 'AM3']:
        demand = first_cust_q1.get(f'{am}_Total_Demand_3Mo', 0)
        trips = first_cust_q1.get(f'{am}_Direct_Num_Trips', 0)
        cost = first_cust_q1.get(f'{am}_Direct_Cost', 0)
        cost_pu = first_cust_q1.get(f'{am}_Direct_Cost_Per_Unit', 0)
        
        demand_str = f"{demand:,.0f}" if pd.notnull(demand) else "N/A"
        trips_str = f"{trips:,.0f}" if pd.notnull(trips) else "N/A"
        # 格式化成本，处理0和N/A
        cost_str = f"${cost:,.2f}" if pd.notnull(cost) and cost != np.inf and not (cost == 0 and demand == 0) else ("$0.00" if cost==0 else "N/A")
        cost_pu_str = f"${cost_pu:,.2f}" if pd.notnull(cost_pu) and cost_pu != np.inf and not (cost_pu == 0 and demand == 0) else ("$0.00" if cost_pu==0 else "N/A")


        output_q1_str += (
            f"  {am}:\n"
            f"    总需求 (3个月): {demand_str}\n"
            f"    直接运输次数: {trips_str}\n"
            f"    直接运输成本: {cost_str}\n"
            f"    单位直接成本: {cost_pu_str}\n"
        )

# 简略打印后续客户 (Q1)
if len(q1_df_for_output) > 1:
    output_q1_str += "\n后续客户 (纯公路运输成本 - 简略格式):\n"
    # 动态确定后续客户名称的最大打印宽度，用于对齐
    q1_subsequent_rows_to_print = q1_df_for_output.iloc[1:min(5, len(q1_df_for_output))]
    max_loc_len_q1 = 0
    if not q1_subsequent_rows_to_print.empty:
        max_loc_len_q1 = q1_subsequent_rows_to_print['Customer_Location'].astype(str).map(len).max()
    max_loc_len_q1 = max(max_loc_len_q1, 15) # 保证一个最小宽度

    for index, row in q1_subsequent_rows_to_print.iterrows():
        location_name_padded = f"{str(row['Customer_Location'])[:max_loc_len_q1]:<{max_loc_len_q1}}" # 左对齐，截断或填充
        
        am_details_lines = []
        for am_idx, am in enumerate(['AM1', 'AM2', 'AM3']):
            demand = row.get(f'{am}_Total_Demand_3Mo', 0)
            trips = row.get(f'{am}_Direct_Num_Trips', 0)
            cost_pu = row.get(f'{am}_Direct_Cost_Per_Unit', 0)

            demand_str = f"{demand:,.0f}" if pd.notnull(demand) else "N/A"
            trips_str = f"{trips:,.0f}" if pd.notnull(trips) else "N/A"
            cost_pu_str = f"${cost_pu:,.2f}" if pd.notnull(cost_pu) and cost_pu != np.inf and not (cost_pu == 0 and demand == 0) else ("$0.00" if cost_pu==0 else "N/A")
            
            am_detail_str = f"{am}[需求:{demand_str},次数:{trips_str},单位成本:{cost_pu_str}]"
            
            if am_idx == 0: # 第一行AM，前面加客户名称
                am_details_lines.append(f"{location_name_padded} {am_detail_str}")
            else: # 后续AM，前面加空格对齐
                am_details_lines.append(f"{' ' * max_loc_len_q1} {am_detail_str}")
        
        output_q1_str += "\n".join(am_details_lines) + "\n"
else:
    output_q1_str += "\n后续客户 (纯公路运输成本 - 简略格式): 无更多数据可展示。\n"


# --- 问题2 输出 ---
output_q2_str = "\n\n问题2：适合转向沿海运输的客户地点识别 (为简洁起见，展示前5个客户):\n"
output_q2_str += "判断标准：如果任何一种多式联运单位成本 < 该AM的纯公路运输单位成本，则为“适合”。\n"
output_q2_str += "MM_Cost列显示为该AM和船舶组合找到的最小多式联运单位成本 (后续行仅展示船A信息)。\n"

if not df_q2_summary_for_print.empty:
    first_cust_q2 = df_q2_summary_for_print.iloc[0]
    output_q2_str += f"\n客户地点: {first_cust_q2['Customer_Location']}\n"
    for am in ['AM1', 'AM2', 'AM3']:
        direct_pu = first_cust_q2.get(f'{am}_Direct_Cost_PU', np.inf)
        mm_cost_a = first_cust_q2.get(f'{am}_Min_MM_Cost_PU_ShipA', np.inf)
        suitable_a = first_cust_q2.get(f'{am}_Suitable_ShipA', "N/A (无有效海运路径)") # 默认值
        mm_cost_b = first_cust_q2.get(f'{am}_Min_MM_Cost_PU_ShipB', np.inf)
        suitable_b = first_cust_q2.get(f'{am}_Suitable_ShipB', "N/A (无有效海运路径)") # 默认值
        
        direct_pu_str = f"${direct_pu:,.2f}" if pd.notnull(direct_pu) and direct_pu != np.inf and direct_pu !=0 else ("$0.00" if direct_pu==0 else "N/A")
        mm_cost_a_str = f"${mm_cost_a:,.2f}" if pd.notnull(mm_cost_a) and mm_cost_a != np.inf and mm_cost_a !=0 else ("$0.00" if mm_cost_a==0 else "N/A")
        mm_cost_b_str = f"${mm_cost_b:,.2f}" if pd.notnull(mm_cost_b) and mm_cost_b != np.inf and mm_cost_b !=0 else ("$0.00" if mm_cost_b==0 else "N/A")

        output_q2_str += (
            f"  {am}:\n"
            f"    单位直接成本: {direct_pu_str}\n"
            f"    船A - 最小多式联运单位成本: {mm_cost_a_str}, 适用性: {suitable_a}\n"
            f"    船B - 最小多式联运单位成本: {mm_cost_b_str}, 适用性: {suitable_b}\n"
        )

if len(df_q2_summary_for_print) > 1:
    output_q2_str += "\n后续客户 (海运适用性 - 简略格式，仅船A):\n"
    # 动态确定后续客户名称的最大打印宽度，用于对齐
    q2_subsequent_rows_to_print = df_q2_summary_for_print.iloc[1:min(5, len(df_q2_summary_for_print))]
    max_loc_len_q2 = 0
    if not q2_subsequent_rows_to_print.empty:
        max_loc_len_q2 = q2_subsequent_rows_to_print['Customer_Location'].astype(str).map(len).max()
    max_loc_len_q2 = max(max_loc_len_q2, 15) # 保证一个最小宽度

    for index, row in q2_subsequent_rows_to_print.iterrows():
        location_name_padded = f"{str(row['Customer_Location'])[:max_loc_len_q2]:<{max_loc_len_q2}}"
        
        am_details_lines_q2 = []
        for am_idx, am in enumerate(['AM1', 'AM2', 'AM3']):
            direct_pu = row.get(f'{am}_Direct_Cost_PU', np.inf)
            mm_cost_a = row.get(f'{am}_Min_MM_Cost_PU_ShipA', np.inf)
            suitable_a = row.get(f'{am}_Suitable_ShipA', "N/A (无有效海运路径)")

            direct_pu_str = f"${direct_pu:,.2f}" if pd.notnull(direct_pu) and direct_pu != np.inf and direct_pu !=0 else ("$0.00" if direct_pu==0 else "N/A")
            mm_cost_a_str = f"${mm_cost_a:,.2f}" if pd.notnull(mm_cost_a) and mm_cost_a != np.inf and mm_cost_a !=0 else ("$0.00" if mm_cost_a==0 else "N/A")
            
            am_detail_str = f"{am}[直运单位成本:{direct_pu_str},船A海运单位成本:{mm_cost_a_str},船A适用:{suitable_a}]"
            
            if am_idx == 0:
                am_details_lines_q2.append(f"{location_name_padded} {am_detail_str}")
            else:
                am_details_lines_q2.append(f"{' ' * max_loc_len_q2} {am_detail_str}")
        
        output_q2_str += "\n".join(am_details_lines_q2) + "\n"
else:
    output_q2_str += "\n后续客户 (海运适用性 - 简略格式，仅船A): 无更多数据可展示。\n"


# 最终合并输出
final_print_output = output_q1_str + "\n" + output_q2_str
print(final_print_output)