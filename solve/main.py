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
    # 创建 results 文件夹
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"创建 '{results_dir}' 文件夹")

    # --- 金额格式化字符串 ---
    float_format_csv = '%.2f' # 用于CSV文件中的浮点数格式 (金额保留两位小数)
    # 命令行打印时，f-string中的 :.2f 已经处理了两位小数

    # 加载和预处理数据
    print("--- 开始加载和预处理数据 ---")
    data_loader = DataLoader("data/DATA_original.xlsx")
    df_locations_raw, df_port_cust_dist_raw = data_loader.load_data()
    df_locations_processed = data_loader.preprocess_data(df_locations_raw, df_port_cust_dist_raw)
    print("--- 数据加载和预处理完毕 ---")

    # --- 解决问题1 ---
    print("\n--- 开始执行问题1 ---")
    q1_solver = Q1Solver(df_locations_processed)
    total_direct_cost_q1, df_with_q1_costs = q1_solver.solve()
    print(f"问题1估算的当前纯公路直接运输总成本 (3个月): ${total_direct_cost_q1:,.2f}") # 命令行输出已是两位小数
    q1_output_path = os.path.join(results_dir, "q1_direct_transport_costs_and_details.csv")
    df_with_q1_costs.to_csv(q1_output_path, index=False, float_format=float_format_csv) # CSV输出两位小数
    print(f"问题1详细成本和计算过程已保存到 {q1_output_path}")
    print("--- 问题1 执行完毕 ---")

    # --- 解决问题2 ---
    print("\n--- 开始执行问题2 ---")
    q2_solver = Q2Solver(df_with_q1_costs)
    df_final_q2, df_q2_summary = q2_solver.solve()
    q2_output_path = os.path.join(results_dir, "q2_multimodal_analysis_details.csv")
    df_final_q2.to_csv(q2_output_path, index=False, float_format=float_format_csv) # CSV输出两位小数
    print(f"问题2多式联运详细分析已保存到 {q2_output_path}")
    q2_summary_output_path = os.path.join(results_dir, "q2_multimodal_summary_by_customer.csv")
    df_q2_summary.to_csv(q2_summary_output_path, index=False, float_format=float_format_csv) # CSV输出两位小数
    print(f"问题2各客户多式联运摘要已保存到 {q2_summary_output_path}")
    print("--- 问题2 执行完毕 ---")

    # --- 解决问题3 ---
    print("\n--- 开始执行问题3 (COPT Optimizer) ---")
    cost_params_q3 = {'port_handling_charge_usd_unit': PORT_HANDLING_CHARGE_USD_UNIT}
    q3_optimizer = Q3Optimizer(df_with_q1_costs, AM_CAPACITY_ANNUAL, SHIPS_DATA, cost_params_q3)
    results_q3_optimization = q3_optimizer.optimize()
    print("--- 问题3 执行完毕 ---")

    # --- 保存问题3的结果到CSV (确保金额为两位小数) ---
    q3_base_path = os.path.join(results_dir, "q3_optimization_results")
    print(f"\n保存问题3的优化结果到CSV文件 (前缀: {q3_base_path})...")

    # 1. 总体摘要 (状态和总成本)
    total_optimal_cost_annual = results_q3_optimization.get('total_optimal_cost')
    annualized_q1_cost = total_direct_cost_q1 * 4 if total_direct_cost_q1 is not None else None

    summary_q3_data = {
        'Status': [results_q3_optimization['status']],
        'TotalOptimalCost_Annual': [total_optimal_cost_annual], # 金额
        'Annualized_DirectCost_Q1_Reference': [annualized_q1_cost], # 金额
        'TotalDirectVolume_Optimal': [results_q3_optimization.get('total_direct_volume')],
        'TotalCoastalVolume_Optimal': [results_q3_optimization.get('total_coastal_volume')],
        'ShipsUsed_Optimal': [', '.join(results_q3_optimization.get('ships_used', []))]
    }
    df_summary_q3 = pd.DataFrame(summary_q3_data)
    df_summary_q3.to_csv(f"{q3_base_path}_summary.csv", index=False, float_format=float_format_csv) # CSV输出两位小数
    print(f"  - 优化摘要已保存到: {q3_base_path}_summary.csv")

    # 2. 直接运输量 (优化后) - Volume 不是金额
    if results_q3_optimization.get('direct_shipments'):
        direct_shipments_data = []
        for (am, cust), volume in results_q3_optimization['direct_shipments'].items():
            direct_shipments_data.append({'AM': am, 'Customer': cust, 'AnnualVolume': volume})
        df_direct_q3 = pd.DataFrame(direct_shipments_data)
        df_direct_q3.to_csv(f"{q3_base_path}_direct_shipments.csv", index=False, float_format='%.0f') # 货量通常为整数
        print(f"  - 优化后的直接运输量已保存到: {q3_base_path}_direct_shipments.csv")

    # 3. 沿海运输量 (优化后) - Volume 不是金额
    if results_q3_optimization.get('coastal_shipments'):
        coastal_shipments_data = []
        for (am, cust, p_orig, q_dest, ship), volume in results_q3_optimization['coastal_shipments'].items():
            coastal_shipments_data.append({
                'AM': am, 'Customer': cust,
                'OriginPort': p_orig, 'DestinationPort': q_dest,
                'ShipType': ship, 'AnnualVolume': volume
            })
        df_coastal_q3 = pd.DataFrame(coastal_shipments_data)
        df_coastal_q3.to_csv(f"{q3_base_path}_coastal_shipments.csv", index=False, float_format='%.0f') # 货量通常为整数
        print(f"  - 优化后的沿海运输量已保存到: {q3_base_path}_coastal_shipments.csv")

    # 4. 各航线年航次数 (优化后) - Voyages 通常是1位或0位小数
    if results_q3_optimization.get('voyages_per_route_ship'):
        voyages_data = []
        for (ship, p_orig, q_dest), num_voyages in results_q3_optimization['voyages_per_route_ship'].items():
            voyages_data.append({
                'ShipType': ship, 'OriginPort': p_orig,
                'DestinationPort': q_dest, 'AnnualVoyages': num_voyages
            })
        df_voyages_q3 = pd.DataFrame(voyages_data)
        df_voyages_q3.to_csv(f"{q3_base_path}_annual_voyages.csv", index=False, float_format='%.1f') # 航次数保留1位小数
        print(f"  - 优化后的年航次数已保存到: {q3_base_path}_annual_voyages.csv")

    # --- 新增：保存问题3的优化结果到 TXT 文件 (确保金额为两位小数) ---
    q3_summary_txt_path = os.path.join(results_dir, "q3_optimization_summary.txt")
    print(f"\n保存问题3的优化摘要到 TXT 文件: {q3_summary_txt_path}...")

    with open(q3_summary_txt_path, 'w') as f:
        f.write(f"Status: {results_q3_optimization['status']}\n")
        
        total_optimal_cost_annual_q3 = results_q3_optimization.get('total_optimal_cost')
        if total_optimal_cost_annual_q3 is not None:
            # 对总成本进行格式化，保留两位小数
            formatted_total_cost_txt = "{:.2f}".format(total_optimal_cost_annual_q3)
            f.write(f"Total Optimal Cost: {formatted_total_cost_txt}\n")
        else:
            f.write("Total Optimal Cost: Not available\n")
        
        # 可选：在 TXT 文件中包含其他摘要信息 (已格式化)
        if annualized_q1_cost is not None:
             f.write(f"Annualized_DirectCost_Q1_Reference: {annualized_q1_cost:.2f}\n")
        if results_q3_optimization.get('total_direct_volume') is not None:
             f.write(f"TotalDirectVolume_Optimal: {results_q3_optimization.get('total_direct_volume'):.0f}\n")
        if results_q3_optimization.get('total_coastal_volume') is not None:
             f.write(f"TotalCoastalVolume_Optimal: {results_q3_optimization.get('total_coastal_volume'):.0f}\n")
        if results_q3_optimization.get('ships_used'):
             f.write(f"ShipsUsed_Optimal: {', '.join(results_q3_optimization.get('ships_used', []))}\n")
            
    print(f"  - 问题3优化摘要已保存到: {q3_summary_txt_path}")
    # --- 新增代码结束 ---

    # --- 控制台打印输出 (确保Q3金额为两位小数) ---
    output_q1_str = f"\n问题1：估算的当前纯公路直接运输总成本 (3个月): ${total_direct_cost_q1:,.2f}\n"
    output_q1_str += f"问题1结果详情已保存至 {q1_output_path}\n"

    output_q2_str = f"\n\n问题2：适合转向沿海运输的客户地点识别\n"
    output_q2_str += f"问题2结果详情已保存至 {q2_output_path} 及 {q2_summary_output_path}\n"

    output_q3_str = f"\n\n问题3：优化沿海运输系统设计结果 (使用 COPT 建模):\n"
    output_q3_str += f"求解状态: {results_q3_optimization['status']}\n"
    
    if total_optimal_cost_annual is not None:
        output_q3_str += f"最优/可行年总运输成本: ${total_optimal_cost_annual:,.2f}\n" # 确保两位小数
        if annualized_q1_cost is not None:
            output_q3_str += f"  对比年化纯公路成本 (Q1估算): ${annualized_q1_cost:,.2f}\n" # 确保两位小数
            if total_optimal_cost_annual < annualized_q1_cost:
                savings = annualized_q1_cost - total_optimal_cost_annual
                savings_percent = (savings / annualized_q1_cost) * 100 if annualized_q1_cost > 0 else 0
                output_q3_str += f"  通过优化方案可节省成本: ${savings:,.2f} (约 {savings_percent:.2f}%)\n" # 确保两位小数
    else:
        output_q3_str += "未能计算出优化方案的总成本。\n"
        
    output_q3_str += f"问题3优化结果详情已保存至多个CSV文件，前缀为: {q3_base_path}\n"
    output_q3_str += f"问题3优化摘要也已保存至TXT文件: {q3_summary_txt_path}\n" # 在控制台输出中也提及TXT文件
    # 可以选择性地打印Q3的其他摘要信息到控制台，例如：
    output_q3_str += f"  优化方案中使用的船型: {', '.join(results_q3_optimization.get('ships_used', ['无']))}\n"
    output_q3_str += f"  优化方案中总直接运输量: {results_q3_optimization.get('total_direct_volume', 0):,.0f} 辆/年\n"
    output_q3_str += f"  优化方案中总沿海运输量: {results_q3_optimization.get('total_coastal_volume', 0):,.0f} 辆/年\n"


    final_print_output = output_q1_str + "\n" + output_q2_str + "\n" + output_q3_str
    print(final_print_output)

if __name__ == "__main__":
    main()