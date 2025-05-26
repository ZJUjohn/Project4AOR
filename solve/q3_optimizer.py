import pandas as pd
import numpy as np
import coptpy as cp
from coptpy import COPT
from solve.config import (
    AM_TO_PORT_DIST, PORT_STAY_SHIP_DAYS_COSTING,
    SEA_DISTANCE_NM_ONE_WAY,
    get_optimized_land_unit_cost,
    TIME_PER_ONE_WAY_VOYAGE_LEG_DAYS,
    ANNUAL_OPERATING_DAYS,
    CUSTOMER_SHIP_ELIGIBILITY
)

class Q3Optimizer:
    def __init__(self, df_with_q1_costs_input, am_annual_capacity_map, ships_data_map_config, cost_params_map_config):
        self.df_model_input = df_with_q1_costs_input.copy()
        self.am_annual_capacity_map = am_annual_capacity_map
        self.ships_data_map = ships_data_map_config
        self.cost_params_map = cost_params_map_config
        self.customer_ship_eligibility = CUSTOMER_SHIP_ELIGIBILITY

    def optimize(self):
        """
        使用COPT优化器解决沿海运输系统设计和装运计划问题。
        返回:
            dict: 包含求解状态、最优成本和决策变量值的优化结果。
        """
        print("\n--- 开始执行问题3：COPT优化模型 (基于计算的航行时间[config.py]和365天运营) ---")

        AMS = ['AM1', 'AM2', 'AM3']
        if 'Cust_ID' not in self.df_model_input.columns:
            self.df_model_input['Cust_ID'] = self.df_model_input['Customer_Location']
        CUSTOMERS = self.df_model_input['Cust_ID'].dropna().unique().tolist()
        PORTS = ['Chennai Port', 'Pipavav Port']
        SHIPS = list(self.ships_data_map.keys())
        SEA_ROUTES_TUPLES = [(p_orig, p_dest) for p_orig in PORTS for p_dest in PORTS if p_orig != p_dest]

        annual_demands = {}
        for idx, row in self.df_model_input.iterrows():
            cust_id = row['Cust_ID']
            if pd.isna(cust_id): continue
            for am in AMS:
                demand_3mo = 0
                if f'{am}_M1' in row and f'{am}_M2' in row and f'{am}_M3' in row:
                    for m_col in [f'{am}_M1', f'{am}_M2', f'{am}_M3']:
                        demand_3mo += pd.to_numeric(row.get(m_col, 0), errors='coerce')
                else:
                    demand_3mo = pd.to_numeric(row.get(f'{am}_Total_Demand_3Mo', 0), errors='coerce')
                annual_demands[(am, cust_id)] = demand_3mo * 4

        direct_truck_unit_costs = {}
        fml_unit_costs = {}
        lml_unit_costs = {}

        for idx, row in self.df_model_input.iterrows():
            cust_id = row['Cust_ID']
            if pd.isna(cust_id): continue
            for am in AMS:
                cost = row.get(f'{am}_Direct_Cost_Per_Unit', np.inf)
                is_demand_positive = annual_demands.get((am, cust_id), 0) > 0
                if pd.isna(cost) or cost == np.inf or (cost == 0 and is_demand_positive) or cost < 0 :
                    direct_truck_unit_costs[(am, cust_id)] = 1e9 if is_demand_positive else 0
                else:
                    direct_truck_unit_costs[(am, cust_id)] = cost

        for am in AMS:
            for port in PORTS:
                fml_unit_costs[(am, port)] = get_optimized_land_unit_cost(AM_TO_PORT_DIST[am][port])

        for idx, row in self.df_model_input.iterrows():
            cust_id = row['Cust_ID']
            if pd.isna(cust_id): continue
            for port in PORTS:
                dist_col = 'Dist_to_Chennai_Port' if port == 'Chennai Port' else 'Dist_to_Pipavav_Port'
                lml_unit_costs[(port, cust_id)] = get_optimized_land_unit_cost(row.get(dist_col))

        port_handling_cost = self.cost_params_map['port_handling_charge_usd_unit']

        sea_voyage_var_cost_per_trip = {}
        ship_annual_fixed_cost_map = {}
        ship_capacity_map = {}
        valid_voyage_triplets = set()

        for s_name in SHIPS:
            ship_annual_fixed_cost_map[s_name] = self.ships_data_map[s_name]['fixed_cost_3months_usd'] * 4
            ship_capacity_map[s_name] = self.ships_data_map[s_name]['capacity']
            s_data = self.ships_data_map[s_name]
            if s_data['speed_avg_knots'] is None or s_data['speed_avg_knots'] == 0:
                voyage_time_days_one_way_transit = np.inf
            else:
                voyage_time_days_one_way_transit = SEA_DISTANCE_NM_ONE_WAY / s_data['speed_avg_knots'] / 24
            vcs_one_leg_transit = voyage_time_days_one_way_transit * s_data['vcs_usd_day']
            vcp_at_ports_for_leg = (PORT_STAY_SHIP_DAYS_COSTING * 2) * s_data['vcp_usd_day']
            var_cost_per_one_way_trip = vcs_one_leg_transit + vcp_at_ports_for_leg
            for r_orig, r_dest in SEA_ROUTES_TUPLES:
                sea_voyage_var_cost_per_trip[(s_name, r_orig, r_dest)] = var_cost_per_one_way_trip
                valid_voyage_triplets.add((s_name, r_orig, r_dest))

        env = cp.Envr()
        model = env.createModel("CoastalShippingOptimization_COPT_NoOrderSplit_ChineseComments")

        x_direct = {}
        for i in AMS:
            for j in CUSTOMERS:
                if annual_demands.get((i, j), 0) > 0:
                     x_direct[i,j] = model.addVar(lb=0, name=f"X_direct_{i}_{j.replace(' ','_')}")

        x_coastal = {}
        for i in AMS:
            for j in CUSTOMERS:
                if annual_demands.get((i, j), 0) > 0:
                    for p_orig in PORTS:
                        for q_dest in PORTS:
                            if p_orig != q_dest:
                                for s in SHIPS:
                                    is_eligible = True
                                    if self.customer_ship_eligibility:
                                        am_eligibility = self.customer_ship_eligibility.get(i, {})
                                        cust_eligibility = am_eligibility.get(j, {})
                                        if not cust_eligibility.get(s, True):
                                            is_eligible = False
                                    if is_eligible:
                                        x_coastal[i,j,p_orig,q_dest,s] = model.addVar(
                                            lb=0, name=f"X_coastal_{i}_{j.replace(' ','_')}_{p_orig.replace(' ','_')}_{q_dest.replace(' ','_')}_{s}"
                                        )
        # --- 新增：选择运输模式的二元变量 ---
        # y_direct_chosen[am, cust]: 1 如果 (AM, Cust) 的需求通过公路直运满足
        # y_coastal_shipA_chosen[am, cust]: 1 如果 (AM, Cust) 的需求通过 ShipA 沿海运输满足
        # y_coastal_shipB_chosen[am, cust]: 1 如果 (AM, Cust) 的需求通过 ShipB 沿海运输满足
        y_direct_chosen = {}
        y_coastal_shipA_chosen = {}
        y_coastal_shipB_chosen = {}

        for i in AMS:
            for j in CUSTOMERS:
                if annual_demands.get((i, j), 0) > 0:
                    y_direct_chosen[i,j] = model.addVar(vtype=COPT.BINARY, name=f"Y_direct_{i}_{j.replace(' ','_')}")
                    if 'ShipA' in SHIPS:
                        y_coastal_shipA_chosen[i,j] = model.addVar(vtype=COPT.BINARY, name=f"Y_coastal_ShipA_{i}_{j.replace(' ','_')}")
                    if 'ShipB' in SHIPS:
                        y_coastal_shipB_chosen[i,j] = model.addVar(vtype=COPT.BINARY, name=f"Y_coastal_ShipB_{i}_{j.replace(' ','_')}")
        # --- 新增结束 ---

        v_annual_voyages = {}
        for s_trip in SHIPS:
            for p_orig_trip in PORTS:
                for q_dest_trip in PORTS:
                    if p_orig_trip != q_dest_trip:
                        if (s_trip, p_orig_trip, q_dest_trip) in valid_voyage_triplets:
                            v_annual_voyages[s_trip,p_orig_trip,q_dest_trip] = model.addVar(
                                lb=0, name=f"V_annual_voyages_{s_trip}_{p_orig_trip.replace(' ','_')}_{q_dest_trip.replace(' ','_')}"
                            )
        u_use_ship_type = {}
        for s_use in SHIPS:
            u_use_ship_type[s_use] = model.addVar(vtype=COPT.BINARY, name=f"U_use_ship_{s_use}")

        obj = cp.LinExpr()
        for (i_obj_d, j_obj_d), var in x_direct.items():
            cost = direct_truck_unit_costs.get((i_obj_d,j_obj_d), 1e9)
            obj += float(cost) * var
        for (i_obj_c,j_obj_c,p_orig_obj_c,q_dest_obj_c,s_obj_c), var in x_coastal.items():
            fml_c = fml_unit_costs.get((i_obj_c,p_orig_obj_c), 1e9)
            lml_c = lml_unit_costs.get((q_dest_obj_c,j_obj_c), 1e9)
            leg_cost = fml_c + (2 * port_handling_cost) + lml_c
            obj += float(leg_cost) * var
        for (s_obj_v, p_orig_obj_v, q_dest_obj_v), var in v_annual_voyages.items():
            var_c_sea_trip = sea_voyage_var_cost_per_trip.get((s_obj_v, p_orig_obj_v, q_dest_obj_v), 1e9)
            obj += float(var_c_sea_trip) * var
        for s_obj_fc, var in u_use_ship_type.items():
            fixed_c = ship_annual_fixed_cost_map.get(s_obj_fc, 0)
            obj += float(fixed_c) * var
        model.setObjective(obj, sense=COPT.MINIMIZE)

        # --- 修改/新增：约束条件 ---
        for i_dem in AMS:
            for j_dem in CUSTOMERS:
                demand_val = annual_demands.get((i_dem,j_dem), 0)
                if demand_val > 0:
                    # 约束1.1: 每个 (AM, Cust) 必须选择一种运输模式
                    mode_selection_lhs = cp.LinExpr()
                    mode_selection_lhs += y_direct_chosen[i_dem, j_dem]
                    if 'ShipA' in SHIPS:
                        mode_selection_lhs += y_coastal_shipA_chosen[i_dem, j_dem]
                    if 'ShipB' in SHIPS:
                        mode_selection_lhs += y_coastal_shipB_chosen[i_dem, j_dem]
                    model.addConstr(mode_selection_lhs == 1, name=f"ChooseOneMode_{i_dem}_{j_dem.replace(' ','_')}")

                    # 约束1.2: 如果选择了公路直运，则公路直运量等于总需求，否则为0
                    if (i_dem,j_dem) in x_direct:
                        model.addConstr(x_direct[i_dem,j_dem] == demand_val * y_direct_chosen[i_dem,j_dem],
                                        name=f"Link_DirectVolume_{i_dem}_{j_dem.replace(' ','_')}")

                    # 约束 1.3 & 1.4: 如果选择了特定船型沿海运输，则该船型总沿海运输量等于总需求，否则为0
                    # 同时，如果未选择该船型，则其他船型的沿海运输量必须为0
                    total_coastal_shipA_for_am_cust = cp.LinExpr()
                    if 'ShipA' in SHIPS:
                        for p_orig_dem in PORTS:
                            for q_dest_dem in PORTS:
                                if p_orig_dem != q_dest_dem:
                                    if (i_dem,j_dem,p_orig_dem,q_dest_dem,'ShipA') in x_coastal:
                                        total_coastal_shipA_for_am_cust += x_coastal[i_dem,j_dem,p_orig_dem,q_dest_dem,'ShipA']
                        model.addConstr(total_coastal_shipA_for_am_cust == demand_val * y_coastal_shipA_chosen[i_dem,j_dem],
                                        name=f"Link_CoastalShipAVolume_{i_dem}_{j_dem.replace(' ','_')}")

                    total_coastal_shipB_for_am_cust = cp.LinExpr()
                    if 'ShipB' in SHIPS:
                        for p_orig_dem in PORTS:
                            for q_dest_dem in PORTS:
                                if p_orig_dem != q_dest_dem:
                                    if (i_dem,j_dem,p_orig_dem,q_dest_dem,'ShipB') in x_coastal:
                                        total_coastal_shipB_for_am_cust += x_coastal[i_dem,j_dem,p_orig_dem,q_dest_dem,'ShipB']
                        model.addConstr(total_coastal_shipB_for_am_cust == demand_val * y_coastal_shipB_chosen[i_dem,j_dem],
                                        name=f"Link_CoastalShipBVolume_{i_dem}_{j_dem.replace(' ','_')}")
                    
                    # 约束1.5 (原需求满足约束仍然需要，以确保总运输量等于需求)
                    # 这个约束现在因为上面的等式约束，其实是冗余的，但保留也没问题，或者可以移除
                    # lhs_demand = cp.LinExpr()
                    # if (i_dem,j_dem) in x_direct:
                    #     lhs_demand += x_direct[i_dem,j_dem]
                    # for p_orig_dem in PORTS:
                    #     for q_dest_dem in PORTS:
                    #         if p_orig_dem != q_dest_dem:
                    #             for s_dem in SHIPS:
                    #                 if (i_dem,j_dem,p_orig_dem,q_dest_dem,s_dem) in x_coastal:
                    #                     lhs_demand += x_coastal[i_dem,j_dem,p_orig_dem,q_dest_dem,s_dem]
                    # model.addConstr(lhs_demand >= demand_val, name=f"Demand_Satisfy_{i_dem}_{j_dem.replace(' ','_')}")

        for i_cap in AMS:
            lhs_supply = cp.LinExpr()
            for j_cap_d in CUSTOMERS:
                if (i_cap,j_cap_d) in x_direct:
                    lhs_supply += x_direct[i_cap,j_cap_d]
            for j_cap_c in CUSTOMERS:
                for p_orig_cap in PORTS:
                    for q_dest_cap in PORTS:
                        if p_orig_cap != q_dest_cap:
                            for s_cap in SHIPS:
                                if (i_cap,j_cap_c,p_orig_cap,q_dest_cap,s_cap) in x_coastal:
                                    lhs_supply += x_coastal[i_cap,j_cap_c,p_orig_cap,q_dest_cap,s_cap]
            model.addConstr(lhs_supply <= self.am_annual_capacity_map[i_cap], name=f"Supply_Capacity_{i_cap}")

        for s_sea_cap in SHIPS:
            for p_orig_sea_cap in PORTS:
                for q_dest_sea_cap in PORTS:
                    if p_orig_sea_cap == q_dest_sea_cap: continue
                    voyage_key_constr = (s_sea_cap, p_orig_sea_cap, q_dest_sea_cap)
                    if voyage_key_constr in v_annual_voyages:
                        lhs_sea_volume_on_route_ship = cp.LinExpr()
                        for i_sea_vol in AMS:
                            for j_sea_vol in CUSTOMERS:
                                if (i_sea_vol,j_sea_vol,p_orig_sea_cap,q_dest_sea_cap,s_sea_cap) in x_coastal:
                                    lhs_sea_volume_on_route_ship += x_coastal[i_sea_vol,j_sea_vol,p_orig_sea_cap,q_dest_sea_cap,s_sea_cap]
                        rhs_capacity = v_annual_voyages[voyage_key_constr] * ship_capacity_map[s_sea_cap]
                        model.addConstr(lhs_sea_volume_on_route_ship <= rhs_capacity,
                                        name=f"Sea_Route_Capacity_{s_sea_cap}_{p_orig_sea_cap.replace(' ','_')}_{q_dest_sea_cap.replace(' ','_')}")

        M_large_number_voyages = sum(annual_demands.values()) * 2 if annual_demands else 1e7
        for s_link_v in SHIPS:
            if s_link_v in u_use_ship_type:
                total_voyages_for_ship_s = cp.LinExpr()
                for p_orig_link in PORTS:
                    for q_dest_link in PORTS:
                        if (s_link_v, p_orig_link, q_dest_link) in v_annual_voyages:
                            total_voyages_for_ship_s += v_annual_voyages[s_link_v, p_orig_link, q_dest_link]
                model.addConstr(total_voyages_for_ship_s <= M_large_number_voyages * u_use_ship_type[s_link_v],
                                name=f"Link_TotalVoyages_To_UseShip_{s_link_v}")
        
        for s_max_voy_time in SHIPS:
            if s_max_voy_time in u_use_ship_type:
                total_operational_time_for_ship_s = cp.LinExpr()
                time_per_leg_s = TIME_PER_ONE_WAY_VOYAGE_LEG_DAYS.get(s_max_voy_time)
                if time_per_leg_s is None or time_per_leg_s <= 0 or time_per_leg_s == np.inf:
                    print(f"  警告: 船型 {s_max_voy_time} 的单向航程时间无效: {time_per_leg_s}。将跳过为其添加最大航行时间约束。")
                    continue
                for p_orig_mv_time in PORTS:
                    for q_dest_mv_time in PORTS:
                        if (s_max_voy_time, p_orig_mv_time, q_dest_mv_time) in v_annual_voyages:
                            total_operational_time_for_ship_s += v_annual_voyages[s_max_voy_time, p_orig_mv_time, q_dest_mv_time] * time_per_leg_s
                model.addConstr(total_operational_time_for_ship_s <= ANNUAL_OPERATING_DAYS * u_use_ship_type[s_max_voy_time],
                                name=f"Max_Annual_OpTime_{s_max_voy_time}")

        model.setParam("Logging", 1)
        model.setParam("TimeLimit", 300) # 可根据需要调整时间限制

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

        results_q3 = {
            "status": status_str, "total_optimal_cost": None,
            "direct_shipments": {}, "coastal_shipments": {},
            "voyages_per_route_ship": {}, "ships_used": [],
            "total_coastal_volume": 0, "total_direct_volume":0,
            "model_solve_status_code": status_code
        }

        if status_code == COPT.OPTIMAL or status_code == COPT.SUBOPTIMAL:
            if hasattr(model, 'objval') and model.objval is not None:
                results_q3["total_optimal_cost"] = model.objval

            for (i_res_d,j_res_d), var in x_direct.items():
                if var.x > 1e-4:
                    results_q3["direct_shipments"][(i_res_d,j_res_d)] = var.x
                    results_q3["total_direct_volume"] += var.x
            for (i_res_c,j_res_c,p_orig_c,q_dest_c,s_c), var in x_coastal.items():
                if var.x > 1e-4:
                    key_c = (i_res_c,j_res_c,p_orig_c,q_dest_c,s_c)
                    results_q3["coastal_shipments"][key_c] = var.x
                    results_q3["total_coastal_volume"] += var.x
            for s_res_u, var in u_use_ship_type.items():
                if var.x > 0.5:
                    results_q3["ships_used"].append(s_res_u)
            for (s_res_v, p_res_v, q_res_v), var in v_annual_voyages.items():
                 if var.x > 1e-4:
                    results_q3["voyages_per_route_ship"][(s_res_v,p_res_v,q_res_v)] = var.x
        return results_q3