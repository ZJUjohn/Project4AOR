import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from .config import (
    PLOTS_DIR, AM_COLORS, SHIP_COLORS, PORT_NODE_COLOR,
    DIRECT_TRUCK_COLOR, COASTAL_TRUCK_COLOR, SEA_LINK_COLOR, MODE_SPLIT_NODE_COLOR,
    PLOTLY_FONT_FAMILY, plotly_font_dict, plotly_title_font_dict,
    plotly_node_label_font_dict, plotly_link_label_font_dict
)

def plot_q3_sankey_flows_plotly(df_q3_direct_vol, df_q3_coastal_vol): # output_dir removed, uses PLOTS_DIR from config
    """
    Generates two Sankey diagrams using Plotly for Q3 shipment flows.
    Chart 4a: AM -> Mode Split (Direct/Coastal) -> Aggregated Destinations.
    Chart 4b: AM -> Mode Split -> Detailed Path (Coastal: AM->Origin->Ship->Dest) -> Individual Customers.
    Saves both PNG and HTML versions.
    """
    print("Generating Chart 4 (Plotly Sankey): Q3 Optimized Shipment Flows...")

    if (df_q3_direct_vol is None or df_q3_direct_vol.empty) and \
       (df_q3_coastal_vol is None or df_q3_coastal_vol.empty):
        print("Warning (Chart 4 - Plotly): Both direct and coastal Q3 shipment data are empty or None. Skipping Sankey diagrams.")
        return

    # Create copies to safely modify
    df_q3_direct_vol_processed = df_q3_direct_vol.copy() if df_q3_direct_vol is not None else pd.DataFrame(columns=['AM', 'Customer', 'AnnualVolume'])
    df_q3_coastal_vol_processed = df_q3_coastal_vol.copy() if df_q3_coastal_vol is not None else pd.DataFrame(columns=['AM', 'OriginPort', 'ShipType', 'DestinationPort', 'Customer', 'AnnualVolume'])


    # Ensure 'ShipType' column exists in coastal data
    if not df_q3_coastal_vol_processed.empty:
        if 'ShipType' not in df_q3_coastal_vol_processed.columns:
            if 'Ship' in df_q3_coastal_vol_processed.columns:
                df_q3_coastal_vol_processed = df_q3_coastal_vol_processed.rename(columns={'Ship': 'ShipType'})
            else:
                print("Warning (Chart 4 - Plotly): 'ShipType' or 'Ship' column not found in coastal data. Using default 'UnknownShip'.")
                df_q3_coastal_vol_processed['ShipType'] = 'UnknownShip'
    else: # If it was None or became empty
        df_q3_coastal_vol_processed = pd.DataFrame(columns=['AM', 'OriginPort', 'ShipType', 'DestinationPort', 'Customer', 'AnnualVolume'])

    if df_q3_direct_vol_processed.empty:
        df_q3_direct_vol_processed = pd.DataFrame(columns=['AM', 'Customer', 'AnnualVolume'])
    
    # Ensure required columns exist even if dataframes were initially empty
    for df, cols in [(df_q3_direct_vol_processed, ['AM', 'Customer', 'AnnualVolume']), 
                     (df_q3_coastal_vol_processed, ['AM', 'OriginPort', 'ShipType', 'DestinationPort', 'Customer', 'AnnualVolume'])]:
        for col_check in cols:
            if col_check not in df.columns:
                df[col_check] = pd.NA # Or appropriate dtype like '' for strings, 0 for numbers


    # --- Chart 4a: AM -> Mode Split -> Aggregated Destinations ---
    print("  Generating Chart 4a: AM -> Mode Split -> Aggregated Destinations...")
    try:
        nodes_4a_labels, nodes_4a_colors, nodes_4a_x, nodes_4a_y = [], [], [], []
        links_4a_source, links_4a_target, links_4a_value, links_4a_color, links_4a_hoverlabel, links_4a_textlabel = [], [], [], [], [], []
        label_to_idx_4a = {}
        
        y_pos_total_nodes_in_x = {} 

        def add_node_4a(label, x_coord, color):
            if label not in label_to_idx_4a:
                label_to_idx_4a[label] = len(nodes_4a_labels)
                nodes_4a_labels.append(label)
                nodes_4a_colors.append(color)
                nodes_4a_x.append(x_coord)
                y_pos_total_nodes_in_x[x_coord] = y_pos_total_nodes_in_x.get(x_coord, 0) + 1
        
        all_ams_4a = sorted(pd.concat([df_q3_direct_vol_processed['AM'], df_q3_coastal_vol_processed['AM']]).dropna().unique())
        for am in all_ams_4a: add_node_4a(f"AM: {am}", 0.05, AM_COLORS.get(am, 'grey'))
        
        direct_mode_label = "直运 (公路)"
        coastal_mode_label = "海运路径 (多式联运)"
        add_node_4a(direct_mode_label, 0.35, MODE_SPLIT_NODE_COLOR)
        add_node_4a(coastal_mode_label, 0.35, MODE_SPLIT_NODE_COLOR)

        unique_origin_ports_4a = sorted(df_q3_coastal_vol_processed['OriginPort'].dropna().unique())
        for op in unique_origin_ports_4a: add_node_4a(f"始发港: {op.replace(' Port','')}", 0.65, PORT_NODE_COLOR)

        dest_direct_label = "客户 (直运)"
        dest_coastal_label = "客户 (经海运)"
        add_node_4a(dest_direct_label, 0.95, DIRECT_TRUCK_COLOR.replace('0.7','0.9')) # Make color more solid
        add_node_4a(dest_coastal_label, 0.95, SEA_LINK_COLOR.replace('0.7','0.9'))   # Make color more solid

        # Assign Y positions
        # Ensure x-coordinates are processed in sorted order for consistent y-assignment
        sorted_x_coords = sorted(y_pos_total_nodes_in_x.keys())
        for x_coord_key in sorted_x_coords:
            count_in_layer = y_pos_total_nodes_in_x[x_coord_key]
            current_y_idx_in_layer = 0
            for i_node, node_x_val in enumerate(nodes_4a_x):
                if node_x_val == x_coord_key: # Check if this node belongs to the current x-layer
                    nodes_4a_y.append((current_y_idx_in_layer + 0.5) / count_in_layer if count_in_layer > 0 else 0.5)
                    current_y_idx_in_layer +=1
        if len(nodes_4a_y) < len(nodes_4a_labels): # Fallback
             nodes_4a_y.extend([0.5] * (len(nodes_4a_labels) - len(nodes_4a_y)))

        # Links for 4a
        if not df_q3_direct_vol_processed.empty and 'AM' in df_q3_direct_vol_processed and 'AnnualVolume' in df_q3_direct_vol_processed:
            direct_grouped = df_q3_direct_vol_processed.groupby('AM')['AnnualVolume'].sum()
            for am, volume in direct_grouped.items():
                if volume > 0.1:
                    s_idx, t_idx = label_to_idx_4a.get(f"AM: {am}"), label_to_idx_4a.get(direct_mode_label)
                    if s_idx is not None and t_idx is not None:
                        links_4a_source.append(s_idx); links_4a_target.append(t_idx); links_4a_value.append(volume)
                        links_4a_color.append(AM_COLORS.get(am, 'grey').replace('0.9', '0.6')) # Lighter for link
                        links_4a_hoverlabel.append(f"{am} → {direct_mode_label}: {volume:,.0f} 台"); links_4a_textlabel.append(f"{volume/1000:,.1f}K")

        if not df_q3_coastal_vol_processed.empty and 'AM' in df_q3_coastal_vol_processed and 'AnnualVolume' in df_q3_coastal_vol_processed:
            coastal_grouped_am = df_q3_coastal_vol_processed.groupby('AM')['AnnualVolume'].sum()
            for am, volume in coastal_grouped_am.items():
                if volume > 0.1:
                    s_idx, t_idx = label_to_idx_4a.get(f"AM: {am}"), label_to_idx_4a.get(coastal_mode_label)
                    if s_idx is not None and t_idx is not None:
                        links_4a_source.append(s_idx); links_4a_target.append(t_idx); links_4a_value.append(volume)
                        links_4a_color.append(AM_COLORS.get(am, 'grey').replace('0.9', '0.6'))
                        links_4a_hoverlabel.append(f"{am} → {coastal_mode_label}: {volume:,.0f} 台"); links_4a_textlabel.append(f"{volume/1000:,.1f}K")

        total_direct_volume = df_q3_direct_vol_processed['AnnualVolume'].sum() if not df_q3_direct_vol_processed.empty else 0
        if total_direct_volume > 0.1:
            s_idx, t_idx = label_to_idx_4a.get(direct_mode_label), label_to_idx_4a.get(dest_direct_label)
            if s_idx is not None and t_idx is not None:
                links_4a_source.append(s_idx); links_4a_target.append(t_idx); links_4a_value.append(total_direct_volume)
                links_4a_color.append(DIRECT_TRUCK_COLOR);
                links_4a_hoverlabel.append(f"{direct_mode_label} → {dest_direct_label}: {total_direct_volume:,.0f} 台"); links_4a_textlabel.append(f"{total_direct_volume/1000:,.1f}K")

        if not df_q3_coastal_vol_processed.empty and 'AM' in df_q3_coastal_vol_processed and 'OriginPort' in df_q3_coastal_vol_processed and 'AnnualVolume' in df_q3_coastal_vol_processed:
            coastal_am_op_grouped = df_q3_coastal_vol_processed.groupby(['AM', 'OriginPort'])['AnnualVolume'].sum().reset_index()
            for _, row in coastal_am_op_grouped.iterrows():
                if row['AnnualVolume'] > 0.1:
                    s_idx, t_idx = label_to_idx_4a.get(coastal_mode_label), label_to_idx_4a.get(f"始发港: {row['OriginPort'].replace(' Port','')}")
                    if s_idx is not None and t_idx is not None:
                        links_4a_source.append(s_idx); links_4a_target.append(t_idx); links_4a_value.append(row['AnnualVolume'])
                        links_4a_color.append(COASTAL_TRUCK_COLOR)
                        links_4a_hoverlabel.append(f"{coastal_mode_label} (来自 {row['AM']}) → {row['OriginPort']}: {row['AnnualVolume']:,.0f} 台"); links_4a_textlabel.append(f"{row['AnnualVolume']/1000:,.1f}K")
        
        if not df_q3_coastal_vol_processed.empty and 'OriginPort' in df_q3_coastal_vol_processed and 'AnnualVolume' in df_q3_coastal_vol_processed:
            coastal_op_total_dest = df_q3_coastal_vol_processed.groupby('OriginPort')['AnnualVolume'].sum().reset_index()
            for _, row in coastal_op_total_dest.iterrows():
                if row['AnnualVolume'] > 0.1:
                    s_idx, t_idx = label_to_idx_4a.get(f"始发港: {row['OriginPort'].replace(' Port','')}"), label_to_idx_4a.get(dest_coastal_label)
                    if s_idx is not None and t_idx is not None:
                        links_4a_source.append(s_idx); links_4a_target.append(t_idx); links_4a_value.append(row['AnnualVolume'])
                        links_4a_color.append(SEA_LINK_COLOR)
                        links_4a_hoverlabel.append(f"{row['OriginPort']} → {dest_coastal_label}: {row['AnnualVolume']:,.0f} 台"); links_4a_textlabel.append(f"{row['AnnualVolume']/1000:,.1f}K")

        if not links_4a_source: # No links were created
            print("  Warning (Chart 4a): No links to draw for Sankey. Skipping Chart 4a.")
        else:
            fig4a = go.Figure(data=[go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=30, thickness=25, line=dict(color="black", width=0.5),
                    label=nodes_4a_labels, color=nodes_4a_colors, x=nodes_4a_x, y=nodes_4a_y,
                    hovertemplate='%{label}<extra></extra>' # Show only label on node hover
                ),
                link=dict(
                    source=links_4a_source, target=links_4a_target, value=links_4a_value,
                    color=links_4a_color,
                    label=links_4a_textlabel, 
                    hovertemplate='%{customdata}<extra></extra>', customdata=links_4a_hoverlabel
                ),
                textfont=plotly_node_label_font_dict 
            )])
            fig4a.update_layout(
                title_text="图 4a: 聚合运输流程分析 (AM → 模式 → 目的地类型)",
                font=plotly_font_dict, 
                title_font=plotly_title_font_dict,
                height=800, width=1400,
                margin=dict(t=80, b=40, l=40, r=40)
            )
            fig4a.write_html(os.path.join(PLOTS_DIR, "04a_aggregated_flows_sankey.html"))
            fig4a.write_image(os.path.join(PLOTS_DIR, "04a_aggregated_flows_sankey.png"), scale=2) # Increased scale
            print(f"  Chart 4a (Plotly Sankey) saved to {os.path.join(PLOTS_DIR, '04a_aggregated_flows_sankey.png')} and .html")

    except Exception as e:
        print(f"Error generating Chart 4a (Aggregated Plotly Sankey): {e}")
        import traceback
        traceback.print_exc()

    # --- Chart 4b: Detailed AM -> Mode Split -> Path -> Individual Customers ---
    print("  Generating Chart 4b: Detailed End-to-End Coastal and Direct Flows...")
    try:
        node_labels_4b, node_colors_4b, node_x_4b, node_y_4b = [], [], [], []
        link_source_4b, link_target_4b, link_value_4b, link_color_4b, link_hover_4b = [], [], [], [], []
        label_to_idx_4b = {}
        
        # Helper to add nodes and store them in layer lists for Y-position calculation
        def add_node_4b(label, x_coord, color, layer_nodes_list):
            if label not in label_to_idx_4b:
                label_to_idx_4b[label] = len(node_labels_4b)
                node_labels_4b.append(label)
                node_colors_4b.append(color)
                node_x_4b.append(x_coord)
                layer_nodes_list.append(label) # Add to the layer list

        layer_am, layer_op, layer_ship, layer_dp, layer_cust = [], [], [], [], []

        # Layer 1: AMs
        all_ams_b = sorted(pd.concat([df_q3_direct_vol_processed['AM'], df_q3_coastal_vol_processed['AM']]).dropna().unique())
        for am in all_ams_b: add_node_4b(f"AM: {am}", 0.01, AM_COLORS.get(am, 'grey'), layer_am)

        # Layer 2: Origin Ports (Coastal Only)
        unique_ops_b = sorted(df_q3_coastal_vol_processed['OriginPort'].dropna().unique())
        for op in unique_ops_b: add_node_4b(f"Origin: {op.replace(' Port','')}", 0.25, PORT_NODE_COLOR, layer_op)

        # Layer 3: Ship Types (Coastal Only)
        unique_ships_b = sorted(df_q3_coastal_vol_processed['ShipType'].dropna().unique())
        for st in unique_ships_b: add_node_4b(f"Ship: {st}", 0.50, SHIP_COLORS.get(st, 'grey'), layer_ship)
        
        # Layer 4: Destination Ports (Coastal Only)
        unique_dps_b = sorted(df_q3_coastal_vol_processed['DestinationPort'].dropna().unique())
        for dp in unique_dps_b: add_node_4b(f"Dest Port: {dp.replace(' Port','')}", 0.75, PORT_NODE_COLOR, layer_dp)

        # Layer 5: Customers
        all_customers_df = pd.concat([
            df_q3_direct_vol_processed[['Customer', 'AM']],
            df_q3_coastal_vol_processed[['Customer', 'AM']]
        ]).dropna(subset=['Customer']).drop_duplicates(subset=['Customer'])
        
        MAX_CUSTOMERS_DISPLAYED = 50 
        if len(all_customers_df) > MAX_CUSTOMERS_DISPLAYED:
            print(f"  Warning (Chart 4b): Number of unique customers ({len(all_customers_df)}) is large. Consider aggregation for clarity.")
        
        for _, cust_row in all_customers_df.iterrows():
            cust_label = f"Cust: {cust_row['Customer']}"
            cust_color = AM_COLORS.get(cust_row['AM'], 'rgba(180,180,180,0.7)') 
            add_node_4b(cust_label, 0.99, cust_color, layer_cust)

        # Calculate Y positions for nodes in each layer
        all_layers_for_y = [layer_am, layer_op, layer_ship, layer_dp, layer_cust]
        temp_node_y_map = {} # Store y positions before assigning to node_y_4b to ensure correct order

        for layer_nodes_list in all_layers_for_y:
            num_in_layer = len(layer_nodes_list)
            if num_in_layer == 0: continue
            for i, node_label_in_layer in enumerate(layer_nodes_list):
                idx_in_main_list = label_to_idx_4b.get(node_label_in_layer)
                if idx_in_main_list is not None:
                     temp_node_y_map[idx_in_main_list] = (i + 0.5) / num_in_layer if num_in_layer > 0 else 0.5
        
        # Populate node_y_4b in the correct order of node_labels_4b
        for i in range(len(node_labels_4b)):
            node_y_4b.append(temp_node_y_map.get(i, 0.5)) # Default to 0.5 if not found (should not happen)


        # Create Links for 4b
        # 1. Direct AM -> Customer
        if not df_q3_direct_vol_processed.empty:
            for _, row in df_q3_direct_vol_processed.iterrows():
                if row['AnnualVolume'] > 0:
                    s_idx = label_to_idx_4b.get(f"AM: {row['AM']}")
                    t_idx = label_to_idx_4b.get(f"Cust: {row['Customer']}")
                    if s_idx is not None and t_idx is not None:
                        link_source_4b.append(s_idx); link_target_4b.append(t_idx); link_value_4b.append(row['AnnualVolume'])
                        link_color_4b.append(DIRECT_TRUCK_COLOR); link_hover_4b.append(f"Direct Truck from {row['AM']} to {row['Customer']}: {row['AnnualVolume']:,.0f} units")

        # Coastal Links
        if not df_q3_coastal_vol_processed.empty:
            # 2. AM -> Origin Port
            am_op_agg_b = df_q3_coastal_vol_processed.groupby(['AM', 'OriginPort'])['AnnualVolume'].sum().reset_index()
            for _, row in am_op_agg_b.iterrows():
                if row['AnnualVolume'] > 0:
                    s_idx = label_to_idx_4b.get(f"AM: {row['AM']}")
                    t_idx = label_to_idx_4b.get(f"Origin: {row['OriginPort'].replace(' Port','')}")
                    if s_idx is not None and t_idx is not None:
                        link_source_4b.append(s_idx); link_target_4b.append(t_idx); link_value_4b.append(row['AnnualVolume'])
                        link_color_4b.append(COASTAL_TRUCK_COLOR); link_hover_4b.append(f"Truck from {row['AM']} to {row['OriginPort']}: {row['AnnualVolume']:,.0f} units")
            
            # 3. Origin Port -> ShipType
            op_ship_agg_b = df_q3_coastal_vol_processed.groupby(['OriginPort', 'ShipType'])['AnnualVolume'].sum().reset_index()
            for _, row in op_ship_agg_b.iterrows():
                if row['AnnualVolume'] > 0:
                    s_idx = label_to_idx_4b.get(f"Origin: {row['OriginPort'].replace(' Port','')}")
                    t_idx = label_to_idx_4b.get(f"Ship: {row['ShipType']}")
                    if s_idx is not None and t_idx is not None:
                        link_source_4b.append(s_idx); link_target_4b.append(t_idx); link_value_4b.append(row['AnnualVolume'])
                        link_color_4b.append(SHIP_COLORS.get(row['ShipType'], 'grey').replace('0.8','0.6')); link_hover_4b.append(f"Loading on {row['ShipType']} at {row['OriginPort']}: {row['AnnualVolume']:,.0f} units")

            # 4. ShipType -> Destination Port
            ship_dp_agg_b = df_q3_coastal_vol_processed.groupby(['ShipType', 'DestinationPort'])['AnnualVolume'].sum().reset_index()
            for _, row in ship_dp_agg_b.iterrows():
                if row['AnnualVolume'] > 0:
                    s_idx = label_to_idx_4b.get(f"Ship: {row['ShipType']}")
                    t_idx = label_to_idx_4b.get(f"Dest Port: {row['DestinationPort'].replace(' Port','')}")
                    if s_idx is not None and t_idx is not None:
                        link_source_4b.append(s_idx); link_target_4b.append(t_idx); link_value_4b.append(row['AnnualVolume'])
                        link_color_4b.append(SHIP_COLORS.get(row['ShipType'], 'grey')); link_hover_4b.append(f"{row['ShipType']} to {row['DestinationPort']}: {row['AnnualVolume']:,.0f} units (Sea)")

            # 5. Destination Port -> Customer
            dp_cust_agg_b = df_q3_coastal_vol_processed.groupby(['DestinationPort', 'Customer'])['AnnualVolume'].sum().reset_index()
            for _, row in dp_cust_agg_b.iterrows():
                if row['AnnualVolume'] > 0:
                    s_idx = label_to_idx_4b.get(f"Dest Port: {row['DestinationPort'].replace(' Port','')}")
                    t_idx = label_to_idx_4b.get(f"Cust: {row['Customer']}")
                    if s_idx is not None and t_idx is not None:
                        link_source_4b.append(s_idx); link_target_4b.append(t_idx); link_value_4b.append(row['AnnualVolume'])
                        link_color_4b.append(COASTAL_TRUCK_COLOR); link_hover_4b.append(f"Truck from {row['DestinationPort']} to {row['Customer']}: {row['AnnualVolume']:,.0f} units")
        
        if not link_source_4b:
             print("  Warning (Chart 4b): No links to draw for detailed Sankey. Skipping Chart 4b.")
        else:
            fig4b = go.Figure(data=[go.Sankey(
                arrangement="snap", 
                node=dict(
                    pad=15, thickness=20, line=dict(color="black", width=0.5),
                    label=node_labels_4b, color=node_colors_4b, x=node_x_4b, y=node_y_4b,
                    hovertemplate='%{label}<extra></extra>'
                ),
                link=dict(
                    source=link_source_4b, target=link_target_4b, value=link_value_4b,
                    color=link_color_4b,
                    label=[f"{v/1000:.1f}K" if v > 100 else "" for v in link_value_4b], 
                    hovertemplate='%{customdata}<extra></extra>', customdata=link_hover_4b
                ),
                textfont=plotly_node_label_font_dict 
            )])
            fig4b.update_layout(
                title_text="Chart 4b: Detailed End-to-End Shipment Flows (with Customers)",
                font_family=PLOTLY_FONT_FAMILY, font_size=10, # Smaller base font for dense diagram
                font=plotly_font_dict, 
                title_font=plotly_title_font_dict,
                height=max(1000, len(all_customers_df) * 25), # Dynamic height
                width=1800, 
                margin=dict(t=70, b=50, l=50, r=50)
            )
            fig4b.write_image(os.path.join(PLOTS_DIR, "04b_detailed_end_to_end_sankey.png"), scale=1.5)
            # fig4b.write_html(os.path.join(PLOTS_DIR, "04b_detailed_end_to_end_sankey.html")) # Optional HTML
            print(f"  Chart 4b saved to {os.path.join(PLOTS_DIR, '04b_detailed_end_to_end_sankey.png')}")
    except Exception as e:
        print(f"Error generating Chart 4b (Detailed Plotly Sankey): {e}")
        import traceback
        traceback.print_exc()