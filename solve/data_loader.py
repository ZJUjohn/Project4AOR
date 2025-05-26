import pandas as pd

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        """Loads data from the specified Excel file sheets."""
        try:
            df_locations_input = pd.read_excel(self.file_path, sheet_name="Customer Locations", header=2)
            df_port_cust_dist_input = pd.read_excel(self.file_path, sheet_name="PortCustDist", header=1)
            return df_locations_input, df_port_cust_dist_input
        except FileNotFoundError as e:
            print(f"错误：Excel文件 '{self.file_path}' 未找到。请确保文件路径正确：{e}")
            raise
        except Exception as e:
            print(f"加载数据时发生错误: {e}")
            raise

    def preprocess_data(self, df_locations_input, df_port_cust_dist_input):
        """Preprocesses and merges the loaded dataframes."""
        # 清洗客户位置数据 (df_locations)
        df_locations_input.columns = ['Cust_Location_Num', 'Customer_Location', 'State_UT',
                                'AM1_Dist', 'AM1_M1', 'AM1_M2', 'AM1_M3',
                                'AM2_Dist', 'AM2_M1', 'AM2_M2', 'AM2_M3',
                                'AM3_Dist', 'AM3_M1', 'AM3_M2', 'AM3_M3']
        df_locations_input.dropna(how='all', inplace=True)

        num_cols_loc = ['AM1_Dist', 'AM1_M1', 'AM1_M2', 'AM1_M3',
                        'AM2_Dist', 'AM2_M1', 'AM2_M2', 'AM2_M3',
                        'AM3_Dist', 'AM3_M1', 'AM3_M2', 'AM3_M3']
        for col in num_cols_loc:
            df_locations_input[col] = pd.to_numeric(df_locations_input[col], errors='coerce')

        # 清洗港口到客户距离数据 (df_port_cust_dist)
        df_port_cust_dist_input.rename(columns={
            'Distance (km)':'Customer_Location_PortFile',
            'Chennai':'Dist_to_Chennai_Port',
            'Pipavav':'Dist_to_Pipavav_Port'
            }, inplace=True)
        df_port_cust_dist_input = df_port_cust_dist_input[['Customer_Location_PortFile', 'Dist_to_Chennai_Port', 'Dist_to_Pipavav_Port']].copy()
        df_port_cust_dist_input.dropna(subset=['Customer_Location_PortFile'], inplace=True)
        df_port_cust_dist_input['Customer_Location'] = df_port_cust_dist_input['Customer_Location_PortFile']
        df_port_cust_dist_input['Dist_to_Chennai_Port'] = pd.to_numeric(df_port_cust_dist_input['Dist_to_Chennai_Port'], errors='coerce')
        df_port_cust_dist_input['Dist_to_Pipavav_Port'] = pd.to_numeric(df_port_cust_dist_input['Dist_to_Pipavav_Port'], errors='coerce')

        # 合并港口距离数据到主位置数据表 (df_locations)
        df_locations = pd.merge(df_locations_input, df_port_cust_dist_input[['Customer_Location', 'Dist_to_Chennai_Port', 'Dist_to_Pipavav_Port']],
                                on="Customer_Location", how="left")
        df_locations['Cust_ID'] = df_locations['Customer_Location'] # 使用 Customer_Location 作为后续模型中的唯一客户ID
        return df_locations