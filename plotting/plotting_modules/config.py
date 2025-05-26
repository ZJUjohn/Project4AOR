import os
import traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm # Added for font finding
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.graph_objects as go

# --- Font Configuration ---
PREFERRED_FONT_LIST = [
    'Microsoft YaHei', 'SimHei', 'Songti SC', 'STKaiti', # Chinese fonts
    'Latin Modern Roman', 'CMU Serif', 'Times New Roman', 'DejaVu Serif', # Serif
    'Georgia', 'DejaVu Sans', 'Calibri', 'Arial', 'Helvetica' # Sans-serif
]
PLOTLY_FONT_FAMILY = "Songti SC" # Default for Plotly, choose one that works on your system
FONT_SETTINGS_SUCCESS = False

print("--- Font Configuration ---")
print(f"Attempting to use font list, prioritizing: {PREFERRED_FONT_LIST[0]}")

try:
    first_font_is_chinese = PREFERRED_FONT_LIST[0] in ['Microsoft YaHei', 'SimHei', 'Songti SC', 'STKaiti']
    selected_font_family = None

    # Attempt to find and set a font from the list
    for font_name in PREFERRED_FONT_LIST:
        try:
            fm.findfont(font_name, fallback_to_default=False)
            selected_font_family = font_name
            print(f"Font '{font_name}' found by font_manager.")
            break
        except:
            print(f"Font '{font_name}' not found by font_manager.")
            continue
    
    if selected_font_family:
        if first_font_is_chinese:
            plt.rcParams['font.family'] = 'sans-serif' # Important for Chinese fonts
            plt.rcParams['font.sans-serif'] = [selected_font_family] + [f for f in PREFERRED_FONT_LIST if f != selected_font_family]
            print(f"Configured for Chinese font priority. Selected: {selected_font_family}. Sans-serif list: {plt.rcParams['font.sans-serif']}")
        else: # Prioritize serif for English if that's the first in list
            plt.rcParams['font.family'] = 'serif'
            plt.rcParams['font.serif'] = [selected_font_family] + [f for f in PREFERRED_FONT_LIST if f != selected_font_family]
            # Provide a sans-serif fallback as well
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Calibri', 'Arial', 'Helvetica']
            print(f"Configured for English serif font priority. Selected: {selected_font_family}. Serif list: {plt.rcParams['font.serif']}")
    else:
        print("No preferred fonts found. Using Matplotlib default sans-serif.")
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']


    plt.rcParams['axes.unicode_minus'] = False # Ensure minus sign displays correctly

    # Seaborn theme and font settings
    sns_rc_params = {
        "font.family": plt.rcParams['font.family'], # Use the primary family set
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": ".8",
        "axes.labelcolor": ".15",
        "axes.grid": True,
        "grid.color": ".8",
        "grid.linestyle": "--",
        "text.color": ".15",
        "xtick.color": ".15",
        "ytick.color": ".15",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.solid_capstyle": "round",
        "patch.edgecolor": "w",
        "patch.force_edgecolor": True,
        "image.cmap": "rocket",
        "font.size": 10.5,
        "axes.labelsize": 10.5, # Matching original plotting.py
        "axes.titlesize": 12.5, # Matching original plotting.py
        "xtick.labelsize": 9.5,  # Matching original plotting.py
        "ytick.labelsize": 9.5,  # Matching original plotting.py
        "legend.fontsize": 9,    # Matching original plotting.py
        "figure.titlesize": 14   # Matching original plotting.py (used as suptitle size)
    }
    # Add specific font family lists to sns_rc_params if they were set
    if 'font.serif' in plt.rcParams:
        sns_rc_params["font.serif"] = plt.rcParams['font.serif']
    if 'font.sans-serif' in plt.rcParams:
        sns_rc_params["font.sans-serif"] = plt.rcParams['font.sans-serif']

    sns.set_theme(style="whitegrid", rc=sns_rc_params)
    FONT_SETTINGS_SUCCESS = True
    print(f"Font settings applied. Matplotlib using family: {plt.rcParams['font.family']}, specific list for sans-serif: {plt.rcParams.get('font.sans-serif', 'N/A')}, specific list for serif: {plt.rcParams.get('font.serif', 'N/A')}.")
    print(f"Seaborn theme set. Effective font family: {sns.plotting_context()['font.family']}")

except Exception as e:
    print(f"Error setting preferred font: {e}")
    print(traceback.format_exc())
    print(f"Please ensure one of the preferred fonts ({PREFERRED_FONT_LIST}) is installed and Matplotlib's cache is updated if necessary.")
    print("Graphs may use Matplotlib's default font.")
    # Basic fallback
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_theme(style="whitegrid")
    print(f"Fallback font settings applied. Matplotlib using: {plt.rcParams['font.family']}")
print("--- End Font Configuration ---")

# --- Global Settings & Directories ---
# Assuming this config.py is in Project/src/plotting_modules/
# To get to Project/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
DATA_DIR = os.path.join(PROJECT_ROOT, "data") # If you have a data directory at project root

# --- Colors (from original plotting.py) ---
AM_COLORS = {'AM1': 'rgba(31, 119, 180, 0.9)', 'AM2': 'rgba(255, 127, 14, 0.9)', 'AM3': 'rgba(44, 160, 44, 0.9)'}
SHIP_COLORS = {'ShipA': 'rgba(214, 39, 40, 0.8)', 'ShipB': 'rgba(148, 103, 189, 0.8)'}
PORT_NODE_COLOR = 'rgba(127, 127, 127, 0.6)'
DIRECT_TRUCK_COLOR = 'rgba(255, 187, 120, 0.7)'
COASTAL_TRUCK_COLOR = 'rgba(188, 189, 34, 0.7)'
SEA_LINK_COLOR = 'rgba(23, 190, 207, 0.7)'
MODE_SPLIT_NODE_COLOR = 'rgba(150, 150, 150, 0.6)'

# --- Standard Plotting Aesthetics (Matplotlib/Seaborn, from original plotting.py) ---
STD_FIG_SIZE = (10, 7)
STD_TITLE_FONTSIZE = 14
STD_LABEL_FONTSIZE = 12
STD_TICK_FONTSIZE = 10
STD_LEGEND_FONTSIZE = 10 # For general legends, specific ones might use different sizes
STD_LEGEND_LOC = 'upper left'
STD_LEGEND_BBOX = (1.02, 1)
LEGEND_ADJUST_RIGHT = 0.75
FIGURE_TITLE_Y_ADJUST = 1.03

# --- Plotly Font Dictionaries (from original plotting.py, plot_q3_sankey_flows_plotly) ---
plotly_font_dict = dict(
    family=PLOTLY_FONT_FAMILY,
    size=22,
    color="black"
)
plotly_title_font_dict = dict(
    family=PLOTLY_FONT_FAMILY,
    size=28,
    color="black"
)
plotly_node_label_font_dict = dict(
    family=PLOTLY_FONT_FAMILY,
    size=20,
    color="black"
)
plotly_link_label_font_dict = dict(
    family=PLOTLY_FONT_FAMILY,
    size=12,
    color="black"
)

# Ensure plots directory exists
if not os.path.exists(PLOTS_DIR):
    try:
        os.makedirs(PLOTS_DIR)
        print(f"Created directory for plots: {PLOTS_DIR}")
    except OSError as e:
        print(f"Error creating directory {PLOTS_DIR}: {e}. Plots may not save correctly.")
        # Fallback to a local directory or raise an error if critical
        PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots_fallback") # Example fallback
        if not os.path.exists(PLOTS_DIR): os.makedirs(PLOTS_DIR)
        print(f"Warning: Plots will be saved to fallback directory: {os.path.abspath(PLOTS_DIR)}")

if not os.path.exists(RESULTS_DIR):
    print(f"Warning: Results directory {RESULTS_DIR} does not exist. Data loading might fail if it relies on this path.")

if not os.path.exists(DATA_DIR):
    print(f"Warning: Data directory {DATA_DIR} does not exist. Data loading might fail if it relies on this path.")