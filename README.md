# Project 13 for Applied Operations Research I
#### Advised by Prof. ZHOU Weihua and CUI Zheng

This project aims to optimize transportation logistics for a vehicle distribution system using various methods, including direct trucking and coastal shipping. The project is structured into several modules, each responsible for specific tasks, making it easier to maintain and extend.

## Project Structure

```
Project4AOR
├── solve/
│   ├── __init__.py          # Marks the directory as a Python package
│   ├── main.py              # Entry point for the application
│   ├── data_loader.py       # Handles data loading and preprocessing
│   ├── q1_solver.py         # Solves question 1: Direct transportation costs
│   ├── q2_solver.py         # Solves question 2: Suitable locations for coastal transport
│   ├── q3_optimizer.py      # Optimizes the coastal transportation system design
│   ├── enhanced_plotting.py # Draw some pictures for question 1, 2 and 3
│   └── config.py            # Configuration constants and parameters
├── plotting/
│   ├── plotting.py
│   └── plotting_modules/
│       ├──── __init__.py
│       ├──── config.py
│       ├──── chart_1_cost_volume.py
│       ├──── chart_2_am_split.py
│       ├──── chart_3_q2_suitability.py
│       ├──── chart_4_sankey_flows.py
│       ├──── chart_5_kpi_summary.py
│       └──── generate_all_plots_logic.py
├── data
│   └── DATA_original.xlsx    # Original data for calculations
├── README.md                 # Project documentation
└── requirements.txt          # Python dependencies
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```shell
pip install -r requirements.txt
```

## Usage

To run the application, execute the `main.py` file:

```shell
python -m solve.main
```

This will sequentially solve questions 1 and 2, and then optimize the coastal transportation system for question 3.

To get the pictures in `results/academic_plots` by yourself:

```shell
python plotting/main.py
```

## Module Descriptions

- **data_loader.py**: Contains functions to load and preprocess data from the Excel file. It ensures that the data is in the correct format for further analysis.

- **q1_solver.py**: Implements the logic to calculate direct transportation costs based on customer demand and distances.

- **q2_solver.py**: Analyzes the results from question 1 to identify customer locations that are suitable for switching to coastal transportation.

- **q3_optimizer.py**: Uses the results from the previous questions to optimize the transportation system design, minimizing costs while meeting demand.

- **config.py**: Stores configuration parameters such as cost constants and other fixed values used throughout the project.