# Project 13 for Applied Operations Research I
#### Advised by Prof. ZHOU Weihua and CUI Zheng (School of Management, Zhejiang University)

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
│   ├── plotting.py          # Draw some pictures for question 1, 2 and 3
│   └── config.py            # Configuration constants and parameters
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

To get the pictures in `plots/` by yourself:

```shell
python -m solve.plotting
```
## Module Descriptions

- **`solve/main.py`**: The main entry point for the project. It coordinates the execution of various modules, sequentially running the solvers for Question 1 and Question 2, and the optimizer for Question 3. It may also trigger the output of results or plotting.

- **`solve/data_loader.py`**: Contains functions to load and preprocess data from the Excel file (`data/DATA_original.xlsx`). It ensures that data is correctly read, cleaned, and transformed into a format suitable for subsequent analysis and model input.

- **`solve/q1_solver.py`**: Implements the solution logic for Question 1. It calculates the transportation costs for all customer locations via direct trucking, based on customer demand, distances, and other relevant data.

- **`solve/q2_solver.py`**: Implements the solution logic for Question 2. It analyzes the results from Question 1 and other relevant data (such as potential cost savings from coastal shipping) to identify customer locations suitable for switching from direct trucking to coastal transportation.

- **`solve/q3_optimizer.py`**: Implements the optimization model for Question 3. Using the results from the previous two questions, it builds and solves an optimization model to design the optimal transportation system (potentially combining trucking and coastal shipping) to minimize total transportation costs while meeting all customer demands and operational constraints.

- **`solve/plotting.py`**: Contains functions to generate visualizations and plots related to the results of Questions 1, 2, and 3. These plots aid in results analysis and presentation, and are saved to the `plots/` directory.

- **`solve/config.py`**: Stores global configuration parameters for the project, such as cost constants, rates, distance thresholds, file paths, and other fixed values or settings shared across different modules.