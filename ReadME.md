# Curve Forecasting for Day-Ahead Electricity Markets  

## 📌 Overview  
This repository contains the code for **forecasting demand and supply price/volume curves** in the **day-ahead electricity market**. The approach involves **decomposing the bid curves into structurally significant components (B, M, and E)** and using **statistical models** for forecasting. It contains the code for both the linear and the combined variant of forecasting, as detailed in the submission (ASMB-25-70) "BME Model: Forecasting Electricity Supply and Demand Curves Using a Tokenization Technique".

The models are evaluated using **data from the Italian IPEX Nord bidding zone** and compared against baseline methods.  

---
# BME_Model Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/nabangshu/BME_Model/blob/main/BME_Model_Demo.ipynb)

---
## Installation and Running Instructions

1. **Clone the Repository**

    ```bash
    git clone https://github.com/nabangshu/BME_Model.git
    cd BME_Model
    ```

2. **Install Required Dependencies**

    Install dependencies listed in `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```

    *(Make sure to have `pip` installed and upgraded.)*

3. **Run the Code**

    Run the main script (`main.py`) to generate results and plots:

    - **Mac/Linux:**
	  ```bash
	  python3 src/main.py
		```
		
	- **Windows:**
	  ```bash
	  python3 src\main.py
		```

4. **Outputs**

    After running `main.py` inside the src folder, you'll see generated results and plots in your terminal or as pop-up windows (depending on your plotting configuration).

---



## 📂 Repository Structure  

📦 Forecasting_Project
│-- 📂 src/                         # Contains all the core scripts
│   ├── 📂 data/                        # Stores raw and processed data files
│	│   ├── total_demand_price_data.npy
│	│   ├── total_demand_volume_data.npy
│	│   ├── total_supply_price_data.npy
│	│   ├── total_supply_volume_data.npy
│	│   ├── holidays.npy
│   ├── utilities.py                # Common helper functions
│   ├── preprocess.py               # Data loading and preprocessing functions
│   ├── demand_model.py             # Demand curve forecasting
│   ├── supply_model.py             # Supply curve forecasting
│   ├── visualize_demand.py         # Visualization of demand forecasts
│   ├── visualize_supply.py         # Visualization of supply forecasts
│   ├── main.py                     # Main script to run forecasting
│-- BME_Model_Demo.ipynb        	# Jupyter Notebook for running everything to obtain the plots
│-- README.md                       # Project documentation
│-- requirements.txt                # Dependencies required to run the project


