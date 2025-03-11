# Curve Forecasting for Day-Ahead Electricity Markets  

## 📌 Overview  
This repository contains the code for **forecasting demand and supply price/volume curves** in the **day-ahead electricity market**. The approach involves **decomposing the bid curves into structurally significant components (B, M, and E)** and using **statistical models** for forecasting. It contains the code for both the linear and the combined variant of forecasting, as detailed in the submission (ASMB-25-70) "BME Model: Forecasting Electricity Supply and Demand Curves Using a Tokenization Technique".

The models are evaluated using **data from the Italian IPEX Nord bidding zone** and compared against baseline methods.  

---

## 📂 Repository Structure  

📦 Forecasting_Project
│-- 📂 src/                         # Contains all the core scripts
│   ├── 📂 data/                        # Stores raw and processed data files
│	    ├── total_demand_price_data.npy
│	    ├── total_demand_volume_data.npy
│	    ├── total_supply_price_data.npy
│	    ├── total_supply_volume_data.npy
│	    ├── holidays.npy
│   ├── utilities.py                # Common helper functions
│   ├── preprocess.py               # Data loading and preprocessing functions
│   ├── demand_model.py             # Demand curve forecasting
│   ├── supply_model.py             # Supply curve forecasting
│   ├── visualize_demand.py         # Visualization of demand forecasts
│   ├── visualize_supply.py         # Visualization of supply forecasts
│   ├── main.py                          # Main script to run forecasting
│-- README.md                       # Project documentation
│-- requirements.txt                 # Dependencies required to run the project

