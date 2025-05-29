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


3. **Outputs**

    The code for all the results and figures are present inside the notebook titled "BME_Model.ipynb". 

---



## 📂 Repository Structure  

│   ├── 📂 data/                        # Stores raw and processed data files
│	│   ├── total_demand_price_data.npy
│	│   ├── total_demand_volume_data.npy
│	│   ├── total_supply_price_data.npy
│	│   ├── total_supply_volume_data.npy
│	│   ├── holidays.npy
│   ├── 📂 Results/                        # Contains the results of both the pure variant and the combined variant
│	│   ├── Demand_comb_p.npy
│	│   ├── Demand_comb_v.npy
│	│   ├── Demand_pure_p.npy
│	│   ├── Demand_pure_v.npy
│	│   ├── Supply_comb_p.npy
│	│   ├── Supply_comb_v.npy
│	│   ├── Supply_pure_p.npy
│	│   ├── Supply_pure_v.npy
│   ├── 📂 FAR_curves                        # Contains the results of the FAR model run on R
│-- BME_Model.ipynb        	# Jupyter Notebook for running everything to obtain the plots
│-- README.md                       # Project documentation
│-- requirements.txt                # Dependencies required to run the project


