import numpy as np
import pandas as pd
import datetime

def load_data():
    """
    Load demand and supply price & volume data along with holiday information.
    """
    public_holidays = np.load("Data/holidays.npy")

    demand_price = np.load("Data/total_demand_price_data.npy", allow_pickle=True)
    demand_volume = np.load("Data/total_demand_volume_data.npy", allow_pickle=True)

    supply_price = np.load("Data/total_supply_price_data.npy", allow_pickle=True)
    supply_volume = np.load("Data/total_supply_volume_data.npy", allow_pickle=True)

    return public_holidays, demand_price, demand_volume, supply_price, supply_volume


def handle_missing_values(demand_price, demand_volume, supply_price, supply_volume):
    """
    Replace missing values in demand and supply datasets at specific timestamps.
    """
    # Define indices with missing values
    missing_indices = [2015, 10919]

    for idx in missing_indices:
        demand_price[idx] = demand_price[idx - (24 * 7)]
        demand_volume[idx] = demand_volume[idx - (24 * 7)]

        supply_price[idx] = supply_price[idx - (24 * 7)]
        supply_volume[idx] = supply_volume[idx - (24 * 7)]

    return demand_price, demand_volume, supply_price, supply_volume


def generate_weekend_monday_vector(start_date=datetime.date(2018, 1, 1), end_date=datetime.date(2019, 12, 31)):
    """
    Generate a binary vector indicating whether a given date is a weekend or Monday.
    1 = Weekend or Monday, 0 = Other weekdays.
    """
    date_range = pd.date_range(start=start_date, end=end_date)
    vector = np.array([1 if date.weekday() == 0 or date.weekday() >= 5 else 0 for date in date_range])

    return vector
