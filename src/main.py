from preprocess import load_data, handle_missing_values, generate_weekend_monday_vector
from Demand_model import train_demand_model
from Supply_model import train_supply_model
from visualize_demand import plot_demand_forecast
from visualize_supply import plot_supply_forecast

if __name__ == "__main__":
    # Load data
    public_holidays, demand_price, demand_volume, supply_price, supply_volume = load_data()
    demand_price, demand_volume, supply_price, supply_volume = handle_missing_values(demand_price, demand_volume, supply_price, supply_volume)
    vector = generate_weekend_monday_vector()
    # Train demand and supply models
    demand_results = train_demand_model(demand_price, demand_volume, public_holidays, vector)
    print("✅ Demand Model Training Completed")

    supply_results = train_supply_model(supply_price, supply_volume, public_holidays, vector)
    print("✅ Supply Model Training Completed")

    # Visualize results
    plot_demand_forecast(demand_price, demand_volume, **demand_results)
    plot_supply_forecast(supply_price, supply_volume, **supply_results)
