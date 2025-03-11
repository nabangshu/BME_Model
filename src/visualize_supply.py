import matplotlib.pyplot as plt
import numpy as np

def plot_supply_forecast(supply_price, supply_volume, sm_predv_h, sm_predp_h, naive_predv_h, naive_predp_h, 
                         combined_predv_h_t2, combined_predp_h_t2, hours=None, idx=10, save_path="Supply_forecast.png"):
    """
    Plots supply forecast comparisons for selected hours.

    Parameters:
    - supply_price (numpy array): Historical supply price data.
    - supply_volume (numpy array): Historical supply volume data.
    - sm_predv_h (dict): Smarter Naive predictions (volume).
    - sm_predp_h (dict): Smarter Naive predictions (price).
    - naive_predv_h (dict): Naive predictions (volume).
    - naive_predp_h (dict): Naive predictions (price).
    - combined_predv_h_t2 (dict): Combined model predictions (volume).
    - combined_predp_h_t2 (dict): Combined model predictions (price).
    - hours (list, optional): List of hours to visualize. Defaults to `[3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 23]`.
    - idx (int, optional): Index of the forecast to visualize. Defaults to `10`.
    - save_path (str, optional): Path to save the figure.

    Returns:
    - None (Displays the plot and saves it as a file).
    """

    if hours is None:
        hours = [3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 23]

    fig, axs = plt.subplots(3, 4, figsize=(12, 9))
    axs = axs.flatten()
    
    ax_empty = axs[0]
    ax_empty.axis('off')  # Hide the first subplot for legend placement
    
    for i, ax in enumerate(axs[1:], start=1):
        supp = supply_price[np.arange(hours[i], len(supply_price), 24)]
        supv = supply_volume[np.arange(hours[i], len(supply_price), 24)]
        testv = supv[-160:]
        testp = supp[-160:]

        # Plot ground truth
        ax.plot(testv[idx], testp[idx], color="black", linewidth=2.0, label="Ground Truth")
        
        # Plot different forecasting methods
        ax.plot(sm_predv_h[hours[i]][idx], sm_predp_h[hours[i]][idx], label="Smarter Naive")
        ax.plot(naive_predv_h[hours[i]][idx], naive_predp_h[hours[i]][idx], label="Naive")
        ax.plot(combined_predv_h_t2[hours[i]][idx], combined_predp_h_t2[hours[i]][idx], label="Combined Model")
        
        ax.set_title(f'Hour {hours[i]+1}')

    # Adjust layout and add legend
    plt.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    ax_empty.legend(handles, labels, loc='upper center', ncol=1)

    # Save and show plot
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    public_holidays, demand_price, demand_volume, supply_price, supply_volume = load_data()
    supply_price, supply_volume = handle_missing_values(supply_price, supply_volume)

    # Visualize supply forecast results
    plot_supply_forecast(supply_price, supply_volume, sm_predv_h, sm_predp_h, naive_predv_h, naive_predp_h, 
                         combined_predv_h_t2, combined_predp_h_t2, 
                         hours=[3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 23], idx=10, save_path="Supply_forecast_summer.png")

    plot_supply_forecast(supply_price, supply_volume, sm_predv_h, sm_predp_h, naive_predv_h, naive_predp_h, 
                         combined_predv_h_t2, combined_predp_h_t2, 
                         hours=[3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 23], idx=-10, save_path="Supply_forecast_winter.png")
