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
    
    ###############################################################
    ########################   Fig 1   ############################
    ###############################################################
    
    # Choose hour offset (8 AM)
    h = 8

    # Select demand price and volume for hour h across all days
    demp = demand_price[np.arange(h, len(demand_price), 24)]
    demv = demand_volume[np.arange(h, len(demand_price), 24)]

    # Select supply price and volume for hour h across all days
    supp = supply_price[np.arange(h, len(supply_price), 24)]
    supv = supply_volume[np.arange(h, len(supply_price), 24)]

    # Choose index i (day) to visualize
    i = 1

    # Create subplot with 2 plots side by side
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # B_demand: Last volume point where price is exactly 3000 €/MWh
    P = demv[i][demp[i] == 3000][-1]

    # M_demand: First point where price is less than 3000
    Q_v = demv[i][demp[i] < 3000][0]
    Q_p = demp[i][demp[i] < 3000][0]


    # E_demand: Last point in the curve
    K_v = demv[i][-1]
    K_p = demp[i][-1]


    # --- Plot Demand Curve ---
    ax[0].plot(demv[i], demp[i], label="Demand Curve")
    ax[0].scatter(P, 3000, label="$\mathbf{B}_{demand}$", color="red", s=200, marker="x")
    ax[0].scatter(Q_v, Q_p, label="$\mathbf{M}_{demand}$", color="blue", s=200, marker="s", facecolor='none')
    ax[0].scatter(K_v, K_p, label="$\mathbf{E}_{demand}$", color="green", s=200, marker="o", facecolor='none')
    ax[0].scatter(demv[i], demp[i], label="Demand bids", color="tab:blue", s=20, marker="v")
    ax[0].set_xlabel("Volume (MWh)")
    ax[0].set_ylabel("Price (€/MWh)")
    ax[0].set_title("(a)")
    ax[0].legend()

    # B_supply: First point in the curve (lowest volume)
    P = supv[i][0]


    # M_supply: Last point where price is less than 3000
    Q_v = supv[i][supp[i] < 3000][-1]
    Q_p = supp[i][supp[i] < 3000][-1]

    # E_supply: First point where price is exactly 3000
    K = supv[i][supp[i] == 3000][0]

    # --- Plot Supply Curve ---
    ax[1].plot(supv[i], supp[i], label="Supply Curve")
    ax[1].scatter(P, 0, label="$\mathbf{B}_{supply}$", color="red", s=200, marker="x")
    ax[1].scatter(Q_v, Q_p, label="$\mathbf{M}_{supply}$", color="blue", s=200, marker="s", facecolor='none')
    ax[1].scatter(K, 3000, label="$\mathbf{E}_{supply}$", color="green", s=200, marker="o", facecolor='none')
    ax[1].scatter(supv[i], supp[i], label="Supply bids", color="tab:blue", s=20, marker="^")
    ax[1].set_xlabel("Volume (MWh)")
    ax[1].set_ylabel("Price (€/MWh)")
    ax[1].set_title("(b)")
    ax[1].legend()

    # Adjust layout and show plots
    plt.tight_layout()
    plt.show()

    
    ###############################################################
    ########################   Fig 4   ############################
    ###############################################################
    
    # Initialize an empty list to store interpolated price curves
    c_p = []

    # Loop over all daily demand curves
    for i in range(len(demp)):
            # Select price values strictly below 3000 €/MWh for the i-th curve
            tp = (demp[i][demp[i] < 3000])

            # Select corresponding volume values for prices < 3000
            tv = (demv[i][demp[i] < 3000])

            # Create a uniform grid of 100 points between the min and max of tv
            grid = np.linspace(tv[0], tv[-1], num=100)

            # Interpolate prices onto the new grid and store
            c_p.append(np.interp(grid, tv, tp))

    # Convert the list of interpolated curves to a NumPy array
    c_p = np.array(c_p)

    # Create a figure with 2 subplots side by side
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # --- (a) Plot raw demand curves below 3000 €/MWh ---
    for i in range(len(demp)):
            ax[0].plot(demv[i][demp[i] < 3000], demp[i][demp[i] < 3000])
    ax[0].set_xlabel("Volume (MWh)")
    ax[0].set_ylabel("Price (€/MWh)")
    ax[0].set_title("(a)")
    # ax[0].legend()  # Disabled

    # --- (b) Plot interpolated demand curves ---
    for i in range(len(demp)):
        ax[1].plot(c_p[i])
    ax[1].set_xlabel("Volume (MWh)")
    ax[1].set_ylabel("Price (€/MWh)")
    ax[1].set_title("(b)")
    # ax[1].legend()  # Disabled

    # Adjust subplot layout
    plt.tight_layout()
    plt.show()


    ###############################################################
    ########################   Fig 5   ############################
    ###############################################################

    # Create a figure with 2 subplots side by side
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Select index i = 2 (i.e., third day)
    i = 2

    # --- Plot (a): Compare supply curves for two consecutive days ---

    # Plot the supply curve for January 3, 2018 at hour 9
    ax[0].plot(supv[i], supp[i], color="tab:blue", label="Curve on January 3, 2018 at hour 9", linestyle='--')

    # Scatter the bid points on the same curve in red, using downward-pointing triangles
    ax[0].scatter(supv[i], supp[i], color="red", s=20, marker="v")

    # Plot the supply curve for January 4, 2018 at hour 9
    ax[0].plot(supv[i+1], supp[i+1], color="tab:orange", label="Curve on January 4, 2018 at hour 9", linestyle='--')

    # Scatter the bid points on the next day's curve in black, using upward-pointing triangles
    ax[0].scatter(supv[i+1], supp[i+1], color="black", s=20, marker="^")

    # Axis labels and title
    ax[0].set_xlabel("Volume (MWh)")
    ax[0].set_ylabel("Price (€/MWh)")
    ax[0].set_title("(a)")
    ax[0].legend()

    # --- Plot (b): Same data but zoomed in to highlight volume/price ranges ---

    ax[1].plot(supv[i], supp[i], color="tab:blue", label="Curve on January 3, 2018 at hour 9", linestyle='--')
    ax[1].scatter(supv[i], supp[i], color="red", s=20, marker="v")
    ax[1].plot(supv[i+1], supp[i+1], color="tab:orange", label="Curve on January 4, 2018 at hour 9", linestyle='--')
    ax[1].scatter(supv[i+1], supp[i+1], color="black", s=20, marker="^")

    # Axis labels and title
    ax[1].set_xlabel("Volume (MWh)")
    ax[1].set_ylabel("Price (€/MWh)")
    ax[1].set_title("(b)")

    # Zoom in to a specific volume and price range to highlight heterogeneity
    ax[1].set_xlim([15000, 20000])
    ax[1].set_ylim([15, 100])

    ax[1].legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Display the plots
    plt.show()



    ###############################################################
    ####################   Rest of the plots  #####################
    ###############################################################
    
    # Train demand and supply models
    demand_results = train_demand_model(demand_price, demand_volume, public_holidays, vector)
    print("✅ Demand Results Completed")

    supply_results = train_supply_model(supply_price, supply_volume, public_holidays, vector)
    print("✅ Supply Results Completed")

    # Visualize results
    plot_demand_forecast(demand_price, demand_volume, **demand_results)
    plot_supply_forecast(supply_price, supply_volume, **supply_results)
