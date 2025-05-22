import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import differential_evolution
from sklearn.metrics import mean_squared_error

# Import utility functions and preprocessing functions
from utilities import metric, reconstruct_sup, model_1  # model_1 is used for LR forecasts
from preprocess import load_data, handle_missing_values, generate_weekend_monday_vector


# Define a neural network for forecasting L (supply curves)
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # The input dimension is 200 (from the interpolated grid)
        # and the output dimension is 200 (forecasting a 200-element curve)
        self.fc1 = nn.Linear(200, 164)
        self.fc3 = nn.Linear(164, 200)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc3(x)
        return x

def model_2(L, num_epochs=100, batch_size=32, hidden_size=32):
    """
    Trains a neural network model for forecasting the interpolated supply curves L.
    
    Parameters:
      - L (numpy array): Interpolated supply curve data with shape (n_samples, 200).
      - num_epochs (int): Number of training epochs.
      - batch_size (int): Batch size for training.
      - hidden_size (int): Not used here since our network is fixed, but kept for consistency.
      
    Returns:
      - Y_valid (numpy array): Forecasts for the validation set.
      - Y_test (numpy array): Forecasts for the test set.
    """
    # Split L into train and test parts (last 173 samples for test)
    train = L[:-172]
    test = L[-173:]
    split = int(0.7 * len(train))
    valid = train[split-1:]
    train = train[:split]

    # Prepare input-output pairs: forecasting next point from the current
    x_train = train[:-1]
    y_train = train[1:]
    
    x_valid = valid[:-1]
    y_valid = valid[1:]
    
    x_test = test[:-1]
    y_test = test[1:]
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_valid_tensor = torch.tensor(x_valid, dtype=torch.float32)
    X_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    
    # Initialize model, loss function, and optimizer
    model = SimpleNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    
    # Training loop
    for epoch in range(num_epochs):
        for i in range(0, len(X_train_tensor), batch_size):
            inputs = X_train_tensor[i:i+batch_size]
            targets = y_train_tensor[i:i+batch_size]
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Evaluate on validation and test sets
    model.eval()
    with torch.no_grad():
        outputs_valid = model(X_valid_tensor)
        outputs_test = model(X_test_tensor)
    
    Y_valid = outputs_valid.numpy()
    Y_test = outputs_test.numpy()
    return Y_valid, Y_test

def train_supply_model(supply_price, supply_volume, public_holidays, vector):
    """
    Trains a supply forecasting model for each hour of the day using multiple techniques.
    
    It:
      1. Extracts hourly supply curves.
      2. Interpolates each supply curve to create L (an array of fixed length 200).
      3. Uses model_2 (MLB/Neural Network) to forecast L.
      4. Obtains additional forecasts using a linear model (model_1).
      5. Uses differential evolution to optimize a combined forecast.
      
    Barameters:
      - supply_price (numpy array): Supply price data.
      - supply_volume (numpy array): Supply volume data.
      - public_holidays (numpy array): Holiday indicator data.
      - vector (numpy array): Indicator vector for weekends/Mondays.
      
    Returns:
      - results (dict): Contains forecasts and loss metrics for each hour.
    """
    results = {
        "naive_predv_h": [],
        "naive_predp_h": [],
        "sm_predv_h": [],
        "sm_predp_h": [],
        "lr_predv_h": [],
        "lr_predp_h": [],
        "combined_predv_h_t2": [],
        "combined_predp_h_t2": []
    }
    
    for h in range(24):
        print(f"Processing Hour {h+1}...")
        # Extract hourly data for supply curves
        sup_B = []
        sup_M = []
            #sup_M_p.append(supp[i][supp[i]<3000][-1])
        sup_E = []
        
        supp = supply_price[np.arange(h,len(supply_price),24)]
        supv = supply_volume[np.arange(h,len(supply_price),24)]
        
        
        for i in range(len(supv)):
            supv[i][supp[i] == 3000] = [supv[i][supp[i] == 3000][0]]
            supp[i][supp[i] == 3000] = [supp[i][supp[i] == 3000][0]]
        
        
        for i in range(len(supp)):
            sup_B.append(supv[i][0])
            sup_M.append(supv[i][supp[i]<3000][-1])
            #sup_M_p.append(supp[i][supp[i]<3000][-1])
            sup_E.append(supv[i][supp[i]==3000][0])
        
        
        sup_B = np.array(sup_B)
        sup_M = np.array(sup_M)
        sup_E = np.array(sup_E)
        
        
        #### LR data
        sup_B_lr_val_train = model_1(sup_B, public_holidays)[0][7:]    
        sup_M_lr_val_train = model_1(sup_M, public_holidays)[0][7:]
        sup_E_lr_val_train = model_1(sup_E, public_holidays)[0][7:]
        
        sup_B_lr_val_test = model_1(sup_B, public_holidays)[-1] 
        sup_M_lr_val_test = model_1(sup_M, public_holidays)[-1]
        sup_E_lr_val_test = model_1(sup_E, public_holidays)[-1]
        
        
        ### Smarter Naive
        
        split = 7
        testv = supv[split:]
        testp = supp[split:]
        
        sm_sup_B = []
        sm_sup_M = []
        sm_sup_E = []
        
        sm_predv = []
        sm_predp = []
        for i in range(split, len(testv)):
            sm_sup_B.append(supv[i - 1][0])
            sm_sup_M.append(supv[i - 1][supp[i - 1]<3000][-1])
            sm_sup_E.append(supv[i - 1][supp[i - 1]==3000][0])
            sm_predv.append(supv[i - 1])
            sm_predp.append(supp[i - 1])
            
            #pred = np.array(pred)
            #rmse = np.sqrt(mean_squared_error(test, pred))
        
        sm_sup_B = np.array(sm_sup_B)
        sm_sup_M = np.array(sm_sup_M)
        sm_sup_E = np.array(sm_sup_E)
        
        sup_B_sm_val_test = sm_sup_B[-172:]
        sup_M_sm_val_test = sm_sup_M[-172:]
        sup_E_sm_val_test = sm_sup_E[-172:]
        
        
        #### Division of curve dataset ##############
        testv = supv[-172:]
        testp = supp[-172:]
        
        valv = supv[-len(sup_B_lr_val_train)-172:-172]
        valp = supp[-len(sup_B_lr_val_train)-172:-172]
        
        #############################################
        
        sm_curve_p_val = []
        sm_curve_p_test = []
        
        for i in range(-len(sup_B_lr_val_train)-172,-172):
            sm_curve_p_val.append(sm_predp[i][sm_predp[i]<3000])
        
        for i in range(len(sm_predp)-172,len(sm_predp)):
            sm_curve_p_test.append(sm_predp[i][sm_predp[i]<3000])
        
        sup_B_sm_val_train = sm_sup_B[-len(sup_B_lr_val_train)-172:-172]
        sup_M_sm_val_train = sm_sup_M[-len(sup_M_lr_val_train)-172:-172]
        sup_E_sm_val_train = sm_sup_E[-len(sup_E_lr_val_train)-172:-172]
        
        sup_B_valid_target = sup_B[-len(sup_B_lr_val_train)-172:-172]
        sup_M_valid_target = sup_M[-len(sup_M_lr_val_train)-172:-172]
        sup_E_valid_target = sup_E[-len(sup_E_lr_val_train)-172:-172]
        
        sup_B_test_target = sup_B[-172:]
        sup_M_test_target = sup_M[-172:]
        sup_E_test_target = sup_E[-172:]
        
        
        ## Models and result
        
        L = []
        #c_v = []
        
        for i in range(len(supp)):
                tp = (supp[i][supp[i]<3000])
                tv = (supv[i][supp[i]<3000])
                grid = np.linspace(tv[0], tv[-1], num = 200)
                L.append(np.interp(grid, tv, tp))
        L = np.array(L)
            
        Y_valid, Y_test = model_2(L)
    
        #############################################################################
        ##################### Validation of the DE ##################################
        
        split = 7
        j = split
        testv = supv[split:]
        testp = supp[split:]
        sm_predv = []
        sm_predp = []
        for i in range(len(testv)):
            if vector[i+split] == 0:
                sm_predv.append(supv[j - 1])
                sm_predp.append(supp[j - 1])
            else:
                sm_predv.append(supv[j - 7])
                sm_predp.append(supp[j - 7])
            j+=1
    
        testv = supv[-172:]
        testp = supp[-172:]
    
        
        predv = supv[-161:-1]
        predp = supp[-161:-1]
        comb_loss = []
        for i in range(len(testv[12:])):
            comb_loss.append(metric(testv[i + 12],testp[i + 12],predv[i],predp[i]))
        rmse_n = np.mean(comb_loss)
        print('Naive loss for hour ' + str(h+1) + ': ' + str(rmse_n))
        #naive_predv_h.append(predv)
        #naive_predp_h.append(predp)

        
        comb_loss = []
        for i in range(len(testv[12:])):
            comb_loss.append(metric(testv[i + 12],testp[i + 12],sm_predv[-160:][i],sm_predp[-160:][i]))
        rmse_sm = np.mean(comb_loss)
        print('Smarter Naive loss for hour ' + str(h+1) + ': ' + str(rmse_sm))
        
        sup_B_h = sup_B_lr_val_test[12:]
        sup_M_h = sup_M_lr_val_test[12:]
        sup_E_h = sup_E_lr_val_test[12:]
        
        comb_loss = []
        for i in range(len(testv)-12):
            sup_B_t = sup_B_h[i]
            sup_M_t = sup_M_h[i]
            sup_E_t = sup_E_h[i]
        
            curve = Y_test[12:][i]
            pred = reconstruct_sup(sup_B_t,sup_M_t,sup_E_t,curve)
            comb_loss.append(metric(testv[i+12],testp[i+12],pred[0],pred[1]))
        rmse_lr = np.mean(comb_loss)
        print('Linear Variant loss for hour ' + str(h+1) + ': ' + str(rmse_lr))
        
        def objective(n):
            sup_B_h = n[0]*sup_B_sm_val_train + n[1]*sup_B_lr_val_train + n[2]
            sup_M_h = n[0]*sup_M_sm_val_train + n[1]*sup_M_lr_val_train + n[3]
            sup_E_h = n[0]*sup_E_sm_val_train + n[1]*sup_E_lr_val_train + n[4]
        
            comb_loss = []
            for i in range(len(valv)):
                sup_B_t = sup_B_h[i]
                sup_M_t = sup_M_h[i]
                sup_E_t = sup_E_h[i]
        
                curve = Y_valid[7:][i]
                pred = reconstruct_sup(sup_B_t,sup_M_t,sup_E_t,curve)
                comb_loss.append(metric(valv[i],valp[i],pred[0],pred[1]))
            rmse = np.mean(comb_loss)
            return rmse
        
        bounds = [(0, 1),(0, 1),(-1000,1000),(-1000,1000),(-1000,1000)]  # Bounds for two parameters
        
        # Set the population size and number of generations
        popsize = 100  # sup_Bopulation size
        maxiter = 10 # Maximum number of generations
        
        # sup_Berform differential evolution optimization with specified parameters
        result = differential_evolution(objective, bounds, popsize=popsize, maxiter=maxiter,mutation=(0.5, 1.0),strategy='best1bin')
        
        # Extract the optimized parameters
        optimized_params = result.x
        
        predictionp = []
        predictionv = []
        n = result.x #[::-1]#[0.75, 0.0, 0.25]
        sup_B_h = n[0]*sup_B_sm_val_test[12:] + n[1]*sup_B_lr_val_test[12:] + n[2]
        sup_M_h = n[0]*sup_M_sm_val_test[12:] + n[1]*sup_M_lr_val_test[12:] + n[3]
        sup_E_h = n[0]*sup_E_sm_val_test[12:] + n[1]*sup_E_lr_val_test[12:] + n[4]
        
        comb_loss = []
        for i in range(len(testv)-12):
            sup_B_t = sup_B_h[i]
            sup_M_t = sup_M_h[i]
            sup_E_t = sup_E_h[i]
        
            curve = Y_test[12:][i]
            pred = reconstruct_sup(sup_B_t,sup_M_t,sup_E_t,curve)
            predictionp.append(pred[1])
            predictionv.append(pred[0])
            comb_loss.append(metric(testv[i+12],testp[i+12],pred[0],pred[1]))
        rmse2 = np.mean(comb_loss)
        print('Combined Variant loss for hour ' + str(h+1) + ': ' + str(rmse2))
    
        # Store results for this hour
        results["naive_predv_h"].append(supv[-161:-1])
        results["naive_predp_h"].append(supp[-161:-1])
        results["sm_predv_h"].append(sm_predv[-160:])
        results["sm_predp_h"].append(sm_predp[-160:])
        results["lr_predv_h"].append(sup_B_lr_val_test)
        results["lr_predp_h"].append(sup_M_lr_val_test)
        results["combined_predv_h_t2"].append(predictionv)
        results["combined_predp_h_t2"].append(predictionp)
    
    return results


if __name__ == "__main__":
    # Load data
    public_holidays, demand_price, demand_volume, supply_price, supply_volume = load_data()
    demand_price, demand_volume, supply_price, supply_volume = handle_missing_values(demand_price, demand_volume, supply_price, supply_volume)
    vector = generate_weekend_monday_vector()
    
    # Train the supply model
    results = train_supply_model(supply_price, supply_volume, public_holidays, vector)
    print("✅ Supply Model Training Completed")
