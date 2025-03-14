import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from scipy.optimize import differential_evolution

from utilities import metric, reconstruct, model_1  # model_1 is used for LR forecasts
from preprocess import load_data, handle_missing_values, generate_weekend_monday_vector


def train_demand_model(demand_price, demand_volume, public_holidays, vector):
    """
    Trains the demand forecasting model for 24-hour intervals.

    Barameters:
    - demand_price (array): Historical demand price data.
    - demand_volume (array): Historical demand volume data.
    - public_holidays (array): Holiday indicator data.
    - vector (array): Weekend and Monday indicator vector.

    Returns:
    - results (dict): Dictionary containing model outputs for each hour.
    """

    results = {
        "coefficients": [],
        "combined_pred_volume": [],
        "combined_pred_price": [],
        "naive_pred_volume": [],
        "naive_pred_price": [],
        "smarter_naive_pred_volume": [],
        "smarter_naive_pred_price": [],
        "linear_pred_volume": [],
        "linear_pred_price": [],
        "rmse_combined": []
    }

    for h in range(24):
        print(f"Processing Hour {h+1}...")
        demp = demand_price[np.arange(h,len(demand_price),24)]
        demv = demand_volume[np.arange(h,len(demand_price),24)]
    
        lr_predv = []
        eesn_predv = []
        combined_predv = []
    
        lr_predp = []
        eesn_predp = []
        combined_predp = []
        
        B = []
        M = []
        E = []
        
        for i in range(len(demp)):
            B.append(demv[i][demp[i]==3000][-1])
            M.append(demv[i][demp[i]<3000][0])
            #M_p.append(demp[i][demp[i]<3000][0])
            E.append(demv[i][-1])
            #E_p.append(demp[i][-1])
        
        
        B = np.array(B)
        M = np.array(M)
        E = np.array(E)
        
        #### LR data
        B_lr_val_train = model_1(B, public_holidays)[0][7:]    
        M_lr_val_train = model_1(M, public_holidays)[0][7:]
        E_lr_val_train = model_1(E, public_holidays)[0][7:]
        
        B_lr_val_test = model_1(B, public_holidays)[-1] 
        M_lr_val_test = model_1(M, public_holidays)[-1]
        E_lr_val_test = model_1(E, public_holidays)[-1]
        
        
        ### Smarter Naive
        
        split = 7
        testv = demv[split:]
        testp = demp[split:]
        j = split
        
        sm_B = []
        sm_M = []
        sm_E = []
        
        sm_predv = []
        sm_predp = []
        for i in range(len(testv)):
            if vector[i+split] == 0:
                sm_B.append(demv[j - 1][demp[j - 1]==3000][-1])
                sm_M.append(demv[j - 1][demp[j - 1]<3000][0])
                sm_E.append(demv[j - 1][-1])
                sm_predv.append(demv[j - 1])
                sm_predp.append(demp[j - 1])
            else:
                sm_B.append(demv[j - 7][demp[j - 7]==3000][-1])
                sm_M.append(demv[j - 7][demp[j - 7]<3000][0])
                sm_E.append(demv[j - 7][-1])
                sm_predv.append(demv[j - 7])
                sm_predp.append(demp[j - 7])
            j+=1
        
        sm_B = np.array(sm_B)
        sm_M = np.array(sm_M)
        sm_E = np.array(sm_E)
        
        B_sm_val_test = sm_B[-172:]
        M_sm_val_test = sm_M[-172:]
        E_sm_val_test = sm_E[-172:]
        
        
        #### Division of curve dataset ##############
        testv = demv[-172:]
        testp = demp[-172:]
        
        valv = demv[-len(B_lr_val_train)-172:-172]
        valp = demp[-len(B_lr_val_train)-172:-172]
        
        #############################################        
        B_sm_val_train = sm_B[-len(B_lr_val_train)-172:-172]
        M_sm_val_train = sm_M[-len(M_lr_val_train)-172:-172]
        E_sm_val_train = sm_E[-len(E_lr_val_train)-172:-172]
        
        B_valid_target = B[-len(B_lr_val_train)-172:-172]
        M_valid_target = M[-len(M_lr_val_train)-172:-172]
        E_valid_target = E[-len(E_lr_val_train)-172:-172]
        
        B_test_target = B[-172:]
        M_test_target = M[-172:]
        E_test_target = E[-172:]
        
        
        ## Models and result
        
        L = []
        #c_v = []
        
        for i in range(len(demp)):
                tp = (demp[i][demp[i]<3000])
                tv = (demv[i][demp[i]<3000])
                grid = np.linspace(tv[0], tv[-1], num = 100)
                L.append(np.interp(grid, tv, tp))
        L = np.array(L)
        
        
        
        train = L[:-172]
        test = L[-173:]          
        split = int(0.7*len(train))    
        valid = train[split-1:]
        train = train[:split]
                       
        x_train = train[0:-1]
        y_train = train[1:]
              
        x_test = test[0:-1]
        y_test = test[1:]
           
        x_valid = valid[0:-1]
        y_valid = valid[1:]
        
        
        x_v = x_valid
        X = x_train  # 1D array, each element is a feature
        #X = np.hstack((X, du_train.reshape(len(du_train),1)))
            # Example target vector
        y = y_train  # 1D array, each element is a target value
        min = 99999
        min_alpha = 0
        grid = np.linspace(10, 1.5e+2, 100)
        for i in grid:
            alpha = i
            lasso_model = Lasso(alpha)  # You can adjust the alpha parameter as needed
            lasso_model.fit(X, y)
            y_pred = lasso_model.predict(x_v)  # Bredict for the input samples
            
            rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
            if rmse < min:
                min = rmse
                min_alpha = i
        alpha = min_alpha
        #x_t = np.hstack((x_test, du_test.reshape(len(du_test),1)))
        x_t = x_test
        y_pred = lasso_model.predict(x_v)
        Y_valid = y_pred
        y_pred = lasso_model.predict(x_t)  # Bredict for the input samples
        Y_test = y_pred  
        
        def objective(n):
            B_h = n[0]*B_sm_val_train + n[1]*B_lr_val_train + n[2]
            M_h = n[0]*M_sm_val_train + n[1]*M_lr_val_train + n[3]
            E_h = n[0]*E_sm_val_train + n[1]*E_lr_val_train + n[4]
        
            comb_loss = []
            for i in range(len(valv)):
                B_t = B_h[i]
                M_t = M_h[i]
                E_t = E_h[i]
        
                curve = Y_valid[7:][i]
                pred = reconstruct(B_t,M_t,E_t,curve)
                comb_loss.append(metric(valv[i],valp[i],pred[0],pred[1]))
            rmse = np.mean(comb_loss)
            return rmse
            
        bounds = [(0.5, 1),(0, 0.5),(-1000, 1000),(-1000, 1000),(-1000, 1000)]
        # Set the population size and number of generations
        popsize = 100  # Bopulation size
        maxiter = 10 # Maximum number of generations
        
        # Perform differential evolution optimization with specified parameters
        result = differential_evolution(objective, bounds, popsize=popsize, maxiter=maxiter,mutation=(0.5, 1.0),strategy='best1bin')
    
        
        predv = demv[-161:-1]
        predp = demp[-161:-1]
        comb_loss = []
        for i in range(len(testv[12:])):
            comb_loss.append(metric(testv[i + 12],testp[i + 12],predv[i],predp[i]))
        rmse_n = np.mean(comb_loss)
        print('Naive loss for hour ' + str(h+1) + ': ' + str(rmse_n))
        
        comb_loss = []
        for i in range(len(testv[12:])):
            comb_loss.append(metric(testv[i + 12],testp[i + 12],sm_predv[-160:][i],sm_predp[-160:][i]))
        rmse_sm = np.mean(comb_loss)
        print('Smarter Naive loss for hour ' + str(h+1) + ': ' + str(rmse_sm))
        
        B_h = B_lr_val_test[12:]
        M_h = M_lr_val_test[12:]
        E_h = E_lr_val_test[12:]
        
        comb_loss = []
        for i in range(len(testv[12:])):
            B_t = B_h[i]
            M_t = M_h[i]
            E_t = E_h[i]
        
            curve = Y_test[12:][i]
            pred = reconstruct(B_t,M_t,E_t,curve)
            comb_loss.append(metric(testv[i + 12],testp[i + 12],pred[0],pred[1]))
            lr_predv.append(pred[0])
            lr_predp.append(pred[1])
        rmse_lr = np.mean(comb_loss)
        print('Linear Variant loss for hour ' + str(h+1) + ': ' + str(rmse_lr))
        
        # Extract the optimized parameters
        
        n = result.x
        B_h = n[0]*B_sm_val_test[12:] + n[1]*B_lr_val_test[12:] + n[2]
        M_h = n[0]*M_sm_val_test[12:] + n[1]*M_lr_val_test[12:] + n[3]
        E_h = n[0]*E_sm_val_test[12:] + n[1]*E_lr_val_test[12:] + n[4]
        
        comb_loss = []
        combined_predv = []
        combined_predp = []
    
        for i in range(len(testv[12:])):
            B_t = B_h[i]
            M_t = M_h[i]
            E_t = E_h[i]
        
            curve = Y_test[12:][i]
            pred = reconstruct(B_t,M_t,E_t,curve)
            comb_loss.append(metric(testv[i + 12],testp[i + 12],pred[0],pred[1]))
            combined_predv.append(pred[0])
            combined_predp.append(pred[1])
        rmse = np.mean(comb_loss)
        print('Combined Variant loss for hour ' + str(h+1) + ': ' + str(rmse))

        # Store results
        results["coefficients"].append(result.x)
        results["combined_pred_volume"].append(combined_predv)
        results["combined_pred_price"].append(combined_predp)
        results["naive_pred_volume"].append(predv[-160:])
        results["naive_pred_price"].append(predp[-160:])
        results["smarter_naive_pred_volume"].append(sm_predv[-160:])
        results["smarter_naive_pred_price"].append(sm_predp[-160:])
        results["linear_pred_volume"].append(lr_predv[-160:])
        results["linear_pred_price"].append(lr_predp[-160:])
        results["rmse_combined"].append(rmse)

    return results



if __name__ == "__main__":
    # Load data
    public_holidays, demand_price, demand_volume, supply_price, supply_volume = load_data()
    
    # Handle missing values
    demand_price, demand_volume, supply_price, supply_volume = handle_missing_values(demand_price, demand_volume, supply_price, supply_volume)
    
    # Generate weekend and Monday indicator vector
    vector = generate_weekend_monday_vector()
    
    # Train the demand model
    demand_results = train_demand_model(demand_price, demand_volume, public_holidays, vector)
    
    print("✅ Demand Model Training Completed")
