import numpy as np
import matplotlib.pyplot as plt
import math
# Parameters for GBM

# Function to simulate the Geometric Brownian Motion

def GBM(sigma, N,  S_0 , r = 0.01 , Nsim = 10000):
    dt = 1/252
    # Create a 2D array to store the stock prices for each simulation
    S = np.zeros([Nsim,N])
    S[:,0] = S_0

    for i in range(Nsim):
        for j in range(N-1):
            Z = np.random.normal(0, 1)      
            # Calculate the stock price at time t+1 using the Geometric Brownian Motion model with risk-free interet rate
            S[i,j+1] = S[i,j] * np.exp( (r - 0.5 * sigma ** 2) * dt + sigma * Z * np.sqrt(dt) )
    return S


def GBM_AV(sigma, N,  S_0, r = 0.01 , Nsim = 10000):
    dt = 1/252
    # Create a 2D array to store the stock prices for each simulation
    S = np.zeros([Nsim,N])
    S[:,0] = S_0

    for i in range(Nsim//2):
        for j in range(N-1):
            Z = np.random.normal(0, 1)      
            # Calculate the stock price at time t+1 using the Geometric Brownian Motion model with risk-free interet rate
            S[2*i,j+1] = S[2*i,j] * np.exp( (r - 0.5 * sigma ** 2) * dt + sigma * Z * np.sqrt(dt) )
            S[2*i+1,j+1] = S[2*i+1,j] * np.exp( (r - 0.5 * sigma ** 2) * dt + sigma * -Z * np.sqrt(dt) )

    return S

#Payoff function for the derivative given a price path
def payoff(price_path, barrier_price = 6566.9596, initial_price = 11130.44):
    barrier_hit = min(price_path) <= barrier_price
    asset_final_price = price_path[-1]

    if barrier_hit:
        return 1000 * (asset_final_price / initial_price)
    else:
        return 1000 * max(1, 1 + 1.25 * (asset_final_price / initial_price - 1))

# Payoff for a given set of simulations
def payoff_sim(S, N, r = 0.01, barrier_price = 6566.9596, initial_price = 11130.44):
    dt = 1/252
    num_simulations = S.shape[0]
    payoff_values = np.zeros(num_simulations)
    for i in range(num_simulations):
        # Need to discount the payoff to present value
        payoff_values[i] = math.exp(-r*N*dt) * payoff(S[i,:], barrier_price, initial_price)
    return payoff_values


def gbm_SS(sigma, N, S_0, num_bins = 5, r = 0.01 , Nsim = 10000):
    dt = 1/252
    # Create a 2D array to store the stock prices for each simulation
    S = np.zeros([Nsim,N])
    S[:,0] = S_0

    for i in range(Nsim//num_bins):
        Z = np.random.normal(0, 1)
        for k in range(num_bins):
            S[num_bins*i+k,1] = S[num_bins*i+k,0] * np.exp( (r - 0.5 * sigma ** 2) * dt + sigma * (Z+k)/num_bins * np.sqrt(dt) )

    for i in range(Nsim):   
        for j in range(1,N-1):
            Z = np.random.normal(0, 1)
            # Calculate the stock price at time t+1 using the Geometric Brownian Motion model with risk-free interet rate
            S[i,j+1] = S[i,j] * np.exp( (r - 0.5 * sigma ** 2) * dt + sigma * Z * np.sqrt(dt) )
    return S
