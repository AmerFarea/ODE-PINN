import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize

def sim(state, t, params):
    X = state[0]
    alpha, K = params
    if X <= 0:  # Avoid invalid log operations
        return 0
    dXdt = alpha * np.log(K / X) * X
    return dXdt

def initial_loss_function(params, timepoints, population):
    y0 = [population[0]]
    t = np.linspace(timepoints[0], timepoints[-1], num=len(timepoints))
    solution = odeint(sim, y0, t, args=(params,), atol=1e-4, rtol=1e-4)  # Relax tolerances
    residuals = population - solution[:, 0]
    return np.mean(residuals**2)

def optimize_parameters(timepoints, population):
    params0 = np.array([1.0, 10.0])  # Initial guesses for alpha and K
    result = minimize(
        initial_loss_function,
        params0,
        args=(timepoints, population),
        method='L-BFGS-B',
        options={'disp': True, 'gtol': 1e-4}
    )
    return result.x
