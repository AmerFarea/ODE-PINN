import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

def plot_results(timepoints, population, t_dense, solution):
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=timepoints, y=population, mode='markers', name='Data'), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_dense, y=solution, mode='lines', name='Fitted Curve'), row=1, col=1)
    fig.update_layout(title='Population Dynamics', xaxis_title='Time', yaxis_title='Population')
    pio.show(fig)

def plot_loss(loss_history):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=loss_history, mode='lines', name='Loss'))
    fig.update_layout(title='Training Loss History', xaxis_title='Epoch', yaxis_title='Loss')
    pio.show(fig)

def plot_parameters(alpha_history, K_history, true_alpha, true_K):
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Alpha History", "K History"))
    
    fig.add_trace(go.Scatter(y=alpha_history, mode='lines', name='Alpha'), row=1, col=1)
    fig.add_trace(go.Scatter(y=[true_alpha]*len(alpha_history), mode='lines', name='True Alpha', line=dict(dash='dash')), row=1, col=1)
    
    fig.add_trace(go.Scatter(y=K_history, mode='lines', name='K'), row=2, col=1)
    fig.add_trace(go.Scatter(y=[true_K]*len(K_history), mode='lines', name='True K', line=dict(dash='dash')), row=2, col=1)
    
    fig.update_layout(title='Parameter Evolution', xaxis_title='Epoch', yaxis_title='Value')
    pio.show(fig)
