import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from plotnine import *
import warnings


warnings.filterwarnings("ignore", category=UserWarning, module='plotnine')

def plot_training_results(t_train, data_S_train, data_I_train, data_R_train, history_with_physics, epochs_list):
    data_frames = []

    for epoch in epochs_list:
        if epoch in history_with_physics:
            S_train_df = pd.DataFrame({
                'time': t_train.flatten(),
                'value': data_S_train.flatten(),
                'epoch': int(epoch),
                'legends': 'True_S'
            })
            S_pred_df = pd.DataFrame({
                'time': t_train.flatten(),
                'value': history_with_physics[epoch]['S_train_pred'].flatten(),
                'epoch': int(epoch),
                'legends': 'Pred_S'
            })
            I_train_df = pd.DataFrame({
                'time': t_train.flatten(),
                'value': data_I_train.flatten(),
                'epoch': int(epoch),
                'legends': 'True_I'
            })
            I_pred_df = pd.DataFrame({
                'time': t_train.flatten(),
                'value': history_with_physics[epoch]['I_train_pred'].flatten(),
                'epoch': int(epoch),
                'legends': 'Pred_I'
            })
            R_train_df = pd.DataFrame({
                'time': t_train.flatten(),
                'value': data_R_train.flatten(),
                'epoch': int(epoch),
                'legends': 'True_R'
            })
            R_pred_df = pd.DataFrame({
                'time': t_train.flatten(),
                'value': history_with_physics[epoch]['R_train_pred'].flatten(),
                'epoch': int(epoch),
                'legends': 'Pred_R'
            })
            data_frames.extend([S_train_df, I_train_df, R_train_df, S_pred_df, I_pred_df, R_pred_df])
    
    combined_df = pd.concat(data_frames)
    combined_df.sort_values(by='epoch', inplace=True)

    color_mapping = {
        'True_S': 'blue',
        'Pred_S': 'red',
        'True_I': 'green',
        'Pred_I': 'orange',
        'True_R': 'purple',
        'Pred_R': 'cyan'
    }

    linetype_mapping = {
        'True_S': 'solid',
        'Pred_S': 'dashed',
        'True_I': 'solid',
        'Pred_I': 'dashed',
        'True_R': 'solid',
        'Pred_R': 'dashed'
    }

    legend_order = ['True_S', 'Pred_S', 'True_R', 'Pred_R', 'True_I', 'Pred_I']
    combined_df['legends'] = pd.Categorical(combined_df['legends'], categories=legend_order, ordered=True)

    combined_df['time'] = combined_df['time'].astype(float)
    combined_df['value'] = combined_df['value'].astype(float)

    p = (
        ggplot(combined_df, aes(x='time', y='value', color='legends', linetype='legends'))
        + geom_line(data=combined_df[combined_df['legends'].isin(['True_S', 'True_I', 'True_R'])], size=3, linetype='solid', alpha=0.8)
        + geom_line(data=combined_df[combined_df['legends'].isin(['Pred_S', 'Pred_I', 'Pred_R'])], size=1.5, linetype='dashed', alpha=0.8)
        + scale_color_manual(values=color_mapping)
        + scale_linetype_manual(values=linetype_mapping)
        + labs(x='Time', y='SIR populations')
        + theme(legend_position='bottom', axis_text_x=element_text(rotation=0))
        + facet_wrap('~epoch', ncol=2)
    )
    
    p.save("DI_FS_training_plot.png", width=9, height=7.5, units='in', dpi=300)
    p.show()

def plot_loss1(train_losses, val_losses):
    loss_df = pd.DataFrame({
        'Epoch': np.arange(len(train_losses)),
        'Training Loss': train_losses,
        'Testing Loss': val_losses
    })

    loss_df = loss_df.melt('Epoch', var_name='Loss Type', value_name='Loss')

    p = (ggplot(loss_df, aes(x='Epoch', y='Loss', color='Loss Type'))
         + geom_line(size=2)
         + labs(x='Epoch', y='Loss')
    )

    p.save("DI_FS_loss_plot1.png", width=4.5, height=3, units='in', dpi=300)
    p.show()

   
def plot_testing_results(t_test_tensor, data_S_test_tensor, data_I_test_tensor, data_R_test_tensor, S_test_pred, I_test_pred, R_test_pred):
    # Convert tensors to numpy
    t_test = t_test_tensor.detach().cpu().numpy().flatten()
    data_S_test = data_S_test_tensor.detach().cpu().numpy().flatten()
    data_I_test = data_I_test_tensor.detach().cpu().numpy().flatten()
    data_R_test = data_R_test_tensor.detach().cpu().numpy().flatten()

    # Ensure the predicted arrays have the same length as the true data
    S_test_pred = S_test_pred.flatten()
    I_test_pred = I_test_pred.flatten()
    R_test_pred = R_test_pred.flatten()

    # Check if lengths are consistent
    assert len(t_test) == len(data_S_test) == len(S_test_pred), "Length mismatch in Susceptible data."
    assert len(t_test) == len(data_I_test) == len(I_test_pred), "Length mismatch in Infected data."
    assert len(t_test) == len(data_R_test) == len(R_test_pred), "Length mismatch in Recovered data."

    # Create a DataFrame for ggplot
    test_results_df = pd.DataFrame({
        'Time': np.concatenate([t_test, t_test, t_test, t_test, t_test, t_test]),
        'Values': np.concatenate([
            data_S_test, data_I_test, data_R_test,
            S_test_pred, I_test_pred, R_test_pred
        ]),
        'Type': ['True'] * len(t_test) * 3 + ['Pred'] * len(t_test) * 3,
        'Legend': (
            ['True_S'] * len(t_test) +
            ['True_I'] * len(t_test) +
            ['True_R'] * len(t_test) +
            ['Pred_S'] * len(t_test) +
            ['Pred_I'] * len(t_test) +
            ['Pred_R'] * len(t_test)
        )
    })

    # Convert 'Legend' to a categorical type with a specified order
    test_results_df['Legend'] = pd.Categorical(
        test_results_df['Legend'],
        categories=['True_S', 'Pred_S', 'True_R', 'Pred_R', 'True_I', 'Pred_I'],
        ordered=True
    )

    # Define line type and color mapping
    line_type_map = {
        'True_S': 'solid', 'Pred_S': 'dashed', 
        'True_R': 'solid', 'Pred_R': 'dashed', 
        'True_I': 'solid', 'Pred_I': 'dashed'
    }
    
    # Define color mapping
    color_map = {
        'True_S': 'blue',
        'Pred_S': 'orange',
        'True_R': 'red',
        'Pred_R': 'cyan',
        'True_I': 'green',
        'Pred_I': 'purple'
    }
    
    # Plot
    p = (ggplot(test_results_df, aes(x='Time', y='Values', color='Legend', linetype='Legend')) 
         + geom_line(data=test_results_df[test_results_df['Legend'].isin(['True_S', 'True_I', 'True_R'])],
                     size=3, linetype='solid', alpha=0.8)  
         + geom_line(data=test_results_df[test_results_df['Legend'].isin(['Pred_S', 'Pred_I', 'Pred_R'])],
                     size=1.5, linetype='dashed', alpha=0.8)  
         + scale_linetype_manual(values=line_type_map)  
         + scale_color_manual(values=color_map)  
         + labs(y='SIR populations')  
         + theme(legend_position='bottom',
                axis_text_x=element_text(rotation=0))
         
    )
    
    # Save the plot
    p.save("DI_FS_testing_plot1.png", width=4.5, height=3, units='in', dpi=300)
    p.show()  





# def plot_testing_results(t_test, data_S_test, data_I_test, data_R_test, S_pred, I_pred, R_pred):
    # # Convert predictions to NumPy arrays if they are PyTorch tensors
    # if isinstance(S_pred, torch.Tensor):
        # S_pred_np = S_pred.numpy()
        # I_pred_np = I_pred.numpy()
        # R_pred_np = R_pred.numpy()
    # else:
        # S_pred_np = S_pred
        # I_pred_np = I_pred
        # R_pred_np = R_pred

    # # Convert test data to NumPy arrays if they are PyTorch tensors
    # if isinstance(data_S_test, torch.Tensor):
        # data_S_test_np = data_S_test.numpy()
        # data_I_test_np = data_I_test.numpy()
        # data_R_test_np = data_R_test.numpy()
    # else:
        # data_S_test_np = data_S_test
        # data_I_test_np = data_I_test
        # data_R_test_np = data_R_test

    # # Create a single plot
    # plt.figure(figsize=(12, 8))

    # # Plot S compartment
    # plt.plot(t_test, data_S_test_np, label='True S', color='blue')
    # plt.plot(t_test, S_pred_np, label='Predicted S', color='red', linestyle='--')

    # # Plot I compartment
    # plt.plot(t_test, data_I_test_np, label='True I', color='green')
    # plt.plot(t_test, I_pred_np, label='Predicted I', color='orange', linestyle='--')

    # # Plot R compartment
    # plt.plot(t_test, data_R_test_np, label='True R', color='purple')
    # plt.plot(t_test, R_pred_np, label='Predicted R', color='magenta', linestyle='--')

    # # Set plot title and labels
    # plt.title('Predictions vs True Values')
    # plt.xlabel('Time')
    # plt.ylabel('Values')
    # plt.legend()
    
    # # Show plot
    # plt.tight_layout()
    # plt.show()




def plot_parameters_IP(parameter_estimates, true_values, epochs):
    # Convert parameter estimates to DataFrame
    parameter_df = pd.DataFrame(parameter_estimates)
    parameter_df['Epoch'] = np.arange(len(parameter_df))

    # Create DataFrame for true values
    true_values_df = pd.DataFrame({
        'Epoch': np.arange(len(parameter_df)),
        'beta': [true_values['beta']] * len(parameter_df),
        'gamma': [true_values['gamma']] * len(parameter_df),
    })

    # Melt DataFrames for plotting
    melted_df = parameter_df.melt(id_vars='Epoch', var_name='Parameter', value_name='Estimated Value')
    true_values_melted = true_values_df.melt(id_vars='Epoch', var_name='Parameter', value_name='True Value')

    # Add a 'Type' column for merging
    melted_df['Type'] = 'Estimated Value'
    true_values_melted['Type'] = 'True Value'
    
    # Combine the melted DataFrames
    combined_df = pd.concat([melted_df, true_values_melted])

    # Define custom colors for each parameter
    color_map = {
        'beta': ['#2ca02c', '#98df8a'],  # Green shades
        'gamma': ['#d62728', '#ff9896'],  # Red shades
    }
    
    # LaTeX-style labels for parameters
    latex_labels = {
        'beta': '$\\beta$',
        'gamma': '$\\gamma$'
    }
    
    # Plot each parameter
    for param in true_values.keys():
        param_df = combined_df[combined_df['Parameter'] == param]
        
        # Start the plot
        p = (ggplot(param_df, aes(x='Epoch', color='Type'))
             + geom_line(aes(y='Estimated Value'), size=2, linetype='solid')
             + geom_line(aes(y='True Value'), linetype='dashed', size=2)
             + labs(
                x='Epoch',
                y=f'{latex_labels[param]}',
                color='Legend'
             )
             + scale_color_manual(values={
                 'Estimated Value': color_map[param][0], 
                 'True Value': color_map[param][1]
             },
             labels={
                 'Estimated Value': f'Estimated {latex_labels[param]}',
                 'True Value': f'True {latex_labels[param]}'
             })
             + theme(axis_title=element_text(size=12),
                     axis_text=element_text(size=10),
                     legend_title=element_text(size=12),
                     legend_text=element_text(size=10))
             # + set_font_sizes1()
        )
        
        # Add custom y-ticks for the gamma parameter
        if param == 'beta':
            p += scale_y_continuous(breaks=[0.5, 0.6, 0.7, 0.8, 0.9, 1.00])
            
        if param == 'gamma':
            p += scale_y_continuous(breaks=[0.05, 0.25, 0.50, 0.75, 1.00])
        
        # Save the plot
        p.save(f"DI_IP_{param}_plot.png", width=4, height=2.5, units='in', dpi=300)
        p.show()
        print(f"Saved plot as DI_IP_{param}_plot.png")
