import pandas as pd
from plotnine import *
import numpy as np


def set_font_sizes(legend_size=12, axis_title_size=14, axis_tick_size=12):
    return theme(
        #legend_position=(0.98, 0.08),  # Place legend in lower-right corner
        legend_position='bottom',  # Place legend at the top
        # legend_text=element_text(size=legend_size),  # Font size for legend text
        # axis_title_x=element_text(size=axis_title_size),  # Font size for x-axis label
        # axis_title_y=element_text(size=axis_title_size),  # Font size for y-axis label
        # axis_text_x=element_text(size=axis_tick_size),  # Font size for x-ticks
        # axis_text_y=element_text(size=axis_tick_size),   # Font size for y-ticks
        legend_title=element_text(size=0),  # Hide legend title
        legend_background=element_blank(),  # Remove legend background
        legend_box_background=element_blank(),  # Remove background of the legend box
        #legend_box_spacing=0.5,  # Adjust spacing if needed
        #legend_key_width=0.5,  # Adjust legend key width if needed
        #legend_key_height=0.5  # Adjust legend key height if needed
    )


def set_font_sizes1(legend_size=12, axis_title_size=14, axis_tick_size=12):
    return theme(
        #legend_position=(0.98, 0.98),  # Place legend in lower-right corner
        legend_position='bottom',  # Place legend at the top
        # legend_text=element_text(size=legend_size),  # Font size for legend text
        # axis_title_x=element_text(size=axis_title_size),  # Font size for x-axis label
        # axis_title_y=element_text(size=axis_title_size),  # Font size for y-axis label
        # axis_text_x=element_text(size=axis_tick_size),  # Font size for x-ticks
        # axis_text_y=element_text(size=axis_tick_size),   # Font size for y-ticks
        legend_title=element_text(size=0),  # Hide legend title
        legend_background=element_blank(),  # Remove legend background
        legend_box_background=element_blank(),  # Remove background of the legend box
        #legend_box_spacing=0.5,  # Adjust spacing if needed
        #legend_key_width=0.5,  # Adjust legend key width if needed
        #legend_key_height=0.5  # Adjust legend key height if needed
    )
 
def set_font_sizes2(legend_size=12, axis_title_size=14, axis_tick_size=12):
    return theme(
        legend_position=(0.98, 0.98),  # Place legend in lower-right corner
        #legend_position='bottom',  # Place legend at the top
        # legend_text=element_text(size=legend_size),  # Font size for legend text
        # axis_title_x=element_text(size=axis_title_size),  # Font size for x-axis label
        # axis_title_y=element_text(size=axis_title_size),  # Font size for y-axis label
        # axis_text_x=element_text(size=axis_tick_size),  # Font size for x-ticks
        # axis_text_y=element_text(size=axis_tick_size),   # Font size for y-ticks
        legend_title=element_text(size=0),  # Hide legend title
        legend_background=element_blank(),  # Remove legend background
        legend_box_background=element_blank(),  # Remove background of the legend box
        #legend_box_spacing=0.5,  # Adjust spacing if needed
        #legend_key_width=0.5,  # Adjust legend key width if needed
        #legend_key_height=0.5  # Adjust legend key height if needed
    )
    
   
    
# def plot_training_results(t_train, data_m_train, data_p_train, history_with_physics, epochs_list):
    # data_frames = []

    # for epoch in epochs_list:
        # if epoch in history_with_physics:
            # # Training data
            # m_train_df = pd.DataFrame({
                # 'time': t_train.flatten(),
                # 'value': data_m_train.flatten(),
                # 'epoch': f'Epoch: {epoch}',
                # 'legends': 'True_mRNA'
            # })
            
            # p_train_df = pd.DataFrame({
                # 'time': t_train.flatten(),
                # 'value': data_p_train.flatten(),
                # 'epoch': f'Epoch: {epoch}',
                # 'legends': 'True_Protein'
            # })

            # # Use precomputed PINN predictions for training data
            # m_pred_df = pd.DataFrame({
                # 'time': t_train.flatten(),
                # 'value': history_with_physics[epoch]['m_train_pred'],
                # 'epoch': f'Epoch: {epoch}',
                # 'legends': 'Pred_mRNA'
            # })

            # p_pred_df = pd.DataFrame({
                # 'time': t_train.flatten(),
                # 'value': history_with_physics[epoch]['p_train_pred'],
                # 'epoch': f'Epoch: {epoch}',
                # 'legends': 'Pred_Protein'
            # })

            # data_frames.extend([m_train_df, p_train_df, m_pred_df, p_pred_df])

    # combined_df = pd.concat(data_frames, ignore_index=True)
    # combined_df['epoch'] = pd.Categorical(combined_df['epoch'], categories=[f'Epoch: {e}' for e in epochs_list], ordered=True)
    # combined_df.sort_values(by='epoch', inplace=True)

    # # Define color mapping
    # color_mapping = {
        # 'True_mRNA': '#1f77b4',
        # 'True_Protein': '#008000',
        # 'Pred_mRNA': 'red',
        # 'Pred_Protein': '#8A2BE2'
    # }
    
    # # Define linetype mapping
    # linetype_mapping = {
        # 'True_mRNA': 'solid',
        # 'True_Protein': 'solid',
        # 'Pred_mRNA': 'dashed',
        # 'Pred_Protein': 'dashed',
       
    # }

    # # Create the plot for training data
    # p_train = (ggplot(combined_df, aes(x='time', y='value', color='legends', linetype='legends'))
               # # + geom_line(aes(color='legends'), size=2, linetype='dashed')  # Draw all lines first
               # # + geom_point(data=combined_df[combined_df['legends'].isin(['True_mRNA', 'True_Protein'])], size=3, alpha=0.6, shape='x')  # Draw points for specific legends
               # + geom_line(size=2.5, alpha=0.3)
       
               # + geom_line(alpha=0.8, size=2.5)
               # + scale_color_manual(values=color_mapping)  # Apply color mapping
               # + scale_linetype_manual(values=linetype_mapping)  # Apply linetype mapping
               # + labs(
                  # # title='Training Data with PINN Predictions',
                  # x='Time (t)',
                  # y='Concentration',
                  # #color='Legend'  # Label for the legend
               # )
               # + facet_wrap('~epoch', ncol=2)  # Adjust to show facets in 2 columns
               # # + theme(axis_title=element_text(size=12),
                       # # axis_text=element_text(size=10),
                       # # legend_title=element_text(size=12),
                       # # legend_text=element_text(size=10))
               # + set_font_sizes())

    # # Save the training plot
    # p_train.save("GE_FS_training_plot.png", width=8, height=5, units='in', dpi=300)
    # p_train.show()  # Show plot
    
    

def plot_training_results(t_train, data_m_train, data_p_train, history_with_physics, epochs_list):
    data_frames = []

    for epoch in epochs_list:
        if epoch in history_with_physics:
            # Training data
            m_train_df = pd.DataFrame({
                'time': t_train.flatten(),
                'value': data_m_train.flatten(),
                'epoch': f'Epoch: {epoch}',
                'legends': 'True_mRNA'
            })
            
            p_train_df = pd.DataFrame({
                'time': t_train.flatten(),
                'value': data_p_train.flatten(),
                'epoch': f'Epoch: {epoch}',
                'legends': 'True_Protein'
            })

            # Use precomputed PINN predictions for training data
            m_pred_df = pd.DataFrame({
                'time': t_train.flatten(),
                'value': history_with_physics[epoch]['m_train_pred'],
                'epoch': f'Epoch: {epoch}',
                'legends': 'Pred_mRNA'
            })

            p_pred_df = pd.DataFrame({
                'time': t_train.flatten(),
                'value': history_with_physics[epoch]['p_train_pred'],
                'epoch': f'Epoch: {epoch}',
                'legends': 'Pred_Protein'
            })

            data_frames.extend([m_train_df, p_train_df, m_pred_df, p_pred_df])

    combined_df = pd.concat(data_frames, ignore_index=True)
    combined_df['epoch'] = pd.Categorical(combined_df['epoch'], categories=[f'Epoch: {e}' for e in epochs_list], ordered=True)
    combined_df.sort_values(by='epoch', inplace=True)

    # Define color mapping
    color_mapping = {
        'True_mRNA': '#1f77b4',
        'True_Protein': '#008000',
        'Pred_mRNA': 'red',
        'Pred_Protein': '#8A2BE2'
    }
    
    # Define linetype mapping
    linetype_mapping = {
        'True_mRNA': 'solid',
        'True_Protein': 'solid',
        'Pred_mRNA': 'dashed',
        'Pred_Protein': 'dashed'
    }

    # Create the plot for training data
    p_train = (ggplot(combined_df, aes(x='time', y='value', color='legends', linetype='legends'))
               #+ geom_line(size=3, alpha=0.8)  # Apply size to all lines first
               + geom_line(data=combined_df[combined_df['legends'].isin(['True_mRNA', 'True_Protein'])],
                           size=3, linetype='solid', alpha=0.8)  # Apply size and linetype specifically for solid lines
               
               + geom_line(data=combined_df[combined_df['legends'].isin(['Pred_mRNA', 'Pred_Protein'])],
                           size=1.5, linetype='dashed', alpha=0.8)  # Apply size and linetype specifically for dashed lines
               + scale_color_manual(values=color_mapping)  # Apply color mapping
               + scale_linetype_manual(values=linetype_mapping)  # Apply linetype mapping
               + labs(
                  x='Time (t)',
                  y='Concentration'
               )
               + facet_wrap('~epoch', ncol=2)  # Adjust to show facets in 2 columns
               + set_font_sizes())

    # Save the training plot
    p_train.save("GE_FS_training_plot.png", width=8, height=5, units='in', dpi=300)
    p_train.show()  # Show plot




def plot_testing_results(t_test, data_m_test, data_p_test, history_with_physics, epochs_list):
    data_frames = []
    epoch = max(history_with_physics.keys(), default=None)
    if epoch is not None:

    # for epoch in epochs_list:
        # if epoch in history_with_physics:
        # Test data
        m_test_df = pd.DataFrame({
            'time': t_test.flatten(),
            'value': data_m_test.flatten(),
            'epoch': f'Epoch: {epoch}',
            'legends': 'True_mRNA'
        })
        
        p_test_df = pd.DataFrame({
            'time': t_test.flatten(),
            'value': data_p_test.flatten(),
            'epoch': f'Epoch: {epoch}',
            'legends': 'True_Protein'
        })

        # Use precomputed PINN predictions for test data
        m_pred_df = pd.DataFrame({
            'time': t_test.flatten(),
            'value': history_with_physics[epoch]['m_test_pred'],
            'epoch': f'Epoch: {epoch}',
            'legends': 'Pred_mRNA'
        })

        p_pred_df = pd.DataFrame({
            'time': t_test.flatten(),
            'value': history_with_physics[epoch]['p_test_pred'],
            'epoch': f'Epoch: {epoch}',
            'legends': 'Pred_Protein'
        })

        data_frames.extend([m_test_df, p_test_df, m_pred_df, p_pred_df])

    combined_df = pd.concat(data_frames, ignore_index=True)
    combined_df['epoch'] = pd.Categorical(combined_df['epoch'], categories=[f'Epoch: {e}' for e in epochs_list], ordered=True)
    combined_df.sort_values(by='epoch', inplace=True)

  
     ##Define color mapping
    color_mapping = {
        'True_mRNA': '#1E90FF',
        'True_Protein': '#FF4500',
        'Pred_mRNA': '#FFD700',
        'Pred_Protein': '#8A2BE2'
    }
    
     # Define linetype mapping
    linetype_mapping = {
        'True_mRNA': 'solid',
        'True_Protein': 'solid',
        'Pred_mRNA': 'dashed',
        'Pred_Protein': 'dashed',
       
    }

    # Create the plot for testing data
    p_test = (ggplot(combined_df, aes(x='time', y='value', color='legends', linetype='legends'))
              #+ geom_line(aes(color='legends'), size=2, linetype='dashed')  # Draw all lines first
              #+ geom_point(data=combined_df[combined_df['legends'].isin(['True_mRNA', 'True_Protein'])], size=3, shape='x')  # Draw points for specific legends
              
       
              # + geom_line(alpha=0.8, size=2.5)
         
              # + geom_line(size=2.5, alpha=0.3)
              
              + geom_line(data=combined_df[combined_df['legends'].isin(['True_mRNA', 'True_Protein'])],
                           size=3, linetype='solid', alpha=0.8)  # Apply size and linetype specifically for solid lines
               
               + geom_line(data=combined_df[combined_df['legends'].isin(['Pred_mRNA', 'Pred_Protein'])],
                           size=1.5, linetype='dashed', alpha=0.8)  # Apply size and linetype specifically for dashed lines
              
              + scale_color_manual(values=color_mapping)  # Apply color mapping
              + scale_linetype_manual(values=linetype_mapping)  # Apply linetype mapping
              + labs(
                 # title='Testing Data with PINN Predictions',
                 x='Time (t)',
                 y='Concentration',
                 #color='Legend'  # Label for the legend
              )
              #+ facet_wrap('~epoch', ncol=2)
              # + theme(axis_title=element_text(size=12),
                      # axis_text=element_text(size=10),
                      # legend_title=element_text(size=12),
                      # legend_text=element_text(size=10))
              + set_font_sizes())

    # Save the testing plot
    p_test.save("GE_FS_testing_plot1.png", width=4.5, height=3, units='in', dpi=300)
    p_test.show()  # Show plot


def plot_loss_curves(train_losses, val_losses):
    loss_df = pd.DataFrame({
        'Epoch': np.arange(len(train_losses)),
        'Loss': train_losses,
        'legends': 'Training Loss'
    })
    val_loss_df = pd.DataFrame({
        'Epoch': np.arange(len(val_losses)),
        'Loss': val_losses,
        'legends': 'Testing Loss'
    })
    combined_loss_df = pd.concat([loss_df, val_loss_df])
    
    p = (ggplot(combined_loss_df, aes(x='Epoch', y='Loss', color='legends'))
         + geom_line(size=2)
         + labs(x='Epoch', y='Loss')
         + set_font_sizes1())
    
    p.save("GE_FS_Loss.png", width=4.5, height=3, units='in', dpi=300)
    p.show()



################# for single plot

def plot_training_results1(t_train, data_m_train, data_p_train, history_with_physics, epochs_list):
    data_frames = []

    for epoch in epochs_list:
        if epoch in history_with_physics:
            # Training data
            m_train_df = pd.DataFrame({
                'time': t_train.flatten(),
                'value': data_m_train.flatten(),
                'epoch': f'Epoch: {epoch}',
                'legends': 'True_mRNA'
            })
            
            p_train_df = pd.DataFrame({
                'time': t_train.flatten(),
                'value': data_p_train.flatten(),
                'epoch': f'Epoch: {epoch}',
                'legends': 'True_Protein'
            })

            # Use precomputed PINN predictions for training data
            m_pred_df = pd.DataFrame({
                'time': t_train.flatten(),
                'value': history_with_physics[epoch]['m_train_pred'],
                'epoch': f'Epoch: {epoch}',
                'legends': 'Pred_mRNA'
            })

            p_pred_df = pd.DataFrame({
                'time': t_train.flatten(),
                'value': history_with_physics[epoch]['p_train_pred'],
                'epoch': f'Epoch: {epoch}',
                'legends': 'Pred_Protein'
            })

            data_frames.extend([m_train_df, p_train_df, m_pred_df, p_pred_df])

    combined_df = pd.concat(data_frames, ignore_index=True)
    combined_df['epoch'] = pd.Categorical(combined_df['epoch'], categories=[f'Epoch: {e}' for e in epochs_list], ordered=True)
    combined_df.sort_values(by='epoch', inplace=True)

 
    
 ##Define color mapping
    color_mapping = {
        'True_mRNA': '#000000',
        'True_Protein': '#008000',
        'Pred_mRNA': '#FFD700',
        'Pred_Protein': '#8A2BE2'
        }


    # Create the plot for training data
    p_train = (ggplot(combined_df, aes(x='time', y='value', color='legends'))
               + geom_line(aes(color='legends'), size=2, linetype='dashed')  # Draw all lines first
               + geom_point(data=combined_df[combined_df['legends'].isin(['True_mRNA', 'True_Protein'])], size=3, alpha=0.6, shape='x')  # Draw points for specific legends
               + scale_color_manual(values=color_mapping)  # Apply color mapping
               + labs(
                  # title='Training Data with PINN Predictions',
                  x='Time (t)',
                  y='Concentration',
                  color='Legend'  # Label for the legend
               )
               + facet_wrap('~epoch', ncol=2)
               # + theme(axis_title=element_text(size=12),
                       # axis_text=element_text(size=10),
                       # legend_title=element_text(size=12),
                       # legend_text=element_text(size=10))
               + set_font_sizes())

    # Save the training plot
    p_train.save("GE_FS_training_plot1.png", width=4, height=2.5, units='in', dpi=300)
    p_train.show()  # Show plot


def plot_testing_results1(t_test, data_m_test, data_p_test, history_with_physics, epochs_list):
    data_frames = []

    for epoch in epochs_list:
        if epoch in history_with_physics:
            # Test data
            m_test_df = pd.DataFrame({
                'time': t_test.flatten(),
                'value': data_m_test.flatten(),
                'epoch': f'Epoch: {epoch}',
                'legends': 'True_mRNA'
            })
            
            p_test_df = pd.DataFrame({
                'time': t_test.flatten(),
                'value': data_p_test.flatten(),
                'epoch': f'Epoch: {epoch}',
                'legends': 'True_Protein'
            })

            # Use precomputed PINN predictions for test data
            m_pred_df = pd.DataFrame({
                'time': t_test.flatten(),
                'value': history_with_physics[epoch]['m_test_pred'],
                'epoch': f'Epoch: {epoch}',
                'legends': 'Pred_mRNA'
            })

            p_pred_df = pd.DataFrame({
                'time': t_test.flatten(),
                'value': history_with_physics[epoch]['p_test_pred'],
                'epoch': f'Epoch: {epoch}',
                'legends': 'Pred_Protein'
            })

            data_frames.extend([m_test_df, p_test_df, m_pred_df, p_pred_df])

    combined_df = pd.concat(data_frames, ignore_index=True)
    combined_df['epoch'] = pd.Categorical(combined_df['epoch'], categories=[f'Epoch: {e}' for e in epochs_list], ordered=True)
    combined_df.sort_values(by='epoch', inplace=True)

  
     ##Define color mapping
    color_mapping = {
        'True_mRNA': '#1E90FF',
        'True_Protein': '#FF4500',
        'Pred_mRNA': '#FFD700',
        'Pred_Protein': '#8A2BE2'
    }
    
     # Define linetype mapping
    linetype_mapping = {
        'True_mRNA': 'solid',
        'True_Protein': 'solid',
        'Pred_mRNA': 'dashed',
        'Pred_Protein': 'dashed',
       
    }

    # Create the plot for testing data
    p_test = (ggplot(combined_df, aes(x='time', y='value', color='legends', linetype='legends'))
              #+ geom_line(aes(color='legends'), size=2, linetype='dashed')  # Draw all lines first
              #+ geom_point(data=combined_df[combined_df['legends'].isin(['True_mRNA', 'True_Protein'])], size=3, shape='x')  # Draw points for specific legends
              
       
              + geom_line(alpha=0.8, size=2.5)
              + geom_line(size=2.5, alpha=0.3)
              + scale_color_manual(values=color_mapping)  # Apply color mapping
              + scale_linetype_manual(values=linetype_mapping)  # Apply linetype mapping
              + labs(
                 # title='Testing Data with PINN Predictions',
                 x='Time (t)',
                 y='Concentration',
                 #color='Legend'  # Label for the legend
              )
              #+ facet_wrap('~epoch', ncol=2)
              # + theme(axis_title=element_text(size=12),
                      # axis_text=element_text(size=10),
                      # legend_title=element_text(size=12),
                      # legend_text=element_text(size=10))
              + set_font_sizes())

    # Save the testing plot
    p_test.save("GE_FS_testing_plot1.png", width=4.5, height=3, units='in', dpi=300)
    p_test.show()  # Show plot


def plot_loss_curves1(train_losses, val_losses):
    loss_df = pd.DataFrame({
        'Epoch': np.arange(len(train_losses)),
        'Loss': train_losses,
        'legends': 'Training Loss'
    })
    val_loss_df = pd.DataFrame({
        'Epoch': np.arange(len(val_losses)),
        'Loss': val_losses,
        'legends': 'Testing Loss'
    })
    combined_loss_df = pd.concat([loss_df, val_loss_df])
    
    p = (ggplot(combined_loss_df, aes(x='Epoch', y='Loss', color='legends'))
         + geom_line(size=2)
         + labs(x='Epoch', y='Loss')
         + set_font_sizes1())
    
    p.save("GE_FS_Loss1.png", width=4.5, height=3, units='in', dpi=300)
    p.show()
    
    
    
########################### INVERSE PROBLEM #####################################


# Plotting functions using ggplot
def plot_training_results_IP(t_train, data_m_train, data_p_train, history_with_physics, epochs_list):
    data_frames = []
    
    for epoch in epochs_list:
        if epoch in history_with_physics:
            m_train_df = pd.DataFrame({
                'time': t_train.flatten(),
                'value': data_m_train.flatten(),
                'epoch': f'Epoch: {epoch}',
                'legends': 'True_mRNA'
            })
            
            m_pred_df = pd.DataFrame({
                'time': t_train.flatten(),
                'value': history_with_physics[epoch]['m_train_pred'].flatten(),
                'epoch': f'Epoch: {epoch}',
                'legends': 'Pred_mRNA'
            })

            p_train_df = pd.DataFrame({
                'time': t_train.flatten(),
                'value': data_p_train.flatten(),
                'epoch': f'Epoch: {epoch}',
                'legends': 'True_Protein'
            })
            p_pred_df = pd.DataFrame({
                'time': t_train.flatten(),
                'value': history_with_physics[epoch]['p_train_pred'].flatten(),
                'epoch': f'Epoch: {epoch}',
                'legends': 'Pred_Protein'
            })
            data_frames.extend([m_train_df, p_train_df, m_pred_df, p_pred_df])
    
    combined_df = pd.concat(data_frames)

    # Define color mapping
    color_mapping = {
        'True_mRNA': '#00A3E0',
        'Pred_mRNA': '#FFC107',
        'True_Protein': '#FF4500',
        'Pred_Protein': '#C71585'
    }

    # Create the plot
    p = (ggplot(combined_df, aes(x='time', y='value', color='legends'))
         + geom_point(data=combined_df[combined_df['legends'] == 'True_mRNA'], size=3, shape='x')
         + geom_line(aes(color='legends'), size=2, linetype='dashed')
         + geom_point(data=combined_df[combined_df['legends'] == 'True_Protein'], size=3, shape='x')
         + scale_color_manual(values=color_mapping)  # Apply color mapping
         + labs(
            # title='Training Data and PINN Predictions',
            x='Time (t)',
            y='Concentration',
            color='Legend'  # Label for the legend
         )
         + facet_wrap('~epoch', ncol=2)
         # + theme(axis_title=element_text(size=12),
                 # axis_text=element_text(size=10),
                 # legend_title=element_text(size=12),
                 # legend_text=element_text(size=10))
         + set_font_sizes())
    
    # Save the plot
    p.save("GE_IP_training_plot.png", width=4, height=2.5, units='in', dpi=300)
    p.show() 

def plot_loss_IP(train_losses):
    # Create a DataFrame with the training losses
    loss_df = pd.DataFrame({
        'Epoch': np.arange(len(train_losses)),
        'Loss': train_losses,
        'legends': 'Training Loss'  # Add a new column for the legend
    })

    # Create the plot
    p = (ggplot(loss_df, aes(x='Epoch', y='Loss', color='legends'))  # Map the color to the 'legends' column
         + geom_line(size=2)  # Use the default color mapping for the legend
         + scale_color_manual(values={'Training Loss': '#7F3A6B'})  # Set the color for 'Training Loss'
         + labs(
            # title='Training Loss Curve',
            x='Epoch',
            y='Loss',
            color='Legend'  # Label for the legend
         )
         # + theme(axis_title=element_text(size=12),
                 # axis_text=element_text(size=10))
         + set_font_sizes1()
    )

    # Save the plot
    p.save("GE_IP_loss_plot.png", width=4, height=2.5, units='in', dpi=300)
    p.show()




def plot_parameters_IP(parameter_estimates, true_values, epochs):
    # Convert parameter estimates to DataFrame
    parameter_df = pd.DataFrame(parameter_estimates)
    parameter_df['Epoch'] = np.arange(len(parameter_df))

    # Create DataFrame for true values
    true_values_df = pd.DataFrame({
        'Epoch': np.arange(len(parameter_df)),
        'k_m': [true_values['k_m']] * len(parameter_df),
        'gamma_m': [true_values['gamma_m']] * len(parameter_df),
        'k_p': [true_values['k_p']] * len(parameter_df),
        'gamma_p': [true_values['gamma_p']] * len(parameter_df),
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
        'k_m': ['#1f77b4', '#aec7e8'],  # Blue shades
        'gamma_m': ['#ff7f0e', '#ffbb78'],  # Orange shades
        'k_p': ['#2ca02c', '#98df8a'],  # Green shades
        'gamma_p': ['#d62728', '#ff9896'],  # Red shades
    }
    
    # LaTeX-style labels for parameters
    latex_labels = {
        'k_m': '$k_{m}$',
        'gamma_m': '$\\gamma_{m}$',
        'k_p': '$k_{p}$',
        'gamma_p': '$\\gamma_{p}$'
    }
    
    # Plot each parameter
    for param in true_values.keys():
        param_df = combined_df[combined_df['Parameter'] == param]
        
        p = (ggplot(param_df, aes(x='Epoch', color='Type'))
             + geom_line(aes(y='Estimated Value'), size=2, linetype='solid')
             + geom_line(aes(y='True Value'), linetype='dashed', size=2)
             + labs(
                # title=f'{latex_labels[param]} vs True Value',
                    x='Epoch',
                    y=f'{latex_labels[param]}',
                    color='Legend')
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
             + set_font_sizes2()
        )
        
        # Save the plot
        p.save(f"GE_IP_{param}_plot.png", width=4, height=2.5, units='in', dpi=300)
        p.show()



