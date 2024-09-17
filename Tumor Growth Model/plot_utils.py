from plotnine import ggplot, aes, geom_point, geom_line, labs, theme, element_text, facet_wrap, scale_color_manual, scale_x_continuous, scale_y_continuous, scale_linetype_manual,element_blank
import numpy as np
import pandas as pd

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
 

    
def set_font_sizes_loss(legend_size=12, axis_title_size=14, axis_tick_size=12):
    return theme(
        #legend_position=(0.98, 0.98),  # Place legend in lower-right corner
        legend_position='bottom',  # Place legend at the top
        # legend_text=element_text(size=legend_size),  # Font size for legend text
        # axis_title_x=element_text(size=axis_title_size),  # Font size for x-axis label
        # axis_title_y=element_text(size=axis_title_size),  # Font size for y-axis label
        # axis_text_x=element_text(size=axis_tick_size),  # Font size for x-ticks
        # axis_text_y=element_text(size=axis_tick_size),   # Font size for y-ticks
        legend_title=element_text(size=0),  # Hide legend title   # Font size for y-ticks
        legend_background=element_blank(),  # Remove legend background
        legend_box_background=element_blank()  # Remove background of the legend box
                   
    )   


 


def plot_training_results(t_train, data_X_train, history, epochs_list, K, x_ticks, y_ticks):
    data_frames = []
    
    for epoch in epochs_list:
        if epoch in history:
            train_df = pd.DataFrame({
                'time': t_train.flatten(),
                'population': data_X_train.flatten(),
                'epoch': f'Epoch: {epoch}',
                'legends': 'Training Samples'
            })
            carrying_capacity_df = pd.DataFrame({
                'time': t_train.flatten(),
                'population': np.full_like(t_train.flatten(), K),
                'epoch': f'Epoch: {epoch}',
                'legends': 'Carrying Capacity'
            })
            pred_train_df = pd.DataFrame({
                'time': t_train.flatten(),
                'population': history[epoch]['train_pred'].flatten(),
                'epoch': f'Epoch: {epoch}',
                'legends': 'PINN Prediction'
            })

            min_length = min(len(train_df), len(pred_train_df), len(carrying_capacity_df))
            train_df = train_df.iloc[:min_length]
            pred_train_df = pred_train_df.iloc[:min_length]
            carrying_capacity_df = carrying_capacity_df.iloc[:min_length]

            data_frames.append(train_df)
            data_frames.append(pred_train_df)
            data_frames.append(carrying_capacity_df)
    
    combined_df = pd.concat(data_frames, ignore_index=True)
    combined_df['epoch'] = pd.Categorical(combined_df['epoch'], categories=[f'Epoch: {e}' for e in epochs_list], ordered=True)
    combined_df.sort_values(by='epoch', inplace=True)

    color_map = {
        'Training Samples': '#1f77b4',
        'PINN Prediction': 'red',
        'Carrying Capacity': 'green'
    }
  
    p = (ggplot(combined_df, aes(x='time', y='population', color='legends', linetype='legends'))
         + geom_line(data=combined_df[combined_df['legends'] == 'Training Samples'], size=3, alpha=0.8)
         + geom_line(data=combined_df[combined_df['legends'] == 'PINN Prediction'], size=1.5, alpha=0.8)  # Thinner dashed line for PINN Prediction
         + geom_line(data=combined_df[combined_df['legends'] == 'Carrying Capacity'], size=2.0, alpha=0.5)  # Thinner dashed line for Carrying Capacity
         + scale_color_manual(values=color_map)
         + scale_linetype_manual(values={'Carrying Capacity': 'dashed', 'Training Samples': 'solid', 'PINN Prediction': 'dashed'})
         + labs(x='Time (t)', y='Tumor Size (X)')
         + facet_wrap('~epoch', ncol=2)
         + scale_x_continuous(breaks=x_ticks)
         + scale_y_continuous(breaks=y_ticks)
         + set_font_sizes())
    
    p.save("TG_FS_Training.png", width=8, height=5, units='in', dpi=300)
    p.show()



# def plot_testing_results(t_test, data_X_test, history, epochs_list, K, x_ticks, y_ticks):
    # data_frames = []
    
    # for epoch in epochs_list:
        # if epoch in history:
            # test_df = pd.DataFrame({
                # 'time': t_test.flatten(),
                # 'population': data_X_test.flatten(),
                # 'epoch': f'Epoch: {epoch}',
                # 'legends': 'Test Samples'
            # })
            # pred_test_df = pd.DataFrame({
                # 'time': t_test.flatten(),
                # 'population': history[epoch]['test_pred'].flatten(),
                # 'epoch': f'Epoch: {epoch}',
                # 'legends': 'PINN Prediction'
            # })
            # carrying_capacity_df = pd.DataFrame({
                # 'time': t_test.flatten(),
                # 'population': np.full_like(t_test.flatten(), K),
                # 'epoch': f'Epoch: {epoch}',
                # 'legends': 'Carrying Capacity'
            # })

            # data_frames.append(test_df)
            # data_frames.append(pred_test_df)
            # data_frames.append(carrying_capacity_df)
    
    # combined_df = pd.concat(data_frames, ignore_index=True)
    # combined_df['epoch'] = pd.Categorical(combined_df['epoch'], categories=[f'Epoch: {e}' for e in epochs_list], ordered=True)
    # combined_df.sort_values(by='epoch', inplace=True)

    # color_map = {
        # 'Test Samples': 'orange',
        # 'PINN Prediction': 'blue',
        # 'Carrying Capacity': 'green'
    # }

    # p = (ggplot(combined_df, aes(x='time', y='population', color='legends', linetype='legends'))
         # # + geom_point(data=combined_df[combined_df['legends'] == 'Test Samples'], size=3)
         # + geom_line(size=2.5, alpha=0.3)
       
         # + geom_line(alpha=0.8, size=2.5)
         # + scale_color_manual(values=color_map)
         # + scale_linetype_manual(values={'Carrying Capacity': 'dashed', 'Test Samples': 'solid', 'PINN Prediction': 'dashed'})
         # + labs(x='Time (t)', y='Tumor Size (X)')
         # + facet_wrap('~epoch', ncol=2)
         # + scale_x_continuous(breaks=x_ticks)
         # + scale_y_continuous(breaks=y_ticks)
         # + set_font_sizes())
    
    # p.save("TG_FS_Testing.png", width=8, height=5, units='in', dpi=300)
    # p.show()

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
         + set_font_sizes_loss())
    
    p.save("TG_FS_Loss.png", width=4.5, height=3, units='in', dpi=300)
    p.show()

################## Legengs inside the plots########################################################################


def set_font_sizes1(legend_size=12, axis_title_size=14, axis_tick_size=12):
    return theme(
        legend_position=(0.98, 0.08),  # Place legend in lower-right corner
        # legend_position='top',  # Place legend at the top
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
    
def set_font_sizes_loss1(legend_size=12, axis_title_size=14, axis_tick_size=12):
    return theme(
        legend_position=(0.98, 0.98),  # Place legend in lower-right corner
        #legend_position='top',  # Place legend at the top
        #legend_text=element_text(size=legend_size),  # Font size for legend text
        #axis_title_x=element_text(size=axis_title_size),  # Font size for x-axis label
        #axis_title_y=element_text(size=axis_title_size),  # Font size for y-axis label
        #axis_text_x=element_text(size=axis_tick_size),  # Font size for x-ticks
        #axis_text_y=element_text(size=axis_tick_size),   # Font size for y-ticks
        legend_title=element_text(size=0),  # Hide legend title   # Font size for y-ticks
        legend_background=element_blank(),  # Remove legend background
        legend_box_background=element_blank()  # Remove background of the legend box
                   
    ) 

def plot_training_results1(t_train, data_X_train, history, epochs_list, K, x_ticks, y_ticks):
    data_frames = []
    
    for epoch in epochs_list:
        if epoch in history:
            train_df = pd.DataFrame({
                'time': t_train.flatten(),
                'population': data_X_train.flatten(),
                'epoch': f'Epoch: {epoch}',
                'legends': 'Training Samples'
            })
            carrying_capacity_df = pd.DataFrame({
                'time': t_train.flatten(),
                'population': np.full_like(t_train.flatten(), K),
                'epoch': f'Epoch: {epoch}',
                'legends': 'Carrying Capacity'
            })
            pred_train_df = pd.DataFrame({
                'time': t_train.flatten(),
                'population': history[epoch]['train_pred'].flatten(),
                'epoch': f'Epoch: {epoch}',
                'legends': 'PINN Prediction'
            })
            

            min_length = min(len(train_df), len(pred_train_df), len(carrying_capacity_df))
            train_df = train_df.iloc[:min_length]
            pred_train_df = pred_train_df.iloc[:min_length]
            carrying_capacity_df = carrying_capacity_df.iloc[:min_length]

            data_frames.append(train_df)
            data_frames.append(pred_train_df)
            data_frames.append(carrying_capacity_df)
    
    combined_df = pd.concat(data_frames, ignore_index=True)
    combined_df['epoch'] = pd.Categorical(combined_df['epoch'], categories=[f'Epoch: {e}' for e in epochs_list], ordered=True)
    combined_df.sort_values(by='epoch', inplace=True)

    color_map = {
        'Training Samples': 'brown',
        'PINN Prediction': 'blue',
        'Carrying Capacity': 'green'
    }

    p = (ggplot(combined_df, aes(x='time', y='population', color='legends', linetype='legends'))
         + geom_point(data=combined_df[combined_df['legends'] == 'Training Samples'], size=3, shape='o')
         + geom_line(alpha=0.6, size=2)
         + scale_color_manual(values=color_map)
         + scale_linetype_manual(values={'Carrying Capacity': 'dashed', 'Training Samples': 'solid', 'PINN Prediction': 'solid'})
         + labs(x='Time (t)', y='Tumor Size (X)')
        #  + theme(axis_title=element_text(size=12),
        #          axis_text=element_text(size=10),
        #          legend_title=element_text(size=12),
        #          legend_text=element_text(size=10))
         + facet_wrap('~epoch', ncol=2)
         + scale_x_continuous(breaks=x_ticks)
         + scale_y_continuous(breaks=y_ticks)
         + set_font_sizes1())
    
    p.save("TG_FS_Training1.png", width=4, height=2.5, units='in', dpi=300)
    p.show()

def plot_testing_results1(t_test, data_X_test, history, epochs_list, K, x_ticks, y_ticks):
    data_frames = []
    epoch = max(history.keys(), default=None)
    if epoch is not None:
    # for epoch in epochs_list:
        # if epoch in history:
        test_df = pd.DataFrame({
            'time': t_test.flatten(),
            'population': data_X_test.flatten(),
            'epoch': f'Epoch: {epoch}',
            'legends': 'Test Samples'
        })
        pred_test_df = pd.DataFrame({
            'time': t_test.flatten(),
            'population': history[epoch]['test_pred'].flatten(),
            'epoch': f'Epoch: {epoch}',
            'legends': 'PINN Prediction'
        })
        carrying_capacity_df = pd.DataFrame({
            'time': t_test.flatten(),
            'population': np.full_like(t_test.flatten(), K),
            'epoch': f'Epoch: {epoch}',
            'legends': 'Carrying Capacity'
        })

        data_frames.append(test_df)
        data_frames.append(pred_test_df)
        data_frames.append(carrying_capacity_df)
    
    combined_df = pd.concat(data_frames, ignore_index=True)
    # combined_df['epoch'] = pd.Categorical(combined_df['epoch'], categories=[f'Epoch: {e}' for e in epochs_list], ordered=True)
    # combined_df.sort_values(by='epoch', inplace=True)

    color_map = {
        'Test Samples': 'orange',
        'PINN Prediction': 'blue',
        'Carrying Capacity': 'green'
    }

    p = (ggplot(combined_df, aes(x='time', y='population', color='legends', linetype='legends'))
         #+ geom_point(data=combined_df[combined_df['legends'] == 'Test Samples'], size=3)
         # + geom_line(size=2.5, alpha=0.3)
       
         # + geom_line(alpha=0.8, size=2.5)
         + geom_line(data=combined_df[combined_df['legends'] == 'Test Samples'], size=3, alpha=0.8)
         + geom_line(data=combined_df[combined_df['legends'] == 'PINN Prediction'], size=1.5, alpha=0.8)  # Thinner dashed line for PINN Prediction
         + geom_line(data=combined_df[combined_df['legends'] == 'Carrying Capacity'], size=2.0, alpha=0.5) 
         
         
         + scale_color_manual(values=color_map)
         + scale_linetype_manual(values={'Carrying Capacity': 'dashed', 'Test Samples': 'solid', 'PINN Prediction': 'dashed'})
         + labs(x='Time (t)', y='Tumor Size (X)')
         #+ facet_wrap('~epoch', ncol=2)
         + scale_x_continuous(breaks=x_ticks)
         + scale_y_continuous(breaks=y_ticks)
         + set_font_sizes())
    
    p.save("TG_FS_Testing1.png", width=4.5, height=3, units='in', dpi=300)
    p.show()

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
         + set_font_sizes_loss1())
    
    p.save("TG_FS_Loss1.png", width=4, height=2.5, units='in', dpi=300)
    p.show()
