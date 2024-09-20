

# Practical Understanding of Physics-Informed Neural Networks: Insights from Dynamic Systems Modeled by ODEs
 

Physics-Informed Neural Networks (PINNs) have gained recognition as a robust method for incorporating physical laws into deep learning models, providing enhanced capabilities for tackling complex problems. However, the practical implementation of PINNs remains challenging. This paper examines the use of PINNs in solving systems of ordinary differential equations (ODEs), focusing on two main challenges: the forward problem of approximating solutions and the inverse problem of estimating parameters. 

We provide detailed case studies on dynamic systems such as tumor growth, gene expression, and disease spread modeled by the Susceptible, Infected, Recovered (SIR) model, showcasing the effectiveness of PINNs in both accurately approximating solutions and estimating parameters. Our findings reveal that while PINNs can deliver precise and reliable solutions, their performance is highly dependent on the careful tuning of neural network architecture and hyperparameters. The results underscore the importance of customized configurations and robust optimization strategies. Overall, this study confirms the significant potential of PINNs in enhancing the understanding and management of dynamic systems, with promising applications across a wide range of real-world scenarios.

Depending on the nature and structure of the available data and the dynamic system modeled with ODEs, we provide implementations for both forward and inverse problems as follows:

## Forward Problem Implementation

Implementing a forward solution for a dynamic system using PINNs involves several steps:

1. **Define the Differential Equation**: The first step involves defining the differential equation governing the dynamic system.

2. **Define the Neural Network**: In this step, the neural network is defined, which includes specifying the architecture, choosing activation functions, and setting an input range. For example, the input could be time $t$ and the output a state variable $x(t)$, with suitable activation functions (e.g., GELU or Tanh) and a defined input range (e.g., $[0, T]$).

3. **Define the Loss Functions**: The loss functions are set up to guide the training process. The total loss function comprises different components corresponding to individual loss functions, each describing a specific aspect:

    - **Data Loss**: Measures the difference between predicted values and observed data:

      $$\text{Data Loss} = \sum_i \left| \hat{x}(t_i) - x_i \right|^2.$$

    - **Physics Loss**: Ensures the network adheres to the physical law (i.e., the differential equation) by computing the derivative of $x(t)$ and comparing it to the expected form:

      $$\text{Physics Loss} = \left| \frac{dx}{dt} - f(x, t) \right|^2.$$

    The total loss combines these individual losses:

    $$\text{Total Loss} = \lambda_1 \times \text{Data Loss} + \lambda_2 \times \text{Physics Loss},$$

    where $\lambda_1$ and $\lambda_2$ are weighting factors that balance the contributions.

4. **Train the Neural Network**: The network is trained by initializing its parameters, sampling points in the domain of $t$, and using an optimizer like Adam or SGD to minimize the total loss over multiple epochs until the model converges.

5. **Evaluate the Model**: The model's performance is assessed by comparing the predicted outputs $\hat{x}(t)$ with known values or analytical solutions when available.

## Inverse Problem Implementation

Using PINNs to estimate unknown parameters from observable data (the inverse problem) for dynamic systems based on ODEs involves several key steps:

1. **Load Data**: Begin by reading time points $t$ and corresponding observations $y$ from available sources (e.g., a CSV file) and converting this data into numpy arrays.

2. **Initial Parameter Estimation**: Define the ODE, where $\theta$ represents the parameters to be estimated. An initial loss function is set up to minimize the mean squared error between the observed data and the values predicted by the ODE. Optimization techniques may be used to estimate the initial values for the parameters $\theta$, denoted as $\theta_{\text{init}}$.

3. **Define the PINN Model**: Construct a neural network model to represent the dynamic system described by the ODE. The architecture includes:

    - **Input**: Time $t$
    - **Output**: Predicted observations $\hat{x}(t)$
    - **Hidden Layers**: Two layers with 100 units each, using ReLU activation functions.

    The parameters $\theta$ are initialized with $\theta_{\text{init}}$.

4. **Train the PINN Model**: During training, an optimizer such as Adam is used with a specified learning rate. Data and time points are normalized. The training process includes:

    - **Compute Total Loss**: 

      $$\text{Total\ Loss} = \text{Data\ Loss} + \text{Physics\ Loss}$$

      where Data Loss reflects how well the model fits the observed data, and Physics Loss enforces the ODE constraints.

    - **Backpropagation**: The loss is backpropagated to update the model parameters.

    - **Monitoring**: Epoch number and current loss are printed at regular intervals, such as every 100 epochs.

5. **Generate Predictions**: After training, predictions are made by creating a dense time grid $t_{\text{dense}}$ and using the trained PINN model to predict observations over this grid.

6. **Plot and Output Results**: Finally, results are visualized by plotting observed data alongside the model’s predictions. The optimized parameters $\theta$ can also be plotted.


The libraries utilized in our code offer essential tools for analyzing forward and inverse problems across a wide range of ordinary differential equation systems. We use `scipy.integrate.odeint` to solve ordinary differential equations, with `numpy` and `pandas` handling numerical computations and data management. To partition data into training and testing sets, we employ `train_test_split` from `sklearn.model_selection`. Our deep learning models are implemented with `PyTorch`, using `torch` as the neural network framework. For efficient data loading and batching, we utilize `DataLoader` and `TensorDataset` from `torch.utils.data`. Network architectures are constructed with `torch.nn`, and optimization during training is handled via `torch.optim`. We use `xavier_uniform_` from `torch.nn.init` for initializing network weights. Finally, `plotnine` is employed for data visualization.


For more information about PINN, please refer to the following: 
- Amer Farea, Olli Yli-Harja, and Frank Emmert-Streib. ["Understanding Physics-Informed Neural Networks: Techniques, Applications, Trends, and Challenges"](https://www.mdpi.com/2673-2688/5/3/74). *AI*, 2024, 5(3), 1534-1557. [https://doi.org/10.3390/ai5030074](https://doi.org/10.3390/ai5030074)


## Citation

```
@article{farea2024understanding,
  title={Understanding Physics-Informed Neural Networks: Techniques, Applications, Trends, and Challenges},
  author={Farea, Amer and Yli-Harja, Olli and Emmert-Streib, Frank},
  journal={AI},
  volume={5},
  number={3},
  pages={1534--1557},
  year={2024},
  publisher={MDPI}
}
```

