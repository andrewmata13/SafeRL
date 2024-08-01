import time

import numpy as np
import os.path
import random
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from numpy.linalg import norm
import statistics
import autokoopman.observable as kobs

from autokoopman import auto_koopman
import autokoopman.core.trajectory as traj

"""set the variable PATH to the directory with measurements folder"""
PATH = os.getcwd()


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def load_data(benchmark):
    """load the measured data"""

    path = os.path.join(PATH, benchmark)
    cnt = 0
    data = []

    while True:
        dirpath = os.path.join(path, 'measurement_' + str(cnt))
        if os.path.isdir(dirpath):
            states = np.asarray(pd.read_csv(os.path.join(dirpath, 'trajectory.csv')))
            inputs = np.asarray(pd.read_csv(os.path.join(dirpath, 'input.csv')))
            time = np.asarray(pd.read_csv(os.path.join(dirpath, 'time.csv'))) * 0.02
            time = np.resize(time, (time.shape[0],))
            data.append(traj.Trajectory(time, states, inputs))
            cnt += 1
        else:
            break

    if len(data) > 100:
        print("truncate data to 100 trajectories")
        data = data[0:100]


    return data


def split_data(data, num_test=10):
    """randomly split data into training and test set"""

    random.seed(0)
    ind = random.sample(range(0, len(data)), num_test)

    test_data = [data[i] for i in ind]
    training_data = [data[i] for i in range(0, len(data)) if i not in ind]

    ids = np.arange(0, len(training_data)).tolist()
    training_data = traj.TrajectoriesData(dict(zip(ids, training_data)))

    ids = np.arange(0, len(test_data)).tolist()

    if ids:
        test_data = traj.TrajectoriesData(dict(zip(ids, test_data)))
    else:
        test_data = []

    return training_data, test_data


def train_model(data):
    """train the Koopman model using the AutoKoopman library"""

    dt = data._trajs[0].times[1] - data._trajs[0].times[0]

    # learn model from data
    dim = data._trajs[0].states[0].size
    num_rff_obs = 200
    rff_gamma = 1e-4
    poly_degree = 2
    observables = 'rff' #kobs.IdentityObservable() | kobs.PolynomialObservable(dim, poly_degree) | kobs.RFFObservable(dim, num_rff_obs, rff_gamma)   

    experiment_results = auto_koopman(
        data,  # list of trajectories
        sampling_period=dt,
        obs_type=observables, #'rff',
        opt='grid',
        n_obs=200,
        rank=(1,200,20),
        grid_param_slices=5,
        n_splits=None, # was 5
        max_opt_iter=100
    )

    # get the model from the experiment results
    model = experiment_results['tuned_model']

    print(f"max: {np.max(model._A)}")
    print(f"min: {np.min(model._A)}")

    print(experiment_results['hyperparameters'])
    print(experiment_results['hyperparameter_values'])

    print()

    return model


def compute_trajectory(model, times, states, inputs, resample):
    test_traj = []
    test_traj.append(list(states[0]))

    current_state = states[0]
    
    for time_step in range(len(times) - 1):

        # Reset to actual ground truth
        #if time_step % resample == 0:
        #    current_state = states[time_step]
        
        time = times[time_step]

        if inputs is not None:
            action = inputs[time_step]
        else:
            action = None

        current_state = model.step(time, current_state, action)

        test_traj.append(list(current_state))
    
    return np.array(test_traj)
    

def compute_error(model, test_data):
    """compute error between model prediction and real data"""
    euc_norms = []

    # loop over all test trajectories
    tmp = list(test_data._trajs.values())


    for t in tmp:
        try:
            # simulate using the learned model
            iv = t.states[0, :]
            start_time = t.times[0]
            end_time = t.times[len(t.times) - 1]
            teval = np.linspace(start_time, end_time, len(t.times))

            trajectory = model.solve_ivp(
                initial_state=iv,
                tspan=(start_time, end_time),
                sampling_period=t.times[1] - t.times[0],
                inputs=t.inputs,
                teval=teval
            )

            # compute error
            y_true = np.matrix.flatten(t.states)
            y_pred = np.matrix.flatten(trajectory.states)
            euc_norm = norm(y_true - y_pred) / norm(y_true)
            euc_norms.append(euc_norm)
        except:
            print("ERROR--solve_ivp failed (likely unstable model)")
            # NOTE: Robot has constant 0 states, resulting in high error numbers (MSE is good)
            euc_norms.append(np.infty)


    '''
    for t in tmp:
        y_true = np.matrix.flatten(t.states)
        y_pred = np.matrix.flatten(compute_trajectory(model, t.times, t.states, t.inputs, 1))
        euc_norm = norm(y_true - y_pred) / norm(y_true)
        euc_norms.append(euc_norm)
    '''
    
    return statistics.mean(euc_norms)


def plot(trajectory, true_trajectory, var_1, var_2):
    plt.figure(figsize=(10, 6))
    # plot the results
    if var_2 == -1:  # plot against time
        plt.plot(trajectory.states[:, var_1], label='Trajectory Prediction')
        plt.plot(true_trajectory.states[:, var_1], label='Ground truth')
    else:
        plt.plot(trajectory.states.T[var_1], trajectory.states.T[var_2], label='Trajectory Prediction')
        plt.plot(true_trajectory.states.T[var_1], true_trajectory.states.T[var_2], label='Ground Truth')

    plt.grid()
    plt.show()

if __name__ == '__main__':
    benchmark = 'DubinsRejoin'

    # load data
    set_seed()
    data = load_data(benchmark)
    
    # split into training and validation set

    if len(data) == 100:
        print(f"using 20 trajectories for test data")
        training_data, test_data = split_data(data, 20)
    else:
        print(f"using all data from training and testing (num traj != 100)")
        training_data, _ = split_data(data, 0)
        test_data = training_data

    start = time.time()
    model = train_model(training_data)
    end = time.time()


    # compute error
    euc_norm = compute_error(model, test_data)
    comp_time = round(end - start, 3)

    print(benchmark)
    err_str = f"{round(euc_norm * 100, 2)}%"
    print(f"The average euc norm perc error is {err_str}")
    print("time taken: ", comp_time)

    # loop over all test trajectories and plot
    tmp = list(test_data._trajs.values())

    # plot first 6 trajectories (or less) on the same plot, using the data from tmp[0] ... tmp[5]
    # if it's less than 6, plot all of them, otherwise use a 2x3 plot. use subplots
    figsize = (16, 9)

    if len(tmp) <= 2:
        fig, axs = plt.subplots(1, 2, figsize=figsize)
    elif len(tmp) <= 4:
        fig, axs = plt.subplots(2, 2, figsize=figsize)
    else:
        fig, axs = plt.subplots(2, 3, figsize=figsize)
     
    for plot_index in range(0, min(6, len(tmp))):
        print(f"Plotting trajectory index={plot_index}")
        t = tmp[plot_index]

        test_traj = compute_trajectory(model, t.times, t.states, t.inputs, 1)

        if len(tmp) <= 2:
            ax = axs[plot_index]
        elif len(tmp) <= 4:
            ax = axs[plot_index // 2][plot_index % 2]
        else:
            ax = axs[plot_index // 3][plot_index % 3]

        #plt.figure(figsize=(10, 6))
        ax.plot(test_traj.T[0], test_traj.T[1], '-o', ms=1.5, lw=0.5, label='Trajectory Prediction')
        ax.plot(t.states.T[0], t.states.T[1], '-', lw=1, label='Ground Truth')

        #plt.legend()
        ax.grid()
        
    plt.tight_layout()
    plt.show()

    '''
    t = tmp[0]
    # simulate using the learned model
    iv = t.states[0, :]
    start_time = t.times[0]
    sampling_period = t.times[1] - t.times[0]
    end_time = t.times[len(t.times) - 1]
    teval = np.linspace(start_time, end_time, len(t.times))

    trajectory = model.solve_ivp(
        initial_state=iv,
        tspan=(start_time, end_time),
        sampling_period=t.times[1] - t.times[0],
        inputs=t.inputs,
        teval=teval
    )
    plot(trajectory, t, 0, 1)

    #print(trajectory.states)
    '''

