import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import json
import time
import os
import pickle
import argparse
import sys

import koopman_util

from cachier import cachier

def make_states(lead_x_list, lead_y_list, wingman_x_list, wingman_y_list, lead_speed_list, wingman_speed_list, lead_heading_list, wingman_heading_list,
    wingman=True):
    '''make states from raw data'''

    lead_vx_normalized = np.cos(lead_heading_list)
    lead_vy_normalized = np.sin(lead_heading_list)

    wingman_vx_normalized = np.cos(wingman_heading_list)
    wingman_vy_normalized = np.sin(wingman_heading_list)

    wingman_vx = wingman_speed_list * np.cos(wingman_heading_list)
    wingman_vy = wingman_speed_list * np.sin(wingman_heading_list)

    lead_vx = lead_speed_list * np.cos(lead_heading_list)
    lead_vy = lead_speed_list * np.sin(lead_heading_list)

    ############ LEAD ###############
    if wingman:
        states = list(zip(lead_x_list, lead_y_list, lead_heading_list, lead_speed_list))
    #states = list(zip(lead_x_list, lead_y_list, lead_vx_normalized, lead_vy_normalized, lead_speed_list)) # magnorm of angle

    ############ WINGMAN ###########
    if not wingman:
        states = list(zip(wingman_x_list, wingman_y_list, wingman_heading_list, wingman_speed_list))
    
    #states = list(zip(wingman_x_list, wingman_y_list, wingman_heading_list, wingman_speed_list, wingman_vx, wingman_vy, )) # magnorm of angle


    
    #states = list(zip(lead_x_list, lead_y_list, lead_heading_list, lead_speed_list, lead_vx_normalized, lead_vy_normalized))


    return states



@cachier(cache_dir='cachier')
def load_json(file_path):

    start = time.time()

    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]

    diff = time.time() - start

    print(f"Loaded {len(data)} lines from {file_path} in {diff:.2f} seconds") 

    return data

def extract_states_actions(data, max_num_traj=np.inf):
    '''convert loaded data in test and train data with correct obserables
    returns list of 2-d np.array of states and actions
    
    each np.array has the following shape:
    dim 0 -> state/action
    dim 1 -> time step

    note different episodes may have different lengths
    '''
    
    start = time.time()

    # next get the data
    states_np_list = []
    actions_np_list = []
    batch = []

    for data_item in data:
        if not (data_item['info']['failure'] or data_item['info']['success']):
            batch.append(data_item)
        else:
            #print(f"Batch index={traj_index} of length {len(batch)}")
            lead_x = np.array([entry['info']['lead']['x'] for entry in batch])
            lead_y = np.array([entry['info']['lead']['y'] for entry in batch])
            wingman_x = np.array([entry['info']['wingman']['x'] for entry in batch])
            wingman_y = np.array([entry['info']['wingman']['y'] for entry in batch])
            lead_speed = np.array([entry['info']['lead']['v'] for entry in batch])
            wingman_speed = np.array([entry['info']['wingman']['v'] for entry in batch])
            lead_heading = np.array([entry['info']['lead']['heading'] for entry in batch])
            wingman_heading = np.array([entry['info']['wingman']['heading'] for entry in batch])

            #actions = [entry['actions'] for entry in batch] ### figure out what is actions??

            # note that control is the value at the PREVIOUS step (so ignore the first action)
            wing_actions = np.array([entry["info"]["wingman"]["controller"]["control"] for entry in batch[1:] + [batch[-1]]])
            lead_actions = np.array([entry["info"]["lead"]["controller"]["control"] for entry in batch[1:] + [batch[-1]]])
            batch = []

            for wingman in [True, False]: # both wingman and lead
                states = make_states(lead_x, lead_y, wingman_x, wingman_y, lead_speed, wingman_speed, lead_heading, wingman_heading)
                states_np_list.append(np.array(states).T)

                if wingman:
                    actions_np_list.append(np.array(wing_actions).T)
                else:
                    actions_np_list.append(np.array(lead_actions).T)

            if len(states_np_list) >= max_num_traj:
                break

    diff = time.time() - start
    print(f"Extracted states/actions from {len(states_np_list)} episodes in {diff:.2f} seconds")

    return states_np_list, actions_np_list

def make_rotation_matrix(theta):
    '''make a 2D rotation matrix'''

    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

def data_to_x_xp_rel(state_np, state_prime_np, action_np, get_extended_state_func):
    '''convert a single data point to x and x' for koopman training'''

    assert state_np.shape[1] == 1, f"state_np.shape[1]={state_np.shape[1]}, expected 1"
    assert state_prime_np.shape[1] == 1, f"state_prime_np.shape[1]={state_prime_np.shape[1]}, expected 1"

    # data is wingman x, y, theta, speed
    x, y, theta, speed = state_np[:, 0].copy()
    #print(f"x: {x}, y: {y}, theta: {theta}, speed: {speed}")
    x_prime, y_prime, theta_prime, speed_prime = state_prime_np[:, 0].copy()
    #print(f"x_prime: {x_prime}, y_prime: {y_prime}, theta_prime: {theta_prime}, speed_prime: {speed_prime}")

    pt = np.array([x, y])
    pt_prime = np.array([x_prime, y_prime])

    rel_pt = pt_prime - pt

    if REL_THETA:
        # rotate by -theta
        R_neg_theta = make_rotation_matrix(-theta)
        rel_pt_rotated = R_neg_theta @ rel_pt

        #delta_x = x_prime - x
        #delta_y = y_prime - y
        delta_theta = theta_prime - theta
        #delta_speed = speed_prime - speed

        x = np.array([[0], [0], [0], [speed]], dtype=float)
        x_prime = np.array([[rel_pt_rotated[0]], [rel_pt_rotated[1]], [delta_theta], [speed_prime]], dtype=float)
    else:
        # absolute theta
        x = np.array([[0], [0], [theta], [speed]], dtype=float)

        x_prime = np.array([[rel_pt[0]], [rel_pt[1]], [theta_prime], [speed_prime]], dtype=float)

    x_ext = get_extended_state_func(x)
    x_ext_prime = get_extended_state_func(x_prime)

    return x_ext, x_ext_prime

def data_to_x_xp_abs(state_np, state_prime_np, action_np, get_extended_state_func):
    '''convert a single data point to x and x' for koopman training'''

    x = get_extended_state_func(state_np)
    x_prime = get_extended_state_func(state_prime_np)

    return x, x_prime

def predict_with_koopman_abs(A, B, x_extended, u, get_extended_state_func, get_extended_action_func):
    '''predict the next state using the koopman model
    
    returns the next extended state
    '''

    return A @ x_extended + B @ get_extended_action_func(u)

def predict_with_koopman_rel(A, B, x_extended, u, get_extended_state_func, get_extended_action_func):
    '''predict the next state using the koopman model
    
    returns the next extended state
    '''

    #print(f"state:\n{x_extended[:4]}\nu: {u}")

    rel_x = x_extended[:4, :].copy() # drop the observables since we have to first predict the real state (similar to resetting)
    rel_x[0, 0] = 0 # set the x to 0
    rel_x[1, 0] = 0 # set the y to 0

    if REL_THETA:
        rel_x[2, 0] = 0 # set the theta to 0

    rel_x_extended = get_extended_state_func(rel_x)

    #x = get_extended_state_func(x)
    u = get_extended_action_func(u)

    x_rel_prime = A @ rel_x_extended + B @ u

    #print(f"rel_x_prime: {x_rel_prime[:4, 0]}")

    theta =  x_extended[2, 0]

    if REL_THETA:
        R_theta = make_rotation_matrix(theta)

        pos_offset = R_theta @ x_rel_prime[0:2, 0]

        #print(f"pos_offset: {pos_offset}")

        x_prime = np.zeros(x_rel_prime.shape)

        x_prime[0, 0] = pos_offset[0] + x_extended[0, 0]
        x_prime[1, 0] = pos_offset[1] + x_extended[1, 0]
        x_prime[2, 0] = x_extended[2, 0] + x_rel_prime[2, 0]
        x_prime[3, 0] = x_rel_prime[3, 0]
    else:
        # absolute theta
        pos_offset = x_rel_prime[0:2, 0]

        x_prime = np.zeros(x_rel_prime.shape)
        x_prime[0, 0] = pos_offset[0] + x_extended[0, 0]
        x_prime[1, 0] = pos_offset[1] + x_extended[1, 0]
        x_prime[2, 0] = x_rel_prime[2, 0]
        x_prime[3, 0] = x_rel_prime[3, 0]

    return x_prime


def load_data(eval_file_path, max_num_traj=np.inf, cache_filename="dubins_data.pkl"):
    """load data from file and return states and actions"""


    start = time.time()

    # if cache_filename exists, load it
    if os.path.exists(cache_filename):
        with open(cache_filename, 'rb') as file:
            data = pickle.load(file)
        
        print(f"Loaded data from {cache_filename}")
        states_np_list, actions_np_list = data

    else:
        assert cache_filename == "dubins_data.pkl", f"filename={cache_filename} not found, expected dubins_data.pkl if you want to load from SafeRL rollouts"
        
        
        data = load_json(eval_file_path)
        states_np_list, actions_np_list = extract_states_actions(data, max_num_traj=max_num_traj)

        with open(cache_filename, 'wb') as file:
            pickle.dump((states_np_list, actions_np_list), file)
            print(f"Saved data to {cache_filename}")

    diff = time.time() - start
    print(f"Loaded data in {diff:.2f} seconds")

    return states_np_list, actions_np_list

def main_f16_koopman():
    ''' main entry point'''

    dubins_file_path = "dubins_data.pkl"
    f16_file_path = "f16_trajs.pkl"

    with open(dubins_file_path, 'rb') as file:
        dubins_data = pickle.load(file)
        
    print(f"Loaded data from {dubins_file_path}")
    dubins_states_np_list, dubins_actions_np_list = dubins_data

    num_trajectories = len(dubins_states_np_list)
    print(f"Loaded {num_trajectories} trajectories from {dubins_file_path}, first with {dubins_states_np_list[0].shape[1]} time steps")

    with open(f16_file_path, 'rb') as file:
        f16_data = pickle.load(file)

    print(f"Loaded {len(f16_data)} trajectories from {f16_file_path}, first with {len(f16_data[0])} time steps")
    assert num_trajectories == len(f16_data), f"num_trajectories={num_trajectories}, expected {len(f16_data)}"

    # fix data format from f16 to match dubins

    for traj_index in range(num_trajectories):#[14]:
        dubins_traj_one = dubins_states_np_list[traj_index]
        
        f16_data[traj_index] = np.array(f16_data[traj_index][:dubins_traj_one.shape[1]]) # truncate to same length as dubins
        # f16_first_state[2] = -f16_first_state[2] + np.pi/2
        f16_data[traj_index][:, 2] = -f16_data[traj_index][:, 2] + np.pi/2 # adjustment between phi and heading
        f16_traj_one = f16_data[traj_index]

        # translate to origin and rotate to zero heading
        init_x = f16_traj_one[0][0]
        init_y = f16_traj_one[0][1]
        init_heading = f16_traj_one[0][2]
        #rotate_matrix = np.array([[np.cos(init_heading), -np.sin(init_heading)], [np.sin(init_heading), np.cos(init_heading)]])
        rotate_matrix = make_rotation_matrix(-init_heading)
        #rotate_matrix = make_rotation_matrix(0)

        #dx = f16_traj_one[1][0] - f16_traj_one[0][0]
        #dy = f16_traj_one[1][1] - f16_traj_one[0][1]
        #print(f"dx: {dx}, dy: {dy}, atan2: {np.arctan2(dy, dx)}, init_heading: {init_heading}")

        #print(f"init_heading: {init_heading}")

        for step in range(f16_traj_one.shape[0]):
            f16_traj_one[step][0] -= init_x
            f16_traj_one[step][1] -= init_y

            scale_pos = 5e4
            scale_speed = 1e2
            x = f16_traj_one[step][0] / scale_pos
            y = f16_traj_one[step][1] / scale_pos
            pt = np.array([[x], [y]], dtype=float)

            rot_pt = rotate_matrix @ pt

            f16_traj_one[step][0] = rot_pt[0, 0]
            f16_traj_one[step][1] = rot_pt[1, 0]

            new_x = f16_traj_one[step][0]
            new_y = f16_traj_one[step][1]

            #print(f"{step}. {x}, {y} -> {new_x}, {new_y}")

            f16_traj_one[step][2] -= init_heading
            f16_traj_one[step][3] /= scale_speed
            

        # plot the first trajectory x and y
        #plt.plot(f16_traj_one[:, 0], f16_traj_one[:, 1], 'b-', label='F16')
        # make it square
        #plt.axis('equal')
        #plt.show()
        #exit(1)

        f16_data[traj_index] = f16_data[traj_index].T

    # shuffle f16_data and dubins_acions_np_list in the same way using seed 5000
    seed = 500
    np.random.seed(seed)

    ####################### make koopman model here from f16 data and dubins_actions_np_list ######################

    split_data = koopman_util.split_test_validation_train(f16_data, dubins_actions_np_list, num_test=3, num_validation=3)
    test_states, test_actions, validation_states, validation_actions, training_states, training_actions = split_data

    # Set NumPy to raise an error on overflow
    np.seterr(over='raise', invalid='raise')
    np.set_printoptions(suppress=True) # no scientific notation

    for plot_index in [4]: #[1, 2, 3, 4]:
        ko_list = []

        if plot_index == 1:
            plot_name = "plots/f16_koop1_dmd_vs_rff.png"
            ko_list.append(koopman_util.KoopmanIdentity())
            #ko_list.append(koopman_util.KoopmanRFF(gamma=1e-4, num_features=500))
            ko_list.append(koopman_util.KoopmanExperimental(gamma=1e-2, num_features=500))
            ko_list.append(koopman_util.KoopmanExperimental(gamma=1e0, num_features=500))
            ko_list.append(koopman_util.KoopmanExperimental(gamma=1e2, num_features=500))
            #ko_list.append(koopman_util.KoopmanExperimental(gamma=1e4, num_features=500))
        
        if plot_index == 2:
            plot_name = "plots/f16_koop2_dmd_rel.png"
            ko_list.append(koopman_util.KoopmanIdentity())
            ko_list.append(koopman_util.KoopmanIdentityRelative())
            ko_list.append(koopman_util.KoopmanIdentityRelative(rotate=True))
            ko_list.append(koopman_util.KoopmanRFFRelative(gamma=1e-2, num_features=500, rotate=True))

        if plot_index == 3:
            plot_name = "plots/f16_koop3_rff_rel_gamma.png"
            ko_list.append(koopman_util.KoopmanRFFRelative(gamma=1e0, num_features=500))
            ko_list.append(koopman_util.KoopmanRFFRelative(gamma=1e-1, num_features=500))
            ko_list.append(koopman_util.KoopmanRFFRelative(gamma=1e-2, num_features=500))
            ko_list.append(koopman_util.KoopmanRFFRelative(gamma=1e-3, num_features=500))
            ko_list.append(koopman_util.KoopmanRFFRelative(gamma=1e-4, num_features=500))

        if plot_index == 3.5:
            plot_name = "plots/f16_koop3.5_rff_rel_gamma_rotate.png"
            
            ko_list.append(koopman_util.KoopmanRFFRelative(gamma=1e-2, num_features=500))
            ko_list.append(koopman_util.KoopmanRFFRelative(gamma=1e-2, num_features=500, rotate=True))

            ko_list.append(koopman_util.KoopmanRFFRelative(gamma=1e-3, num_features=500))
            ko_list.append(koopman_util.KoopmanRFFRelative(gamma=1e-3, num_features=500, rotate=True))
            #ko_list.append(koopman_util.KoopmanRFFRelative(gamma=1e-4, num_features=500))
            #ko_list.append(koopman_util.KoopmanRFFRelative(gamma=1e-4, num_features=500, rotate=True))

            ko_list.append(koopman_util.KoopmanRFFRelative(gamma=1e-4, num_features=500))
            ko_list.append(koopman_util.KoopmanRFFRelative(gamma=1e-4, num_features=500, rotate=True))
            
        
        if plot_index == 4:
            plot_name = "plots/f16_koop4_rff_rel_features.png"
            ko_list.append(koopman_util.KoopmanIdentity())
            #ko_list.append(koopman_util.KoopmanRFFRelative(gamma=1e-3, num_features=100, rotate=rotate))
            #ko_list.append(koopman_util.KoopmanRFFRelative(gamma=1e-3, num_features=75, rotate=rotate))
            ko_list.append(koopman_util.KoopmanRFF(gamma=1e3, num_features=25))
            ko_list.append(koopman_util.KoopmanRFF(gamma=1e2, num_features=50))
            ko_list.append(koopman_util.KoopmanRFF(gamma=1e1, num_features=100))
            ko_list.append(koopman_util.KoopmanRFF(gamma=1e0, num_features=200))
        

        print("training...")
        for kobj in ko_list:
            kobj.train(training_states, training_actions)

        print("plotting...")
        koopman_util.analyze_predictions(test_states, training_states, test_actions, ko_list, plot_name=plot_name)

    print("Exit before plots 3.1415")
    exit(1)

    matplotlib.use('TkAgg') # set backend
    plt.style.use(['bmh', 'bak_matplotlib.mlpstyle'])

    for traj_index in [0]: #range(num_trajectories):#[14]:
        traj_one = dubins_states_np_list[traj_index]
              
        f16_first_state = f16_data[traj_index][:, 0]
        dubins_first_state = traj_one[:, 0]

        #actions_one = actions_np_list[traj_index]
        dubins_xs = traj_one[0, :] # x0, y0, heading0, v0
        dubins_ys = traj_one[1, :]
    
        f16_xs = f16_data[traj_index][0, :] # x0, y0, heading0, v0
        f16_ys = f16_data[traj_index][1, :]

        ######        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)

        ax.plot(dubins_xs, dubins_ys, 'r-', label='Dubins')
        ax.plot(f16_xs, f16_ys, 'b-', label='F16')

        # set x and y label
        ax.set_xlabel('X Position (ft)')
        ax.set_ylabel('Y Position (ft)')

        # add label 'start' to the first point
        ax.text(dubins_xs[0], dubins_ys[0], ' Start', fontsize=14, color='black')

        ax.set_title(f'Trajectory #{traj_index}')
        ax.legend()
        filename = f"plots/traj_{traj_index}.png"
        plt.savefig(filename)
        plt.close()

        print(f"Made {filename}")

def main_dubins():
    '''main entry point using dubins data'''

    #file_path="../output/expr_20240522_143535/PPO_DubinsRejoin_15bc3_00000_0_2024-05-22_14-35-38/eval/ckpt_200/eval.log"
    eval_file_path="../output/expr_20240918_143039/PPO_DubinsRejoin_1c75e_00000_0_2024-09-18_14-30-42/eval/ckpt_200/eval.log"

    # Set NumPy to raise an error on overflow
    np.seterr(over='raise', invalid='raise')
    np.set_printoptions(suppress=True) # no scientific notation

    num_traj = np.inf #100

    print("loading data from {eval_file_path}...")
    states_np_list, actions_np_list = load_data(eval_file_path, max_num_traj=num_traj)

    split_data = koopman_util.split_test_validation_train(states_np_list, actions_np_list, num_test=3, num_validation=3)
    test_states, test_actions, validation_states, validation_actions, training_states, training_actions = split_data

    ko_list = []

    plot_index = 4

    if plot_index == 1:
        plot_name = "koop1_dmd_vs_rff.png"
        #ko_list.append(koopman_util.KoopmanIdentity())
        #ko_list.append(koopman_util.KoopmanRFF(gamma=1e-4, num_features=500))
        ko_list.append(koopman_util.KoopmanExperimental(gamma=1e-2, num_features=500))
        ko_list.append(koopman_util.KoopmanExperimental(gamma=1e0, num_features=500))
        ko_list.append(koopman_util.KoopmanExperimental(gamma=1e2, num_features=500))
        ko_list.append(koopman_util.KoopmanExperimental(gamma=1e4, num_features=500))
    
    if plot_index == 2:
        plot_name = "koop2_dmd_rel.png"
        ko_list.append(koopman_util.KoopmanIdentity())
        ko_list.append(koopman_util.KoopmanIdentityRelative())
        ko_list.append(koopman_util.KoopmanIdentityRelative(rotate=True))


    if plot_index == 3:
        plot_name = "koop3_rff_rel_gamma.png"
        ko_list.append(koopman_util.KoopmanRFFRelative(gamma=1e0, num_features=500))
        ko_list.append(koopman_util.KoopmanRFFRelative(gamma=1e-1, num_features=500))
        ko_list.append(koopman_util.KoopmanRFFRelative(gamma=1e-2, num_features=500))
        ko_list.append(koopman_util.KoopmanRFFRelative(gamma=1e-3, num_features=500))
        ko_list.append(koopman_util.KoopmanRFFRelative(gamma=1e-4, num_features=500))
    
    if plot_index == 4:
        plot_name = "koop4_rff_rel_features.png"
        ko_list.append(koopman_util.KoopmanRFFRelative(gamma=1e-2, num_features=500))
        ko_list.append(koopman_util.KoopmanRFFRelative(gamma=1e-2, num_features=200))
        ko_list.append(koopman_util.KoopmanRFFRelative(gamma=1e-2, num_features=100))
        ko_list.append(koopman_util.KoopmanIdentityRelative())
    
    #ko_list.append(koopman_util.KoopmanIdentityNormalized())

    gamma = 1e-3
    num_features = 500

    #ko_list.append(koopman_util.KoopmanRFFRelative(gamma=gamma, num_features=num_features))
    #ko_list.append(koopman_util.KoopmanRFFNormalized(gamma=gamma, num_features=num_features))

    print("training...")
    for kobj in ko_list:
        kobj.train(training_states, training_actions)

    print("plotting...")
    koopman_util.analyze_predictions(test_states, training_states, test_actions, ko_list, plot_name=plot_name)

def main2():
    '''main entry point (original)'''

    # Set NumPy to raise an error on overflow
    np.seterr(over='raise', invalid='raise')
    np.set_printoptions(suppress=True) # no scientific notation

    num_traj = 100
    num_test = 3 # num validation equals num test

    seed = 1985

    states_np_list, actions_np_list = load_data(max_num_traj=num_traj)
    
    

    best_hyperparams = {"percent_error": np.inf, "gamma": None, "num_features": None}

    if DO_HYPERPARM_TUNING:
        num_features_list = [20, 50, 100, 200, 500]
        gamma_list = [1.0, 1e-2, 1e-3, 1e-4]
    else:
        num_features_list = [200]
        gamma_list = [1e-4]

    for num_features in num_features_list:
        for gamma in gamma_list:
            print(f"Training with gamma={gamma}, num_features={num_features}")
            
            percent_error = try_koopman_model(training_states, training_actions, validation_states, validation_actions, seed, gamma, num_features, plot=False)
            
            print(f"Average percent error at last time step: {percent_error:.2f}%")

            if best_hyperparams["percent_error"] == np.inf or percent_error < best_hyperparams["percent_error"]:
                best_hyperparams["percent_error"] = percent_error
                best_hyperparams["gamma"] = gamma
                best_hyperparams["num_features"] = num_features

    print(f"\nPlotting best hyperparameters: {best_hyperparams}")

    try_koopman_model(training_states, training_actions, test_states, test_actions, seed, best_hyperparams["gamma"], best_hyperparams["num_features"], plot=True)


def main():
    #main_dubins()
    main_f16_koopman()

if __name__ == '__main__':
    main()
