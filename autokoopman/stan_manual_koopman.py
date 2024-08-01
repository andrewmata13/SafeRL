import numpy as np
from matplotlib import pyplot as plt
import json
import time

from cachier import cachier

def get_extended_state_rff(X, seed=1985):
    '''get X with rff observables'''

    assert len(X.shape) == 2, f"X.shape={X.shape}, expected 2D array"

    observations = []

    np.random.seed(seed)
    GAMMA = 1e-4
    NUM_FEATURES = 200 # was 200, 500 = low error, look at 330
    dimension = X.shape[0]

    w = np.sqrt(2 * GAMMA) * np.random.normal(size=(NUM_FEATURES, dimension))
    
    # Generate D iid samples from Uniform(0,2*pi)
    u = 2 * np.pi * np.random.rand(1, NUM_FEATURES)

    s = np.sqrt(2 / NUM_FEATURES)

    # each observatio is gamma * cos(col <dot> rand_vec + rand_phase)    

    #print(f"X.shape={X.shape}")
    #print(f"w.shape={w.shape}")
    #print(f"u.shape={u.shape}")

    Z = s * np.cos(w @ X + u.T) # was X.T @ w

    #print(f"Z.shape={Z.shape}")

    #one = np.ones((1, Z.shape[1]))
    #rv = np.vstack((X, one, Z))

    
    rv = np.vstack((X, Z))

    return rv

def make_states(lead_x_list, lead_y_list, wingman_x_list, wingman_y_list, lead_speed_list, wingman_speed_list, lead_heading_list, wingman_heading_list):
    '''make states from raw data'''

    lead_vx_normalized = np.cos(lead_heading_list)
    lead_vy_normalized = np.sin(lead_heading_list)

    lead_vx = lead_speed_list * np.cos(lead_heading_list)
    lead_vy = lead_speed_list * np.sin(lead_heading_list)

    states = list(zip(lead_x_list, lead_y_list, lead_heading_list, lead_speed_list))
    #states = list(zip(lead_x_list, lead_y_list, lead_heading_list, lead_speed_list, lead_vx_normalized, lead_vy_normalized))

    #sigma = [1000, 1, 1, 1000, 1, 1, 400, 1, 1, 400, 1, 1] # normalization

    #states = []

    #for i in range(len(lead_x_list)):
    #    lead_x = lead_x_list[i]
    #    lead_y = lead_y_list[i]
    #    wingman_x = wingman_x_list[i]
    #    wingman_y = wingman_y_list[i]
    #    lead_speed = lead_speed_list[i]
    #    wingman_speed = wingman_speed_list[i]
    #    lead_heading = lead_heading_list[i]
    #    wingman_heading = wingman_heading_list[i]

     #lead_vx = lead_speed * np.cos(lead_heading)
    #lead_vy = lead_speed * np.sin(lead_heading)
    
    #states = list(zip(lead_x, lead_y, lead_heading, lead_speed))
    #states = list(zip(lead_x, lead_y, lead_vx, lead_vy, wingman_x, wingman_y))
    #states = list(zip(lead_x, lead_y, wingman_x, wingman_y))
    #states = list(zip(lead_x, lead_y, wingman_x, wingman_y, lead_speed, wingman_speed, lead_heading, wingman_heading))
    #states = list(zip(lead_x, lead_y, lead_speed, lead_heading))
    #states = list(zip(wingman_x, wingman_y, wingman_speed, wingman_heading))

    #states = list(zip(lead_x - wingman_x, lead_y - wingman_y, lead_speed - wingman_speed, lead_heading - wingman_heading))
    
    #output.append([actions, times, states])

    #if traj_index == 6:
    #    break

    return states

def split_test_train(states_np_list, actions_np_list, num_test=3):
    '''split into test and train data'''

    assert len(states_np_list) == len(actions_np_list), f"states_np_list={len(states_np_list)} != actions_np_list={len(actions_np_list)}"
    assert num_test < len(states_np_list), f"num_test={num_test} >= len(states_np_list)={len(states_np_list)}"

    test_states = states_np_list[:num_test]
    test_actions = actions_np_list[:num_test]
    training_states = states_np_list[num_test:]
    training_actions = actions_np_list[num_test:]

    return test_states, test_actions, training_states, training_actions

@cachier(cache_dir='cachier')
def load_json(file_path="../output/expr_20240522_143535/PPO_DubinsRejoin_15bc3_00000_0_2024-05-22_14-35-38/eval/ckpt_200/eval.log"):

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

            actions = [entry['actions'] for entry in batch]
            batch = []

            states = make_states(lead_x, lead_y, wingman_x, wingman_y, lead_speed, wingman_speed, lead_heading, wingman_heading)

            states_np_list.append(np.array(states).T)
            actions_np_list.append(np.array(actions).T)

            if len(states_np_list) >= max_num_traj:
                break

    diff = time.time() - start
    print(f"Extracted states/actions from {len(states_np_list)} episodes in {diff:.2f} seconds")

    return states_np_list, actions_np_list

def train_koopman_model(states_np_list, actions_np_list, get_extended_state_func):
    '''train a koopman model and return it

    returns A, B
    '''

    start = time.time()
    # add rff observables to states
    ext_states_np_list = [get_extended_state_func(states_np) for states_np in states_np_list]
    diff = time.time() - start
    print(f"Constructed extended states in {diff:.2f} seconds")

    X = np.hstack([mat[:,:-1] for mat in ext_states_np_list])
    X_prime = np.hstack([mat[:,1:] for mat in ext_states_np_list])
    Gamma = np.hstack([mat[:,:-1] for mat in actions_np_list])

    #print(f"X.shape={X.shape}")
    #print(f"X_prime.shape={X_prime.shape}")
    #print(f"Gamma.shape={Gamma.shape}")

    Omega = np.vstack((X, Gamma))
    #print(f"Omega.shape={Omega.shape}")

    start2 = time.time()
    pseudoinv = np.linalg.pinv(Omega)
    diff = time.time() - start2
    print(f"Computed pseudoinv in {diff:.2f} seconds")

    A_B = X_prime @ pseudoinv
    #print(f"A_B.shape={A_B.shape}")

    A = A_B[:, :X.shape[0]]
    B = A_B[:, X.shape[0]:]
    #print(f"A.shape={A.shape}")
    #print(f"B.shape={B.shape}")

    diff = time.time() - start
    print(f"Total Koopman training: {diff:.2f} seconds")

    return A, B

def plot_predictions(A, B, states_np_list, actions_np_list, get_extended_state_func):
    num_plots = len(states_np_list)
    fig, axs = plt.subplots(num_plots, 4, figsize=(16, 9))

    for index in range(num_plots):

        states_np = states_np_list[index]
        actions_np = actions_np_list[index]

        # replay using koopman
        first_state = states_np[:, 0:1]

        #print(f"first_state.shape={first_state.shape}")
        first_state_observed = get_extended_state_func(first_state)

        #print(f"first_state_observed.shape={first_state_observed.shape}")
        num_steps = states_np.shape[1]
        num_vars = states_np.shape[0]
        test_traj = first_state_observed
        percent_errors = []
        observed_state_norms = []

        # normal koopman prediction
        for step in range(num_steps - 1):
            x = test_traj[:, -1:]
            observed_state_norms.append(np.linalg.norm(x))

            # first update error
            real_vars_predicted = x[0:num_vars, :]
            #print(f"real_vars_predicted.shape={real_vars_predicted.shape}")
            actual_vars = states_np[:, step:step+1]
            #print(f"actual_vars.shape={actual_vars.shape}")
        
            error = np.linalg.norm(real_vars_predicted - actual_vars)
            #print(f"step={step} error={error}")
            rel_error = error / np.linalg.norm(actual_vars)
            percent_errors.append(100 * rel_error)

            u = actions_np[:, step:step+1]

            x_prime = A @ x + B @ u

            #test_traj.append(x_prime)
            test_traj = np.hstack((test_traj, x_prime))

        # resetting koopman prediction
        
        test_traj = test_traj.T

        ax = axs[index, 0]
        xs = [x[0] for x in test_traj]
        ys = [x[1] for x in test_traj]
        ax.plot(xs, ys, '-o', ms=1.5, lw=0.5, label='Trajectory Prediction', color='r')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
    
        ax.plot(states_np[0, :], states_np[1, :], '-', color='g', lw=1, label='Ground Truth')
        ax.grid()

        ax = axs[index, 1]
        

        for other_i, other_states_np in enumerate(states_np_list):
            color = 'g' if other_states_np is states_np else 'k'
            zorder = 2 if other_states_np is states_np else 1
            lw = 2 if other_states_np is states_np else 0.5

            if other_i > 2 or color == 'g':
                xs = other_states_np[0, :]
                ys = other_states_np[1, :]
                ax.plot(xs, ys, '-', color=color, lw=lw, zorder=zorder)

        #xs = [x[0] for x in test_traj]
        #ys = [x[0] for x in test_traj]
        #ax.plot(xs, ys, '-o', ms=1.5, lw=0.5, label='Trajectory Prediction', color='r')
        #ax.set_xlabel("Heading")
        #ax.set_ylabel("Speed")
        #ax.plot(states_np[2, :], states_np[3, :], '-', color='g', lw=1, label='Ground Truth')
        #ax.grid()

        ax = axs[index, 2]
        ax.plot(percent_errors, label='Orig Koopman', color='r')
        
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Percent Error (%)")

        ax = axs[index, 3]
        ax.plot(observed_state_norms, label='Orig Koopman', color='r')
        ax.set_xlabel("Time Step")
        ax.set_ylabel("2-Norm of Observed State")

        ############ now do resetting koopman prediction
        reset_test_traj = first_state_observed[0:num_vars, :]
        reset_percent_errors = []
        reset_observed_state_norms = []

        for step in range(num_steps - 1):
            x = reset_test_traj[:, -1:]
            x_obs = get_extended_state_func(x)

            reset_observed_state_norms.append(np.linalg.norm(x_obs))

            # first update error
            real_vars_predicted = x[0:num_vars, :]
            #print(f"real_vars_predicted.shape={real_vars_predicted.shape}")
            actual_vars = states_np[:, step:step+1]
            #print(f"actual_vars.shape={actual_vars.shape}")
        
            error = np.linalg.norm(real_vars_predicted - actual_vars)
            #print(f"step={step} error={error}")
            reset_rel_error = error / np.linalg.norm(actual_vars)
            reset_percent_errors.append(100 * reset_rel_error)

            u = actions_np[:, step:step+1]

            x_prime = A @ x_obs + B @ u

            # pop off the non-real vars
            x_prime = x_prime[0:num_vars, :]

            #test_traj.append(x_prime)
            reset_test_traj = np.hstack((reset_test_traj, x_prime))

        reset_test_traj = reset_test_traj.T

        ax = axs[index, 0]
        xs = [x[0] for x in reset_test_traj]
        ys = [x[1] for x in reset_test_traj]
        ax.plot(xs, ys, '-o', ms=1.5, lw=0.5, label='Prediction with Resets', color='b')
        ax.legend()

        ax = axs[index, 2]
        ax.plot(reset_percent_errors, label='With Resets', color='b')
        ax.legend()

        ax = axs[index, 3]
        ax.plot(reset_observed_state_norms, label='With Resets', color='b')
        ax.legend()

    plt.tight_layout()
    plt.show()

def get_centers_ranges(states_np_list):
    '''get the centers and ranges of the data
    
    returns centers, ranges'''

    min_states = np.min(np.array([np.min(states_np, axis=1) for states_np in states_np_list]), axis=0)
    max_states = np.max(np.array([np.max(states_np, axis=1) for states_np in states_np_list]), axis=0)

    centers = (min_states + max_states) / 2
    ranges = max_states - min_states

    return centers, ranges

def load_data(max_num_traj=np.inf):
    """load data from file and return states and actions"""
    start = time.time()

    data = load_json()
    states_np_list, actions_np_list = extract_states_actions(data, max_num_traj=max_num_traj)

    diff = time.time() - start
    print(f"Loaded data in {diff:.2f} seconds")

    return states_np_list, actions_np_list

def normalize_matrix(mat, centers, ranges):
    '''normalize a single 2-d matrix, with columns being the snapshots'''

    # multiply by 2 to make it between -1 and 1
    norm_data_mat = 2 * (mat - centers[:, np.newaxis]) / ranges[:, np.newaxis]

    assert norm_data_mat.shape == mat.shape, f"norm_data_mat.shape={norm_data_mat.shape} != data_mat.shape={mat.shape}"
    return norm_data_mat

def denormalize_matrix(norm_data_mat, centers, ranges):
    '''denormalize a single 2-d matrix, with columns being the snapshots'''

    # multiply by 0.5 since range was 2.0 (between -1 and 1)
    data_mat = 0.5 * norm_data_mat * ranges[:, np.newaxis] + centers[:, np.newaxis]

    assert data_mat.shape == norm_data_mat.shape, f"data_mat.shape={data_mat.shape} != norm_data_mat.shape={norm_data_mat.shape}"
    return data_mat

def normalize_single_list(data, centers, ranges):
    '''normalize a single data set
    
    data is a list of 2-d np.arrays, normalization is done on the rows
    '''

    normalized_data = []
    for data_mat in data:        
        norm_data_mat = normalize_matrix(data_mat, centers, ranges)

        normalized_data.append(norm_data_mat)

    #centers, ranges = get_centers_ranges(normalized_data)
    #print(f"after normalization centers={centers} ranges={ranges}")

    return normalized_data

def normalize_data_lists(test_states, test_actions, training_states, training_actions):
    '''normalize the data based on ranges in training data

    returns norm_tup, test_states, test_actions, training_states, training_actions
    '''

    start = time.time()

    centers_states, ranges_states = get_centers_ranges(training_states)
    centers_actions, ranges_actions  = get_centers_ranges(training_actions)

    test_states = normalize_single_list(test_states, centers_states, ranges_states)

    test_actions = normalize_single_list(test_actions, centers_actions, ranges_actions)
    training_states = normalize_single_list(training_states, centers_states, ranges_states)
    training_actions = normalize_single_list(training_actions, centers_actions, ranges_actions)

    norm_tup = (centers_states, ranges_states, centers_actions, ranges_actions)

    diff = time.time() - start
    print(f"Normalized data lists in {diff:.2f} seconds")

    return norm_tup, test_states, test_actions, training_states, training_actions

def main():
    '''main entry point'''

    num_traj = 10
    obs_func = get_extended_state_rff

    start = time.time()

    states_np_list, actions_np_list = load_data(max_num_traj=num_traj)
    test_states, test_actions, training_states, training_actions = split_test_train(states_np_list, actions_np_list, num_test=3)

    norm_tup, test_states, test_actions, training_states, training_actions = normalize_data_lists(test_states, test_actions, training_states, training_actions)

    A, B = train_koopman_model(training_states, training_actions, obs_func)
    diff = time.time() - start
    print(f"Total time: {diff:.2f} seconds")

    plot_predictions(A, B, test_states, test_actions, obs_func)

if __name__ == '__main__':
    main()
