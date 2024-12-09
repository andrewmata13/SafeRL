import pickle
import os
from cachier import cachier
import numpy as np
from numpy import deg2rad
import matplotlib.pyplot as plt

from aerobench.util import get_script_path, StateIndex, get_state_names
from aerobench.run_f16_sim import run_f16_sim, F16SimState
from aerobench.visualize import plot
from wingman_autopilot import WingmanAutopilot

from scipy.optimize import linprog

class DubinsPredictor:
    '''predictor using dubins car model'''
    def __init__(self):
        pass

    def predict(self, res_dict, start_step, num_steps_to_predict):# state0, actions_rollout):
        '''return the predicted state after applying the passed-in actions'''

        f16_np_states, actions_rollout_full = extract_np_states_actions(res_dict)
        state0 = f16_np_states[start_step].copy()
        actions_rollout = actions_rollout_full[:, start_step:start_step+(num_steps_to_predict)]

        NUM_STEPS = actions_rollout.shape[1]
        #traj_linear = np.zeros((4, NUM_STEPS))
        
        # create initial cur_state
        cur_state = list(state0) # x0, y0, theta0, vel0

        for i, action in enumerate(actions_rollout.T):
            # extract current state
            x, y, theta, vel = cur_state

            # apply action to get next state
            next_state = cur_state.copy()
            next_state[0] = x + vel * np.cos(theta)
            next_state[1] = y + vel * np.sin(theta)

            # update targets
            next_state[2] = theta + action[0] # update theta
            next_state[3] = vel + action[1]

            cur_state = next_state

        return cur_state
    
@cachier(cache_dir='./cachier')
def linf_best_fit(input_array, output_array):
    """
    Solve for A in A x = y minimizing the L_inf norm of residuals.
    For multiple output columns, solve each column independently.

    Parameters:
        input_array: (M, N) array of input features
        output_array: (M, K) array of outputs

    Returns:
        A_lstsq: (K, N) array of solutions (each row corresponds to one output dimension)
        linf_lstsq_residuals: (K,) array of L_inf residuals per output dimension
        linf_minimize_residuals: (K,) array of sum of absolute residuals per output dimension
    """

    input_array = np.asarray(input_array)
    output_array = np.asarray(output_array)
    M, N = input_array.shape
    if output_array.ndim == 1:
        output_array = output_array.reshape(-1, 1)
    M2, K = output_array.shape

    if M != M2:
        raise ValueError("Number of rows in input_array and output_array must match.")

    # We will solve column by column
    A_list = []
    linf_per_output = []
    sum_abs_per_output = []

    # Bounds: x are unbounded, t >= 0
    bounds = [(None, None)] * N + [(0, None)]
    # Objective: minimize t
    c = np.zeros(N + 1)
    c[-1] = 1

    for k_idx in range(K):
        b = output_array[:, k_idx]

        # Build constraints for L_inf norm:
        # |A_i x - b_i| <= t
        # => A_i x - b_i <= t AND -(A_i x - b_i) <= t
        # => A_i x - t <= b_i AND -A_i x - t <= -b_i
        # Coeff: [A_i, -1], RHS: b_i
        # and   [-A_i, -1], RHS: -b_i

        A_ub_list = []
        b_ub_list = []
        for i in range(M):
            A_ub_list.append(np.concatenate([input_array[i], [-1.0]]))
            b_ub_list.append(b[i])

            A_ub_list.append(np.concatenate([-input_array[i], [-1.0]]))
            b_ub_list.append(-b[i])

        A_ub = np.array(A_ub_list)
        b_ub = np.array(b_ub_list)

        # Solve LP
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if not result.success:
            raise ValueError(f"L_inf optimization failed for output dimension {k_idx}: {result.message}")

        x_solution = result.x[:-1]
        A_list.append(x_solution)

        # Compute residuals for this output dimension
        predicted = input_array @ x_solution
        residuals = b - predicted
        # L_inf residual for this dimension
        linf_res = np.max(np.abs(residuals))
        linf_per_output.append(linf_res)

    A_lstsq = np.vstack(A_list)
    linf_lstsq_residuals = np.array(linf_per_output)

    return A_lstsq, linf_lstsq_residuals
    
class RegressionPredictor:
    '''predictor from regression'''

    def __init__(self, input_obs_func, output_obs_func):
        self.input_obs_func = input_obs_func
        self.output_obs_func = output_obs_func

        self.A_dict = {}

    def fit(self, all_f16_res_dicts, num_steps):
        '''train a linear model and assign it to A_dict'''

        assert num_steps not in self.A_dict, f"{num_steps=} already set"
        input_list = [] # each is a 1d array
        output_list = [] # each is a 1d array

        for _, res_dict in enumerate(all_f16_res_dicts):
            f16_np_states, actions_rollout = extract_np_states_actions(res_dict)

            for start_step in range(len(f16_np_states) - (num_steps + 1)):
                #state0 = f16_np_states[start_step].copy()

                #actions_rollout_trimmed = actions_rollout[:, start_step:start_step+(num_steps)]

                input_obs = self.input_obs_func(res_dict, start_step, num_steps)
                output_obs = self.output_obs_func(res_dict, start_step+num_steps, num_steps)

                input_list.append(input_obs)
                output_list.append(output_obs)

                #print(f"{input_obs=}, {output_obs=}")
        
        input_array = np.array(input_list)
        output_array = np.array(output_list)

        A, residuals = linf_best_fit(input_array, output_array)
        print(f"{num_steps=}, {A.shape=}, {residuals=}")
 
        self.A_dict[num_steps] = A

    def predict(self, res_dict, start_step, num_steps_to_predict):
        '''predict using the trained model'''

        assert num_steps_to_predict in self.A_dict, f"A matrix not set for {num_steps_to_predict=}. call fit() first"
        A = self.A_dict[num_steps_to_predict]

        #print(f'predict() with {num_steps_to_predict=} {A.shape=}')

        input_obs = self.input_obs_func(res_dict, start_step, num_steps_to_predict)
        predicted_output = input_obs @ A.T
        
        return predicted_output


    
def make_linear_predictor(all_f16_res_dicts, MAX_STEPS):
    '''make a RegressionPredictor from the passed-in data'''

    def input_obs_func(res_dict, start_step, num_steps):
        '''return the input observation vector'''

        list_of_arrays = []

        f16_state_13d = res_dict['states'][start_step]

        # dubins state
        x = f16_state_13d[StateIndex.POS_E]
        y = f16_state_13d[StateIndex.POS_N]
        heading = np.pi / 2 - f16_state_13d[StateIndex.PSI]
        vel = f16_state_13d[StateIndex.VT]

        dubins_state = np.array([x, y, heading, vel, vel*np.sin(heading), vel*np.cos(heading)])
        list_of_arrays.append(dubins_state)

        # add f16 state
        list_of_arrays.append(f16_state_13d)
        dubins_rollout_cur_state = dubins_state.copy() # also do a dubins rollout as part of the observation
        ideal_dubins_rollout_cur_state = dubins_state.copy()

        # use res_dict['vel_targets'] and res_dict['psi_targets'] to get the ideal dubins rollout
        ideal_dubins_rollout_cur_state[3] = res_dict['vel_targets'][start_step]
        ideal_dubins_rollout_cur_state[2] = np.pi / 2 - res_dict['psi_targets'][start_step]

        #list_of_arrays.append([res_dict['vel_targets'][start_step], np.pi / 2 - res_dict['psi_targets'][start_step]])
            
        for step in range(start_step, start_step + num_steps):
            action = res_dict['actions'][:, step]
            list_of_arrays.append(action)

            # apply action to get next state
            next_state = dubins_rollout_cur_state.copy()
            next_state[0] = dubins_rollout_cur_state[0] + dubins_rollout_cur_state[3] * np.cos(dubins_rollout_cur_state[2])
            next_state[1] = dubins_rollout_cur_state[1] + dubins_rollout_cur_state[3] * np.sin(dubins_rollout_cur_state[2])
            next_state[2] = dubins_rollout_cur_state[2] + action[0] # update theta
            next_state[3] = dubins_rollout_cur_state[3] + action[1] # update vel
            next_state[4] = next_state[3] * np.cos(next_state[2])
            next_state[5] = next_state[3] * np.sin(next_state[2])

            next_ideal_state = ideal_dubins_rollout_cur_state.copy()
            next_ideal_state[0] = ideal_dubins_rollout_cur_state[0] + ideal_dubins_rollout_cur_state[3] * np.cos(ideal_dubins_rollout_cur_state[2])
            next_ideal_state[1] = ideal_dubins_rollout_cur_state[1] + ideal_dubins_rollout_cur_state[3] * np.sin(ideal_dubins_rollout_cur_state[2])
            next_ideal_state[2] = ideal_dubins_rollout_cur_state[2] + action[0] # update theta
            next_ideal_state[3] = ideal_dubins_rollout_cur_state[3] + action[1] # update vel
            next_ideal_state[4] = next_ideal_state[3] * np.cos(next_ideal_state[2])
            next_ideal_state[5] = next_ideal_state[3] * np.sin(next_ideal_state[2])
            
            dubins_rollout_cur_state = next_state
            ideal_dubins_rollout_cur_state = next_ideal_state

            LAST_N_STEPS = np.inf
            if step >= start_step + num_steps - LAST_N_STEPS:
                # add the last few state
                list_of_arrays.append(dubins_rollout_cur_state)
                list_of_arrays.append(ideal_dubins_rollout_cur_state)

        # also add vel and theta targets
        #list_of_arrays.append([]])

        list_of_arrays.append([1]) # identity term for constant offsets

        return np.concatenate(list_of_arrays)

    def output_obs_func(res_dict, step, num_steps):
        '''return the output observation vector'''

        f16_state_13d = res_dict['states'][step]
        x = f16_state_13d[StateIndex.POS_E]
        y = f16_state_13d[StateIndex.POS_N]
        heading = np.pi / 2 - f16_state_13d[StateIndex.PSI]
        vel = f16_state_13d[StateIndex.VT]
        
        return np.array([x, y, heading, vel])
    
    predictor = RegressionPredictor(input_obs_func, output_obs_func)

    for num_steps in range(1, MAX_STEPS+1):
        print(num_steps)
        predictor.fit(all_f16_res_dicts, num_steps)

    return predictor

class WingmanF16State:
    '''object containing simulation state

    With this interface you can run partial simulations, rather than having to simulate for the entire time bound

    if you just want a single run with a fixed time, it may be easier to use the run_f16_sim function
    '''

    def __init__(self, initial_state):

        NUM_STATES = len(get_state_names())

        assert len(initial_state) in [4, NUM_STATES+3], f"Expected 4 or {NUM_STATES+3} initial state values, got {len(initial_state)}"

        if len(initial_state) == 4:
            ### Initial Conditions ###
            power = 9 # engine power level (0-10)

            # Default alpha & beta
            alpha = deg2rad(2.1215) # Trim Angle of Attack (rad)
            beta = 0                # Side slip angle (rad)

            # Initial Attitude
            alt = 1000        # altitude (ft)
            #vt = 600          # initial velocity (ft/sec)
            phi = 0           # Roll angle from wings level (rad)
            theta = 0         # Pitch angle from nose level (rad)
            #psi = 0           # Yaw angle from North (rad)

            x0, y0, heading0, v0 = initial_state

            vt = v0
            #psi = heading0 + np.pi
            psi = np.pi / 2 - heading0
            pe = x0
            pn = y0

            # Build Initial Condition Vectors
            # state = [vt, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
            initial_state = [vt, alpha, beta, phi, theta, psi, 0, 0, 0, pn, pe, alt, power]

            self.ap = WingmanAutopilot(target_heading=psi, target_vel=v0, target_alt=alt, stdout=True)
        else:

            psi = initial_state[StateIndex.PSI]
            v0 = initial_state[StateIndex.VT]
            alt = initial_state[StateIndex.ALT]
            self.ap = WingmanAutopilot(target_heading=psi, target_vel=v0, target_alt=alt, stdout=True)

        self.tmax = 0.0 # current time in simulation

        
        self.fss = F16SimState(initial_state, self.ap, step=1.0, extended_states=True)
        #self.fss.integrator_kwargs = {'rtol':1e-5, 'atol':1e-8}

    def one_step_with_control(self, control):
        '''run one step with the passed in control (rudder, throttle)'''

        #print(f".calling one_step_with_control({control}), ap targets: {self.ap.targets}")

        ali_limits = False

        self.ap.targets[0] -= control[0] # modify target heading by rudder control input

        if not ali_limits:
            
            self.ap.targets[1] += control[1] # modify target velocity by throttle control input
        else:
            k_v = 1.0 # need to be tuned

            v_min = 700
            v_max = 900

            if self.ap.targets[1] + control[1]*k_v < v_min:
                self.ap.targets[1] = v_min
                print(f"{self.ap.targets[1]:.2f}, trimming due to min")
            elif self.ap.targets[1] + control[1]*k_v > v_max:
                self.ap.targets[1] = v_max
                print(f"{self.ap.targets[1]:.2f}, trimming due to max")
            else:
                self.ap.targets[1] += control[1]*k_v
                print(f"{self.ap.targets[1]:.2f}, no trimming")

        self.tmax += 1.0
        self.fss.simulate_to(self.tmax, update_mode_at_start=True) # TODO: check if we need to be increasing the time by 1.0 here

@cachier(cache_dir='./cachier')
def load_data_and_sim_f16(single_index=None):
    '''load dubins data from file and recreate using f16 sim'''

    dubins_filename = "dubins_data.pkl"
    script_dir = get_script_path(__file__)
    dubins_file_path = os.path.join(script_dir, dubins_filename)

    with open(dubins_file_path, 'rb') as file:
        data = pickle.load(file)
        
    print(f"Loaded data from {dubins_filename}")
    states_np_list, actions_np_list = data

    if single_index is not None:
        print(f"Using single index {single_index}")
        states_np_list = [states_np_list[single_index]]
        actions_np_list = [actions_np_list[single_index]]

    num_trajectories = len(states_np_list)

    all_f16_res_dicts = []

    for traj_index in range(num_trajectories):
    #for traj_index in [DEBUG_INDEX]:
        traj_rollout = states_np_list[traj_index]
        actions_rollout = actions_np_list[traj_index]

        # simulate f16
        print(f'Simulating f16 {traj_index+1}/{num_trajectories}', end='', flush=True)
        f16_res_dict = recreate_trajectory_f16(traj_rollout, actions_rollout)

        all_f16_res_dicts.append(f16_res_dict)

    return all_f16_res_dicts

def extract_np_states_actions(res_dict):
    '''extract the 4d states and actions from a simulation res_dict'''

    f16_states_13d = res_dict['states']
    f16_np_states = []

    for i, state in enumerate(f16_states_13d):
        f16_x = state[StateIndex.POS_E]
        f16_y = state[StateIndex.POS_N]
        f16_heading = state[StateIndex.PSI]
        f16_v = state[StateIndex.VT]

        f16_heading = np.pi / 2 - f16_heading # convert psi to dubins heading(theta)
        f16_np_state = np.array([f16_x, f16_y, f16_heading, f16_v])
        f16_np_states.append(f16_np_state)

    return f16_np_states, res_dict['actions']

def eval_predictor_accuracy(all_f16_res_dicts, predictor, num_steps_to_predict):
    '''evalute the accuracy of a predictive model for a given number of steps
    
    returns max_error_vector, an error vector of max absolute errors for each output dimension
    '''

    # all_f16_trajs, actions_np_list

    num_trajectories = len(all_f16_res_dicts)
    error_vectors = []
    dubins_error_vectors = []
    max_error_sum_index = -1
    max_error_sum = 0
    dubins_max_error_sum_index = -1
    dubins_max_error_sum = 0

    dubins = DubinsPredictor()

    for traj_index in range(num_trajectories):
        res_dict = all_f16_res_dicts[traj_index]
        num_traj_steps = len(res_dict['states'])

        f16_np_states, actions_rollout = extract_np_states_actions(res_dict)

        #f16_np_states = all_f16_trajs[traj_index]
        #actions_rollout = actions_np_list[traj_index]

        #assert len(f16_np_states) == actions_rollout.shape[1] + 1, f"expected same number of f16 trajs and actions, got {len(f16_np_states)} and {actions_rollout.shape[1]}"

        for start_step in range(num_traj_steps - (num_steps_to_predict + 1)):
            #state0 = f16_np_states[start_step].copy()

            #actions_rollout_trimmed = actions_rollout[:, start_step:start_step+(num_steps_to_predict)]

            predicted_state = predictor.predict(res_dict, start_step, num_steps_to_predict)
            dubins_state = dubins.predict(res_dict, start_step, num_steps_to_predict)

            abs_pos_error = get_abs_error(predicted_state, f16_np_states[start_step+num_steps_to_predict])
            dubins_abs_pos_error = get_abs_error(dubins_state, f16_np_states[start_step+num_steps_to_predict])
            #print(f"Step {start_step+num_steps_to_predict}: abs_pos_error: {abs_pos_error}")

            error_vectors.append(abs_pos_error)
            dubins_error_vectors.append(dubins_abs_pos_error)

            # update max indices
            error_sum = np.sum(abs_pos_error)
            dubins_error_sum = np.sum(dubins_abs_pos_error)

            if error_sum > max_error_sum:
                max_error_sum = error_sum
                max_error_sum_index = start_step

            if dubins_error_sum > dubins_max_error_sum:
                dubins_max_error_sum = dubins_error_sum
                dubins_max_error_sum_index = start_step

    max_error_vector = np.max(np.abs(error_vectors), axis=0)
    max_dubins_error_vector = np.max(np.abs(dubins_error_vectors), axis=0)

    print(f"max_error_vector: {max_error_vector}")
    print(f"max_dubins_error_vector: {max_dubins_error_vector}")
    #print(f"max_error_vector: {max_error_vector}")
    print()
    print(f"max_error_sum: {max_error_sum}, index: {max_error_sum_index}")
    print(f"dubins_max_error_sum: {dubins_max_error_sum}, index: {dubins_max_error_sum_index}")

    return max_error_vector

def plot_prediction_vs_dubins(all_f16_res_dicts, predictor, num_steps_to_predict):
    '''for each trajectory plot prediction vs dubins vs f16
    '''

    num_trajectories = len(all_f16_res_dicts)
    dubins = DubinsPredictor()

    for traj_index in range(num_trajectories):
        res_dict = all_f16_res_dicts[traj_index]
        num_traj_steps = len(res_dict['states'])

        print(f"Plotting trajectory {traj_index} with {num_traj_steps} steps")

        f16_np_states, actions_rollout_full = extract_np_states_actions(res_dict)

        ax = plot.plot_overhead(res_dict, figsize=(10, 8))
        plt.plot(res_dict['states'][:, StateIndex.POS_E], res_dict['states'][:, StateIndex.POS_N], 'k-', label='F16 recreation')

        for start_step in range(num_traj_steps - (num_steps_to_predict + 1)):

            x = res_dict['states'][:, StateIndex.POS_E][start_step]
            y = res_dict['states'][:, StateIndex.POS_N][start_step]

            dubins_xs = [x]
            dubins_ys = [y]

            regression_xs = [x]
            regression_ys = [y]

            #state0 = f16_np_states[start_step].copy()

            # do dubins prediction from start step

            #actions_rollout_trimmed = actions_rollout[:, start_step:start_step+(num_steps_to_predict)]

            for s in range(1, num_steps_to_predict):
                predicted_state = predictor.predict(res_dict, start_step, s)
                dubins_state = dubins.predict(res_dict, start_step, s)

                regression_xs.append(predicted_state[0])
                regression_ys.append(predicted_state[1])

                dubins_xs.append(dubins_state[0])
                dubins_ys.append(dubins_state[1])

            # plot
            lw = 0.5
            plt.plot(dubins_xs, dubins_ys, 'g-', lw=lw, label='Dubins' if start_step == 0 else None)
            plt.plot(regression_xs, regression_ys, 'b-', lw=lw, label='Linear' if start_step == 0 else None)


        ax.legend()
        filename = f'plots/overhead_{traj_index}.png'
        plt.savefig(filename)
        #plt.show()
        plt.close()
        print(f"Made {filename}")

        

def recreate_trajectory_f16(traj_one, actions_one, stdout=True):
    '''run f-16 simulation of the passed-in rollout from saferl'''

    init_state = traj_one[:, 0] # x0, y0, heading0, v0

    #print(f"first state: {init_state}")
    #print(f"first action: {actions_one[:, 0]}")
    
    f16 = WingmanF16State(init_state)

    #ap = WingmanAutopilot(target_heading=math.pi/2, target_vel=400, target_alt=alt, stdout=True)

    #step = 1/30
    #extended_states = False
    #res = run_f16_sim(init, tmax, ap, step=step, extended_states=extended_states, integrator_str='rk45')

    # print(f"res['states']: {res['states']}")

    AP_VEL_TARGET_INDEX = 1 
    AP_PSI_TARGET_INDEX = 0
    vel_targets = [f16.ap.targets[AP_VEL_TARGET_INDEX]]
    psi_targets = [f16.ap.targets[AP_PSI_TARGET_INDEX]]
    u_refs = None
    f16_np_states = []
    
    ## append init state
    last_f16_state = f16.fss.x0
    f16_x = last_f16_state[StateIndex.POS_E]
    f16_y = last_f16_state[StateIndex.POS_N]
    f16_heading = last_f16_state[StateIndex.PSI]
    f16_v = last_f16_state[StateIndex.VT]
    f16_np_state = np.array([f16_x, f16_y, f16_heading, f16_v])
    f16_np_states.append(f16_np_state)

    for i, action in enumerate(actions_one.T):
        f16.one_step_with_control(action)

        if stdout:
            print('.', end='', flush=True)

        vel_targets.append(f16.ap.targets[AP_VEL_TARGET_INDEX])
        psi_targets.append(f16.ap.targets[AP_PSI_TARGET_INDEX])

        if u_refs is None:
            first_state = f16.fss.states[0]
            u_refs = [f16.ap.get_u_ref(0, first_state)]

        u_ref = f16.ap.get_u_ref(0, f16.fss.states[-1])
        u_refs.append(u_ref)

        last_f16_state = f16.fss.states[-1]
        f16_x = last_f16_state[StateIndex.POS_E]
        f16_y = last_f16_state[StateIndex.POS_N]
        f16_heading = last_f16_state[StateIndex.PSI]
        f16_v = last_f16_state[StateIndex.VT]
        f16_np_state = np.array([f16_x, f16_y, f16_heading, f16_v])
        f16_np_states.append(f16_np_state)

    if stdout:
        print()

    last_state = traj_one[:, -1].copy() # x0, y0, heading0, v0
    last_state[2] = np.pi / 2 - last_state[2] # convert heading to psi

    assert len(f16_np_states) == len(actions_one.T) + 1, f"expected {len(actions_one.T) + 1} states, got {len(f16_np_states)}"

    # convert heading to psi in f16_np_states
    for state in f16_np_states:
        state[2] = np.pi / 2 - state[2]

    f16_res_dict = {'states': np.array(f16.fss.states), 'actions': actions_one, 'times': np.array(f16.fss.times),
            'u_list': f16.fss.u_list, 'ps_list': f16.fss.ps_list, 'Nz_list': f16.fss.Nz_list, 'Ny_r_list': f16.fss.Ny_r_list,
            'vel_targets': vel_targets, 'psi_targets': psi_targets, 'u_refs': u_refs}
    
    return f16_res_dict

def plot_normal(traj_index, traj_rollout, traj_linear_list, f16_res_dict):
    '''normal overhead plotting'''

    ax = plot.plot_overhead(f16_res_dict)
    plt.plot(f16_res_dict['states'][:, StateIndex.POS_E], f16_res_dict['states'][:, StateIndex.POS_N], 'k-', label='F16 recreation')

    # add overhead plot of first dubins path traj_one rows 0 and 1
    for traj_linear in traj_linear_list:
        label = 'Linear' if traj_linear is traj_linear_list[0] else None
        ax.plot(traj_linear[0], traj_linear[1], 'g', lw=0.7, label=label)

#    ax.plot(traj_linear[0], traj_linear[1], 'lime', label='Linear')
#    ax.plot(traj_linear[0, 0], traj_linear[1, 0], 'go')

    ax.plot(traj_rollout[0], traj_rollout[1], 'r:', label='Rollout (Dubins)')
    ax.plot(traj_rollout[0, 0], traj_rollout[1, 0], 'ro')

    ax.legend()

    filename = f'plots/overhead_{traj_index}.png'
    plt.savefig(filename)
    plt.close()
    print(f"Made {filename}")

def plot_details(traj_linear_list, full_states_linear, res):
    '''extra detailed plotting for a given simulation
    
    res is f16_res_dict
    '''

    assert isinstance(traj_linear_list, list)

    throttle_cmds = [u_tuple[3] for u_tuple in res['u_list']]

    plot.plot_single(res, 'alpha', title='Alpha (angle of attack)')
    filename = 'alpha.png'
    plt.savefig(filename)
    print(f"Made {filename}")

    plot.plot_single(res, 'psi', title='Psi (heading angle, radians)')
    plt.plot(res['times'], res['states'][:, StateIndex.PSI], 'k-', label='actual')

    # plot the target psi as a dotted line
    psi_targets = res['psi_targets']
    plt.plot(res['times'], psi_targets, 'r--', label='target')
    # add reference line for 3pi/2
    plt.axhline(y=3*np.pi/2, color='g', linestyle='--', label='3pi/2')
    plt.legend()
    filename = 'psi.png'
    plt.savefig(filename)
    print(f"Made {filename}")

    plt.clf()
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))

    traj_linear = traj_linear_list[0]

    print(f'num time steps: {res['times'][:-1]}, {traj_linear[3].shape=}')

    axs[0].plot(res['times'], res['states'][:, StateIndex.VT], 'k-', label='f16')
    #axs[0].plot(res['times'], res['states'][:, StateIndex.VT], 'g-', label='linear')
    axs[0].plot(res['times'][:-1], traj_linear[3], 'lime', label='linear')
    vel_targets = res['vel_targets']
    axs[0].plot(res['times'], vel_targets, 'b:', label='target')
    axs[0].set_title('Velocity (ft/sec)')

    #u_refs = res['u_refs']
    axs[1].plot(res['times'], res['states'][:, StateIndex.PSI], 'k-', label='f16')
    axs[1].plot(res['times'][:-1], traj_linear[2], 'lime', label='linear')
    axs[1].set_title('Heading angle (theta in dubins, psi in f16)')

    #axs[1].plot(res['times'], throttle_cmds, 'b-', label='throttle')
    #axs[1].set_title('Throttle Cmds')
    axs[2].plot(res['times'], res['states'][:, StateIndex.PHI], 'k-', label='actual')

    if 'phi' in full_states_linear:
        axs[2].plot(res['times'][:-1], full_states_linear['phi'], 'lime', label='linear')

    #print(f"full_states_linear['phi']: {full_states_linear['phi']}")
    axs[2].set_title('Roll Angle (Phi)')

    # add legend entry for blue '-' line called 'actual'
    for ax in axs:
        ax.legend()

    #plt.legend()
    filename = 'vel.png'
    plt.savefig(filename)
    print(f"Made {filename}")

    ax = plot.plot_overhead(res)
    plt.plot(res['states'][:, StateIndex.POS_E], res['states'][:, StateIndex.POS_N], 'k-', label='F16 recreation')

    # add overhead plot of first dubins path traj_one rows 0 and 1

    for traj_linear in traj_linear_list:
        ax.plot(traj_linear[0], traj_linear[1], 'lime', label='Linear')
        #ax.plot(traj_linear[0, 0], traj_linear[1, 0], 'go')
    ax.legend()

    filename = 'overhead.png'
    plt.savefig(filename)
    print(f"Made {filename}")
    
    # plot inner loop controls + references
    plot.plot_inner_loop(res)
    filename = 'inner_loop.png'
    plt.savefig(filename)
    print(f"Made {filename}")

    # plot outer loop controls + references
    plot.plot_outer_loop(res)
    filename = 'outer_loop.png'
    plt.savefig(filename)
    print(f"Made {filename}")

def get_abs_error(symbolic_state, f16_np_state):
    '''get absolute error between symbolic state and f16 state'''

    error = np.zeros(4)

    for dim in range(4):
        error[dim] = abs(symbolic_state[dim] - f16_np_state[dim])

    return error

def get_abs_position_error(traj_one, f16_np_states):
    '''get relative position error between dubins trajectory and f16 recreation'''

    last_f16_state = f16_np_states[-1]

    last_rollout_state = traj_one[:, -1].copy() # x0, y0, heading0, v0
    last_rollout_state[2] = np.pi / 2 - last_rollout_state[2] # convert heading to psi

    rel_errors = []

    for dim in range(4):
        rel_error = np.linalg.norm(last_rollout_state[dim] - last_f16_state[dim]) / np.linalg.norm(last_rollout_state[dim])
        rel_errors.append(f"{rel_error*100:.3f}%")

    print(f"rel_errors: {rel_errors}")

    pos_dubins = last_rollout_state[:2]
    pos_f16 = last_f16_state[:2]
    abs_pos_error = np.linalg.norm(pos_dubins - pos_f16)

    return abs_pos_error
