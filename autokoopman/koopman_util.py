'''
Generic utilities for Koopman

Stanley Bak, 8/2024
'''


import numpy as np
from matplotlib import pyplot as plt

class KoopmanIdentity:
    '''generic base class for Koopman training and predicting

    override for different types of Koopman models (RFF, relative coords, etc)

    in base version, identity observables are used (plain DMD)
    '''

    def __init__(self, name='Plain DMD'):
        '''initialize the Koopman model'''

        self.name = name

        # these get set once train() is called
        self.A = None
        self.B = None

    def get_extended_single_state(self, state_np):
        '''get the extended state, for the first state in a trajectory'''

        return get_extended_state_identity(state_np)

    def data_to_x_xprime(self, state_np, state_prime_np):
        '''convert a single data point to x and x' for koopman training'''

        x = self.get_extended_single_state(state_np)
        xprime = self.get_extended_single_state(state_prime_np)

        return x, xprime

    def predict(self, x_extended, u):
        '''predict the next state using the koopman model
    
        returns the next extended state
        '''

        assert self.A is not None, "Koopman model not trained, call train() first"

        return self.A @ x_extended + self.B @ u

    def train(self, states_np_list, actions_np_list):
        '''train the koopman model

        states_np_list is a list of numpy arrays of states
        actions_np_list is a list of numpy arrays of actions
        '''

        self.A, self.B = train_koopman(states_np_list, actions_np_list, self.data_to_x_xprime)

class KoopmanRFF(KoopmanIdentity):

    def __init__(self, gamma, num_features, name='RFF', seed=1985):
        fullname = f"{name} ($\gamma$={gamma}, $N_{{obs}}$={num_features})"
        super().__init__(fullname)

        self.seed = seed
        self.gamma = gamma
        self.num_features = num_features

    def get_extended_single_state(self, state_np):
        '''get the extended state, for the first state in a trajectory'''

        return get_extended_state_rff(state_np, self.seed, gamma=self.gamma, num_features=self.num_features, add_ones=False)
    
class KoopmanRFFResets(KoopmanRFF):

    def __init__(self, gamma, num_features, seed=1985):
        super().__init__(gamma, num_features, name="RFF w/Resets", seed=seed)

    def predict(self, x_extended, u):
        '''predict the next state using the koopman model
    
        returns the next extended state
        '''

        assert self.A is not None, "Koopman model not trained, call train() first"

        # pull out just the state variables
        vars = x_extended[0:-self.num_features, :]

        # recompute extended state
        reset_x_extended = self.get_extended_single_state(vars)

        return self.A @ reset_x_extended + self.B @ u

def train_koopman(states_np_list, actions_np_list, data_to_x_xprime_func):
    '''train a koopman model'''

    X_mats = []
    X_prime_mats = []

    for states_np, actions_np in zip(states_np_list, actions_np_list):
        num_steps = states_np.shape[1]

        X_mat_list = []
        X_prime_mat_list = []

        for step in range(num_steps - 1):
            state_np = states_np[:, step:step+1]
            state_prime_np = states_np[:, step + 1:step + 2]

            single_state_x, singe_state_xp = data_to_x_xprime_func(state_np, state_prime_np)

            X_mat_list.append(single_state_x)
            X_prime_mat_list.append(singe_state_xp)
        
        X_mats.append(np.hstack(X_mat_list))
        X_prime_mats.append(np.hstack(X_prime_mat_list))

    X = np.hstack(X_mats)
    X_prime = np.hstack(X_prime_mats)

    #Gamma = np.hstack([mat[:,:-1] for mat in actions_np_list])
    Gamma = np.hstack([mat[:,:-1] for mat in actions_np_list])


    Omega = np.vstack((X, Gamma))
    #print(f"Omega.shape={Omega.shape}")

    pseudoinv = np.linalg.pinv(Omega)

    A_B = X_prime @ pseudoinv
    #print(f"A_B.shape={A_B.shape}")

    A = A_B[:, :X.shape[0]]
    B = A_B[:, X.shape[0]:]
    
    return A, B

def analyze_predictions(states_np_list, training_state_np_list, actions_np_list, koopman_obj_list, plot=False):
    '''analyze predictions from multiple koopman objects
    
    returns the average relative error at the last time step, for each koopman object (a list)

    if plot is set, will plot one row per koopman prediction, with all trajectories on the same plot
    '''

    num_trajectories = len(states_np_list)
    num_koopman_objs = len(koopman_obj_list)

    figsize = (16, 9) if num_koopman_objs > 1 else (16, 5)

    ax_index_vars = [(0, 1), (2, 3)]
    ax_index_labels = [("X", "Y"), ("Heading", "Speed")]

    #ax_index_vars = [(0, 1)]
    #ax_index_labels = [("X", "Y")]

    if plot:
        horz_plots = len(ax_index_vars) + 3

        fig, axs = plt.subplots(num_koopman_objs, horz_plots, figsize=figsize)

        if num_koopman_objs == 1:
            # make axs a 2-d array for compatibility
            axs = np.array([axs])

    koopman_obj_percent_errors = []

    for k_index, koopman_obj in enumerate(koopman_obj_list):
        last_percent_errors = []

        for trajectory_index in range(num_trajectories):
            states_np = states_np_list[trajectory_index]
            actions_np = actions_np_list[trajectory_index]

            # replay using koopman
            first_state = states_np[:, 0:1]
            first_state_extended = koopman_obj.get_extended_single_state(first_state)

            print(f"k_index={k_index} trajectory_index={trajectory_index} first_state_extended.shape={first_state_extended.shape}")

            #print(f"first_state_observed.shape={first_state_observed.shape}")
            num_steps = states_np.shape[1]
            num_vars = states_np.shape[0]
            test_traj = first_state_extended # test_traj is stored as 2-d np.array that gets extended
            percent_errors = []
            observed_state_norms = []

            # normal koopman prediction
            for step in range(num_steps - 1):
                x_extended = test_traj[:, -1:]
                observed_state_norms.append(np.linalg.norm(x_extended))

                # first update error
                real_vars_predicted = x_extended[0:num_vars, :]
                #print(f"real_vars_predicted.shape={real_vars_predicted.shape}")
                actual_vars = states_np[:, step:step+1]
                #print(f"actual_vars.shape={actual_vars.shape}")
            
                error = np.linalg.norm(real_vars_predicted - actual_vars)
                #print(f"step={step} error={error}")
                rel_error = error / np.linalg.norm(actual_vars)
                percent_errors.append(100 * rel_error)

                u = actions_np[:, step:step+1]

                x_prime_exteneded = koopman_obj.predict(x_extended, u)
                
                test_traj = np.hstack((test_traj, x_prime_exteneded))

            last_percent_errors.append(percent_errors[-1])
            print(f"last_percent_error: {percent_errors[-1]}")

            if plot == False:
                continue
            
            test_traj = test_traj.T
            traj_color = ['r', 'm', 'b', 'c', 'm', 'y', 'k'][trajectory_index % 7]

            for ax_col_index, (var_index, label) in enumerate(zip(ax_index_vars, ax_index_labels)):
                ax = axs[k_index, ax_col_index]
                xs = [x[var_index[0]] for x in test_traj]
                ys = [x[var_index[1]] for x in test_traj]

                ax.plot(xs, ys, '-o', ms=1.7, lw=0.6, color=traj_color)
                ax.set_xlabel(label[0])
                ax.set_ylabel(label[1])
            
                ax.plot(states_np[var_index[0], :], states_np[var_index[1], :], '-', color='g', lw=1, label='Ground Truth')
                ax.grid()
                #ax.set_aspect('equal', adjustable='box')

            if trajectory_index == 0: # once per row
                # add title to subplot
                for ax_col_index in range(len(ax_index_vars)):
                    axs[k_index, ax_col_index].set_title(f"{koopman_obj.name}")

                axs[k_index, len(ax_index_vars)].set_title(f"Training Data")
                axs[k_index, len(ax_index_vars) + 1].set_title(f"Percent Error")
                axs[k_index, len(ax_index_vars) + 2].set_title(f"Extended State Norm")

                # plot training data 
                ax = axs[k_index, len(ax_index_vars)]

                for other_states_np in training_state_np_list:
                    color = 'gray'
                    zorder = 1
                    lw = 0.1

                    xs = other_states_np[0, :]
                    ys = other_states_np[1, :]
                    ax.plot(xs, ys, '-', color=color, lw=lw, zorder=zorder)

            ax = axs[k_index, len(ax_index_vars)]
            
            # plot test data in green
            xs = states_np[0, :]
            ys = states_np[1, :]
            ax.plot(xs, ys, '-', color='g', lw=2, zorder=2)

            # indicate first state using traj_color
            ax.plot(xs[0], ys[0], 'o', ms=5, color=traj_color, zorder=3)

            # plot percent error
            ax = axs[k_index, 1 + len(ax_index_vars)]
            ax.plot(percent_errors, label=koopman_obj.name, color=traj_color)
            
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Percent Error (%)")

            # plot norm of state
            ax_index = 2 + len(ax_index_vars)
            ax = axs[k_index, ax_index]
            ax.plot(observed_state_norms, label=koopman_obj.name, color=traj_color)
            ax.set_xlabel("Time Step")
            ax.set_ylabel("2-Norm of Extended (Observed) State")

        avg = np.average(last_percent_errors)
        koopman_obj_percent_errors.append(avg)

    if plot:
        plt.tight_layout()
        plt.savefig("stan_koopman.png")
        plt.show()
        
    return koopman_obj_percent_errors

def get_extended_state_identity(X):
    '''get X with idnetity observables'''

    assert len(X.shape) == 2, f"X.shape={X.shape}, expected 2D array"

    rv = X.copy()

    return rv

def get_extended_state_rff(X, seed, gamma=1e-4, num_features=200, add_ones=False):
    '''get X with rff observables
    
    this also duplicates the variables as part of the observables
    if add_one=True, this also adds a constant 1 to the observables at each step
    '''

    assert len(X.shape) == 2, f"X.shape={X.shape}, expected 2D array"

    np.random.seed(seed)
    
    dimension = X.shape[0]

    w = np.sqrt(2 * gamma) * np.random.normal(size=(num_features, dimension))
    
    # Generate D iid samples from Uniform(0,2*pi)
    u = 2 * np.pi * np.random.rand(1, num_features)

    s = np.sqrt(2 / num_features)

    # each observation is gamma * cos(col <dot> rand_vec + rand_phase)    

    Z = s * np.cos(w @ X + u.T) # was X.T @ w

    if add_ones:
        one = np.ones((1, Z.shape[1]))
        rv = np.vstack((X, one, Z))
    else:
        rv = np.vstack((X, Z))

    return rv

def split_test_train(states_np_list, actions_np_list, num_test=3):
    '''split into test and train data'''

    assert len(states_np_list) == len(actions_np_list), f"states_np_list={len(states_np_list)} != actions_np_list={len(actions_np_list)}"
    assert num_test < len(states_np_list), f"num_test={num_test} >= len(states_np_list)={len(states_np_list)}"

    training_states = states_np_list[:-num_test]
    training_actions = actions_np_list[:-num_test]

    test_states = states_np_list[-num_test:]
    test_actions = actions_np_list[-num_test:]

    return test_states, test_actions, training_states, training_actions

def split_test_validation_train(states_np_list, actions_np_list, num_test=3, num_validation=3):
    '''split data into test, train, validation'''

    test_states, test_actions, rest_states, rest_actions = split_test_train(states_np_list, actions_np_list, num_test=num_test)
    validation_states, validation_actions, training_states, training_actions = split_test_train(rest_states, rest_actions, num_test=num_validation)

    return test_states, test_actions, validation_states, validation_actions, training_states, training_actions