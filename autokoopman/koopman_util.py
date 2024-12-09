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

    def __init__(self, name='Plain $DMD$'):
        '''initialize the Koopman model'''

        self.name = name

        # these get set once train() is called
        self.A = None
        self.B = None

    def get_extended_single_state(self, state_np):
        '''get the extended state, for the first state in a trajectory'''

        return get_extended_state_identity(state_np)
    
    def preprocess_trajectory(self, states_np, actions_np=None):
        '''preprocess a test trajectory, possibly modifying it (for example to normalize)'''

        if actions_np is None:
            return states_np
        
        return states_np, actions_np

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

        assert self.A is None, "Koopman model already trained"

        self.A, self.B = train_koopman(states_np_list, actions_np_list, self.data_to_x_xprime)

    def max_plot_steps(self):
        '''maximum number of steps to plot'''

        return np.inf
    
def make_rotation_matrix(theta):
    '''make a 2D rotation matrix'''


    assert isinstance(theta, float), f"theta={theta} is not a float: {theta} (type={type(theta)})"

    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        
class KoopmanIdentityRelative(KoopmanIdentity):

    def __init__(self, rotate=False):
        name = f"Relative $DMD$"

        if rotate:
            name += f" (w/ rotation)"

        KoopmanIdentity.__init__(self, name=name)
        
        self.rotate = rotate

    def data_to_x_xprime(self, state_np, state_prime_np):
        '''convert a single data point to x and x' for koopman training'''

        x = state_np.copy()
        xp = state_prime_np.copy()

        for dim in (0, 1):
            x[dim, :] -= state_np[dim, :] # should make the var value zero
            xp[dim, :] -= state_np[dim, :]

        if self.rotate:
            THETA_DIM = 2

            theta = state_np[THETA_DIM, 0]
            R_neg_theta = make_rotation_matrix(-theta)

            x[0:2, :] = R_neg_theta @ x[0:2, :]
            x[THETA_DIM, :] -= theta

            xp[0:2, :] = R_neg_theta @ xp[0:2, :]
            xp[THETA_DIM, :] -= theta

            # assert x[0], x[1], and x[2] is now zero
            assert np.linalg.norm(x[0:3, :]) < 1e-6, f"np.linalg.norm(x[0:3, :])={np.linalg.norm(x[0:3, :])}"

        return x, xp
    
    def predict(self, x_extended, u):
        '''predict the next state using the koopman model
    
        returns the next extended state
        '''

        assert self.A is not None, "Koopman model not trained, call train() first"

        x_relative = x_extended.copy()

        # translate
        for dim in (0, 1):
            x_relative[dim, :] -= x_extended[dim, :]

        # rotate
        if self.rotate:
            THETA_DIM = 2

            theta = x_extended[THETA_DIM, 0]
            R_neg_theta = make_rotation_matrix(-theta)
            x_relative[0:2, :] = R_neg_theta @ x_relative[0:2, :]
            x_relative[THETA_DIM, :] -= theta

        # (optional) make extended state

        # predict
        result = self.A @ x_relative + self.B @ u

        # rotate back
        if self.rotate:
            theta = x_extended[THETA_DIM, 0]
            R_theta = make_rotation_matrix(theta)
            result[0:2, :] = R_theta @ result[0:2, :]
            result[THETA_DIM, :] += theta

        # translate back
        for dim in (0, 1):
            result[dim, :] += x_extended[dim, :]

        return result

class KoopmanRFF(KoopmanIdentity):

    def __init__(self, gamma, num_features, name='$RFF$', seed=1985):
        fullname = f"{name} ($\\gamma$={gamma}, $N_{{obs}}$={num_features})"
        super().__init__(fullname)

        self.seed = seed
        self.gamma = gamma
        self.num_features = num_features

    def get_extended_single_state(self, state_np):
        '''get the extended state, for the first state in a trajectory'''

        rv = get_extended_state_rff(state_np, self.seed, gamma=self.gamma, num_features=self.num_features)

        return rv
    
class KoopmanExperimental(KoopmanRFF):
    """copied from relative RFF"""

    def __init__(self, gamma, num_features, name='Experimental', seed=1985):

        super().__init__(gamma, num_features, name=name, seed=seed)
        
    def data_to_x_xprime(self, state_np, state_prime_np):
        '''convert a single data point to x and x' for koopman training'''

        x_extended = get_extended_state_rff(state_np, self.seed, gamma=self.gamma, num_features=self.num_features)
        xp_extended = get_extended_state_rff(state_prime_np, self.seed, gamma=self.gamma, num_features=self.num_features)

        rel_x = state_np.copy()
        rel_xp = state_prime_np.copy()

        for dim in (0, 1, 2, 3):
            rel_x[dim, :] -= state_np[dim, :] # should make the var value zero
            rel_xp[dim, :] -= state_np[dim, :]

        # extend states
        x_stack = np.vstack((rel_x, x_extended))
        xp_stack = np.vstack((rel_xp, xp_extended))

        return x_stack, xp_stack
    
    def get_extended_single_state(self, state_np):
        '''get the extended state, for the first state in a trajectory'''

        #rv = get_extended_state_rff(state_np, self.seed, gamma=self.gamma, num_features=self.num_features)

        return state_np #rv
    
    def predict(self, x_orig_extended, u):
        '''predict the next state using the koopman model
    
        returns the next extended state
        '''

        assert self.A is not None, "Koopman model not trained, call train() first"

        x_relative = x_orig_extended.copy()[0:4]

        x_extended = get_extended_state_rff(x_relative, self.seed, gamma=self.gamma, num_features=self.num_features)

        # translate
        for dim in (0, 1, 2, 3):
            x_relative[dim, :] -= x_relative[dim, :]

        # make extended state
        x_stack = np.vstack((x_relative, x_extended))

        assert x_stack.shape[0] == self.A.shape[1], f"x_stack.shape={x_stack.shape} != self.A.shape[1]={self.A.shape[1]}"

        # predict
        result = self.A @ x_stack + self.B @ u

        # translate back
        for dim in (0, 1, 2, 3):
            result[dim, :] += x_orig_extended[dim, :]

        result = result[0:4, :]
        assert result.shape == x_orig_extended.shape, f"result.shape={result.shape} != x_orig_extended.shape={x_orig_extended.shape}"

        return result 

class KoopmanRFFRelative(KoopmanRFF):

    def __init__(self, gamma, num_features, rotate=False, name='Relative $RFF$', seed=1985):
        # rotate is not an option since RFF is not needed if rotate is used

        if rotate:
            name += f" (w/ rotation)"

        super().__init__(gamma, num_features, name=name, seed=seed)


        self.rotate = rotate

        
    def data_to_x_xprime(self, state_np, state_prime_np):
        '''convert a single data point to x and x' for koopman training'''

        x = state_np.copy()
        xp = state_prime_np.copy()

        for dim in (0, 1):
            x[dim, :] -= state_np[dim, :] # should make the var value zero
            xp[dim, :] -= state_np[dim, :]

        if self.rotate:
            THETA_DIM = 2

            theta = state_np[THETA_DIM, 0]
            R_neg_theta = make_rotation_matrix(-theta)

            x[0:2, :] = R_neg_theta @ x[0:2, :]
            x[THETA_DIM, :] -= theta

            xp[0:2, :] = R_neg_theta @ xp[0:2, :]
            xp[THETA_DIM, :] -= theta

            # assert x[0], x[1], and x[2] is now zero
            assert np.linalg.norm(x[0:3, :]) < 1e-6, f"np.linalg.norm(x[0:3, :])={np.linalg.norm(x[0:3, :])}"

        # extend states
        extended_x = self.get_extended_single_state(x)
        extended_xp = self.get_extended_single_state(xp)

        return extended_x, extended_xp
    
    def predict(self, x_extended, u):
        '''predict the next state using the koopman model
    
        returns the next extended state
        '''

        assert self.A is not None, "Koopman model not trained, call train() first"

        x_relative = x_extended.copy()[0:4]

        # translate
        for dim in (0, 1):
            x_relative[dim, :] -= x_extended[dim, :]

        # rotate
        if self.rotate:
            THETA_DIM = 2

            theta = x_extended[THETA_DIM, 0]
            R_neg_theta = make_rotation_matrix(-theta)
            x_relative[0:2, :] = R_neg_theta @ x_relative[0:2, :]
            x_relative[THETA_DIM, :] -= theta

        # make extended state
        x_rel_extended = self.get_extended_single_state(x_relative)

        # predict
        result = self.A @ x_rel_extended + self.B @ u

        # rotate back
        if self.rotate:
            theta = x_extended[THETA_DIM, 0]
            R_theta = make_rotation_matrix(theta)
            result[0:2, :] = R_theta @ result[0:2, :]
            result[THETA_DIM, :] += theta

        # translate back
        for dim in (0, 1):
            result[dim, :] += x_extended[dim, :]

        return result 
    
class NormalizationMixin:
    '''a mix-in class to be used for multiple inheritance to normalize data before training and testing a Koopman model'''

    def __init__(self):
        # assigned on train()
        self.centers_states = None 
        self.ranges_states = None

    def get_extended_single_state(self, state_np):
        '''get the extended state, for the first state in a trajectory'''

        extended_state_np = super().get_extended_single_state(state_np)

        return add_ones(extended_state_np) # add ones since we're normalizing

    def preprocess_trajectory(self, states_np, actions_np=None):
        '''Preprocess a test trajectory, possibly modifying it (for example, to normalize)'''
        
        assert self.centers_states is not None, "Koopman model not trained, call train() first"
        
        normalized_states_np = normalize_single_list([states_np], self.centers_states, self.ranges_states)[0]

        if actions_np is None:
            return normalized_states_np

        return normalized_states_np, actions_np

    def train(self, states_np_list, actions_np_list):
        '''Train the Koopman model
        
        states_np_list is a list of numpy arrays of states
        actions_np_list is a list of numpy arrays of actions
        '''
        
        assert self.A is None, "Koopman model already trained"

        # Normalize states and store the normalization factors
        norm_tup, norm_states_np_list = normalize_state_lists(states_np_list)
        self.centers_states, self.ranges_states = norm_tup

        # note we do not normalize actions since error is not based on actions so it wouldn't matter (plus you'd need to add identity action)
        #print(".in train(); 2 / range[0]=", 2 / self.ranges_states[0])

        # Call the parent class's train method with normalized data
        super().train(norm_states_np_list, actions_np_list)

class KoopmanRFFNormalized(NormalizationMixin, KoopmanRFF):

    def __init__(self, gamma, num_features, name='$RFF_{Norm}$', seed=1985):
        # Initialize both parent classes
        KoopmanRFF.__init__(self, gamma, num_features, name=name, seed=seed)
        NormalizationMixin.__init__(self)
    
    def max_plot_steps(self):
        '''maximum number of steps to plot'''

        return np.inf #2

class KoopmanIdentityNormalized(NormalizationMixin, KoopmanIdentity):
    
    def __init__(self, name='Plain $DMD_{Norm}$'):
        # Initialize both parent classes
        KoopmanIdentity.__init__(self, name=name)
        NormalizationMixin.__init__(self)

class KoopmanRFFResets(KoopmanRFF):

    def __init__(self, gamma, num_features, name='$RFF_{Reset}$', seed=1985):
        super().__init__(gamma, num_features, name=name, seed=seed)

    def predict(self, x_extended, u):
        '''predict the next state using the koopman model
    
        returns the next extended state
        '''

        assert self.A is not None, "Koopman model not trained, call train() first"

        # pull out just the state variables
        vars = x_extended[0:-self.num_features, :]

        # recompute (reset) observables
        reset_x_extended = self.get_extended_single_state(vars)

        return self.A @ reset_x_extended + self.B @ u
    
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

def normalize_single_list(data, centers, ranges, print_label=None):
    '''normalize a single data set
    
    data is a list of 2-d np.arrays, normalization is done on the rows
    '''

    normalized_data = []
    for data_mat in data:        
        norm_data_mat = normalize_matrix(data_mat, centers, ranges)

        normalized_data.append(norm_data_mat)

    return normalized_data

def get_centers_ranges(states_np_list, stdout=False):
    '''get the centers and ranges of the data
    
    returns centers, ranges'''

    min_states = np.min(np.array([np.min(states_np, axis=1) for states_np in states_np_list]), axis=0)
    max_states = np.max(np.array([np.max(states_np, axis=1) for states_np in states_np_list]), axis=0)

    centers = (min_states + max_states) / 2
    ranges = max_states - min_states

    if stdout:
        print(f"min_states={min_states}")
        print(f"max_states={max_states}")
        print(f"centers={centers}")
        print(f"ranges={ranges}")

    return centers, ranges

def normalize_state_lists(training_states):
    '''normalize the data based on ranges in training data

    returns norm_tup, normalized_training_states

    where norm_tup is a tuple of (centers_states, ranges_states)
    '''

    centers_states, ranges_states = get_centers_ranges(training_states, stdout=False)

    training_states = normalize_single_list(training_states, centers_states, ranges_states)

    norm_tup = (centers_states, ranges_states)

    return norm_tup, training_states

def train_koopman(states_np_list, actions_np_list, data_to_x_xprime_func):
    '''train a koopman model'''

    X_mats = []
    X_prime_mats = []

    for states_np in states_np_list:
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

    # print first col of X and X_prime
    #print(f"X[0, :]={X[:, 0]}")
    #print(f"X_prime[0, :]={X_prime[:, 0]}\n")
    #print()

    #print(f"First 6 rows of X for 5 time steps:\n{X[0:6, :5]}")

    #Gamma = np.hstack([mat[:,:-1] for mat in actions_np_list])
    Gamma = np.hstack([mat[:,:-1] for mat in actions_np_list])


    Omega = np.vstack((X, Gamma))
    #print(f"Omega.shape={Omega.shape}")

    pseudoinv = np.linalg.pinv(Omega)

    A_B = X_prime @ pseudoinv
    #print(f"A_B.shape={A_B.shape}")

    A = A_B[:, :X.shape[0]]
    B = A_B[:, X.shape[0]:]

    # print A and B norm
    print(f"A norm={np.linalg.norm(A)} B norm={np.linalg.norm(B)}")

    # print first row of A and B
    #print(f"A[0, :]={A[0, :]}")
    #print(f"B[0, :]={B[0, :]}\n")

    #print(f"num cols: {X.shape[1]}")

    #print(f"B first 4 rows:\n{B[0:4, :]}")

    if False:
        for c in range(1):
            x0 = X[:, c]
            u0 = Gamma[:, c]
            x0p = X_prime[:, c]

            # print A * x0 + B * u0 and compare with x0p
            #print(f"A @ xc + B @ uc={A @ x0 + B @ u0}")
            #print(f"xcp={x0p}\n")
            abs_error = np.linalg.norm((A @ x0 + B @ u0) - x0p)
            print(f"c={c} abs_error={abs_error:.6f}, abs_error_first_4={np.linalg.norm((A @ x0 + B @ u0)[0:4] - x0p[0:4]):.6f}")

            

            predicted = (A @ x0 + B @ u0)
            actual = x0p

            print(f"x first 4: {x0[0:4]}")
            print(f"predicted x' first 4: {predicted[0:4]}")
            print(f"actual x' first 4: {actual[0:4]}")

            # also print what the prediction contribution is from the RFF observables
            masked_state = x0.copy()
            x0[0:4] = 0
            masked_predicted = (A @ x0 + B @ u0)
            print(f"masked predicted x' first 4: {masked_predicted[0:4]}")

            # print abs error for null hypothesis
            abs_error_null = np.linalg.norm(x0p[0:4] - x0[0:4])
            print(f"abs_error_null first 4={abs_error_null:.6f}\n")

    
    return A, B

def analyze_predictions(states_np_list, training_state_np_list, actions_np_list, koopman_obj_list, plot_name=None, plot_norms=False):
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

    if plot_name is not None:
        horz_plots = len(ax_index_vars) + 2

        if plot_norms:
            horz_plots += 1

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

            # possibly normalize
            states_np, actions_np = koopman_obj.preprocess_trajectory(states_np, actions_np)

            # replay using koopman
            first_state = states_np[:, 0:1]
            first_state_extended = koopman_obj.get_extended_single_state(first_state)

            #print(f"k_index={k_index} trajectory_index={trajectory_index} first_state_extended.shape={first_state_extended.shape}")

            #print(f"first_state_observed.shape={first_state_observed.shape}")
            num_steps = states_np.shape[1]
            num_vars = states_np.shape[0]
            test_traj = first_state_extended # test_traj is stored as 2-d np.array that gets extended
            percent_errors = []
            observed_state_norms = []

            # normal koopman prediction
            for step in range(num_steps - 1):
                if step > koopman_obj.max_plot_steps():
                    break

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

                try:
                    x_prime_exteneded = koopman_obj.predict(x_extended, u)
                    test_traj = np.hstack((test_traj, x_prime_exteneded))
                except:
                    print(f"Error predicting step={step} for trajectory_index={trajectory_index} k_index={k_index}")
                    break
                
                

            last_percent_errors.append(percent_errors[-1])
            #print(f"last_percent_error: {percent_errors[-1]}")

            if plot_name is None:
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

                if plot_norms:
                    axs[k_index, len(ax_index_vars) + 2].set_title(f"Extended State Norm")

                # plot training data 
                ax = axs[k_index, len(ax_index_vars)]

                for other_states_np in training_state_np_list:
                    other_states_np = koopman_obj.preprocess_trajectory(other_states_np)
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

            if plot_norms:
                # plot norm of state
                ax_index = 2 + len(ax_index_vars)
                ax = axs[k_index, ax_index]
                ax.plot(observed_state_norms, label=koopman_obj.name, color=traj_color)
                ax.set_xlabel("Time Step")
                ax.set_ylabel("2-Norm Observed State")

        avg = np.average(last_percent_errors)
        koopman_obj_percent_errors.append(avg)

    if plot_name is not None:
        plt.tight_layout()
        plt.savefig(plot_name)
        print(f"Saved plot to {plot_name}")
        #plt.show()
        plt.close()
        
    return koopman_obj_percent_errors

def add_ones(X):
    '''add a row of ones to the bottom of the matrix'''

    assert len(X.shape) == 2, f"X.shape={X.shape}, expected 2D array"

    one = np.ones((1, X.shape[1]))
    rv = np.vstack((X, one))

    return rv

def get_extended_state_identity(X):
    '''get X with identity observables'''

    assert len(X.shape) == 2, f"X.shape={X.shape}, expected 2D array"

    return X.copy()

def get_extended_state_rff(X, seed, gamma=1e-4, num_features=200):
    '''get X with rff observables
    
    this also duplicates the variables as part of the observables
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