 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox

def get_extended_state_rff(X, seed=1985, gamma=1e-4, num_features=200, include_ident=True):
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

    if include_ident:
        rv = np.vstack((X, Z))
    else:
        rv = Z

    return rv

def train_koopman(states_np_list, data_to_x_xprime_func):
    '''train a koopman model, return a'''

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

    
    print(f"X.shape={X.shape}")

    pseudoinv = np.linalg.pinv(X)

    A = X_prime @ pseudoinv
    #print(f"A_B.shape={A_B.shape}")

    # print A and B norm
    print(f"A norm={np.linalg.norm(A)}")
    
    return A

class KoopmanData:
    '''data related to koopman linearization'''

    def __init__(self, length_scale, num_obs, include_ident):
        self.trained = False

        self.A = None

    def train(self, trajectories, length_scale, num_obs, include_ident):
        '''train the koopman linearization'''

        state_np_list = []

        for xs, ys in trajectories:
            state_np = np.vstack((xs, ys))
            state_np_list.append(state_np)

        def data_to_x_xprime_func(x, xp):
            '''convert data to x and xprime'''

            x_rff = get_extended_state_rff(x, gamma=length_scale, num_features=num_obs, include_ident=include_ident)
            xp_rff = get_extended_state_rff(xp, gamma=length_scale, num_features=num_obs, include_ident=include_ident)

            return x_rff, xp_rff
        
        self.A = train_koopman(trajectories, data_to_x_xprime_func)
        self.trained = True

    def get_prediction(self, x, y):
        '''get the prediction from the koopman linearization'''

        assert self.trained, 'Koopman model not trained'

        state_np = np.vstack((x, y))

        res = self.A @ get_extended_state_rff(state_np, gamma=1e-4, num_features=200, include_ident=True)

        return res[0], res[1]



koop_obj = KoopmanData(1e-1, 10, 0)

# Function for vector field
def vector_field(x, y):
    #u = 2 * x**2 + x * y
    #v = 2 * y**2 + x * y
    #return u, v
    return koop_obj.get_prediction(x, y)

# Draw the vector field
def draw_vector_field(ax):
    x = np.linspace(-10, 10, 20)
    y = np.linspace(-10, 10, 20)
    X, Y = np.meshgrid(x, y)

    if koop_obj.trained:
        U, V = vector_field(X, Y)
        ax.quiver(X, Y, U, V, color='grey')
    else:
        # just plot dots '.' at meshgrid points
        ax.plot(X, Y, '.', ms=1, color='grey')

# Callback for drawing lines with the mouse
def on_click(event):
    if event.inaxes == ax1 and event.button == 1:  # Left mouse button
        line, = ax1.plot(event.xdata, event.ydata, 'o-', color='black', lw=2)
        lines.append(line)

def on_motion(event):
    if event.inaxes == ax1 and event.button == 1 and len(lines) > 0:
        xdata, ydata = lines[-1].get_data()
        xdata = np.append(xdata, event.xdata)
        ydata = np.append(ydata, event.ydata)
        lines[-1].set_data(xdata, ydata)
        plt.draw()

def on_release(event):
    if event.button == 1 and len(lines) > 0:
        plt.draw()

# Button click callback
def clear(event):
    while len(lines) > 0:
        line = lines.pop()
        line.remove()

    koop_obj.trained = False

    # Remove the existing vector field without clearing the entire plot
    for coll in ax1.collections:
        coll.remove()
        
    draw_vector_field(ax1)

    plt.draw()

def train(event):
    # Get the data points
    trajectories = [l.get_data() for l in lines]
    length_scale = float(text_length_scale.text)
    num_obs = int(text_num_obs.text)
    include_ident = int(text_include_id.text)
    koop_obj.train(trajectories, length_scale, num_obs, include_ident)

    # Remove the existing vector field without clearing the entire plot
    for coll in ax1.collections:
        coll.remove()

    draw_vector_field(ax1)

# Initialize the plot
fig, ax1 = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(bottom=0.15)

# Draw initial vector field
draw_vector_field(ax1)

# Add text boxes and buttons
axcolor = 'lightgoldenrodyellow'
ax_length_scale = plt.axes([0.2, 0.05, 0.08, 0.04], facecolor=axcolor)
ax_num_obs = plt.axes([0.4, 0.05, 0.08, 0.04], facecolor=axcolor)
ax_include_id = plt.axes([0.6, 0.05, 0.08, 0.04], facecolor=axcolor)

text_length_scale = TextBox(ax_length_scale, 'Length Scale:', initial='1e-1')
text_num_obs = TextBox(ax_num_obs, 'Num Obs:', initial='10')
text_include_id = TextBox(ax_include_id, 'Include Ident:', initial='0')

ax_button_clear = plt.axes([0.7, 0.05, 0.1, 0.04])
ax_button_train = plt.axes([0.81, 0.05, 0.1, 0.04])

button_clear = Button(ax_button_clear, 'Clear')
button_clear.on_clicked(clear)

button_train= Button(ax_button_train, 'Train')
button_train.on_clicked(train)

# List to keep track of lines drawn by user
lines = []

# Connect mouse events to callbacks
fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('button_release_event', on_release)

plt.show()
