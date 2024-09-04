 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox

def get_extended_state_rff(X, seed=1985, gamma=1e-4, num_features=200, include_ident=True, include_ones=True):
    '''get X with rff observables
    
    this also duplicates the variables as part of the observables
    if add_one=True, this also adds a constant 1 to the observables at each step
    '''

    assert len(X.shape) == 2, f"X.shape={X.shape}, expected 2D array"

    np.random.seed(seed)
    
    dimension = X.shape[0]

    assert num_features >= 0, f"num_features={num_features}, expected >= 0"

    if num_features == 0:
        assert include_ident, f"num_features={num_features}, expected 0 if include_ident=True"
        rv = X
    else:
        w = np.sqrt(2 * gamma) * np.random.normal(size=(num_features, dimension))
        
        # Generate D iid samples from Uniform(0,2*pi)
        u = 2 * np.pi * np.random.rand(1, num_features)

        s = np.sqrt(2 / num_features)
        #s = 0.02

        # each observation is gamma * cos(col <dot> rand_vec + rand_phase)    

        Z = s * np.cos(w @ X + u.T) # was X.T @ w

        if include_ident:
            rv = np.vstack((X, Z))
        else:
            rv = Z

    if include_ones:
        ones = np.ones((1, X.shape[1]))
        rv = np.vstack((rv, ones))

    return rv

def train_koopman(states_np_list, data_to_x_xprime_func):
    '''train a koopman model, return a'''

    for states_np in states_np_list:
        assert type(states_np) == np.ndarray, f"states_np type={type(states_np)}, expected np.ndarray"
        assert len(states_np.shape) == 2, f"states_np.shape={states_np.shape}, expected 2D array"

    X_mats = []
    X_prime_mats = []

    for states_np in states_np_list:
        num_steps = states_np.shape[1]

        if num_steps < 2:
            continue

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
    print(f"pseudoinv.shape={pseudoinv.shape}")
    print(f"X shape={X.shape}")
    print(f"X_prime.shape={X_prime.shape}")

    A = X_prime @ pseudoinv
    print(f"A.shape={A.shape}")

    if A.shape[0] < 5 and A.shape[1] < 5:
        print(f"A:\n{A}")

    # print A and B norm
    print(f"A norm={np.linalg.norm(A)}")
    
    return A

class KoopmanData:
    '''data related to koopman linearization'''

    def __init__(self, length_scale, num_obs, include_ident):
        self.trained = False

        self.A = None

        self.length_scale = None
        self.num_obs = None
        self.include_ones = None

        self.include_ident = True
        

    def data_to_x_xprime_func(self, x, xp):
        '''convert state to observed state, x and (optionally) xprime'''

        length_scale = self.length_scale
        num_obs = self.num_obs
        include_ident = self.include_ident
        include_ones = self.include_ones

        x_rff = get_extended_state_rff(x, gamma=length_scale, num_features=num_obs, include_ident=include_ident, include_ones=include_ones)
        xp_rff = get_extended_state_rff(xp, gamma=length_scale, num_features=num_obs, include_ident=include_ident, include_ones=include_ones)

        return x_rff, xp_rff

    def train(self, trajectories, length_scale, num_obs, include_ones):
        '''train the koopman linearization'''

        self.length_scale = length_scale
        self.num_obs = num_obs
        self.include_ones = include_ones

        state_np_list = []

        for xs, ys in trajectories:
            state_np = np.vstack((xs, ys))
            state_np_list.append(state_np)
        
        self.A = train_koopman(state_np_list, self.data_to_x_xprime_func)
        self.trained = True

    def get_prediction(self, x, y):
        '''get the prediction from the koopman linearization'''

        assert self.trained, 'Koopman model not trained'

        state_np = np.vstack((x, y))
        assert state_np.shape == (2, 1), f"state_np.shape={state_np.shape}, expected (2, 1)"

        extended_state_np = get_extended_state_rff(state_np, gamma=self.length_scale, num_features=self.num_obs, 
                                                    include_ident=self.include_ident, include_ones=self.include_ones)
        res = self.A @ extended_state_np

        return res[0, 0], res[1, 0]

koop_obj = KoopmanData(1e-1, 10, 0)

def plot_arrow(ax, pt_1, pt_2, color='gray'):
    """
    Draws an arrow from pt_1 to pt_2 using plot() with a 30-degree arrowhead.
    
    Parameters:
        pt_1 (tuple): Starting point (x, y) of the arrow.
        pt_2 (tuple): Ending point (x, y) of the arrow.
        color (str): Color of the arrow. 
    """
    # Unpack points
    x_start, y_start = pt_1
    x_end, y_end = pt_2
    
    # Plot the line part of the arrow
    ax.plot([x_start, x_end], [y_start, y_end], color=color)
    
    # Arrowhead configuration
    arrow_length = 0.1  # Length of the arrowhead lines
    arrow_angle = np.pi / 6  # 30 degrees in radians
    
    # Calculate the direction of the arrow
    dx = x_end - x_start
    dy = y_end - y_start
    angle = np.arctan2(dy, dx)
    
    # Calculate the points for the arrowhead
    x1 = x_end - arrow_length * np.cos(angle + arrow_angle)
    y1 = y_end - arrow_length * np.sin(angle + arrow_angle)
    
    x2 = x_end - arrow_length * np.cos(angle - arrow_angle)
    y2 = y_end - arrow_length * np.sin(angle - arrow_angle)
    
    # Plot the arrowhead
    ax.plot([x_end, x1], [y_end, y1], color=color)
    ax.plot([x_end, x2], [y_end, y2], color=color)

# Draw the vector field
def draw_vector_field(ax):
    if koop_obj.trained:
        for x in range(-10, 11):
            for y in range(-10, 11):
                pred_x, pred_y = koop_obj.get_prediction(x, y)

                delta_x = pred_x - x
                delta_y = pred_y - y
                delta_vec = np.array([delta_x, delta_y])

                #if x == 0:
                #    print(f"x={x}, y={y}, pred_x={pred_x}, pred_y={pred_y}, delta_x={delta_x}, delta_y={delta_y}")

                max_magnitude = 0.9

                if np.linalg.norm(delta_vec) < max_magnitude:
                    # plot the vector
                    color='gray'                    
                else:
                    # normalize the vector to have a magnitude of `max_magnitude`
                    delta_vec = delta_vec / np.linalg.norm(delta_vec) * max_magnitude
                    color = 'red'
           
                xs = [x, x + delta_vec[0]]
                ys = [y, y + delta_vec[1]]
                #ax.plot(xs, ys, color=color)
                pt1 = (x, y)
                pt2 = (x + delta_vec[0], y + delta_vec[1])

                plot_arrow(ax, pt1, pt2, color=color)

                #do this: ax.annotate("", xy=(0.5, 0.5), xytext=(0, 0),  arrowprops=dict(arrowstyle="->", lw=2))
                #ax.annotate("", xy=(x + delta_vec[0], y + delta_vec[1]), xytext=(x, y), arrowprops=dict(arrowstyle="->", lw=1, color=color))
    else:
        # just plot dots '.' at meshgrid points
        X, Y = np.meshgrid(np.linspace(-10, 10, 20), np.linspace(-10, 10, 20))
        
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

# Initialize the plot
fig, ax1 = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(bottom=0.15)
ax1.set_title(f"(Koopman model not trained)")

# Button click callback
def clear(event):
    while len(lines) > 0:
        line = lines.pop()
        line.remove()

    koop_obj.trained = False

    # Remove all lines from the plot
    for line in ax1.lines:
        line.remove()
        
    draw_vector_field(ax1)

    plt.draw()
    ax1.set_title(f"(Koopman model not trained)")

def train(event):
    # Get the data points
    trajectories = [l.get_data() for l in lines]
    length_scale = float(text_length_scale.text)
    num_obs = int(text_num_obs.text)
    include_ones = int(text_include_ones.text)
    koop_obj.train(trajectories, length_scale, num_obs, include_ones)

    # Remove vector field lines from plot
    for line in ax1.lines:
        if line not in lines:
            line.remove()

    draw_vector_field(ax1)

    # update title to be "A norm = ..."
    ax1.set_title(f"A.shape={koop_obj.A.shape}, norm(A) = {np.linalg.norm(koop_obj.A):.1f}")

# Draw initial vector field
draw_vector_field(ax1)

# Add text boxes and buttons
axcolor = 'lightgoldenrodyellow'
ax_length_scale = plt.axes([0.2, 0.05, 0.08, 0.04], facecolor=axcolor)
ax_num_obs = plt.axes([0.4, 0.05, 0.08, 0.04], facecolor=axcolor)
ax_include_ones = plt.axes([0.6, 0.05, 0.08, 0.04], facecolor=axcolor)

text_length_scale = TextBox(ax_length_scale, 'Length Scale:', initial='1e-1')
text_num_obs = TextBox(ax_num_obs, 'Num Obs:', initial='0')
text_include_ones = TextBox(ax_include_ones, 'Include Ones:', initial='1')

ax_button_clear = plt.axes([0.7, 0.05, 0.1, 0.04])
ax_button_train = plt.axes([0.81, 0.05, 0.1, 0.04])

button_clear = Button(ax_button_clear, 'Clear')
button_clear.on_clicked(clear)

button_train= Button(ax_button_train, 'Train')
button_train.on_clicked(train)

# List to keep track of lines drawn by user
lines = []

# initial data
xs = [-2, -1.5, -1, -0.5, 0, 0.5]
ys = [1, 1, 1, 1, 1, 1]
line, = ax1.plot(xs, ys, 'o-', color='black', lw=2)
lines.append(line)

xs = [-4, -3.5, -3]
ys = [-2, -2, -2]
line, = ax1.plot(xs, ys, 'o-', color='black', lw=2)
lines.append(line)

# Connect mouse events to callbacks
fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('button_release_event', on_release)

plt.show()
