import abc
import copy
import gym
import scipy.spatial
import scipy.integrate
import numpy as np


class BaseEnvObj(abc.ABC):

    @abc.abstractmethod
    def __init__(self, name):
        self.name = name

    @property
    @abc.abstractmethod
    def x(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def y(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def z(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def position(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def orientation(self) -> scipy.spatial.transform.Rotation:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def velocity(self):
        raise NotImplementedError


class BaseActuator(abc.ABC):

    @property
    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def space(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def default(self):
        raise NotImplementedError


class ContinuousActuator(BaseActuator):
    def __init__(self, name, bounds, default):
        self._name = name
        self._space = 'continuous'
        self._bounds = bounds

        if isinstance(default, np.ndarray):
            self._default = default
        else:
            self._default = np.array(default, ndmin=1, dtype=np.float64)

    @property
    def name(self) -> str:
        return self._name

    @property
    def space(self) -> str:
        return self._space

    @property
    def bounds(self) -> list:
        return self._bounds.copy()

    @bounds.setter
    def bounds(self, val):
        self._bounds = val.copy()

    @property
    def default(self):
        return copy.deepcopy(self._default)


class BaseController(abc.ABC):
    @abc.abstractmethod
    def gen_actuation(self, state, action=None):
        raise NotImplementedError


class PassThroughController(BaseController):
    def __init__(self):
        self.action_space = None

    def gen_actuation(self, state, action=None):
        return action


class AgentController(BaseController):

    def __init__(self, actuator_set, config):
        self.actuator_set = actuator_set
        self.actuator_config_list = config['actuators']

        self.action_preprocessors, self.action_space = self.setup_action_space()

    def setup_action_space(self):
        action_preprocessors = []
        action_space_tup = ()

        # loop over actuators in controller config and setup preprocessors
        for i, actuator_config in enumerate(self.actuator_config_list):
            actuator_name = actuator_config['name']

            # get associated actuator from platform's actuator_set
            if actuator_name in self.actuator_set.name_idx_map:
                actuator = self.actuator_set.actuators[self.actuator_set.name_idx_map[actuator_name]]
            else:
                raise ValueError("Actuator name {} not found in platform's actuator set".format(actuator_name))

            if actuator.space == 'continuous':
                # determine upper and lower bounds of actuator range.
                # Should be the intersection of the actuator object bounds and the actuator config bounds
                if 'bounds' in actuator_config:
                    actuator.bounds = actuator_config['bounds']

                bounds_min = actuator.bounds[0]
                bounds_max = actuator.bounds[1]

                if ('space' not in actuator_config) or (actuator_config['space'] == 'continuous'):
                    # validate config values
                    cont_act_config_keys = ['name', 'space', 'bounds', 'rescale', 'zero_centered', 'post_activation']
                    for k in actuator_config.keys():
                        if k not in cont_act_config_keys:
                            raise ValueError(
                                f"Action Config for Actuator {actuator.name} has invalid config entry of {k}. "
                                f"Must be on of {cont_act_config_keys}")

                    zero_centered = actuator_config.get('zero_centered', False)
                    post_activation = actuator_config.get('post_activation', "clip")
                    if ('rescale' not in actuator_config) or (actuator_config['rescale']):
                        preprocessor = ActionPreprocessorContinuousRescale(
                            actuator_name, [bounds_min, bounds_max], zero_centered=zero_centered,
                            post_activation=post_activation)

                        actuator_action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
                    else:
                        assert not zero_centered, \
                            f"zero centered not supported without rescale for actuator {actuator.name}"
                        assert not post_activation, \
                            f"post_activation not supported without rescale for actuator {actuator.name}"

                        preprocessor = ActionPreprocessorPassThrough(actuator_name)
                        actuator_action_space = gym.spaces.Box(low=bounds_min, high=bounds_max, shape=(1,))
                elif actuator_config['space'] == 'discrete':
                    # if actuator in continuous but config is discrete, discretize actuator bounds
                    vals = np.linspace(bounds_min, bounds_max, actuator_config['points'])
                    preprocessor = ActionPreprocessorDiscreteMap(actuator_name, vals)
                    actuator_action_space = gym.spaces.Discrete(actuator_config['points'])
                else:
                    raise ValueError(
                        "Action Config for Actuator {} has invalid space of {}. "
                        "Should be 'continuous' or 'discrete'".format(
                            actuator.name, actuator_config['space']))

            elif actuator.space == 'discrete':
                # if the actuator is discrete, ignore actuator config.
                # Use actuator defined points and pass through value to control
                raise NotImplementedError

            else:
                raise ValueError(
                    "Actuator {} has invalid space of {}. Should be 'continuous' or 'discrete'".format(actuator.name,
                                                                                                       actuator.space))

            # append actuator action space and preprocessor
            action_space_tup += (actuator_action_space,)
            action_preprocessors.append(preprocessor)

        action_space = gym.spaces.Tuple(action_space_tup)

        return action_preprocessors, action_space

    def gen_actuation(self, state, action=None):
        actuation = {}

        if action is not None:
            for i, action_val in enumerate(action):
                if action_val is not None:

                    if not isinstance(action_val, np.ndarray):
                        action_val = np.array(action_val, ndmin=1)

                    actuator_name, action_processed = self.action_preprocessors[i](action_val)
                    actuation[actuator_name] = action_processed

        return actuation


class ActionPreprocessor(abc.ABC):
    def __init__(self, name):
        self.name = name

    @abc.abstractmethod
    def preprocess(self, action):
        raise NotImplementedError

    def __call__(self, action):
        return copy.deepcopy(self.name), self.preprocess(action)


class ActionPreprocessorPassThrough(ActionPreprocessor):

    def preprocess(self, action):
        return action


class ActionPreprocessorContinuousRescale(ActionPreprocessor):
    def __init__(self, name, bounds, zero_centered=False, post_activation="clip"):
        self.bounds = bounds
        self.zero_centered = zero_centered

        if post_activation == "clip":
            self.post_activation = lambda x: np.clip(x, -1, 1)
        elif post_activation == "tanh":
            self.post_activation = np.tanh
        else:
            raise ValueError("Invalid value for post_activation. Must be [None, 'tanh']")

        if self.zero_centered:
            assert bounds[0] <= 0 and bounds[1] >= 0, f"bounds {bounds} must include 0 to use zero_centered"
        super().__init__(name)

    def preprocess(self, action):
        low = self.bounds[0]
        high = self.bounds[1]

        action = self.post_activation(action)

        if self.zero_centered:
            if action < 0:
                scaled_action = -low * action
            if action >= 0:
                scaled_action = high * action
        else:
            scaled_action = low + (action + 1.0) * (high - low) / 2.0

        return scaled_action


class ActionPreprocessorDiscreteMap(ActionPreprocessor):

    def __init__(self, name, vals):
        self.vals = vals
        super().__init__(name)

    def preprocess(self, action):
        return self.vals[action]


class BaseActuatorSet:
    def __init__(self, actuators):
        self.actuators = actuators

        self.name_idx_map = {}
        for i, actuator in enumerate(self.actuators):
            self.name_idx_map[actuator.name] = i

    def gen_control(self, actuation=None):
        control_list = []

        if actuation is None:
            actuation = {}

        for actuator in self.actuators:
            actuator_name = actuator.name

            if actuator_name in actuation:
                actuator_control = actuation[actuator_name]
            else:
                actuator_control = actuator.default

            control_list.append(actuator_control)

        control = np.concatenate(control_list)

        return control


class BasePlatform(BaseEnvObj):

    def __init__(self, name, dynamics, actuator_set, state, controller, rta=None):

        if controller is None:
            controller = PassThroughController()
        elif type(controller) == dict:
            controller = controller["class"](actuator_set, config=controller)

        super().__init__(name)
        self.action_space = controller.action_space

        self.dependent_objs = []

        self.dynamics = dynamics
        self.actuator_set = actuator_set
        self.controller = controller
        self.state = state
        self.next_state = self.state

        # setup rta module with reference to self
        self.rta = rta
        if type(self.rta) == dict:
            if 'config' in self.rta:
                rta_kwargs = self.rta['config']
            else:
                rta_kwargs = {}
            self.rta = self.rta['class'](**rta_kwargs)
        if self.rta is not None:
            self.rta.setup(self)

        self.reset()

    def reset(self, **kwargs):
        self.state.reset(**kwargs)
        self.next_state = self.state

        self.current_actuation = {}
        self.current_control = self.actuator_set.gen_control()
        self.untrimmed_control = copy.deepcopy(self.current_control)

        for obj in self.dependent_objs:
            obj.reset(**kwargs)

    def step(self, sim_state, step_size, action=None):
        self.step_compute(sim_state, step_size, action=action)
        self.step_apply()

    def step_compute(self, sim_state, step_size, action=None):

        actuation = self.controller.gen_actuation(self.state, action)
        control = self.actuator_set.gen_control(actuation)
        
        #print(f"actuation was {actuation}, control is {control} (platforms.py)")

        if self.rta is not None:
            control = self.rta.filter_control(sim_state, step_size, control)

        # OLD: save current actuation and control
        #self.current_actuation = copy.deepcopy(actuation)
        self.untrimmed_control = copy.deepcopy(control)

        # compute new state if dynamics were applied
        self.next_state = self.dynamics.step(step_size, copy.deepcopy(self.state), control)
        
        #print(f"control was {self.current_control}, is now {control}")
        
        # New (after trimming): save current actuation and control
        self.current_actuation = copy.deepcopy(actuation)
        self.current_control = copy.deepcopy(control)

        for obj in self.dependent_objs:
            obj.step_compute(sim_state, action=action)

    def step_apply(self):

        # overwrite platform state with new state from dynamics
        self.state = self.next_state

        for obj in self.dependent_objs:
            obj.step_apply()

    def register_dependent_obj(self, obj):
        self.dependent_objs.append(obj)

    def generate_info(self):
        info = {
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'controller': {
                'actuation': self.current_actuation,
                'control': self.current_control,
                'untrimmed_control': self.untrimmed_control,
            }
        }

        if self.rta is not None:
            info['rta'] = self.rta.generate_info()

        return info

    @property
    def x(self):
        return self.state.x

    @property
    def y(self):
        return self.state.y

    @property
    def z(self):
        return self.state.z

    @property
    def position(self):
        return self.state.position

    @property
    def orientation(self):
        return self.state.orientation

    @property
    def velocity(self):
        return self.state.velocity


class BasePlatformState(BaseEnvObj):

    def __init__(self, **kwargs):
        self.reset(**kwargs)

    @abc.abstractmethod
    def reset(self, **kwargs):
        raise NotImplementedError


class BasePlatformStateVectorized(BasePlatformState):

    def reset(self, vector=None, vector_deep_copy=True, **kwargs):
        if vector is None:
            self._vector = self.build_vector(**kwargs)
        else:
            assert isinstance(vector, np.ndarray)
            assert vector.shape == self.vector_shape
            if vector_deep_copy:
                self._vector = copy.deepcopy(vector)
            else:
                self._vector = vector

    @abc.abstractmethod
    def build_vector(self):
        raise NotImplementedError

    @property
    def vector_shape(self):
        return self.build_vector().shape

    @property
    def vector(self):
        return copy.deepcopy(self._vector)

    @vector.setter
    def vector(self, value):
        self._vector = copy.deepcopy(value)


class BaseDynamics(abc.ABC):

    @abc.abstractmethod
    def step(self, step_size, state, control):
        raise NotImplementedError


class BaseODESolverDynamics(BaseDynamics):

    def __init__(self, integration_method='Euler'):
        self.integration_method = integration_method
        super().__init__()

    @abc.abstractmethod
    def dx(self, t, state_vec, control):
        raise NotImplementedError

    def step(self, step_size, state, control):

        if self.integration_method == "RK45":
            sol = scipy.integrate.solve_ivp(self.dx, (0, step_size), state.vector, args=(control,))

            state.vector = sol.y[:, -1]  # save last timestep of integration solution
        elif self.integration_method == 'Euler':
            state_dot = self.dx(0, state.vector, control)
            state.vector = state.vector + step_size * state_dot
        else:
            raise ValueError("invalid integration method '{}'".format(self.integration_method))

        return state


class BaseLinearODESolverDynamics(BaseODESolverDynamics):

    def __init__(self, integration_method='Euler'):
        self.A, self.B = self.gen_dynamics_matrices()
        super().__init__(integration_method=integration_method)

    @abc.abstractmethod
    def gen_dynamics_matrices(self):
        raise NotImplementedError

    def update_dynamics_matrices(self, state_vec):
        pass

    def dx(self, t, state_vec, control):
        self.update_dynamics_matrices(state_vec)
        dx = np.matmul(self.A, state_vec) + np.matmul(self.B, control)
        return dx

    def step(self, step_size, state, control):
        return super().step(step_size, state, control)
