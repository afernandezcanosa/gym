import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from gym.envs.connected_vehicles.assets.utils import load_dc, calc_mpg
from gym.envs.connected_vehicles.assets.conversions import Conversions as CV

class ConVehLongControl(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self):
        
        # Parameters of the host vehicle
        self.mass = 2800 * CV.LBS_TO_KGS # kg
        self.front_area = 2.5 # m^2
        self.drag_coeff = 0.32 # [-]
        self.air_density = 1.184 # kg/m^3
        self.mu = 0.013
        self.g = 9.81 # m/s^2
        self.tire_radius = 0.35 # m      
        self.min_torque = -1000 # N*m
        self.max_torque = 1400  # N*m 

        # Min and max normalized control inputs
        self.u_min = self.min_torque/(self.mass*self.tire_radius)
        self.u_max = self.max_torque/(self.mass*self.tire_radius)
        # Relative speeds limits
        self.dv_min_threshold = -15 * CV.MPH_TO_MPS
        self.dv_max_threshold = 15 * CV.MPH_TO_MPS
        # Max and minimum relative distances (thresholds)
        self.dx_min_threshold = 10
        self.dx_max_threshold = 50
        self.v_min = 0
        self.v_max = 80 * CV.MPH_TO_MPS
              
        self.dv_min = self.dv_min_threshold
        self.dv_max = self.dv_max_threshold
        self.dx_min = self.dx_min_threshold
        self.dx_max = self.dx_max_threshold
        
        # Action space box: normalized control inputs
        self.action_space = spaces.Box(low=self.u_min, high=self.u_max, shape=(1,), dtype=np.float32)      
        # Observation space
        low  = np.array([self.dv_min, self.dx_min])
        high = np.array([self.dv_max, self.dx_max])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)        

        high = np.array([1., 1., self.max_speed])
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,u):
        th, thdot = self.state # th := theta

        g = 10.
        m = 1.
        l = 1.
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u # for rendering
        costs = angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)

        newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
        newth = th + newthdot*dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111

        self.state = np.array([newth, newthdot])
        return self._get_obs(), -costs, False, {}

    def reset(self):
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self, mode='human'):

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0,0,0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi/2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u/2, np.abs(self.last_u)/2)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
