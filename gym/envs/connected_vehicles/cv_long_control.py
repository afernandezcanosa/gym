"""
Longitudinal control of one Connected Vehicle
"""

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
        
        # Properties of the numerical_scheme
        self.dt = 0.5 # sec
        self.kinematics_integrator = 'euler'
        self.k = 0
        
        self.seed()
        self.viewer = None
        self.state = None  
        
        self.steps_beyond_done = None
        
        # Load driving cycles of the lead vehicle
        self.df_drive_lead = load_dc(self.dt)
        self.lead_speed = self.df_drive_lead['speed_mph'] * CV.MPH_TO_MPS
        self.lead_dist = self.df_drive_lead['distance_m']
               
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def step(self, action):
        dv, dx = self.state
        v_lead = self.lead_speed[self.k] 
        x_lead = self.lead_dist[self.k]
        v = v_lead - dv
        x = x_lead - dx
        
        u = np.clip(action, self.u_min, self.u_max)
        
        aero = (0.5/self.mass)*self.air_density*self.drag_coeff*self.front_area*v**2
        grade = 0.
        rolling = self.mu*self.g
        
        a = u + aero + rolling + grade
        
        if self.kinematics_integrator == 'euler':
            x = x + self.dt * v
            v = v + self.dt * a            
        # Next step
        self.k+=1
        v_lead = self.lead_speed[self.k] 
        x_lead = self.lead_dist[self.k]
        dv = v_lead = v
        dx = x_lead - x
        
        done =  dx < self.dx_min_threshold \
                or dx > self.dx_max_threshold \
                or dv < self.dv_min_threshold \
                or dv > self.dx_max_threshold       
        done = bool(done)
        
        if not done:
            reward = calc_mpg(v, a)
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0 
            
        print(reward)
        self.state = (dv,dx)
        
        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)   

    def render(self, mode='human'):
        pass

    def close(self):
        pass
        
        