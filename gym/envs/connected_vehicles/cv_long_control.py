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
        'video.frames_per_second' : 10
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
        self.dx_max_threshold = 30
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
        
        self.seed()
        self.viewer = None
        self.dt = 1.0 # sec
        
        # Load driving cycles of the lead vehicle
        df_lead = load_dc(self.dt)
        self.speed_lead = df_lead['speed_mph'] * CV.MPH_TO_MPS
        self.dist_lead = df_lead['distance_m']
        
        
        # Properties of the numerical_scheme
        self.p = len(df_lead)
        self.k = int(self.np_random.uniform(0, int(self.p/2), size=(1,)))

               
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def step(self, action):
        dv, dx = self.state
        self.v_lead = self.speed_lead.iloc[self.k]
        self.x_lead = self.dist_lead.iloc[self.k]
        v = self.v_lead - dv
        x = self.x_lead - dx
#        print('x = %.2f, v = %.2f' %(x, v))

        # Action
        u = np.clip(action, self.u_min, self.u_max)[0]
        aero = (0.5/self.mass)*self.air_density*self.drag_coeff*self.front_area*v**2
        grade = 0.
        rolling = self.mu*self.g 
        road_load = aero + rolling + grade
        a = u - road_load
        
        # Temporal step
        x = x + self.dt * v
        v = v + self.dt * a  
#        print('Step number: %i' %self.k)
#        print('dx = %.2f, dv = %.2f' %(dx, dv))
#        print('speed = %.2f, accel = %.2f' %(v, a))
   
        # Next step
        self.k+=1
        self.x_lead = self.dist_lead.iloc[self.k]
        self.v_lead = self.speed_lead.iloc[self.k]      
        dv = self.v_lead - v
        dx = self.x_lead - x

        done =  dx < self.dx_min_threshold \
                or dx > self.dx_max_threshold \
                or v < self.v_min \
                or v > self.v_max
        done = bool(done)
        
        if not done:
            reward = calc_mpg(v, a) + 0*self.k
        else:
            reward = -100
            
            
        self.state = np.array([dv, dx])
        
        return self.state, reward, done, {}

    def reset(self):
        self.k = int(self.np_random.uniform(0, int(self.p/2), size=(1,)))
        self.v_lead = self.speed_lead.iloc[self.k]
        self.x_lead = self.dist_lead.iloc[self.k]   
        low = np.array([self.dv_min, self.dx_min])
        high = np.array([self.dv_max, self.dx_max])
        self.state = self.np_random.uniform(low=low, high=high, size=(2,))
        dv, dx = self.state
        self.state = np.array([dv, dx])
        return self.state

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400
        
        world_width = (self.dx_max_threshold + 10) + (self.dx_min_threshold + 10)
        scale = screen_width/world_width
        cary = 200 # TOP OF CAR
        
        # Size of following vehicle  
        car_width = 70.0
        car_height = 40.0
        
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -car_width/2, car_width/2, car_height/2, -car_height/2
            car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            car.set_color(.8, .3, .3)
            self.car_trans = rendering.Transform()
            car.add_attr(self.car_trans)
            self.viewer.add_geom(car)     
            
            l,r,t,b = -2, 2, screen_height, -screen_height
            line1 = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            line1.set_color(.0, .0, .0)
            self.line1_trans = rendering.Transform()
            line1.add_attr(self.line1_trans)
            self.viewer.add_geom(line1)  
            
            l,r,t,b = -2, 2, screen_height, -screen_height
            line2 = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            line2.set_color(.0, .0, .0)
            self.line2_trans = rendering.Transform()
            line2.add_attr(self.line2_trans)
            self.viewer.add_geom(line2)   
            
        if self.state is None: return None

        offset = 50
        x = self.state
        carx = screen_width - x[1]*scale 
        self.car_trans.set_translation(carx-offset, cary)  
        
        self.line1_trans.set_translation(screen_width -2 - offset, 0)
        self.line2_trans.set_translation(screen_width-self.dx_max*scale - 2 - offset, 0)
        
        
        return self.viewer.render(return_rgb_array = mode=='rgb_array')             
        
        
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        