import pandas as pd
from gym.envs.connected_vehicles.assets.conversions import Conversions as CV
import sys
from os.path import dirname, abspath, join

sys.path.insert(0,dirname(dirname(dirname(abspath(__file__)))))
base_path = dirname(abspath(__file__))

def load_dc(dt):
    df = pd.read_csv(join(base_path,'driving_cycles.csv'))
    dt_old = 1.0
    return df[0:-1:int(dt/dt_old)].reset_index(drop = True)

def calc_mpg(v, a):
    
    b0 = 0.1569
    b1 = 0.02450
    b2 = -0.0007415
    b3 = 0.00005975
    
    c0 = 0.07224
    c1 = 0.09681
    c2 = 0.001075
    
    Fd = 0.1
    
    f_cruise = b0 + b1*v + b2*v**2 + b3*v**3
    f_accel = a*(c0 + c1*v + c2*v**2)
    
    if a < 0 or v == 0:
        zeta = 1
    else:
        zeta = 0       
    mf = (1 - zeta)*(f_cruise + f_accel) + zeta*Fd
    
    mf_gal_h = mf * CV.ML_TO_GAL * 3600
    v = v * CV.MPS_TO_MPH
    
    return v/mf_gal_h

if __name__ == '__main__':
    
    mpg=calc_mpg(4.16, -2.11)