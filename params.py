import numpy as np
#### PARAMETERS #### THIS IS REALLY UGLY
# this is the order of the parameters
"""
K: number of phases in trajectory
delT: fine time discretization
delT_phase: time length of a phase

N: number of contact surfaces
mass: mass
gravity: gravity
mu: friction coefficient

accel_lamb: coefficient for acceleration cost
cont_lamb: coefficient for small contact forces cost
cone_lamb: coefficient for forces to lie in the friction cone
phase_weights: list of weights for the cost functions in this order(Lci, Lphys, Ltask)

the rest are derived:
steps_per_phase
T_steps
T_final
len_s
len_s_aug
len_S
len_S_aug

"""
class Params(object):
    def __init__(self, test_params = {}, num_moveable_objects = 3):
        self.default_params = {'K': 5, 'delT': 0.1, 'delT_phase': 0.5, 'N': 3, \
                                'mass': 1.0, 'gravity': 10.0, 'mu': 0.1, \
                                'accel_lamb': 1.e-3, 'cont_lamb': 1.e-7, \
                                'cone_lamb': 1.e-3,\
                                'phase_weights': [[0.0, 0.0, 1.0],[0.0, 1.0, 1.0]], \
                                'num_objs': num_moveable_objects}
        self.test_params = test_params
        self.set_test_params()
        self.set_default_params()

    def set_test_params(self):
        for param in self.test_params.keys():
            setattr(self, param, self.test_params[param])

    def get_param_values(self):
        out = []
        for param in self.default_params.keys():
            if param in self.test_params:
                self.test_params[param]
            else:
                self.default_params[param]
        return out

    def set_default_params(self):
        for param in self.default_params:
            if not hasattr(self, param):
                setattr(self, param, self.default_params[param])

        # set derived params
        self.steps_per_phase = int(self.delT_phase/self.delT)
        self.T_steps = self.K*self.steps_per_phase
        self.T_final = self.K*self.delT_phase
        self.len_s = int(6*self.num_objs + self.N*5) # each contact surface has 5 associated vars
        self.len_s_aug = int(self.len_s + 3.*self.num_objs) # add accelerations of objects
        self.len_S = int(self.len_s*self.K)
        self.len_S_aug = int(self.len_s_aug*self.K*self.steps_per_phase)

        # objective function weights
        #self.ci_lamb = 1.0       # contact invariance (have to be touching object to activate contact force)
        #self.kin_lamb = .1       # object collisisions
        #self.cone_lamb = 1.0     # contact force in friction cone
        #self.vel_lamb = 1.0      # velocities have to match change in poses

def set_global_params(p):
    global K, delT, delT_phase, N, mass, gravity, mu, accel_lamb, cont_lamb, cone_lamb, phase_weights
    global steps_per_phase, T_steps, T_final, len_s, len_s_aug, len_S, len_S_aug, num_objs, start_phase
    K = p.K
    delT = p.delT
    delT_phase = p.delT_phase
    N = p.N
    mass = p.mass
    gravity = p.gravity
    mu = p.mu
    accel_lamb = p.accel_lamb
    cont_lamb = p.cont_lamb
    cone_lamb = p.cone_lamb
    phase_weights = p.phase_weights
    num_objs = p.num_objs
    # derived
    steps_per_phase = p.steps_per_phase
    T_steps = p.T_steps
    T_final = p.T_final
    len_s = p.len_s
    len_s_aug = p.len_s_aug
    len_S = p.len_S
    len_S_aug = p.len_S_aug
    if hasattr(p, 'start_phase'):
        start_phase = p.start_phase
    else:
        start_phase = 0

# NOT derived ones... don't need to store those
def get_global_params():
    return [K, delT, delT_phase, N, mass, gravity, mu, accel_lamb, cont_lamb, cone_lamb, phase_weights, num_objs]

# define global variables which will be used
K = None
delT = None
delT_phase = None
N = None
mass = None
gravity = None
mu = None
accel_lamb = None
cont_lamb = None
cone_lamb = None
phase_weights = None
num_objs = None
# derived
steps_per_phase = None
T_steps = None
T_final = None
len_s = None
len_s_aug = None
len_S = None
len_S_aug = None
start_phase = None
