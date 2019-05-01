import numpy as np

class Params(object):
    def __init__(self, num_moveable_objects, K=5, delT=0.1, delT_phase=0.5, N=3, mass=1.0, gravity=10.0,
                    mu=0.3, accel_lamb=1.e-3, cont_lamb=1.e-7, cone_lamb=1.e-3,
                    phase_weights=[[0.0, 0.0, 1.0],[0.0, 1.0, 1.0]]):
        """
        Parameters
        ----------
        num_objs : int
            the number of objects in the environment (does not include the ground)
        K : int, optional
            the number of keyframes used to represent the trajectory
        delT : float
            the time step used to calculate derivates using finite differences
        delT_phase : float, optional
            the length of time a keyframe lasts
        N : need to remove...

        mass : float, optional
            the mass (kg) of the manipulated object
        gravity : float, optional
            the magnitude of the gravity force
        mu : float, optional
            the coefficient of friction
        accel_lamb : float, optional
            the weight on the accel function in the optimization function
        cont_lamb : float, optional
            the weight on the cont function in the optimization function
        cone_lamb : float, optional
            the weight on the cone function in the optimization function
        phase_weights : list of PhaseWeights, optional
            the list of weights used during each optimization phase. the length of this list represents the number of optimization phases
        """
        self.num_objs = num_moveable_objects
        self.K = K
        self.delT = delT
        self.delT_phase = delT_phase
        self.N = N
        self.mass = mass
        self.gravity = gravity
        self.mu = mu
        self.accel_lamb = accel_lamb
        self.cont_lamb = cont_lamb
        self.cone_lamb = cone_lamb
        self.phase_weights = phase_weights
        
        # derived parameters
        self.steps_per_phase = int(self.delT_phase/self.delT)
        self.T_steps = self.K*self.steps_per_phase
        self.T_final = self.K*self.delT_phase
        self.len_s = int(6*self.num_objs + self.N*5) # each contact surface has 5 associated vars
        self.len_s_aug = int(self.len_s + 3.*self.num_objs) # add accelerations of objects
        self.len_S = int(self.len_s*self.K)
        self.len_S_aug = int(self.len_s_aug*self.K*self.steps_per_phase)
