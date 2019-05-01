import numpy as np
from collections import namedtuple

PhaseWeights = namedtuple('PhaseWeights', 'w_CI w_physics w_kinematics w_task')
"""
Parameters
----------
w_CI : float
    the weight for the L_CI function
w_physics : float
    the weight for the L_physics function
w_kinematics : float
    the weight for the L_kinematics function
w_task : float
    the weight for the L_task function
"""

class Params(object):
    def __init__(self, world, K=5, delT=0.1, delT_phase=0.5, mass=1.0, gravity=10.0,
                    mu=0.3, lamb=1.e-3, phase_weights=[PhaseWeights(0.0, 0.0, 0.0, 1.0), \
                    PhaseWeights(0.0, 1.0, 0.0, 1.0)]):
        """
        Parameters
        ----------
        world : world.World
            an object representing the objects in the world
        K : int, optional
            the number of keyframes used to represent the trajectory
        delT : float, optional
            the time step used to calculate derivates using finite differences
        delT_phase : float, optional
            the length of time a keyframe lasts
        mass : float, optional
            the mass (kg) of the manipulated object
        gravity : float, optional
            the magnitude of the gravitational force
        mu : float, optional
            the coefficient of friction
        lamb : float, optional
            a regularizer that keeps accelerations and applied forces small
        phase_weights : list of PhaseWeights, optional
            the list of weights used during each optimization phase. the length of this list represents the number of optimization phases
        """
        self.K = K
        self.delT = delT
        self.delT_phase = delT_phase
        self.mass = mass
        self.gravity = gravity
        self.mu = mu
        self.lamb = lamb
        self.phase_weights = phase_weights

        ## DERIVED PARAMETERS
        self.N = world.get_num_contact_objects()
        self.steps_per_phase = int(self.delT_phase/self.delT)
        self.T_steps = self.K*self.steps_per_phase
        self.T_final = self.K*self.delT_phase
        # each dynamic object has a 2D pose and vel and each contact surface has 5 associated vars
        self.len_s = int(6*world.get_num_dynamic_objects() + self.N*5)
        # add accelerations of dynamic objects
        self.len_s_aug = int(self.len_s + 3.*world.get_num_dynamic_objects())
        self.len_S = int(self.len_s*self.K)
        self.len_S_aug = int(self.len_s_aug*self.K*self.steps_per_phase)

    def print_phase_weights(self, phase):
        print('PHASE PARAMETERS:')
        print('     w_CI:', self.phase_weights[phase].w_CI)
        print('     w_kinematics:', self.phase_weights[phase].w_kinematics)
        print('     w_physics:', self.phase_weights[phase].w_physics)
        print('     w_task:', self.phase_weights[phase].w_task)
        print('     lambda:', self.lamb)
