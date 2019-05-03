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
    def __init__(self, world, K=2, delT=0.1, delT_phase=0.5, mass=1.0,
                    mu=0.9, lamb=10.e-3, phase_weights=[PhaseWeights(w_CI=0.1, w_physics=0.1, w_kinematics=0.0, w_task=1.0),
                    PhaseWeights(w_CI=10., w_physics=1., w_kinematics=0., w_task=10.)], init_traj=None):
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
        mu : float, optional
            the coefficient of friction
        lamb : float, optional
            a regularizer that keeps accelerations and applied forces small
        phase_weights : list of PhaseWeights, optional
            the list of weights used during each optimization phase. the length of this list represents the number of optimization phases
        init_traj : function that takes in an initial state
        """
        self.K = K
        self.delT = delT
        self.delT_phase = delT_phase
        self.mass = mass
        self.mu = mu
        self.lamb = lamb
        self.phase_weights = phase_weights

        ## DERIVED PARAMETERS
        self.N = len(world.contact_state)
        self.steps_per_phase = int(self.delT_phase/self.delT)
        self.T_steps = self.K*self.steps_per_phase
        self.T_final = self.K*self.delT_phase
        # each dynamic object has a 2D pose and vel and each contact surface has 5 associated vars
        self.len_s = int(6*len(world.get_all_objects()) + self.N*5)
        # add accelerations of dynamic objects
        self.len_s_aug = int(self.len_s + 3.*len(world.get_all_objects()))
        self.len_S = int(self.len_s*self.K)
        self.len_S_aug = int(self.len_s_aug*self.K*self.steps_per_phase)

    def print_phase_weights(self, phase):
        print('PHASE PARAMETERS:')
        print('     w_CI:', self.phase_weights[phase].w_CI)
        print('     w_kinematics:', self.phase_weights[phase].w_kinematics)
        print('     w_physics:', self.phase_weights[phase].w_physics)
        print('     w_task:', self.phase_weights[phase].w_task)
        print('     lambda:', self.lamb)
