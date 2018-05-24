import numpy as np
import pdb

#### PARAMETERS ####
N_contacts = 3 # 2 grippers and 1 ground contact
T_final = 10 # time steps to optimize over
delT = 1.0
mass = 1.0 # mass
gravity = 10.0 # gravity
mu = 0.1 # friction coefficient
len_s = 33
len_s_aug = len_s + 9 # accelerations
len_S = len_s*(T_final-1)
len_S_aug = len_s_aug*(T_final-1) # will be more when interpolate between s values
small_ang = .25
hw, hh = 5.0, 5.0 # half-width, half-height

ci_lamb = 1.0       # contact invariance (have to be touching object to activate contact force)
kin_lamb = .1       # object collisisions
phys_lamb = 10.0    # physics have to make sense
cone_lamb = 1.0     # contact force in friction cone
cont_lamb = 1.0     # discourage large contact forces
vel_lamb = 1.0      # velocities have to match change in poses
task_lamb = 100.    # task (final position)
accel_lamb = .001   # small accelerations

#### GET FUNCTIONS ####
# TODO: call indice functions in here so don't have to change in 2 places...
def get_gripper1_pos(s):
    i,j = get_gripper1_pos_ind()
    return s[i:j]

def get_gripper1_vel(s):
    i,j = get_gripper1_vel_ind()
    return s[i:j]

def get_gripper2_pos(s):
    i,j = get_gripper2_pos_ind()
    return s[i:j]

def get_gripper2_vel(s):
    i,j = get_gripper2_vel_ind()
    return s[i:j]

def get_object_pos(s):
    i,j = get_object_pos_ind()
    return s[i:j]

def get_object_vel(s):
    i,j = get_object_vel_ind()
    return s[i:j]

def get_contact_info(s):
    fj = np.zeros((N_contacts,2))
    roj = np.zeros((N_contacts,2))
    cj = np.zeros(N_contacts)

    i = get_fj_ind()
    k = get_roj_ind()
    l = get_contact_ind()

    for j in range(N_contacts):
        fj[j,:] = s[i[j][0]:i[j][1]]
        roj[j,:] = s[k[j][0]:k[j][1]]
        cj[j] = s[l[j]]
    return fj, roj, cj

def get_gripper1_accel(s):
    i,j = get_gripper1_accel_ind()
    return s[i:j]

def get_gripper2_accel(s):
    i,j = get_gripper2_accel_ind()
    return s[i:j]

def get_object_accel(s):
    i,j = get_object_accel_ind()
    return s[i:j]

#### GET INDICE FUNCTIONS ####
def get_gripper1_pos_ind():
    return 0,3

def get_gripper1_vel_ind():
    return 3,6

def get_gripper2_pos_ind():
    return 6,9

def get_gripper2_vel_ind():
    return 9,12

def get_object_pos_ind():
    return 12,15

def get_object_vel_ind():
    return 15,18

def get_contact_info_ind():
    return 18,33

def get_fj_ind():
    return (18,20), (23,25), (28,30)

def get_roj_ind():
    return (20,22), (25,27), (30,32)

def get_contact_ind():
    return 22, 27, 32

def get_gripper1_accel_ind():
    return 33,36

def get_gripper2_accel_ind():
    return 36,39

def get_object_accel_ind():
    return 39,42

#### AUGMENT DECISION VARIABLE ####
def interpolate_s(s0, S):
    S_aug = np.zeros(len_S_aug)
    # should interpolate between S values (skip for now)
    for t in range(1,T_final):
        # if t == 0 use initial vel and accel, otherwise use last calculated ones
        # previous time step
        if t == 1:
            # t=0 accels are zero
            s_tm1 = np.concatenate([s0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        else:
            s_tm1 = get_s_aug_t(S_aug, t-1)
        # current time step
        s_t = get_s_t(S, t)

        # get accelerations
        g1accel_t = (get_gripper1_vel(s_t) - get_gripper1_vel(s_tm1))/delT
        g2accel_t = (get_gripper2_vel(s_t) - get_gripper2_vel(s_tm1))/delT
        oaccel_t = (get_object_vel(s_t) - get_object_vel(s_tm1))/delT

        s_aug_t = np.concatenate([s_t, g1accel_t, g2accel_t, oaccel_t])
        S_aug[(t-1)*len_s_aug:(t-1)*len_s_aug+len_s_aug] = s_aug_t

    return S_aug

def get_s_aug_t(S_aug, t):
    return S_aug[(t-1)*len_s_aug:(t-1)*len_s_aug+len_s_aug]

def get_s_t(S, t):
    return S[(t-1)*len_s:(t-1)*len_s+len_s]

#### HELPER FUNCTIONS ####
def get_bounds():
    contact_ind = get_contact_ind()
    ground_ind = get_fj_ind()[2][0]
    bounds = []
    for t in range(1,T_final):
        for v in range(len_s):
            if v in contact_ind:
                bounds.append((0.,1.)) # c's are between 0 and 1
            elif v == ground_ind:
                bounds.append((0.,0.)) # no x direction forces from the ground allowed
            else:
                bounds.append((None,None))
    return bounds
