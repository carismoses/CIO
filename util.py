import numpy as np

#### PARAMETERS ####
N_contacts = 3 # 2 grippers and 1 ground contact
T_final = 10 # time steps to optimize over
delT = 1
mass = 1.0 # mass
gravity = 10.0 # gravity
mu = 0.5 # friction coefficient
len_s = 24
len_s_aug = 24 + 18 # includes vels and accels
len_S = len_s*(T_final-1)
len_S_aug = len_s_aug*(T_final-1) # will be more when interpolate between s values
task_lamb = .1 # L_task parameter (weigh accelerations)
col_lamb = .1 # L_kinematics parameter (weight object collisisions)
phys_lamb = 1.0 # L_physics parameter (weigh contact forces)
small_ang = .25
hw, hh = 5.0, 5.0 # half-width, half-height

#### GET FUNCTIONS ####
def get_gripper1_pos(s):
    return s[0:3]

def get_gripper2_pos(s):
    return s[3:6]

def get_object_pos(s):
    return s[6:9]

def get_contact_info(s):
    fj = np.zeros((N_contacts,2))
    roj = np.zeros((N_contacts,2))
    cj = np.zeros(N_contacts)

    for j in range(N_contacts):
        fj[j,:] = [s[9 + j*5], s[10 + j*5]]
    for j in range(N_contacts):
        roj[j,:] = [s[11 + j*5], s[12 + j*5]]
    for j in range(N_contacts):
        cj[j] = s[13 + j*5]
    return fj, roj, cj

def get_gripper1_vel(s):
    return s[24:27]

def get_gripper1_accel(s):
    return s[27:30]

def get_gripper2_vel(s):
    return s[30:33]

def get_gripper2_accel(s):
    return s[33:36]

def get_object_vel(s):
    return s[36:39]

def get_object_accel(s):
    return s[39:42]

#### GET INDICE FUNCTIONS ####
def get_gripper1_pos_ind():
    return 0,3

def get_gripper2_pos_ind():
    return 3,6

def get_object_pos_ind():
    return 6,9

def get_contact_info_ind():
    return 9,24

def get_gripper1_vel_ind():
    return 24,27,

def get_gripper1_accel_ind():
    return 27,30

def get_gripper2_vel_ind():
    return 30,33

def get_gripper2_accel_ind():
    return 33,36

def get_object_vel_ind():
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
            # initially vel and accel are zero
            s_tm1 = np.concatenate([s0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, \
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        else:
            s_tm1 = get_s_aug_t(S_aug, t-1)
        # current time step
        s_t = get_s_t(S, t)

        # gripper1
        g1vel_t = (get_gripper1_pos(s_t) - get_gripper1_pos(s_tm1))/delT
        g1accel_t = (g1vel_t - get_gripper1_vel(s_tm1))/delT

        # gripper2
        g2vel_t = (get_gripper2_pos(s_t) - get_gripper2_pos(s_tm1))/delT
        g2accel_t = (g2vel_t - get_gripper2_vel(s_tm1))/delT

        # object
        ovel_t = (get_object_pos(s_t) - get_object_pos(s_tm1))/delT
        oaccel_t = (ovel_t - get_object_vel(s_tm1))/delT

        s_aug_t = np.concatenate([s_t, g1vel_t, g1accel_t, g2vel_t, g2accel_t, ovel_t, oaccel_t])
        S_aug[(t-1)*len_s_aug:(t-1)*len_s_aug+len_s_aug] = s_aug_t

    return S_aug

def get_s_aug_t(S_aug, t):
    return S_aug[(t-1)*len_s_aug:(t-1)*len_s_aug+len_s_aug]

def get_s_t(S, t):
    return S[(t-1)*len_s:(t-1)*len_s+len_s]
