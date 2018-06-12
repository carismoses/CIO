import numpy as np
import pdb

#### PARAMETERS ####
K = 5 # phases to optimize over
delT_phase = 0.5
delT = 0.1
steps_per_phase = int(delT_phase/delT)
T_steps = K*steps_per_phase
len_s = 33
len_s_aug = len_s + 9 # includes accelerations
len_S = len_s*K
len_S_aug = len_s_aug*K*steps_per_phase

# physical parameters
N = 3 # 2 grippers and 1 ground contact
mass = 1.0 # mass
gravity = 10.0 # gravity
mu = 0.1 # friction coefficient

# objective function weights
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
    fj = np.zeros((N,2))
    roj = np.zeros((N,2))
    cj = np.zeros(N)

    i = get_fj_ind()
    k = get_roj_ind()
    l = get_contact_ind()

    for j in range(N):
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

#### SET FUNCTIONS ####
def set_fj(fj, s):
    i = get_fj_ind()
    for j in range(N):
        s[i[j][0]:i[j][1]] = fj[j,:]
    return s

def set_roj(roj, s):
    i = get_roj_ind()
    for j in range(N):
        s[i[j][0]:i[j][1]] = roj[j,:]
    return s

def set_contact(c, s):
    i,j,k = get_contact_ind()
    s[i], s[j], s[k] = c
    return s

#### AUGMENT DECISION VARIABLE ####
def augment_s(s0, S):
    S_aug = np.zeros(len_S_aug)

    # interpolate contacts, contact forces, poses and velocities
    for k in range(1, K+1):
        # get enpoints of interpolation phase
        if k == 0:
            s_left = s0
        else:
            s_left = get_s(S, k-1)
        s_right = get_s(S, k)

        # get s values at endpoints
        fj_left, roj_left, cj_left = get_contact_info(s_left)
        fj_right, roj_right, cj_right = get_contact_info(s_right)

        # interpolate the contact forces (linear)
        fjs = linspace_matrices(fj_left, fj_right, steps_per_phase)

        # interpolate the contact poses (linear)
        rojs = linspace_matrices(roj_left, roj_right, steps_per_phase)

        for p in range(steps_per_phase):
            s_aug = np.zeros(len_s_aug)
            # add 1 since t = 0 corresponds to s0
            t = k*steps_per_phase + p + 1

            # interpolate contacts (constant)
            s_aug = set_contact(cj_right, s_aug)

            # interpolate contact forces (linear)
            s_aug = set_fj(fjs[p,:,:], s_aug)

            # interpolate contact poses (linear)
            s_aug = set_roj(rojs[p,:,:], s_aug)

        """
        # if k == 0 use initial accel, otherwise use last calculated ones
        # previous time step
        if t == 1:
            # t=0 accels are zero
            s_tm1 = np.concatenate([s0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        else:
            s_tm1 = get_s_aug_t(S_aug, t-1) # fix
        # current time step
        s_t = get_s_t(S, t) # fix

        # get accelerations
        g1accel_t = calc_accel(get_gripper1_vel(s_t), get_gripper1_vel(s_tm1))
        g2accel_t = calc_accel(get_gripper2_vel(s_t), get_gripper2_vel(s_tm1))
        oaccel_t = calc_accel(get_object_vel(s_t), get_object_vel(s_tm1))

        s_aug_t = np.concatenate([s_t, g1accel_t, g2accel_t, oaccel_t])
        S_aug[(t-1)*len_s_aug:(t-1)*len_s_aug+len_s_aug] = s_aug_t
        """
        
    return S_aug

def get_s(S, t):
    if len(S) == len_S:
        return S[t*len_s:t*len_s+len_s]
    else:
        return S[t*len_s_aug:t*len_s_aug+len_s_aug]

def calc_deriv(x1, x0):
    return (x1 - x0)/delT

#### HELPER FUNCTIONS ####
def get_bounds():
    contact_ind = get_contact_ind()
    ground_ind = get_fj_ind()[2][0]
    bounds = []
    for t in range(K):
        for v in range(len_s):
            if v in contact_ind:
                bounds.append((0.,1.)) # c's are between 0 and 1
            elif v == ground_ind:
                bounds.append((0.,0.)) # no x direction forces from the ground allowed
            else:
                bounds.append((None,None))
    return bounds

def linspace_matrices(mat0, mat1, num_steps):
    w = len(mat0[0])
    h = len(mat0)
    out_mat = np.zeros((num_steps, h, w))
    for j in range(h):
        for i in range(w):
            left = mat0[j,i]
            right = mat1[j,i]
            out_mat[:,j,i] = np.linspace(left, right, num_steps)
    return out_mat
