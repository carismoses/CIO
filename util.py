import numpy as np
import pdb
from scipy.interpolate import BPoly

np.random.seed(0)

#### GET FUNCTIONS ####
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

def get_contact_info(s,p):
    fj = np.zeros((p.N,2))
    roj = np.zeros((p.N,2))
    cj = np.zeros(p.N)

    i = get_fj_ind()
    k = get_roj_ind()
    l = get_contact_ind()

    for j in range(p.N):
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
def set_object_pos(p, s):
    i = get_object_pos_ind()
    s[i[0]:i[1]] = p
    return s

def set_object_vel(v, s):
    i = get_object_vel_ind()
    s[i[0]:i[1]] = v
    return s

def set_object_accel(a, s):
    i = get_object_accel_ind()
    s[i[0]:i[1]] = a
    return s

def set_gripper1_pos(p, s):
    i = get_gripper1_pos_ind()
    s[i[0]:i[1]] = p
    return s

def set_gripper1_vel(v, s):
    i = get_gripper1_vel_ind()
    s[i[0]:i[1]] = v
    return s

def set_gripper1_accel(a, s):
    i = get_gripper1_accel_ind()
    s[i[0]:i[1]] = a
    return s

def set_gripper2_pos(p, s):
    i = get_gripper2_pos_ind()
    s[i[0]:i[1]] = p
    return s

def set_gripper2_vel(v, s):
    i = get_gripper2_vel_ind()
    s[i[0]:i[1]] = v
    return s

def set_gripper2_accel(a, s):
    i = get_gripper2_accel_ind()
    s[i[0]:i[1]] = a
    return s

def set_fj(fj, s, p):
    i = get_fj_ind()
    for j in range(p.N):
        s[i[j][0]:i[j][1]] = fj[j,:]
    return s

def set_roj(roj, s, p):
    i = get_roj_ind()
    for j in range(p.N):
        s[i[j][0]:i[j][1]] = roj[j,:]
    return s

def set_contact(c, s):
    i,j,k = get_contact_ind()
    s[i], s[j], s[k] = c
    return s

#### AUGMENT DECISION VARIABLE ####
def augment_s(s0, S, p):
    S_aug = np.zeros(p.len_S_aug)

    # perform spline interpolation and get velocities and accelerations for all objects
    # object
    set_funcs = set_object_pos, set_object_vel, set_object_accel
    get_funcs = get_object_pos, get_object_vel
    S_aug = interpolate_poses(s0, S, S_aug, set_funcs, get_funcs, p)

    # gripper1
    set_funcs = set_gripper1_pos, set_gripper1_vel, set_gripper1_accel
    get_funcs = get_gripper1_pos, get_gripper1_vel
    S_aug = interpolate_poses(s0, S, S_aug, set_funcs, get_funcs, p)

    # gripper2
    set_funcs = set_gripper2_pos, set_gripper2_vel, set_gripper2_accel
    get_funcs = get_gripper2_pos, get_gripper2_vel
    S_aug = interpolate_poses(s0, S, S_aug, set_funcs, get_funcs, p)

    # interpolate contacts, contact forces, and contact poses
    for k in range(p.K):
        # get enpoints of interpolation phase
        if k == 0:
            s_left = s0
        else:
            s_left = get_s(S, k-1, p)
        s_right = get_s(S, k, p)

        # get s values at endpoints
        fj_left, roj_left, cj_left = get_contact_info(s_left,p)
        fj_right, roj_right, cj_right = get_contact_info(s_right,p)

        # interpolate the contact forces (linear)
        fjs = linspace_matrices(fj_left, fj_right, p.steps_per_phase+1)

        # interpolate the contact poses (linear)
        rojs = linspace_matrices(roj_left, roj_right, p.steps_per_phase+1)

        for ph in range(p.steps_per_phase):
            t = k*p.steps_per_phase + ph
            s_aug = get_s(S_aug, t, p)

            # interpolate contacts (constant)
            s_aug = set_contact(cj_right, s_aug)

            # interpolate contact forces (linear)
            s_aug = set_fj(fjs[ph+1,:,:], s_aug, p)

            # interpolate contact poses (linear)
            s_aug = set_roj(rojs[ph+1,:,:], s_aug, p)
            S_aug = set_s(S_aug, s_aug, t, p)

    return S_aug

def get_s(S, t, p):
    if len(S) == p.len_S:
        return S[t*p.len_s:t*p.len_s+p.len_s]
    else:
        return S[t*p.len_s_aug:t*p.len_s_aug+p.len_s_aug]

def set_s(S, s, t, p):
    if len(S) == p.len_S:
        S[t*p.len_s:t*p.len_s+p.len_s] = s
    else:
        S[t*p.len_s_aug:t*p.len_s_aug+p.len_s_aug] = s
    return S

def calc_deriv(x1, x0, delta):
    return (x1 - x0)/delta

#### HELPER FUNCTIONS ####
def get_bounds(p):
    contact_ind = get_contact_ind()
    ground_ind = get_fj_ind()[2][0]
    bounds = []
    for t in range(p.K):
        for v in range(p.len_s):
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

def cubic_spline_func(s0, S, dim, get_funcs, p):
    get_pos, get_vel = get_funcs
    x = np.linspace(0.,p.T_final, p.K+1)
    y = np.zeros((p.K+1,2))
    for k in range(p.K+1):
        if k == 0:
            y[k,:] = get_pos(s0)[dim], get_vel(s0)[dim]
        else:
            y[k,:] = get_pos(get_s(S,k-1,p))[dim], get_vel(get_s(S,k-1,p))[dim]

    f = BPoly.from_derivatives(x,y,orders=3, extrapolate=False)
    return f

def interpolate_poses(s0, S, S_aug, set_funcs, get_funcs, p):
    #pdb.set_trace()

    # get spline functions using the velocities as tangents
    set_pos, set_vel, set_accel = set_funcs
    get_pos, get_vel = get_funcs
    spline_funcs = []
    for dim in range(3):
        spline_funcs += [cubic_spline_func(s0, S, dim, get_funcs, p)]

    # calc poses using spline function
    i = 0
    j = 0
    times = np.linspace(0.,p.T_final, p.T_steps+1)
    for t in times:
        if i != 0:
            if not t % p.delT_phase: # this is a keyframe
                pose = get_pos(get_s(S,j,p))
                j += 1
            else: # get from spline
                pose = spline_funcs[0](t), spline_funcs[1](t), spline_funcs[2](t)
            s = get_s(S_aug, i-1, p)
            s = set_pos(pose, s)
            S_aug = set_s(S_aug, s, i-1, p)
        i += 1

    # use FD to get velocities and accels #TODO: check that vels are close to ones in dec vars
    for i in range(p.T_steps):
        if i == 0:
            x_tm1 = get_pos(s0)
        else:
            x_tm1 = get_pos(get_s(S_aug, i-1, p))
        x_t = get_pos(get_s(S_aug, i, p))
        v = calc_deriv(x_t, x_tm1, p.delT)
        s = get_s(S_aug, i, p)
        s = set_vel(v, s)
        S_aug = set_s(S_aug, s, i, p)

    for i in range(p.T_steps):
        if i == 0:
            v_tm1 = get_vel(s0)
        else:
            v_tm1 = get_vel(get_s(S_aug,i-1,p))
        v_t = get_vel(get_s(S_aug, i, p))
        a = calc_deriv(v_t, v_tm1, p.delT)
        s = get_s(S_aug, i, p)
        s = set_accel(a, s)
        S_aug = set_s(S_aug, s, i, p)

    return S_aug

def add_noise(vec):
    # perturb all vars by gaussian noise
    mean = 0.
    var = 10.e-2
    for j in range(len(vec)):
        vec[j] += np.random.normal(mean, var)

    return vec
