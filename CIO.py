# CIO implementation
from scipy.optimize import fmin_l_bfgs_b
import numpy as np
import pdb
from world import WorldTraj

#### PARAMETERS ####
N_contacts = 3 # 2 grippers and 1 ground contact
T_final = 10 # time steps to optimize over
delT = 1
mass = 10.0 # mass
gravity = 10.0 # gravity
mu = 0.5 # friction coefficient
len_s = 24
len_s_aug = 24 + 18 # includes vels and accels
len_S = len_s*(T_final-1)
len_S_aug = len_s_aug*(T_final-1) # will be more when interpolate between s values
lamb = 1.0 # lambda is a L_physics parameter
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

#### HELPER FUNCTIONS ####
def calc_e(s, objects):
    _, box, _, _ = objects
    o = box.pose

    # get ro: roj in world frame
    _, roj, cj = get_contact_info(s)
    rj = roj + np.tile(o, (3, 1))

    # get pi_j: project rj onto all contact surfaces
    pi_j = np.zeros((N_contacts, 2))
    for object in objects:
        if object.contact_index != None:
            pi_j[object.contact_index,:] = object.project_point(rj[object.contact_index,:])

    # get pi_o: project rj onto object
    pi_o = np.zeros((N_contacts,2))
    for j in range(N_contacts):
        pi_o[j,:] = box.project_point(rj[j,:])

    e_O = pi_o - rj
    e_H = pi_j - rj
    return e_O, e_H

#### INITIALIZE DECISION VARIABLES ####
def init_vars(objects):
    _, box, _, _ = objects
    # create the initial system state: s0
    s0 = np.zeros(len_s)

    # fill in object poses
    for object in objects:
        if object.pose_index != None:
            s0[3*object.pose_index:3*object.pose_index+2] = object.pose
            s0[3*object.pose_index+2] = object.angle

    # initial contact information (just in contact with the ground):
    # [fxj fyj rOxj rOyj cj for j in N_contacts]
    # j = 0 gripper 1 contact
    # j = 1 gripper 2 contact
    # j = 2 ground contact
    # rO is in object (box) frame
    # ground contact force:
    box_pose = box.pose
    box_width = box.width
    f2 = mass*gravity
    con0 = [0.0, 0.0, 0.0, 0.0, 0.0]
    con1 = [0.0, 0.0, 0.0, 0.0, 0.0]
    con2 = [0.0, f2, box_width/2.0, 0.0, 1.0]

    s0[9:len_s] = (con0 + con1 + con2)

    # initialize traj to all be the same as the starting state
    S0 = np.zeros(len_S)
    for t in range(T_final-1):
        S0[t*len_s:t*len_s+len_s] = s0
    return s0, S0

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
            s_tm1 = get_s_aug_t(S_aug, t)
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

#### OBJECTIVE FUNCTIONS ####
def L_CI(s,t, objects, world_traj):
    e_O, e_H = calc_e(s, objects)
    _, _, cj = get_contact_info(s)
    e_O_tm1, e_H_tm1 = world_traj.e_Os[:,t-1,:], world_traj.e_Hs[:,t-1,:]

    # calculate the edots
    e_O_dot = (e_O - e_O_tm1)/delT
    e_H_dot = (e_H - e_H_tm1)/delT

    # calculate the contact invariance cost
    cost = 0
    for j in range(N_contacts):
        cost += cj[j]*(np.linalg.norm(e_O[j,:])**2 + np.linalg.norm(e_H[j,:])**2 \
                + np.linalg.norm(e_O_dot[j,:])**2 + np.linalg.norm(e_H_dot)**2)
    return cost

def L_physics(s):
    # get relevant state info
    fj, roj, cj = get_contact_info(s)
    o = get_object_pos(s)
    ov = get_object_vel(s)
    oa = get_object_accel(s)

    # calculate sum of forces on object
    # x-direction
    f_x_tot = 0.0
    # calc frictional force only if object is moving in x direction
    if ov[0] > 0.0:
        normal_force = 0
        for j in range(N_contacts):
            normal_force += cj[j]*fj[j][1]
        fric = -mu*normal_force
        f_x_tot += fric
    for j in range(N_contacts):
        f_x_tot += cj[j]*fj[j][0]

    # y-direction
    f_y_tot = -mass*gravity
    for j in range(N_contacts):
        f_y_tot += cj[j]*fj[j][1]
    f_tot = np.array([f_x_tot, f_y_tot])

    # calc change in linear momentum
    p_dot = mass*oa[0:2]

    # calc sum of moments on object (friction acts on COM? gravity does)
    # TODO: correct calc of I (moment of inertia)
    I = mass
    m_tot = np.array([0.0,0.0])
    for j in range(N_contacts):
        m_tot += np.cross(cj[j]*fj[j], roj[j]-o[0:2])

    # calc change in angular momentum
    l_dot = I*oa[2]

    # discourage large contact forces
    term = 0
    for j in range(N_contacts):
        term += np.linalg.norm(fj[j])**2
    term = lamb*term

    L_physics =  np.linalg.norm(f_tot - p_dot)**2 + np.linalg.norm(m_tot - l_dot)**2 + term

    # calc L_cone
    # get unit normal to contact surfaces at pi_j using surface line
    L_cone = 0
    nj = np.array((0.0, 1.0))
    for j in range(N_contacts):
        L_cone += max(np.arccos(np.dot(fj[j], nj)) - np.arctan(mu), 0)**2

    cost = L_physics + L_cone
    return cost

# includes 1) limits on finger and arm joint angles (doesn't apply)
#          2) distance from fingertips to palms limit (doesn't apply)
#          3) TODO: collisions between fingers
def L_kinematics(s, objects):
    cost = 0
    # penalize collisions between all objects


    return cost

# doesn't apply
def L_pad(s):
    return 0

def L_task(s, t):
    T_final, len_sk = args

    # l constraint: get object to desired pos
    I = 1 if t == T_final else 0
    hb = get_object_pos(s)
    h_star = [5.0, 10.0, 0.0] # goal object pos
    l = I*np.linalg.norm(hb - h_star)**2

    # small acceleration constraint (supposed to keep hand accel small, but
    # don't have a central hand so use grippers individually)
    o_dotdot = get_object_accel(s)
    g1_dotdot = get_gripper1_accel(s)
    g2_dotdot = get_gripper2_accel(s)

    cost = l + lamb*(np.linalg.norm(o_dotdot)**2 + np.linalg.norm(g1_dotdot)**2 \
                + np.linalg.norm(g2_dotdot)**2)
    return cost

def L(S, s0, objects):
    # calculate the interpolated values between the key frames (for now skip) -> longer S
    S_aug = interpolate_s(s0, S)
    world_traj = WorldTraj(s0, S_aug, objects, N_contacts, T_final)
    world_traj.e_Os[:,0,:], world_traj.e_Hs[:,0,:] = calc_e(s0, objects)
    cost = 0
    for t in range(1,T_final):
        s_aug_t = get_s_aug_t(S_aug,t)
        cost += L_CI(s_aug_t, t, objects, world_traj) + L_physics(s_aug_t) + L_kinmatics(s_aug_t) #+ \
    #            L_pad(s_aug_t) + L_task(s_aug_t, t)
    return cost

#### MAIN FUNCTION ####
def CIO(goal, objects):
    s0, S0 = init_vars(objects)
    x, f, d = fmin_l_bfgs_b(func=L, x0=S0, args=(s0, objects), approx_grad=True)
    return x,f,d

    # just for visualizing results
    #for t in range(1,T_final):
    #    s_t = get_s_t(x, t)
    #    contact_info = get_contact_info(s_t)
    #    print(contact_info, t)
    #print(x)
