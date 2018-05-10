# CIO implementation
from scipy.optimize import fmin_l_bfgs_b
import numpy as np
import pdb

#### PARAMETERS ####
Ncontacts = 3 # 2 grippers and 1 ground contact
Tfinal = 10 # time steps to optimize over
delT = 1
mass = 10.0 # mass
gravity = 10.0 # gravity
mu = 0.5 # friction coefficient
len_s = 24
len_s_aug = 24 + 18 # includes vels and accels
len_S = len_s*(Tfinal-1)
len_S_aug = len_s_aug*(Tfinal-1) # will be more when interpolate between s values
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
    fj = []
    for j in range(Ncontacts):
        fj += [np.array([s[9 + j*5], s[10 + j*5]])]
    rj = []
    for j in range(Ncontacts):
        rj += [np.array([s[11 + j*5], s[12 + j*5]])]
    cj = []
    for j in range(Ncontacts):
        cj += [s[13 + j*5]]
    return fj, rj, cj

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
def get_line(pos):
    # get the functin of the line: ax + by + c = 0 in the world coordinate frames
    # check if near singularity TODO: test for better small angle or just straight check for singular matrix in inv calc
    if abs(pos[2] - 0.0) < small_ang:
        return np.array([1.0, 0.0, -pos[0]])
    if abs(pos[2] - np.pi/2.0) < small_ang:
        return np.array([0.0, 1.0, -pos[2]])
    delta = 1.0
    A = np.array([[pos[0],                  pos[1],                  1.0],
                  [pos[0] + np.sin(pos[2]), pos[1] - np.cos(pos[2]), 1.0],
                  [pos[0] - np.sin(pos[2]), pos[1] + np.cos(pos[2]), 1.0]])
    b = np.zeros(3)
    abc = np.multiply(np.linalg.inv(A),b)
    return abc

def project_to_line(line, point):
    n = line[1]*point[0] - line[0]*point[1]
    d = line[0]**2 + line[1]**2
    proj_point = np.array([line[1]*n    - line[0]*line[2], \
                           line[0]*-1*n - line[1]*line[2]])/d
    return proj_point

def get_dist(line, point):
    return abs(line[0]*point[0] + line[1]*point[1] + line[2])/ \
            np.sqrt(line[0]**2 + line[1]**2)

def get_e(s):
    # get relevant state info
    _, roj, cj = get_contact_info(s)
    o = get_object_pos(s)
    g1 = get_gripper1_pos(s)
    g2 = get_gripper2_pos(s)

    # contact surface lines in world frame
    abc = np.zeros((Ncontacts,3))
    abc[0,:] = get_line(g1)
    abc[1,:] = get_line(g2)
    abc[2,:] = [0.0, 1.0, 0.0] # ground line

    # rj is the contact point in the world frame
    rj = np.zeros((Ncontacts, 2))
    rj[0,:] = roj[0] + o[0:2]
    rj[1,:] = roj[1] + o[0:2]
    rj[2,:] = roj[2] + o[0:2]

    # projection of rj onto contact surfaces (currently line along surface, NOT necessarily on surface) TODO
    pi_j = np.zeros((Ncontacts,2))
    for j in range(Ncontacts):
        # project rj onto contact region j
        pi_j[j,:] = project_to_line(abc[j,:], rj[j,:])

    for j in range(Ncontacts):
        # get a point and orientation for a point on each side of the object
        points_on_face = np.zeros((4,3))
        # right face
        points_on_face[0,:] = [o[0] + hw*np.cos(o[2]),
                              o[1] - hw*np.sin(o[2]),
                              o[2]]
        # top face
        points_on_face[1,:] = [o[0] + hh*np.sin(o[2]),
                              o[1] + hh*np.cos(o[2]),
                              o[2]]
        # left face
        points_on_face[2,:] = [o[0] - hw*np.cos(o[2]),
                              o[1] + hw*np.sin(o[2]),
                              o[2]]
        # bottom face
        points_on_face[3,:] = [o[0] - hh*np.sin(o[2]),
                              o[1] + hh*np.cos(o[2]),
                              o[2]]

    # projection of rj onto object (currently lines through face of object, NOT necessarily on object) TODO
    pi_o = np.zeros((Ncontacts,2))

    # find dist to each face for each contact and project onto the closest face
    for j in range(Ncontacts):
        face_dists = np.zeros(4)
        face_lines = np.zeros((4,3))
        for face in range(4):
            face_lines[face,:] = get_line(points_on_face[face,:])
            face_dists[face] = get_dist(face_lines[face,:], rj[j,:])
        min_dist = min(face_dists)
        min_face = np.argmin(face_dists)
        pi_o[j,:] = project_to_line(face_lines[min_face,:], rj[j,:])

    e_O = pi_o - rj
    e_H = pi_j - rj
    return e_O, e_H

#### INITIALIZE DECISION VARIABLES ####
def init_vars():
    global s0
    # initial gripper poses: [gx1 gy1 go1 gx2 gy2 go2]
    g0 = [-5.0, 15.0, 0.0, 5.0, 15.0, 0.0]

    # initial object pose: [ox oy oo]
    o0 = [0.0, hh, 0.0]

    # initial contact information (just in contact with the ground):
    # [fxj fyj rOxj rOyj cj for j in Ncontacts]
    # j = 0 gripper 1 contact
    # j = 1 gripper 2 contact
    # j = 2 ground contact
    # ground contact force:
    f2 = mass*gravity
    con0 = [0.0, 0.0, -5.0, 10.0, 0.0]
    con1 = [0.0, 0.0, 5.0, 10.0, 0.0]
    con2 = [0.0, f2, 0.0, -hh, 1.0]

    s0 = np.array(g0 + o0 + con0 + con1 + con2)
    S0 = np.zeros(len_S)
    for t in range(Tfinal-1):
        S0[t*len_s:t*len_s+len_s] = s0
    return S0

#### AUGMENT DECISION VARIABLE ####
def interpolate_s(S):
    S_aug = np.zeros(len_S_aug)
    # should interpolate between S values (skip for now)
    for t in range(1,Tfinal):
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
def L_CI(s,t):
    e_O, e_H = get_e(s)
    _, _, cj = get_contact_info(s)
    e_O_tm1, e_H_tm1 = e_Os[:,t-1,:], e_Hs[:,t-1,:]

    # calculate the edots
    e_O_dot = (e_O - e_O_tm1)/delT
    e_H_dot = (e_H - e_H_tm1)/delT

    # calculate the contact invariance cost
    cost = 0
    for j in range(Ncontacts):
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
        for j in range(Ncontacts):
            normal_force += cj[j]*fj[j][1]
        fric = -mu*normal_force
        f_x_tot += fric
    for j in range(Ncontacts):
        f_x_tot += cj[j]*fj[j][0]

    # y-direction
    f_y_tot = -mass*gravity
    for j in range(Ncontacts):
        f_y_tot += cj[j]*fj[j][1]
    f_tot = np.array([f_x_tot, f_y_tot])

    # calc change in linear momentum
    p_dot = mass*oa[0:2]

    # calc sum of moments on object (friction acts on COM? gravity does)
    # TODO: correct calc of I (moment of inertia)
    I = mass
    m_tot = np.array([0.0,0.0])
    for j in range(Ncontacts):
        m_tot += np.cross(cj[j]*fj[j], roj[j]-o[0:2])

    # calc change in angular momentum
    l_dot = I*oa[2]

    # discourage large forces
    term = 0
    for j in range(Ncontacts):
        term += np.linalg.norm(fj[j])**2
    term = lamb*term

    L_physics =  np.linalg.norm(f_tot - p_dot)**2 + np.linalg.norm(m_tot - l_dot)**2 + term

    # calc L_cone
    # get unit normal to contact surfaces at pi_j using surface line
    nj = np.zeros((Ncontacts,2))
    for j in range(Ncontacts):
        nj[j,:] = [0,1]

    L_cone = 0
    for j in range(Ncontacts):
        L_cone += max(np.acos(np.dot(fj[j], nj[j,:])) - np.arctan(mu), 0)**2

    cost = L_physics + L_cone
    return cost

# includes 1) limits on finger and arm joint angles (doesn't apply)
#          2) distance from fingertips to palms limit (doesn't apply)
#          3) TODO: collisions between fingers
def L_kinematics(s):
    cost = 0
    # penalize collisions between
    # 1) the object and the floor
    corners = get_corners(s) #TODO
    for corner in corners:
        cost += -1*min(0, corner[1])

    # 2) the end points of the gripper and the floor
    endpoints1 = get_endpoints(s,1) # TODO
    endpoints2 = get_endpoints(s,2) # TODO
    cost += -1*min(0,endpoints1[0][1])
    cost += -1*min(0,endpoints1[1][1])
    cost += -1*min(0,endpoints2[0][1])
    cost += -1*min(0,endpoints2[1][1])

    # 3) grippers inside of box #TODO


    return cost


# doesn't apply
def L_pad(s):
    return 0

def L_task(s, t):
    Tfinal, len_sk = args

    # l constraint: get object to desired pos
    I = 1 if t == Tfinal else 0
    hb = get_object_pos(s)
    h_star = [5.0, 10.0, 0.0] # goal object pos
    l = I*np.linalg.norm(hb - h_star)**2

    # small acceleration constraint (supposed to keep hand accel small, but
    # don't have a central hand so use grippers individually)
    o_dotdot = get_object_accel(s)
    g1_dotdot = get_gripper1_accel(s)
    g2_dotdot = get_gripper2_accel(s)

    cost = l + lambda*(linalg.norm(o_dotdot)**2 + linalg.norm(g1_dotdot)**2 + \
                linalg.norm(g2_dotdot)**2)
    return cost

def L(S):
    cost = 0
    # calculate the interpolated values between the key frames (for now skip) -> longer S
    S_aug = interpolate_s(S)
    ## HACK:
    global e_Os, e_Hs
    e_Os, e_Hs = np.zeros((Ncontacts, Tfinal+1, 2)), np.zeros((Ncontacts, Tfinal+1, 2))
    e_Os[:,0,:], e_Hs[:,0,:] = get_e(s0)

    for t in range(1,Tfinal):
        s_aug_t = get_s_aug_t(S_aug,t)
        cost += L_CI(s_aug_t, t) + L_physics(s_aug_t) + L_kinmatics(s_aug_t) + \
                L_pad(s_aug_t) + L_task(s_aug_t, t)
    return cost

#### MAIN FUNCTION ####
def main():
    #pdb.set_trace()
    S0 = init_vars()
    x, f, d = fmin_l_bfgs_b(func=L, x0=S0, approx_grad=True)
    for t in range(1,Tfinal):
        s_t = get_s_t(x, t)
        contact_info = get_contact_info(s_t)
        print(contact_info, t)
    print(x)

if __name__ == '__main__':
    main()
