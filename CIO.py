# CIO implementation
from scipy.optimize import fmin_l_bfgs_b, minimize
import numpy as np
import pdb
from world import WorldTraj
from util import *

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
    con0 = [0.0, 0.0, 0.0, 5.0, 0.0]
    con1 = [0.0, 0.0, 0.0, 0.0, 0.0]
    con2 = [0.0, f2, box_width/2.0, 0.0, 1.0]

    s0[9:len_s] = (con0 + con1 + con2)

    # initialize traj to all be the same as the starting state
    S0 = np.zeros(len_S)
    for t in range(T_final-1):
        S0[t*len_s:t*len_s+len_s] = s0
    return s0, S0

#### HELPER FUNCTIONS ####
def calc_e(s, objects):
    _, box, _, _ = objects
    o = box.pose

    # get ro: roj in world frame
    fj, roj, cj = get_contact_info(s)
    rj = roj + np.tile(o, (3, 1))
    #print("gripper1 x direction force: ",fj[0][0])
    #print("box pose: ", o)
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
    term = phys_lamb*term

    L_physics =  np.linalg.norm(f_tot - p_dot)**2 + np.linalg.norm(m_tot - l_dot)**2 + term

    # calc L_cone
    # get unit normal to contact surfaces at pi_j using surface line
    L_cone = 0
    nj = np.array((0.0, 1.0))
    for j in range(N_contacts):
        if np.linalg.norm(fj[j]) > 0.0: # only calc if there is a contact force
            cosangle_num = np.dot(fj[j], nj)
            cosangle_den = np.dot(np.linalg.norm(fj[j]), np.linalg.norm(nj))
            angle = np.arccos(cosangle_num/cosangle_den)
            L_cone += max(angle - np.arctan(mu), 0)**2

    cost = L_physics + L_cone
    return cost

# includes 1) limits on finger and arm joint angles (doesn't apply)
#          2) distance from fingertips to palms limit (doesn't apply)
#          3) TODO: collisions between fingers
def L_kinematics(s, objects):
    cost = 0
    # penalize collisions between all objects
    obj_num = 0
    while obj_num < len(objects):
        for col_object in objects[obj_num+1:]:
            col_dist = objects[obj_num].check_collisions(col_object)
            cost += col_lamb*col_dist
            obj_num += 1
    return cost

# doesn't apply
def L_pad(s):
    return 0

def L_task(s, goal, t):
    # l constraint: get object to desired pos
    I = 1 if t == (T_final-1) else 0
    hb = get_object_pos(s)
    h_star = goal[1]
    l = I*np.linalg.norm(hb - h_star)**2
    #if t == (T_final-1):
    #    print("this is l: ",l)

    # small acceleration constraint (supposed to keep hand accel small, but
    # don't have a central hand so use grippers individually)
    o_dotdot = get_object_accel(s)
    g1_dotdot = get_gripper1_accel(s)
    g2_dotdot = get_gripper2_accel(s)

    cost = l + task_lamb*(np.linalg.norm(o_dotdot)**2 + np.linalg.norm(g1_dotdot)**2 \
                + np.linalg.norm(g2_dotdot)**2)
    return cost

def L(S, s0, objects, goal):
    # calculate the interpolated values between the key frames (for now skip) -> longer S
    S_aug = interpolate_s(s0, S)
    world_traj = WorldTraj(s0, S_aug, objects, N_contacts, T_final)
    world_traj.e_Os[:,0,:], world_traj.e_Hs[:,0,:] = calc_e(s0, objects)
    cost = 0
    for t in range(1,T_final):
        world_traj.step(t)
        s_aug_t = get_s_aug_t(S_aug,t)
        cost += L_task(s_aug_t, goal, t)
        #cost += L_CI(s_aug_t, t, objects, world_traj) + L_physics(s_aug_t) + \
        #        L_pad(s_aug_t) + L_task(s_aug_t, goal, t) #+ L_kinematics(s_aug_t, objects) +
    print(cost)
    return cost

#### MAIN FUNCTION ####
def CIO(goal, objects, s0 = (), S0 = ()):
    if s0 == ():
        s0, S0 = init_vars(objects)
    x, f, d = fmin_l_bfgs_b(func=L, x0=S0, args=(s0, objects, goal), approx_grad=True)

    # for comparing hand made traj and init traj
    #hms0, hmS0 = s0, S0
    #ins0, inS0 = init_vars(objects)
    #print("Hand made cost: ", L(hmS0, hms0, objects, goal))
    #print("Init cost: ", L(inS0, ins0, objects, goal))

    # output result
    # pose trajectory
    print("pose trajectory:")
    for t in range(1,T_final):
        s_t = get_s_t(x, t)
        box_pose = get_object_pos(s_t)
        print(box_pose, t)

    # contact forces
    #print("forces:")
    for t in range(1,T_final):
        s_t = get_s_t(x, t)
        contact_info = get_contact_info(s_t)
        force = contact_info[0]
        #print(force, t)

    return x,f,d
