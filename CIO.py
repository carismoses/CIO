# CIO implementation
from scipy.optimize import fmin_l_bfgs_b, minimize
import numpy as np
import pdb
from world import WorldTraj
from util import *

#### SURFACE NORMALS ####
def get_normals(angles):
    nj = np.zeros((N_contacts, 2))
    for j in range(N_contacts):
        norm_angle = angles[j] + np.pi/2
        nj[j,:] = np.array([np.cos(norm_angle), np.sin(norm_angle)])
    return nj

#### HELPER FUNCTIONS ####
def calc_e(s, objects):
    _, box, _, _ = objects
    o = box.pose

    # get ro: roj in world frame
    fj, roj, cj = get_contact_info(s)
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

#### OBJECTIVE FUNCTIONS ####
def L_CI(s, t, objects, world_traj):
    e_O, e_H = calc_e(s, objects)
    world_traj.e_Os[:,t,:], world_traj.e_Hs[:,t,:] = e_O, e_H
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

def L_physics(s, objects):
    pdb.set_trace()
    # get relevant state info
    fj, roj, cj = get_contact_info(s)
    ov = get_object_vel(s)
    oa = get_object_accel(s)
    ground, _, gripper1, gripper2 = objects
    contact_objects = [gripper1, gripper2, ground]

    # calculate sum of forces on object
    # calc frictional force only if object is moving in x direction
    f_tot = np.array([0.0, 0.0])
    for j in range(N_contacts):
        f_tot += cj[j]*fj[j]
    f_tot[1] += -mass*gravity

    fric = (-1*np.sign(ov[0]))*mu*cj[2]*fj[2][1]
    f_tot[0] += fric

    # calc change in linear momentum
    p_dot = mass*oa[0:2]

    # calc sum of moments on object (friction acts on COM? gravity does)
    # TODO: correct calc of I (moment of inertia)
    I = mass
    m_tot = np.array([0.0,0.0])
    for j in range(N_contacts):
        # transform to be relative to object COM not lower left corner
        m_tot += np.cross(cj[j]*fj[j], roj[j] + np.array([-5, -5]))

    # calc change in angular momentum
    l_dot = I*oa[2]

    cost =  phys_lamb_2*(np.linalg.norm(f_tot - p_dot)**2 + np.linalg.norm(m_tot - l_dot)**2)
    return cost

def L_cone(s):
    # calc L_cone
    fj, roj, cj = get_contact_info(s)
    cost = 0.0
    # get contact surface angles
    angles = np.zeros((N_contacts))
    for j in range(N_contacts):
        angles[j] = contact_objects[j].angle
    # get unit normal to contact surfaces at pi_j using surface line
    nj = get_normals(angles)
    for j in range(N_contacts):
        if cj[j] > 0.0: # TODO: fix.. don't think it's working..
            cosangle_num = np.dot(fj[j], nj[j,:])
            cosangle_den = np.dot(np.linalg.norm(fj[j]), np.linalg.norm(nj[j,:]))
            if cosangle_den == 0.0: # TODO: is this correct?
                angle = 0.0
            else:
                angle = np.arccos(cosangle_num/cosangle_den)
            cost += max(angle - np.arctan(mu), 0)**2
    return cost

def L_contact(s):
    # discourage large contact forces
    fj, roj, cj = get_contact_info(s)
    cost = 0.
    for j in range(N_contacts):
        cost += np.linalg.norm(fj[j])**2
    cost = phys_lamb*term
    return cost

def L_vels(s, s_tm1):
    # penalize differences between poses and velocities
    cost = sum((get_object_vel(s) - (get_object_pos(s) - get_object_pos(s_tm1))/delT)**2)
    return cost

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

#### GRADIENT OF OBJECTIVE FUNCTIONS ####
def L_physics_diff(s, objects):
    # get relevant state info
    fj, roj, cj = get_contact_info(s)
    o = get_object_pos(s)
    ov = get_object_vel(s)
    oa = get_object_accel(s)
    ground, _, gripper1, gripper2 = objects
    contact_objects = [gripper1, gripper2, ground]

    # calculate sum of forces on object
    # calc frictional force only if object is moving in x direction
    f_tot = np.array([0.0, 0.0])
    for j in range(N_contacts):
        f_tot += cj[j]*fj[j]
    f_tot[1] += -mass*gravity

    fric = (-1*np.sign(ov[0]))*mu*cj[2]*fj[2][1]
    f_tot[0] += fric

    # calc change in linear momentum
    p_dot = mass*oa[0:2]

    # calc sum of moments on object (friction acts on COM? gravity does)
    # TODO: correct calc of I (moment of inertia)
    I = mass
    m_tot = np.array([0.0,0.0])
    for j in range(N_contacts):
        # transform to be relative to object COM not lower left corner
        m_tot += np.cross(cj[j]*fj[j], roj[j] + np.array([-5, -5]))

    # calc change in angular momentum
    l_dot = I*oa[2]

    cost =  phys_lamb_2*(np.linalg.norm(f_tot - p_dot)**2 + np.linalg.norm(m_tot - l_dot)**2)
    return cost

#### MAIN OBJECTIVE FUNCTION ####
def L(S, s0, objects, goal):
    # calculate the interpolated values between the key frames (for now skip) -> longer S
    S_aug = interpolate_s(s0, S)
    world_traj = WorldTraj(s0, S, objects, N_contacts, T_final)
    world_traj.e_Os[:,0,:], world_traj.e_Hs[:,0,:] = calc_e(s0, objects)
    tot_cost = 0.0
    cis, kinems, physs, coness, conts, velss, tasks = \
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    for t in range(1,T_final):
        if t == 1:
            s_tm1 = s0
        else:
            s_tm1 = get_s_aug_t(S_aug, t-1)

        world_traj.step(t)
        s_aug_t = get_s_aug_t(S_aug,t)

        #ci = L_CI(s_aug_t, t, objects, world_traj)
        #kinem = L_kinematics(s_aug_t, objects)
        phys = L_physics(s_aug_t, objects)
        #cones = L_cone(s_aug_t)
        #cont = L_contact(s_aug_t)
        #vels = L_vels(s_aug_t, s_tm1)
        task = L_task(s_aug_t, goal, t)
        cost = phys + task #ci + kinem + cones + cont + vels

        #cis += ci
        #kinems += kinem
        physs += phys
        #coness += cones
        #conts += cont
        #velss += vels
        tasks += task
        tot_cost += cost

        #print("ci:             ", ci)
        #print("kinematics:     ", kinem)
        print("physics:        ", phys)
        #print("cone:           ", cones)
        #print("contact forces: ", cont)
        #print("velocities:     ", vels)
        print("task:           ", task)
        print("TOTAL: ", cost)
    return cost

#### MAIN FUNCTION ####
def CIO(goal, objects, s0, S0):
    #pdb.set_trace()
    bounds = get_bounds()

    """
    c = L(S0, s0, objects, goal)
    print(c)
    return None, None, None
    """

    #x, f, d = fmin_l_bfgs_b(func=L, x0=S0, args=(s0, objects, goal), approx_grad=True, bounds=bounds)
    res = minimize(fun=L, x0=S0, args=(s0, objects, goal), method='BFGS', bounds=bounds)
    x = res['x']

    # output result
    print("differences in final state: \n", S0-x)

    # augement the output
    x_aug = interpolate_s(s0,x)
    # pose trajectory
    print("pose trajectory:")
    for t in range(1,T_final):
        s_t = get_s_t(x, t)
        box_pose = get_object_pos(s_t)
        print(box_pose, t)

    # velocity trajectory
    print("vel trajectory:")
    for t in range(1,T_final):
        s_t = get_s_t(x, t)
        box_vel = get_object_vel(s_t)
        print(box_vel, t)

    # accel trajectory
    print("accel trajectory:")
    for t in range(1,T_final):
        s_t = get_s_aug_t(x_aug, t)
        box_accel = get_object_accel(s_t)
        print(box_accel, t)

    # contact forces
    print("contact forces:")
    for t in range(1,T_final):
        s_t = get_s_t(x, t)
        contact_info = get_contact_info(s_t)
        force = contact_info[0]
        print(t, ":\n", force)

    # contact poses
    print("contact poses:")
    for t in range(1,T_final):
        s_t = get_s_t(x, t)
        contact_info = get_contact_info(s_t)
        pos = contact_info[1]
        print(t, ":\n", pos)

    # contact coefficients
    print("coefficients:")
    for t in range(1,T_final):
        s_t = get_s_t(x, t)
        contact_info = get_contact_info(s_t)
        contact = contact_info[2]
        print(t, ":\n", contact)

    print("Final cost: ", L(x, s0, objects, goal))
    pdb.set_trace()

    return x
