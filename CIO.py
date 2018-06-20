# CIO implementation
from scipy.optimize import fmin_l_bfgs_b, minimize
import numpy as np
import pdb
from world import WorldTraj
from util import *
import theano
import theano.tensor as T
from theano.ifelse import ifelse
import params as p

verbose = False

#### SURFACE NORMALS ####
def get_normals(angles):
    nj = np.zeros((N, 2))
    for j in range(N):
        norm_angle = angles[j] + np.pi/2
        nj[j,:] = np.array([np.cos(norm_angle), np.sin(norm_angle)])
    return nj

#### OBJECTIVE FUNCTIONS ####
def L_CI(s, t, objects, world_traj):
    e_O, e_H = world_traj.calc_e(s, t, objects)
    e_O_tm1, e_H_tm1 = world_traj.e_Os[:,t-1,:], world_traj.e_Hs[:,t-1,:]
    _, _, cj = get_contact_info(s)

    # calculate the edots
    e_O_dot = calc_deriv(e_O, e_O_tm1, p.delT)
    e_H_dot = calc_deriv(e_H, e_H_tm1, p.delT)

    # calculate the contact invariance cost
    cost = 0
    for j in range(p.N):
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
    return kin_lamb*cost

def L_physics(s, objects):
    # get relevant state info
    fj, roj, cj = get_contact_info(s)
    ov = get_object_vel(s)
    oa = get_object_accel(s)
    ground, _, gripper1, gripper2 = objects
    contact_objects = [gripper1, gripper2, ground]

    # calculate sum of forces on object
    # calc frictional force only if object is moving in x direction
    f_tot = np.array([0.0, 0.0])
    for j in range(p.N):
        f_tot += cj[j]*fj[j]
    f_tot[1] += -p.mass*p.gravity

    fric = (-1*np.sign(ov[0]))*p.mu*cj[2]*fj[2][1]
    f_tot[0] += fric

    # calc change in linear momentum
    p_dot = p.mass*oa[0:2]

    # calc sum of moments on object (friction acts on COM? gravity does)
    # TODO: correct calc of I (moment of inertia)
    I = p.mass
    m_tot = np.array([0.0,0.0])
    for j in range(p.N):
        # transform to be relative to object COM not lower left corner
        m_tot += np.cross(cj[j]*fj[j], roj[j] + np.array([-5, -5]))

    # calc change in angular momentum
    l_dot = I*oa[2]

    cost = np.linalg.norm(f_tot - p_dot)**2 + np.linalg.norm(m_tot - l_dot)**2
    return cost

def L_cone(s):
    # calc L_cone
    fj, roj, cj = get_contact_info(s)
    cost = 0.0
    # get contact surface angles
    angles = np.zeros((p.N))
    for j in range(p.N):
        # TODO: this only works currently because all contact surfaces are lines...
        # will need to change if have different shaped contact surfaces
        angles[j] = contact_objects[j].angle
    # get unit normal to contact surfaces at pi_j using surface line
    nj = get_normals(angles)
    for j in range(p.N):
        if cj[j] > 0.0: # TODO: fix.. don't think it's working..
            cosangle_num = np.dot(fj[j], nj[j,:])
            cosangle_den = np.dot(np.linalg.norm(fj[j]), np.linalg.norm(nj[j,:]))
            if cosangle_den == 0.0: # TODO: is this correct?
                angle = 0.0
            else:
                angle = np.arccos(cosangle_num/cosangle_den)
            cost += max(angle - np.arctan(p.mu), 0)**2
    return cone_lamb*cost

def L_contact(s):
    # discourage large contact forces
    fj, _, _ = get_contact_info(s)
    cost = 0.
    for j in range(p.N):
        cost += np.linalg.norm(fj[j])**2
    cost = p.cont_lamb*cost
    return cost

def L_task(s, goal, t):
    # l constraint: get object to desired pos
    I = 1 if t == (p.T_steps-1) else 0
    hb = get_object_pos(s)
    h_star = goal[1]
    cost = I*np.linalg.norm(hb - h_star)**2
    return cost

def L_accel(s):
    # small acceleration constraint (supposed to keep hand accel small, but
    # don't have a central hand so use grippers individually)
    o_dotdot = get_object_accel(s)
    g1_dotdot = get_gripper1_accel(s)
    g2_dotdot = get_gripper2_accel(s)

    cost = p.accel_lamb*(np.linalg.norm(o_dotdot)**2 + np.linalg.norm(g1_dotdot)**2 \
                + np.linalg.norm(g2_dotdot)**2)
    return cost

#### MAIN OBJECTIVE FUNCTION ####
def L(S, s0, objects, goal, phase_weights):
    global cis, kinems, physs, coness, conts, tasks, accels
    # augment by calculating the accelerations from the velocities
    # interpolate all of the decision vars to get a finer trajcetory disretization
    S_aug = augment_s(s0, S)

    world_traj = WorldTraj(s0, S_aug, objects)
    tot_cost = 0.0
    cis, kinems, physs, coness, conts, tasks, accels = \
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    for t in range(1,p.T_steps):
        if t == 1:
            s_tm1 = s0
        else:
            s_tm1 = get_s(S_aug, t-1)

        world_traj.step(t)
        s_aug_t = get_s(S_aug,t)

        wci, wphys, wtask = phase_weights

        ci = L_CI(s_aug_t, t, objects, world_traj)
        kinem = 0.#L_kinematics(s_aug_t, objects)
        phys = L_physics(s_aug_t, objects)
        cones = 0.#L_cone(s_aug_t)
        cont = L_contact(s_aug_t)
        accel = L_accel(s_aug_t)
        task = L_task(s_aug_t, goal, t)
        cost = wtask*(task + accel) + wci*ci + wphys*(phys + kinem + cones + cont)

        cis += ci
        kinems += kinem
        physs += phys
        coness += cones
        conts += cont
        accels += accel
        tasks += task
        tot_cost += cost

        if verbose:
            print_step(tot_cost)
    return tot_cost

#### MAIN FUNCTION ####
def CIO(goal, objects, s0, S0):
    """
    # FOR TESTING A SINGLE traj
    pdb.set_trace()
    x = L(S0, s0, objects, goal)
    print(x)
    pdb.set_trace()
    """
    #pdb.set_trace()
    bounds = get_bounds()

    ret_info = {}
    x_init = S0
    if p.phase_weights == []:
        p.phase_weights = [(1.,1.,1.),]
    for phase in range(p.start_phase, len(p.phase_weights)):
        phase_weights = p.phase_weights[phase]
        res = minimize(fun=L, x0=x_init, args=(s0, objects, goal, phase_weights), method='L-BFGS-B', bounds=bounds)
        x_final = res['x']
        nit = res['nit']
        final_cost = res['fun']

        print_result(x_final, s0)
        print("Final cost: ", final_cost)
        all_final_costs = [cis, kinems, physs, coness, conts, tasks, accels]
        ret_info[phase] = s0, x_final, final_cost, nit, all_final_costs
        x_init = x_final
    return ret_info

def print_step(tot_cost):
    print("cis:            ", cis)
    print("kinematics:     ", kinems)
    print("physics:        ", physs)
    print("cone:           ", coness)
    print("contact forces: ", conts)
    print("accels:         ", accels)
    print("task:           ", tasks)
    print("TOTAL: ", tot_cost)

def print_result(x, s0):
    # augement the output
    x_aug = augment_s(s0,x)

    # pose trajectory
    print("pose trajectory:")
    for t in range(p.T_steps+1):
        if t == 0:
            box_pose = get_object_pos(s0)
        else:
            s_t = get_s(x_aug, t-1)
            box_pose = get_object_pos(s_t)
        print(box_pose, t)

    # velocity trajectory
    print("vel trajectory:")
    for t in range(p.T_steps+1):
        if t == 0:
            box_vel = get_object_vel(s0)
        else:
            s_t = get_s(x_aug, t-1)
            box_vel = get_object_vel(s_t)
        print(box_vel, t)

    # accel trajectory
    print("accel trajectory:")
    for t in range(p.T_steps+1):
        if t == 0:
            box_accel = get_object_accel(s0)
        else:
            s_t = get_s(x_aug, t-1)
            box_accel = get_object_accel(s_t)
        print(box_accel, t)

    # contact forces
    print("contact forces:")
    for t in range(p.T_steps+1):
        if t == 0:
            contact_info = get_contact_info(s0)
            force = contact_info[0]
        else:
            s_t = get_s(x_aug, t-1)
            contact_info = get_contact_info(s_t)
            force = contact_info[0]
        print(t, ":\n", force)

    # contact poses
    print("contact poses:")
    for t in range(p.T_steps+1):
        if t == 0:
            contact_info = get_contact_info(s0)
            pos = contact_info[1]
        else:
            s_t = get_s(x_aug, t-1)
            contact_info = get_contact_info(s_t)
            pos = contact_info[1]
        print(t, ":\n", pos)

    # contact coefficients
    print("coefficients:")
    for t in range(p.T_steps+1):
        if t == 0:
            contact_info = get_contact_info(s0)
            contact = contact_info[2]
        else:
            s_t = get_s(x_aug, t-1)
            contact_info = get_contact_info(s_t)
            contact = contact_info[2]
        print(t, ":\n", contact)
