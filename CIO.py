# CIO implementation
from scipy.optimize import fmin_l_bfgs_b, minimize
import numpy as np
import pdb
from world import WorldTraj
from util import *
import theano
import theano.tensor as T
from theano.ifelse import ifelse

#### THEANO VARIABLES AND HELPER FUNCTIONS ####
fj = T.dmatrix('fj')
roj = T.dmatrix('roj')
cj = T.dvector('cj')
ov = T.dvector('ov')
oa = T.dvector('oa')
one = T.constant(0.)
zero = T.constant(1.)


def TNorm(x):
    return T.sum(T.sqr(x))

# 2D cross product
def TCross(a, b):
    return T.as_tensor([a[0]*b[1] - a[1]*b[0]])

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
def phys_fns():
    # calculate sum of forces on object
    # calc frictional force only if object is moving in x direction
    f_tot = sum([cj[j]*fj[j] for j in range(N_contacts)])
    f_tot = T.inc_subtensor(f_tot[1], -mass*gravity)
    coeff = ifelse(T.gt(ov[0], zero), one, ifelse(T.lt(ov[0], 0), -one, zero))
    fric = -1*coeff*mu*cj[2]*fj[2,1]
    f_tot = T.inc_subtensor(f_tot[0], fric)

    # calc change in linear momentum
    p_dot = mass*oa[0:2]

    # calc sum of moments on object (friction acts on COM? gravity does)
    # TODO: correct calc of I (moment of inertia)
    I = mass
    m_tot = sum(TCross(cj[j]*fj[j,:], roj[j,:] + np.array([-5, -5])) for j in range(N_contacts))

    # calc change in angular momentum
    l_dot = I*oa[2]

    cost_fn =  phys_lamb_2*(TNorm(f_tot - p_dot)**2 + TNorm(m_tot - l_dot)**2)
    cost = theano.function([fj, roj, cj, ov, oa], cost_fn)
    gc_fj = T.grad(cost_fn, fj)
    gc_roj = T.grad(cost_fn, roj)
    gc_cj = T.grad(cost_fn, cj)
    gc_ov = T.grad(cost_fn, ov)
    grad_fn = [theano.function([fj, roj, cj, ov, oa], gc_fj),
                theano.function([fj, roj, cj, ov, oa], gc_roj),
                theano.function([fj, roj, cj, ov, oa], gc_cj),
                theano.function([fj, roj, cj, ov, oa], gc_ov, on_unused_input='ignore')]
    return cost, grad_fn

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

#### HELPER FUNCTIONS ####
def phys_helper(s, cost_fn=None, grad_fns=None):
    fj_val, roj_val, cj_val = get_contact_info(s)
    ov_val = get_object_vel(s)
    oa_val = get_object_accel(s)
    vals = [fj_val, roj_val, cj_val, ov_val, oa_val]

    ind = []
    ind += [get_fj_ind()]
    ind += [get_roj_ind()]
    ind += [get_contact_ind()]
    ind += [get_object_vel_ind()]

    cost = cost_helper(vals, ind, cost_fn=cost_fn, grad_fns=grad_fns)
    return cost

def cost_helper(vals, inds, cost_fn = None, grad_fns = None):
    if cost_fn != None:
        return cost_fn(vals)
    else:
        grad = np.zeros((len_s))
        for i in range(len(grad_fns)):
            this_grad = grad_fns[i](vals[0], vals[1], vals[2], vals[3], vals[4]) #TODO: make this work for variable amounts of vals
            ind = inds[i]
            h = len(ind)
            if type(ind[0]) is tuple:
                w = len(ind[0])
                for k in range(h):
                    for l in range(w):
                        grad[ind[k][l]] = this_grad[k][l]
            else:
                for k in range(h):
                    grad[ind[k]] = this_grad[k]
        return grad

#### MAIN OBJECTIVE FUNCTION AND GRADIENT ####
def L(S, s0, objects, goal, fns):
    # calculate the interpolated values between the key frames (for now skip) -> longer S
    S_aug = interpolate_s(s0, S)
    world_traj = WorldTraj(s0, S, objects, N_contacts, T_final)
    world_traj.e_Os[:,0,:], world_traj.e_Hs[:,0,:] = calc_e(s0, objects)
    tot_cost = 0.0
    cis, kinems, physs, coness, conts, velss, tasks = \
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    phys_cost, phys_grad = fns

    for t in range(1,T_final):
        if t == 1:
            s_tm1 = s0
        else:
            s_tm1 = get_s_aug_t(S_aug, t-1)

        world_traj.step(t)
        s_aug_t = get_s_aug_t(S_aug,t)

        #ci = L_CI(s_aug_t, t, objects, world_traj)
        #kinem = L_kinematics(s_aug_t, objects)
        phys = phys_helper(s_aug_t, cost_fn=phys_cost)
        #cones = L_cone(s_aug_t)
        #cont = L_contact(s_aug_t)
        #vels = L_vels(s_aug_t, s_tm1)
        #task = L_task(s_aug_t, goal, t)
        cost = phys #+ task #ci + kinem + cones + cont + vels

        #cis += ci
        #kinems += kinem
        physs += phys
        #coness += cones
        #conts += cont
        #velss += vels
        #tasks += task #TODO ADD TSK COST BACK AFTER THEANIZE IT
        tot_cost += cost

        #print("ci:             ", ci)
        #print("kinematics:     ", kinem)
        print("physics:        ", phys)
        #print("cone:           ", cones)
        #print("contact forces: ", cont)
        #print("velocities:     ", vels)
        #print("task:           ", task)
        print("TOTAL: ", cost)
    return cost

def L_grad(S, s0, objects, goal, fns):
    # calculate the interpolated values between the key frames (for now skip) -> longer S
    S_aug = interpolate_s(s0, S)
    world_traj = WorldTraj(s0, S, objects, N_contacts, T_final)
    world_traj.e_Os[:,0,:], world_traj.e_Hs[:,0,:] = calc_e(s0, objects)
    tot_cost = 0.0
    cis, kinems, physs, coness, conts, velss, tasks = \
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    phys_cost, phys_grad = fns
    grad_tot = np.zeros(len_s)

    for t in range(1,T_final):
        if t == 1:
            s_tm1 = s0
        else:
            s_tm1 = get_s_aug_t(S_aug, t-1)

        world_traj.step(t)
        s_aug_t = get_s_aug_t(S_aug,t)

        grad = np.zeros(len_s)

        #ci = L_CI(s_aug_t, t, objects, world_traj)
        #kinem = L_kinematics(s_aug_t, objects)
        phys = phys_helper(s_aug_t, grad_fns=phys_grad)
        #cones = L_cone(s_aug_t)
        #cont = L_contact(s_aug_t)
        #vels = L_vels(s_aug_t, s_tm1)
        #task = L_task(s_aug_t, goal, t)
        cost = phys #+ task #ci + kinem + cones + cont + vels

        #cis += ci
        #kinems += kinem
        physs += phys
        #coness += cones
        #conts += cont
        #velss += vels
        #tasks += task
        tot_cost += cost

        #print("ci:             ", ci)
        #print("kinematics:     ", kinem)
        print("physics:        ", phys)
        #print("cone:           ", cones)
        #print("contact forces: ", cont)
        #print("velocities:     ", vels)
        #print("task:           ", task)
        print("TOTAL: ", cost)
    return cost

#### MAIN FUNCTION ####
def CIO(goal, objects, s0, S0):
    pdb.set_trace()
    bounds = get_bounds()

    # get cost functions and their derivatives
    phys_cost, phys_grad = phys_fns()
    #task_cost, task_grad = task_fns()
    fns = [phys_cost, phys_grad]
    """
    c = L(S0, s0, objects, goal)
    print(c)
    return None, None, None
    """

    #x, f, d = fmin_l_bfgs_b(func=L, x0=S0, args=(s0, objects, goal), approx_grad=True, bounds=bounds)
    res = minimize(fun=L, x0=S0, args=(s0, objects, goal, fns), method='L-BFGS-B', jac=L_grad, bounds=bounds)
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
