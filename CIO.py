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
    nj = np.zeros((N, 2))
    for j in range(N):
        norm_angle = angles[j] + np.pi/2
        nj[j,:] = np.array([np.cos(norm_angle), np.sin(norm_angle)])
    return nj

#### OBJECTIVE FUNCTIONS ####
"""
def phys_fns():
    # calculate sum of forces on object
    # calc frictional force only if object is moving in x direction
    f_tot = sum([cj[j]*fj[j] for j in range(N)])
    f_tot = T.inc_subtensor(f_tot[1], -mass*gravity)
    coeff = ifelse(T.gt(ov[0], zero), one, ifelse(T.lt(ov[0], 0), -one, zero))
    fric = -1*coeff*mu*cj[2]*fj[2,1]
    f_tot = T.inc_subtensor(f_tot[0], fric)

    # calc change in linear momentum
    p_dot = mass*oa[0:2]

    # calc sum of moments on object (friction acts on COM? gravity does)
    # TODO: correct calc of I (moment of inertia)
    I = mass
    m_tot = sum(TCross(cj[j]*fj[j,:], roj[j,:] + np.array([-5, -5])) for j in range(N))

    # calc change in angular momentum
    l_dot = I*oa[2]

    cost_fn =  phys_lamb*(TNorm(f_tot - p_dot)**2 + TNorm(m_tot - l_dot)**2)
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
"""

#### OBJECTIVE FUNCTIONS ####
def L_CI(s, t, objects, world_traj):
    e_O, e_H = world_traj.calc_e(s, t, objects)
    e_O_tm1, e_H_tm1 = world_traj.e_Os[:,t-1,:], world_traj.e_Hs[:,t-1,:]
    _, _, cj = get_contact_info(s)

    # calculate the edots
    e_O_dot = calc_deriv(e_O, e_O_tm1, delT)
    e_H_dot = calc_deriv(e_H, e_H_tm1, delT)

    # calculate the contact invariance cost
    cost = 0
    for j in range(N):
        cost += cj[j]*(np.linalg.norm(e_O[j,:])**2 + np.linalg.norm(e_H[j,:])**2 \
                + np.linalg.norm(e_O_dot[j,:])**2 + np.linalg.norm(e_H_dot)**2)
    return ci_lamb*cost

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
    for j in range(N):
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
    for j in range(N):
        # transform to be relative to object COM not lower left corner
        m_tot += np.cross(cj[j]*fj[j], roj[j] + np.array([-5, -5]))

    # calc change in angular momentum
    l_dot = I*oa[2]

    cost =  phys_lamb*(np.linalg.norm(f_tot - p_dot)**2 + np.linalg.norm(m_tot - l_dot)**2)
    return cost

def L_cone(s):
    # calc L_cone
    fj, roj, cj = get_contact_info(s)
    cost = 0.0
    # get contact surface angles
    angles = np.zeros((N))
    for j in range(N):
        angles[j] = contact_objects[j].angle
    # get unit normal to contact surfaces at pi_j using surface line
    nj = get_normals(angles)
    for j in range(N):
        if cj[j] > 0.0: # TODO: fix.. don't think it's working..
            cosangle_num = np.dot(fj[j], nj[j,:])
            cosangle_den = np.dot(np.linalg.norm(fj[j]), np.linalg.norm(nj[j,:]))
            if cosangle_den == 0.0: # TODO: is this correct?
                angle = 0.0
            else:
                angle = np.arccos(cosangle_num/cosangle_den)
            cost += max(angle - np.arctan(mu), 0)**2
    return cone_lamb*cost

def L_contact(s):
    # discourage large contact forces
    fj, roj, cj = get_contact_info(s)
    cost = 0.
    for j in range(N):
        cost += np.linalg.norm(fj[j])**2
    cost = cont_lamb*term
    return cost

def L_vels(s, s_tm1):
    # penalize differences between poses and velocities
    cost = vel_lamb*sum((get_object_vel(s) - (get_object_pos(s) - get_object_pos(s_tm1))/delT)**2)
    return cost

def L_task(s, goal, t):
    # l constraint: get object to desired pos
    I = 1 if t == (T_steps-1) else 0
    hb = get_object_pos(s)
    h_star = goal[1]
    cost = np.linalg.norm(hb - h_star)**2
    return cost

def L_accel(s):
    # small acceleration constraint (supposed to keep hand accel small, but
    # don't have a central hand so use grippers individually)
    o_dotdot = get_object_accel(s)
    g1_dotdot = get_gripper1_accel(s)
    g2_dotdot = get_gripper2_accel(s)

    cost = accel_lamb*(np.linalg.norm(o_dotdot)**2 + np.linalg.norm(g1_dotdot)**2 \
                + np.linalg.norm(g2_dotdot)**2)
    return cost

#### HELPER FUNCTIONS ####
"""
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
        return cost_fn(vals[0], vals[1], vals[2], vals[3], vals[4]) #TODO: same as 4 lines down
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
"""

#### MAIN OBJECTIVE FUNCTION AND GRADIENT ####
"""
def L(S, s0, objects, goal, fns):
"""
def L(S, s0, objects, goal, phase_weights=None, phase=None):
    # augment by calculating the accelerations from the velocities
    # interpolate all of the decision vars to get a finer trajcetory disretization
    S_aug = augment_s(s0, S)

    world_traj = WorldTraj(s0, S_aug, objects)
    tot_cost = 0.0
    cis, kinems, physs, coness, conts, velss, tasks, accels = \
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    """
    phys_cost, phys_grad = fns
    """

    for t in range(1,T_steps):
        if t == 1:
            s_tm1 = s0
        else:
            s_tm1 = get_s(S_aug, t-1)

        world_traj.step(t)
        s_aug_t = get_s(S_aug,t)

        """
        phys = phys_helper(s_aug_t, cost_fn=phys_cost)
        """
        if phase_weights == None:
            wci, wphys, wtask = 1.0, 1.0, 1.0
        else:
            wci, wphys, wtask = phase_weights[phase]
        ci = wci*L_CI(s_aug_t, t, objects, world_traj)
        #kinem = L_kinematics(s_aug_t, objects)
        phys = wphys*L_physics(s_aug_t, objects)
        #cones = L_cone(s_aug_t)
        #cont = L_contact(s_aug_t)
        #vels = L_vels(s_aug_t, s_tm1)
        accel = L_accel(s_aug_t)
        task = wtask*(L_task(s_aug_t, goal, t) + accel)
        cost = ci + phys + task# + ci #vels + ci #+ kinem + cones + cont

        cis += ci
        #kinems += kinem
        physs += phys
        #coness += cones
        #conts += cont
        #velss += vels
        accels += accel
        tasks += task
        tot_cost += cost

    print("cis:             ", cis)
    #print("kinematics:     ", kinems)
    print("physics:        ", physs)
    #print("cone:           ", coness)
    #print("contact forces: ", conts)
    #print("velocities:     ", velss)
    print("accels:     ", accels)
    print("task:           ", tasks)
    print("TOTAL: ", tot_cost)
    return tot_cost

"""
def L_grad(S, s0, objects, goal, fns):
    # calculate the interpolated values between the key frames (for now skip) -> longer S
    S_aug = augment_s(s0, S)
    world_traj = WorldTraj(s0, S, objects)
    phys_cost_fn, phys_grad_fn = fns
    grad_S = []

    for t in range(1,N):
        if t == 1:
            s_tm1 = s0
        else:
            s_tm1 = get_s_aug_t(S_aug, t-1)

        world_traj.step(t)
        s_aug_t = get_s_aug_t(S_aug,t)

        # sum up total for this time step
        phys_grad = phys_helper(s_aug_t, grad_fns=phys_grad_fn)
        grad_s = phys_grad #+ task + vels +ci + kinem + cones + cont + accels

        # append to overall gradient
        grad_S = np.concatenate([grad_S, grad_s])
    return grad_S
"""

#### MAIN FUNCTION ####
def CIO(goal, objects, s0, S0, params=None):
    """
    # get cost functions and their derivatives
    phys_cost, phys_grad = phys_fns()
    #task_cost, task_grad = task_fns()
    fns = [phys_cost, phys_grad]
    """
    """
    x = L(S0, s0, objects, goal, fns)
    print(x)
    """

    """
    # FOR TESTING A SINGLE traj
    pdb.set_trace()
    x = L(S0, s0, objects, goal)
    print(x)
    pdb.set_trace()
    """
    #pdb.set_trace()
    bounds = get_bounds()

    phase_weights = [(0.,0.,1.), (1.,0.1, 1.0), (1.0, 1.0, 1.0)] # (Lci, Lphys, Ltask)
    x = S0
    for phase in range(len(phase_weights)):
        res = minimize(fun=L, x0=x, args=(s0, objects, goal, phase_weights, phase), method='L-BFGS-B', bounds=bounds)
        x = res['x']
        print_result(x, s0)
        """
        print("Final cost: ", L(x, s0, objects, goal, fns))
        """
        print("Final cost: ", L(x, s0, objects, goal))
        input()
    """
    res = minimize(fun=L, x0=S0, args=(s0, objects, goal, fns), method='L-BFGS-B', bounds = bounds, jac=L_grad)
    """
    return x

def print_result(x, s0):
    # augement the output
    x_aug = augment_s(s0,x)

    # pose trajectory
    print("pose trajectory:")
    for t in range(T_steps):
        s_t = get_s(x_aug, t)
        box_pose = get_object_pos(s_t)
        print(box_pose, t)

    # velocity trajectory
    print("vel trajectory:")
    for t in range(T_steps):
        s_t = get_s(x_aug, t)
        box_vel = get_object_vel(s_t)
        print(box_vel, t)

    # accel trajectory
    print("accel trajectory:")
    for t in range(T_steps):
        s_t = get_s(x_aug, t)
        box_accel = get_object_accel(s_t)
        print(box_accel, t)

    # contact forces
    print("contact forces:")
    for t in range(T_steps):
        s_t = get_s(x_aug, t)
        contact_info = get_contact_info(s_t)
        force = contact_info[0]
        print(t, ":\n", force)

    # contact poses
    print("contact poses:")
    for t in range(T_steps):
        s_t = get_s(x_aug, t)
        contact_info = get_contact_info(s_t)
        pos = contact_info[1]
        print(t, ":\n", pos)

    # contact coefficients
    print("coefficients:")
    for t in range(T_steps):
        s_t = get_s(x_aug, t)
        contact_info = get_contact_info(s_t)
        contact = contact_info[2]
        print(t, ":\n", contact)
