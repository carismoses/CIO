# CIO implementation
from scipy.optimize import fmin_l_bfgs_b, minimize
import numpy as np
import pdb
from world import WorldTraj
from util import *
import theano
import theano.tensor as T
from theano.ifelse import ifelse
from params import Params

verbose_step = False

#### SURFACE NORMALS ####
def get_normals(angles,p):
    nj = np.zeros((p.N, 2))
    for j in range(p.N):
        norm_angle = angles[j] + np.pi/2
        nj[j,:] = np.array([np.cos(norm_angle), np.sin(norm_angle)])
    return nj

#### OBJECTIVE FUNCTIONS ####
def L_CI(s, t, world, world_traj, p):
    e_O, e_H = world_traj.calc_e(s, t, world)
    e_O_tm1, e_H_tm1 = world_traj.e_Os[:,t-1,:], world_traj.e_Hs[:,t-1,:]
    _, _, cj = get_contact_info(s,p)

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
#          3) collisions between fingers
def L_kinematics(s, world, p):
    cost = 0
    # any overlap between objects is penalized
    all_objects = world.get_all_objects()
    obj_num = 0
    while obj_num < world.get_num_all_objects():
        for col_object in all_objects[obj_num+1:]:
            col_dist = all_objects[obj_num].check_collisions(col_object)
            cost += col_dist
            obj_num += 1
    return cost

def L_physics(s, p):
    # get relevant state info
    fj, roj, cj = get_contact_info(s,p)
    ov = get_object_vel(s)
    oa = get_object_accel(s)

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

    # removing angular momentum conservation until roj vars optimized through L_CI
    newton_cost = np.linalg.norm(f_tot - p_dot)**2 #+ np.linalg.norm(m_tot - l_dot)**2

    force_reg_cost = 0
    for j in range(p.N):
        force_reg_cost += np.linalg.norm(fj[j])**2
    force_reg_cost = p.lamb*force_reg_cost

    return force_reg_cost + newton_cost

def L_cone(s, world, p):
    # calc L_cone
    fj, _, _ = get_contact_info(s,p)
    contact_objects = world.get_contact_objects()
    cost = 0.0
    # get contact surface angles
    angles = np.zeros((p.N))
    for j in range(p.N):
        # TODO: this only works currently because all contact surfaces are lines...
        # will need to change if have different shaped contact surfaces
        angles[j] = contact_objects[j].angle
    # get unit normal to contact surfaces at pi_j using surface line
    nj = get_normals(angles, p)
    for j in range(p.N):
        cosangle_num = np.dot(fj[j], nj[j,:])
        cosangle_den = np.dot(np.linalg.norm(fj[j]), np.linalg.norm(nj[j,:]))
        if cosangle_den == 0.0: # TODO: is this correct?
            angle = 0.0
        else:
            angle = np.arccos(cosangle_num/cosangle_den)
        cost += max(angle - np.arctan(p.mu), 0)**2
    return cost

def L_task(s, goal, t, p):
    # task constraint: get object to desired pos
    I = 1 if t == (p.T_steps-1) else 0
    hb = get_object_pos(s)
    h_star = goal[1]
    task_cost = I*np.linalg.norm(hb - h_star)**2

    # small acceleration constraint (supposed to keep hand accel small, but
    # don't have a central hand so use grippers individually)
    o_dotdot = get_object_accel(s)
    g1_dotdot = get_gripper1_accel(s)
    g2_dotdot = get_gripper2_accel(s)

    accel_cost = p.lamb*(np.linalg.norm(o_dotdot)**2 + np.linalg.norm(g1_dotdot)**2 \
                + np.linalg.norm(g2_dotdot)**2)
    return accel_cost + task_cost

#### MAIN OBJECTIVE FUNCTION ####
def L(S, s0, world, goal, p, phase=0):
    global cis, physs, kinems, tasks
    # augment by calculating the accelerations from the velocities
    # interpolate all of the decision vars to get a finer trajectory disretization
    S_aug = augment_s(s0, S, p)

    # world traj stores current object information used to calculate e vars for L_CI
    world_traj = WorldTraj(s0, S_aug, world, p)
    tot_cost = 0.0
    cis, physs, kinems, tasks = 0.0, 0.0, 0.0, 0.0

    for t in range(1,p.T_steps):
        # s_tm1 is s from t-1
        if t == 1:
            s_tm1 = s0
        else:
            s_tm1 = get_s(S_aug, t-2, p)

        world_traj.step(t)
        s_aug_t = get_s(S_aug,t-1, p)

        ci = p.phase_weights[phase].w_CI*L_CI(s_aug_t, t, world, world_traj, p)
        phys = p.phase_weights[phase].w_physics*L_physics(s_aug_t, p)
        kin = 0.#p.phase_weights[phase].w_kinematics*L_kinematics(s_aug_t, world, p)
        task = p.phase_weights[phase].w_task*L_task(s_aug_t, goal, t, p)
        cost = ci + phys + kin + task

        cis += ci
        physs += phys
        kinems += kin
        tasks += task
        tot_cost += cost

        if verbose_step:
            print_step(ci, kinem, phys, task)
    return tot_cost

#### MAIN FUNCTION ####
def CIO(goal, world, s0, S0, p, single=False):
    global iter

    if single:
        # FOR TESTING A SINGLE traj
        pdb.set_trace()
        x = L(S0, s0, world, goal, p)
        print_final(cis, kinems, physs, tasks)
        return {}

    print('Cost of initial trajectory:')
    x = L(S0, s0, world, goal, p)
    print_final(cis, kinems, physs, tasks)

    visualize_result(S0, s0, world, goal, p, 'initial.gif')

    bounds = get_bounds(p)

    ret_info = {}
    x_init = S0
    for phase in range(len(p.phase_weights)):
        iter = 0
        if phase == 0:
            x_init = add_noise(x_init)
        print('BEGINNING PHASE:', phase)
        p.print_phase_weights(phase)
        res = minimize(fun=L, x0=x_init, args=(s0, world, goal, p, phase), \
                method='L-BFGS-B', bounds=bounds, options={'eps': 10.e-3}, callback=callback)
        x_final = res['x']
        nit = res['nit']
        final_cost = res['fun']

        visualize_result(x_final, s0, world, goal, p, 'phase_{}.gif'.format(phase))
        print_final(cis, kinems, physs, tasks)
        all_final_costs = [cis, kinems, physs, tasks]
        ret_info[phase] = s0, x_final, final_cost, nit, all_final_costs
        x_init = x_final
    return ret_info

def callback(xk):
    global iter
    #print('iter: ', iter)
    iter += 1
