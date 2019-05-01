# CIO implementation
from scipy.optimize import fmin_l_bfgs_b, minimize
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import shutil
import tempfile
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
def L_CI(s, t, objects, world_traj, p):
    e_O, e_H = world_traj.calc_e(s, t, objects)
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
#          3) TODO: collisions between fingers
def L_kinematics(s, objects, p):
    cost = 0
    # penalize collisions between all objects
    obj_num = 0
    while obj_num < len(objects):
        for col_object in objects[obj_num+1:]:
            col_dist = objects[obj_num].check_collisions(col_object)
            cost += col_lamb*col_dist
            obj_num += 1
    return kin_lamb*cost

def L_physics(s, objects,p):
    # get relevant state info
    fj, roj, cj = get_contact_info(s,p)
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

    # removing angular momentum conservation until roj vars optimized through L_CI
    cost = np.linalg.norm(f_tot - p_dot)**2 #+ np.linalg.norm(m_tot - l_dot)**2
    return cost

def L_cone(s, objects, p):
    # calc L_cone
    fj, _, _ = get_contact_info(s,p)
    ground, _, gripper1, gripper2 = objects
    contact_objects = [gripper1, gripper2, ground]
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
    return p.cone_lamb*cost

def L_contact(s, p):
    # discourage large contact forces
    fj, _, _ = get_contact_info(s,p)
    cost = 0.
    for j in range(p.N):
        cost += np.linalg.norm(fj[j])**2
    cost = p.cont_lamb*cost
    return cost

def L_task(s, goal, t, p):
    # l constraint: get object to desired pos
    I = 1 if t == (p.T_steps-1) else 0
    hb = get_object_pos(s)
    h_star = goal[1]
    cost = I*np.linalg.norm(hb - h_star)**2
    return cost

def L_accel(s, p):
    # small acceleration constraint (supposed to keep hand accel small, but
    # don't have a central hand so use grippers individually)
    o_dotdot = get_object_accel(s)
    g1_dotdot = get_gripper1_accel(s)
    g2_dotdot = get_gripper2_accel(s)

    cost = p.accel_lamb*(np.linalg.norm(o_dotdot)**2 + np.linalg.norm(g1_dotdot)**2 \
                + np.linalg.norm(g2_dotdot)**2)
    return cost

#### MAIN OBJECTIVE FUNCTION ####
def L(S, s0, objects, goal, p, phase_weights):
    global cis, kinems, physs, coness, conts, tasks, accels
    # augment by calculating the accelerations from the velocities
    # interpolate all of the decision vars to get a finer trajectory disretization
    S_aug = augment_s(s0, S, p)

    # world traj stores current object information used to calculate e vars for L_CI
    world_traj = WorldTraj(s0, S_aug, objects, p)
    tot_cost = 0.0
    cis, kinems, physs, coness, conts, tasks, accels = \
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    for t in range(1,p.T_steps):
        # s_tm1 is s from t-1
        if t == 1:
            s_tm1 = s0
        else:
            s_tm1 = get_s(S_aug, t-2, p)

        world_traj.step(t)
        s_aug_t = get_s(S_aug,t-1, p)

        wci, wphys, wtask = phase_weights

        ci = L_CI(s_aug_t, t, objects, world_traj, p)
        kinem = 0.#L_kinematics(s_aug_t, objects)
        phys = L_physics(s_aug_t, objects, p)
        cones = L_cone(s_aug_t, objects, p)
        cont = L_contact(s_aug_t, p)
        accel = L_accel(s_aug_t, p)
        task = L_task(s_aug_t, goal, t, p)
        cost = wtask*(task + accel) + wci*ci + wphys*(phys + kinem + cones + cont)

        cis += ci
        kinems += kinem
        physs += phys
        coness += cones
        conts += cont
        accels += accel
        tasks += task
        tot_cost += cost

        if verbose_step:
            print_step(ci, kinem, phys, cones, cont, accel, task, cost)
    return tot_cost

#### MAIN FUNCTION ####
def CIO(goal, objects, s0, S0, p, single=False):
    global iter

    if single:
        # FOR TESTING A SINGLE traj
        pdb.set_trace()
        x = L(S0, s0, objects, goal, p, (1.,1.,1.))
        print_final(x)
        return {}

    print('Cost of initial trajectory:')
    x = L(S0, s0, objects, goal, p, (1.,1.,1.))
    print_final(x)

    visualize_result(S0, s0, objects, goal, p, 'initial.gif')

    bounds = get_bounds(p)

    ret_info = {}
    x_init = S0
    for phase in range(len(p.phase_weights)):
        iter = 0
        if phase == 0:
            x_init = add_noise(x_init)
        phase_weights = p.phase_weights[phase]
        print("PHASE WEIGHTS:", phase_weights)
        res = minimize(fun=L, x0=x_init, args=(s0, objects, goal, p, phase_weights), \
                method='L-BFGS-B', bounds=bounds, options={'eps': 1.e-2}, callback=callback)
        x_final = res['x']
        nit = res['nit']
        final_cost = res['fun']

        #print_result(x_final, s0)
        visualize_result(x_final, s0, objects, goal, p, 'phase_{}.gif'.format(phase))
        print_final(final_cost)
        all_final_costs = [cis, kinems, physs, coness, conts, tasks, accels]
        ret_info[phase] = s0, x_final, final_cost, nit, all_final_costs
        x_init = x_final
    return ret_info

def callback(xk):
    global iter
    #print('iter: ', iter)
    iter += 1

def print_step(ci, kinem, phys, cones, cont, accel, task):
    print('----- step costs ------')
    print('cis:            ', ci)
    print('kinematics:     ', kinem)
    print('physics:        ', phys)
    print('cone:           ', cones)
    print('contact forces: ', cont)
    print('accels:         ', accel)
    print('task:           ', task)
    print('total:          ', ci + kinem + phys + cones + cont + accel + task)

def print_final(tot_cost):
    print('----- traj costs -----')
    print('cis:            ', cis)
    print('kinematics:     ', kinems)
    print('physics:        ', physs)
    print('cone:           ', coness)
    print('contact forces: ', conts)
    print('accels:         ', accels)
    print('task:           ', tasks)
    print('TOTAL: ', tot_cost)

def print_result(x, s0):
    # augement the output
    x_aug = augment_s(s0,x, p)

    # pose trajectory
    print('pose trajectory:')
    for t in range(p.T_steps+1):
        if t == 0:
            box_pose = get_object_pos(s0)
        else:
            s_t = get_s(x_aug, t-1, p)
            box_pose = get_object_pos(s_t)
        print(box_pose, t)

    # velocity trajectory
    print('vel trajectory:')
    for t in range(p.T_steps+1):
        if t == 0:
            box_vel = get_object_vel(s0)
        else:
            s_t = get_s(x_aug, t-1, p)
            box_vel = get_object_vel(s_t)
        print(box_vel, t)

    # accel trajectory
    print('accel trajectory:')
    for t in range(p.T_steps+1):
        if t == 0:
            box_accel = get_object_accel(s0)
        else:
            s_t = get_s(x_aug, t-1, p)
            box_accel = get_object_accel(s_t)
        print(box_accel, t)

    # contact forces
    print('contact forces:')
    for t in range(p.T_steps+1):
        if t == 0:
            contact_info = get_contact_info(s0,p)
            force = contact_info[0]
        else:
            s_t = get_s(x_aug, t-1, p)
            contact_info = get_contact_info(s_t,p)
            force = contact_info[0]
        print(t, ':\n', force)

    # contact poses
    print('contact poses:')
    for t in range(p.T_steps+1):
        if t == 0:
            contact_info = get_contact_info(s0,p)
            pos = contact_info[1]
        else:
            s_t = get_s(x_aug, t-1, p)
            contact_info = get_contact_info(s_t,p)
            pos = contact_info[1]
        print(t, ':\n', pos)

    # contact coefficients
    print('coefficients:')
    for t in range(p.T_steps+1):
        if t == 0:
            contact_info = get_contact_info(s0,p)
            contact = contact_info[2]
        else:
            s_t = get_s(x_aug, t-1, p)
            contact_info = get_contact_info(s_t,p)
            contact = contact_info[2]
        print(t, ':\n', contact)


def visualize_result(S0, s0, objects, goal, p, outfile):
    x_aug = augment_s(s0,S0,p)

    temp_dirpath = tempfile.mkdtemp()
    image_filenames = []

    ground, box, gripper1, gripper2 = objects

    for t in range(p.T_steps+1):
        plt.figure()

        if t == 0:
            s_t = s0
        else:
            s_t = get_s(x_aug, t-1, p)

        fj, roj, cj = get_contact_info(s_t,p)
        f_contact = np.array([0.0, 0.0])
        for j in range(p.N):
            f_contact += cj[j]*fj[j]
        f_gravity = np.array([0., -p.mass*p.gravity])
        ov = get_object_vel(s_t)
        fric = (-1*np.sign(ov[0]))*p.mu*cj[2]*fj[2][1]

        f_fric = np.array([fric, 0.])

        box_pose = get_object_pos(s_t)
        # get ro: roj in world frame
        rj = roj + np.tile(box_pose[:2], (3, 1))

        for rji, cji in zip(rj, cj):
            rj_circ = plt.Circle(rji, 1., fc='blue', alpha=cji)
            plt.gca().add_patch(rj_circ)

        gripper1_pose = get_gripper1_pos(s_t)
        gripper2_pose = get_gripper2_pos(s_t)

        try:
            box_rect = plt.Rectangle(box_pose[:2], box.width, box.height, fc='r')
            plt.gca().add_patch(box_rect)
        except AttributeError:
            circ = plt.Circle(box_pose[:2], box.radius, fc='r')
            plt.gca().add_patch(circ)

        box_origin = plt.Circle(box_pose[:2], 1., fc='gray')
        plt.gca().add_patch(box_origin)

        gripper1_endpoint = gripper1_pose[:2] + gripper1.length*np.array((np.cos(gripper1_pose[2]), np.sin(gripper1_pose[2])))
        plt.plot([gripper1_pose[0], gripper1_endpoint[0]], [gripper1_pose[1], gripper1_endpoint[1]], c='black', linewidth=3.)

        gripper2_endpoint = gripper2_pose[:2] + gripper2.length*np.array((np.cos(gripper2_pose[2]), np.sin(gripper2_pose[2])))
        plt.plot([gripper2_pose[0], gripper2_endpoint[0]], [gripper2_pose[1], gripper2_endpoint[1]], c='black', linewidth=3.)

        goal_circ = plt.Circle(goal[1][:2], 1., fc='g')
        plt.gca().add_patch(goal_circ)

        plt.arrow(box_pose[0], box_pose[1], f_contact[0], f_contact[1],
            head_width=0.5, head_length=1., fc='k', ec='k')

        plt.arrow(box_pose[0], box_pose[1], f_fric[0], f_fric[1],
            head_width=0.5, head_length=1., fc='k', ec='k')

        plt.arrow(box_pose[0], box_pose[1], f_gravity[0], f_gravity[1],
            head_width=0.5, head_length=1., fc='k', ec='k')

        plt.xlim((-10., 50))
        plt.ylim((-10., 50))
        plt.tight_layout()
        plt.axes().set_aspect('equal', 'datalim')
        image_filename = os.path.join(temp_dirpath, '{}.png'.format(t))
        plt.savefig(image_filename)
        plt.close()
        image_filenames.append(image_filename)
    images = [imageio.imread(filename) for filename in image_filenames]
    imageio.mimsave(outfile, images, fps=10)
    #print("Wrote out to {}.".format(outfile))

    shutil.rmtree(temp_dirpath)
