import numpy as np
import pdb
from scipy.interpolate import BPoly
import imageio
import matplotlib
import matplotlib.pyplot as plt
import shutil
import tempfile
import os
import warnings
import pickle
from copy import deepcopy
from collections import namedtuple
warnings.filterwarnings("ignore")

np.random.seed(0)

"""
Position Attributes
-----
x : float
    x position
y : float
    y position
"""
Position = namedtuple('Position', 'x y')
Pose = namedtuple('Pose', 'x y theta')
"""
LinearVelocity Attributes
-----
x : float
    linear velocity in x direction
y : float
    linear velocity in y direction
"""
LinearVelocity = namedtuple('LinearVelocity', 'x y')
Velocity = namedtuple('Velocity', 'x y theta')
Acceleration = namedtuple('Acceleration', 'x y theta')
"""
Contact Attributes
-----
f : tuple[2]
    x and y force
ro : tuple[2]
    position of applied force in the frame of the manipulated object
c : float in [0,1]
    the probability of being in contact
"""
Contact = namedtuple('Contact', 'f ro c')

Goal = namedtuple('Goal', 'time constraint')

def generate_world_traj(S, world, p):
    # get dyanamic and contact info from S
    dyn_info = {}
    for (i, dyn_obj) in enumerate(world.get_all_objects()):
        poses, vels, accels = calc_obj_dynamics(world.s0, S, p, i)
        dyn_info[i] = [poses, vels, accels]

    dyn_offset = len(world.get_all_objects())*6
    cont_info = {}
    for (ci, cont_obj) in enumerate(world.contact_state):
        fs, ros, cs = get_contact_info(world.s0, S, p, ci, dyn_offset)
        cont_info[ci] = [fs, ros, cs]

    # fill into list of new worlds
    worlds = []
    for t in range(p.T_steps+1):
        world_t = deepcopy(world)
        for i in range(len(world_t.get_all_objects())):
            world_t.set_dynamics(i,dyn_info[i][0][t], dyn_info[i][1][t], dyn_info[i][2][t])
        for ci in range(len(world_t.contact_state)):
            world_t.set_contact_state(ci, cont_info[ci][0][:,t], cont_info[ci][1][:,t], cont_info[ci][2][t])
        if t == 0:
            world_t.set_e_vars(None, p)
        else:
            world_t.set_e_vars(worlds[t-1], p)
        worlds += [world_t]
    return worlds

def stationary_traj(world, goal, p, traj_data=None):
    S = np.zeros(p.len_S)
    for k in range(p.K):
        S[k*p.len_s:k*p.len_s+p.len_s] = world.get_vars()
    return S

def save_run(file_name, goals, world, p, stage_info):
    fname = file_name + '.pickle'
    data = [goals, world, p, stage_info]
    with open(fname, 'wb') as handle:
        pickle.dump(data, handle)
    print('Saved run to', fname)

def data_from_file(file_name):
    fname = file_name + '.pickle'
    with open(fname, 'rb') as handle:
        goals, world, p, stage_info = pickle.load(handle)
    print('Read data from', fname)
    return goals, world, p, stage_info

def normalize(vec):
    mag = np.linalg.norm(vec)
    if mag == 0:
        return vec
    return np.divide(vec, mag)

def calc_deriv(x1, x0, delta):
    return np.divide(np.subtract(x1,x0),delta)

def get_dist(point0, point1):
    return np.linalg.norm(point1 - point0)**2

def get_bounds(world, p):
    cis = []
    dyn_obj_offset = 6*len(world.get_all_objects())
    for i in range(len(world.contact_state)):
        cis += [dyn_obj_offset + 5*i + 4]
    bounds = []
    for t in range(p.K):
        for v in range(p.len_s):
            if v in cis:
                bounds.append((0.,1.))
            else:
                bounds.append((None,None))
    return bounds

def get_contact_info(s0, S, p, ci, dyn_offset):
    offset = dyn_offset+5*ci

    f_traj_K = [s0[offset:offset+2]] + [S[offset+k*p.len_s:offset+k*p.len_s+2] for k in range(p.K)]
    ro_traj_K = [s0[offset+2:offset+4]] + [S[offset+k*p.len_s+2:offset+k*p.len_s+4] for k in range(p.K)]
    c_traj_K = [s0[offset+4]] + [S[offset+k*p.len_s+4] for k in range(p.K)]

    # interpolate contacts, contact forces, and contact poses
    f_traj_T = np.zeros((2,p.T_steps+1))
    ro_traj_T = np.zeros((2,p.T_steps+1))
    c_traj_T = np.zeros(p.T_steps+1)

    for k in range(1,p.K+1):
        f_left, ro_left, c_left = f_traj_K[k-1], ro_traj_K[k-1], c_traj_K[k-1]
        f_right, ro_right, c_right = f_traj_K[k], ro_traj_K[k], c_traj_K[k]

        # interpolate the contact forces (linear)
        f_traj_T[:,(k-1)*p.steps_per_phase:k*p.steps_per_phase+1] = linspace_vectors(f_left, f_right, p.steps_per_phase+1)

        # interpolate the contact poses (linear)
        ro_traj_T[:,(k-1)*p.steps_per_phase:k*p.steps_per_phase+1] = linspace_vectors(ro_left, ro_right, p.steps_per_phase+1)

        for t in range((k-1)*p.steps_per_phase,k*p.steps_per_phase+1):
            c_traj_T[t] = c_traj_K[k-1]

    return f_traj_T, ro_traj_T, c_traj_T

def calc_obj_dynamics(s0, S, p, index):
    # get pose and vel traj from S
    offset = 6*index
    pose_traj_K = [s0[offset:offset+3]] + [S[offset+k*p.len_s:offset+k*p.len_s+3] for k in range(p.K)]
    vel_traj_K = [s0[offset+3:offset+6]] + [S[offset+k*p.len_s+3:offset+k*p.len_s+6] for k in range(p.K)]

    # make spline functions
    spline_funcs = []
    for dim in range(3):
        x = np.linspace(0.,p.T_final, p.K+1)
        y = np.zeros((p.K+1,2))
        for k in range(p.K+1):
            y[k,:] = pose_traj_K[k][dim], vel_traj_K[k][dim]
        spline_funcs += [BPoly.from_derivatives(x,y,orders=3, extrapolate=False)]

    # use to interpolate pose
    pose_traj_T = []
    k = 0
    times = np.linspace(0.,p.T_final, p.T_steps+1)
    for t in times:
        if not t % p.delT_phase: # this is a phase
            pose_traj_T += [pose_traj_K[k]]
            k += 1
        else: # get from spline
            pose_traj_T += [[spline_funcs[0](t), spline_funcs[1](t), spline_funcs[2](t)]]

    # use FD to get velocities and accels
    vel_traj_T = [np.zeros(3)]
    for t in range(1,p.T_steps+1):
        p_tm1 = pose_traj_T[t-1]
        p_t= pose_traj_T[t]
        vel_traj_T += [calc_deriv(p_t, p_tm1, p.delT)]

    accel_traj_T = [np.zeros(3)]
    for t in range(1,p.T_steps+1):
        v_tm1 = vel_traj_T[t-1]
        v_t = vel_traj_T[t]
        accel_traj_T += [calc_deriv(v_t, v_tm1, p.delT)]

    return pose_traj_T, vel_traj_T, accel_traj_T

def linspace_vectors(vec0, vec1, num_steps):
    l = len(vec0)
    out_vec = np.zeros((l, num_steps))
    for j in range(l):
        left = vec0[j]
        right = vec1[j]
        out_vec[j,:] = np.linspace(left, right, num_steps)
    return out_vec

def add_noise(vec):
    # perturb all vars by gaussian noise
    mean = 0.
    var = 10.**-2
    for j in range(len(vec)):
        vec[j] += np.random.normal(mean, var)

    return vec

### FOR PRINTING AND VISUALIZING
def print_final(ci, phys, kinem, task):
    print('TRAJECTORY COSTS:')
    print('     CI:             ', ci)
    print('     kinematics:     ', kinem)
    print('     physics:        ', phys)
    print('     task:           ', task)
    print('  TOTAL: ', ci + kinem + phys + task)

def visualize_result(world, goals, p, outfile, S):
    world_traj = generate_world_traj(S, world, p)

    temp_dirpath = tempfile.mkdtemp()
    image_filenames = []

    for (t,world_t) in enumerate(world_traj):
        plt.figure()

        object = world_t.manip_obj

        obj_pose = object.pose

        # for each contact object plot the object, a circle at the r value showing the
        # c value, the force, and a line from the object to r which defines the
        # friction cone
        for (cont_obj, cont) in world_t.contact_state.items():
            # get ro in world frame
            r = cont.ro + np.array([obj_pose.x, obj_pose.y])
            rj_circ = plt.Circle(r, 1., fc='blue', alpha=cont.c)
            plt.gca().add_patch(rj_circ)

            hand_circ = plt.Circle([cont_obj.pose.x, cont_obj.pose.y], cont_obj.radius, fc='red')
            plt.gca().add_patch(hand_circ)

            plt.arrow(r[0], r[1], cont.f[0], cont.f[1],
                head_width=0.5, head_length=1., fc='k', ec='k')

            plt.plot([cont_obj.pose.x, r[0]], [cont_obj.pose.y, r[1]], c='black', linewidth=1.)

        # plot the manipulated object and the static objects in the environment
        for obj in world_t.static_objects+[world_t.manip_obj]:
            if obj==world_t.manip_obj:
                color = 'r'
            else:
                color='b'
            try:
                rect = plt.Rectangle([obj.pose.x-obj.width/2, obj.pose.y-obj.height/2],
                                                    obj.width, obj.height, fc=color)
                plt.gca().add_patch(rect)
            except:
                circ = plt.Circle([obj.pose.x, obj.pose.y], obj.radius, fc=color)
                plt.gca().add_patch(circ)

        # find position goals and print
        for goal in goals:
            if type(goal.constraint) == Position:
                goal_circ = plt.Circle(goal.constraint[:2], 1., fc='g')
                plt.gca().add_patch(goal_circ)

        # plot the imbalance between contact forces and acceleration
        f_tot = world_t.sum_forces()

        if t==0:
            accel = Acceleration(0., 0., 0.)
        else:
            accel = object.accel
        imbal = np.subtract(f_tot, np.array([accel.x, accel.y]))
        plt.arrow(obj_pose.x, obj_pose.y, imbal[0], imbal[1],
            head_width=0.5, head_length=1., fc='k', ec='k')

        plt.xlim((-25., 25))
        plt.ylim((-5., 30)) # the second arg (max) is set relative to xlim and ylim.min
        plt.tight_layout()
        plt.axes().set_aspect('equal', 'datalim')
        image_filename = os.path.join(temp_dirpath, '{}.png'.format(t))
        plt.savefig(image_filename)
        plt.close()
        image_filenames.append(image_filename)
    images = [imageio.imread(filename) for filename in image_filenames]
    try:
        os.mkdir('output')
    except:
        pass
    outfile = 'output/'+outfile
    imageio.mimsave(outfile, images, fps=10)
    print("Wrote out to {}.".format(outfile))

    shutil.rmtree(temp_dirpath)
