import numpy as np
import pdb
from world import Line, Rectangle
from CIO import CIO
import params as p
from util import *

#### INITIALIZE DECISION VARIABLES ####
def init_vars(objects):
    _, box, _, _ = objects
    # create the initial system state: s0
    s0 = np.zeros(p.len_s)

    # fill in object poses and velocities
    for object in objects:
        if object.pose_index != None:
            s0[6*object.pose_index:6*object.pose_index+2] = object.pose
            s0[6*object.pose_index+2] = object.angle
            s0[6*object.pose_index+3:6*object.pose_index+6] = object.vel

    # initial contact information (just in contact with the ground):
    # [fxj fyj rOxj rOyj cj for j in N]
    # j = 0 gripper 1 contact
    # j = 1 gripper 2 contact
    # j = 2 ground contact
    # rO is in object (box) frame
    con0 = [0.0, 0.0,  0.0, 15.0, 0.5] # gripper1
    con1 = [0.0, 0.0, 10.0, 15.0, 0.5] # gripper2
    con2 = [0.0, 10.0,  5.0, 0.0, 1.0] # ground

    s0[18:p.len_s] = (con0 + con1 + con2)

    # initialize traj to all be the same as the starting state
    S0 = np.zeros(p.len_S)
    for k in range(p.K):
        S0[k*p.len_s:k*p.len_s+p.len_s] = s0

    return s0, S0

#### PARAMETERS #### TODO move them here
def init_objects():
    # ground: origin is left
    ground = Line((0.0, 0.0), 0.0, 30.0, contact_index = 2)

    # circle: origin is center of the circle
    rad = 5.0
    box = Circle(pose=(5.0, rad), angle=0.0, radius=rad, vel = (0.0, 0.0, 0.0), pose_index = 2)
    # box: origin is left bottom of box
    #box = Rectangle((5.0, 0.0), np.pi/2, 10.0, 10.0, vel = (0.0, 0.0, 0.0), pose_index = 2)

    # gripper1: origin is bottom of line
    gripper1 = Line((5.0, 15.0), 3*np.pi/2, 2.0, pose_index = 0, contact_index = 0,\
                    actuated = True)

    # gripper2: origin is bottom of line
    gripper2 = Line((15.0, 15.0), np.pi/2, 2.0, pose_index = 1, contact_index = 1,\
                    actuated = True)

    objects = [ground, box, gripper1, gripper2]
    goal = ("box", (50.0, rad, np.pi/2))
    return goal, objects

#### test trajectories ####
def make_test_traj(s0, S0, goal, objects):
    l,r = get_object_pos_ind()
    vl,vr = get_object_vel_ind()
    init_pos = get_object_pos(s0)
    goal = goal[1]
    interp_poses_x = np.linspace(init_pos[0],goal[0],p.K+1)
    interp_poses_y = np.linspace(init_pos[1],goal[1],p.K+1)
    interp_poses_th = np.linspace(init_pos[2],goal[2],p.K+1)
    for t in range(1,p.K+1):
        S0[(t-1)*p.len_s+l:(t-1)*p.len_s+r] = [interp_poses_x[t], interp_poses_y[t], \
                                            interp_poses_th[t]]
        vx = calc_deriv(interp_poses_x[t], interp_poses_x[t-1], p.delT_phase)
        vy = calc_deriv(interp_poses_y[t], interp_poses_y[t-1], p.delT_phase)
        vth = calc_deriv(interp_poses_th[t], interp_poses_th[t-1], p.delT_phase)
        S0[(t-1)*p.len_s+vl:(t-1)*p.len_s+vr] = [vx, vy, vth]
    return s0, S0

def make_test_traj2(s0, S0, goal, objects):
    l,r = get_object_pos_ind()
    vl,vr = get_object_vel_ind()
    init_pos = get_object_pos(s0)
    goal = goal[1]
    for t in range(1,p.K+1):
        if t == p.K:
            S0[(t-1)*p.len_s+l:(t-1)*p.len_s+r] = goal
    return s0, S0

def make_test_traj_cones(s0, S0, goal, objects):
    #pdb.set_trace()
    # gripper 1 force range (need to also vary the gripper 0 angle)
    f_mag = 10
    f_angles = np.linspace(0., 2*np.pi, p.K)
    for t in range(0,p.K):
        s = get_s(S0, t)
        fj_old, _, _ = get_contact_info(s)
        f_unit = np.array([np.cos(f_angles[t]), np.sin(f_angles[t])])
        fj_old[0] = f_mag*f_unit
        s_new = set_fj(fj_old, s)
        S0 = set_s(S0, s_new, t)
    return s0, S0

def make_test_traj_cones_2(s0, S0, goal, objects):
    f_gripper1 = np.array([1.5, 0.0])
    f_ground = np.array([0.0, 10.0])
    for t in range(0, p.K):
        s = get_s(S0, t)
        fj,_,cj = get_contact_info(s)
        fj[0] = f_gripper1
        fj[2] = f_ground
        cj = [1., 0., 1.]
        s_new = set_fj(fj, s)
        s_new = set_contact(cj, s)
        S0 = set_s(S0, s_new, t)
    return s0, S0

def make_min_traj(s0, S0, goal, objects):
    object_poses = [5.2, 6.1, 7., 8.5, 10]
    object_vel = [.8, 1.9, 2.5, 3.2, 2.8]
    j = 0
    gripper1_fx = [4.6, 4.7, 4.5, 3.9, 3.0]
    for t in range(0, p.K):
        s = get_s(S0, t)
        pos = get_object_pos(s)
        vel = get_object_vel(s)
        pos = [object_poses[t], pos[1], pos[2]]
        vel = [object_vel[t], vel[1], vel[2]]
        s = set_object_pos(pos, s)
        s = set_object_vel(vel, s)
        fj,_,c = get_contact_info(s)
        fj[j][0] = gripper1_fx[t]
        s = set_fj(fj,s)
        s = set_contact(c,s)
    return s0, S0

#### MAIN FUNCTION ####
def main(test_params={}, s0=None, S0=None, single=False, traj=0):
    #pdb.set_trace()

    # initialize objects
    goal, objects = init_objects()
    num_moveable_objects = len(objects) - 1 # don't count the ground

    # set params
    paramClass = p.Params(test_params, num_moveable_objects)
    p.set_global_params(paramClass)

    # initialize decision variables
    if s0 == None:
        s0, S0 = init_vars(objects)

    # potentially update with a test trajcetory
    traj = int(traj)
    if traj==1:
        s0, S0 = make_test_traj(s0, S0, goal, objects)
    if traj==2:
        s0, S0 = make_test_traj2(s0, S0, goal, objects)
    if traj==3:
        s0, S0 = make_test_traj_cones(s0, S0, goal, objects)
    if traj==4:
        s0, S0 = make_test_traj_cones_2(s0, S0, goal, objects)
    if traj==5:
        s0, S0 = make_min_traj(s0, S0, goal, objects)

    # run CIO
    phase_info = CIO(goal, objects, s0, S0, single)

    return phase_info

if __name__ == '__main__':
    main()
