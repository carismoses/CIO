import numpy as np
import pdb
from world import Line, Rectangle
from CIO import CIO
import params as p
from util import *

testing1 = False
testing2 = False

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
    con2 = [0.0, 0.0,  5.0, 0.0, 0.5] # ground

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

    # box: origin is left bottom of box
    box = Rectangle((5.0, 0.0), np.pi/2, 10.0, 10.0, vel = (0.0, 0.0, 0.0), pose_index = 2)

    # gripper1: origin is bottom of line
    gripper1 = Line((5.0, 15.0), 3*np.pi/2, 2.0, pose_index = 0, contact_index = 0,\
                    actuated = True)

    # gripper2: origin is bottom of line
    gripper2 = Line((15.0, 15.0), np.pi/2, 2.0, pose_index = 1, contact_index = 1,\
                    actuated = True)

    objects = [ground, box, gripper1, gripper2]
    goal = ("box", (15.0, 0.0, np.pi/2))
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
        vx = calc_deriv(interp_poses_x[t], interp_poses_x[t-1], delT_phase)
        vy = calc_deriv(interp_poses_y[t], interp_poses_y[t-1], delT_phase)
        vth = calc_deriv(interp_poses_th[t], interp_poses_th[t-1], delT_phase)
        S0[(t-1)*p.len_s+vl:(t-1)*p.len_s+vr] = [vx, vy, vth]
    return s0, S0

def make_test_traj2(s0, S0, goal, objects):
    l,r = get_object_pos_ind()
    vl,vr = get_object_vel_ind()
    init_pos = get_object_pos(s0)
    goal = goal[1]
    for t in range(1,p.K+1):
        if t == p.K:
            S0[(t-1)*p.len_s+l:(t-1)*len_s+r] = goal
    return s0, S0

#### MAIN FUNCTION ####
def main(test_params={}):
    #pdb.set_trace()

    # initialize objects
    goal, objects = init_objects()
    num_moveable_objects = len(objects) - 1 # don't count the ground

    # set params
    paramClass = p.Params(test_params, num_moveable_objects)
    p.set_global_params(paramClass)

    # initialize decision variables
    s0, S0 = init_vars(objects)

    # potentially update with a test trajcetory
    if testing1:
        s0, S0 = make_test_traj(s0, S0, goal, objects)
    if testing2:
        s0, S0 = make_test_traj2(s0, S0, goal, objects)

    # run CIO
    phase_info = CIO(goal, objects, s0, S0)

    return phase_info

if __name__ == '__main__':
    main()
