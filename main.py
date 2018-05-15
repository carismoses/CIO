import numpy as np
import pdb
from world import Line, Rectangle
from CIO import CIO

testing = False

#### PARAMETERS #### TODO move them here
def init_objects():
    # ground: origin is left
    ground = Line((0.0, 0.0), 0.0, 30.0, contact_index = 2)

    # box: origin is left bottom of box
    box = Rectangle((5.0, 0.0), np.pi/2, 10.0, 10.0, pose_index = 2)

    # gripper1: origin is bottom of line
    gripper1 = Line((5.0, 15.0), np.pi/2, 2.0, pose_index = 0, contact_index = 0,\
                    actuated = True)

    # gripper2: origin is bottom of line
    gripper2 = Line((15.0, 15.0), np.pi/2, 2.0, pose_index = 1, contact_index = 1,\
                    actuated = True)

    objects = [ground, box, gripper1, gripper2]
    goal = ("box", (15.0, 0.0, np.pi/2))
    return goal, objects

#### test trajectories ####
from CIO import init_vars
from util import *

def make_test_traj(goal, objects):
    s0, S0 = init_vars(objects)
    l,r = get_object_pos_ind()
    init_pos = get_object_pos(s0)
    goal = goal[1]
    interp_poses_x = np.linspace(init_pos[0],goal[0],T_final+1)
    interp_poses_y = np.linspace(init_pos[1],goal[1],T_final+1)
    interp_poses_th = np.linspace(init_pos[2],goal[2],T_final+1)
    for t in range(1,T_final):
        S0[(t-1)*len_s+l:(t-1)*len_s+r] = [interp_poses_x[t], interp_poses_y[t], \
                                            interp_poses_th[t],]
    return s0, S0

#### MAIN FUNCTION ####
def main():
    goal, objects = init_objects()
    if testing:
        s0, S0 = make_test_traj(objects)
        x,f,d = CIO(goal, objects, s0, S0)
    else:
        x,f,d = CIO(goal, objects)
if __name__ == '__main__':
    main()
