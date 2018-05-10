import numpy as np
import pdb
from world import Line, Rectangle
from CIO import CIO

#### PARAMETERS ####
Ncontacts = 3 # 2 grippers and 1 ground contact
Tfinal = 10 # time steps to optimize over
delT = 1
mass = 10.0 # mass
gravity = 10.0 # gravity
mu = 0.5 # friction coefficient
len_s = 24
len_s_aug = 24 + 18 # includes vels and accels
len_S = len_s*(Tfinal-1)
len_S_aug = len_s_aug*(Tfinal-1) # will be more when interpolate between s values
lamb = 1.0 # lambda is a L_physics parameter
small_ang = .25
hw, hh = 5.0, 5.0 # half-width, half-height

def init_objects():
    # world origin is left bottom with 0 deg being along the x-axis, all poses are in world frame
    ground = Line(pose_index = None, contact_index = 2, pose = (0.0, 0.0), angle = 0.0, \
                    actuated = False, length = 30.0)

    # box: origin is left bottom of box
    box = Rectangle(pose_index = 2, contact_index = None, pose = (15.0, 0.0), angle = 0.0, \
                actuated = False, width = 10.0, height = 10.0)

    # gripper1: origin is bottom of line
    gripper1 = Line(pose_index = 0, contact_index = 0, pose = (15.0, 15.0), angle = np.pi/2, \
                actuated = True, clength = 2.0)

    # gripper2: origin is bottom of line
    gripper2 = Line(pose_index = 1, contact_index = 1, pose = (15.0, 15.0), angle = np.pi/2, \
                actuated = True, length = 2.0)

    objects = [ground, box, gripper1, gripper2]
    goal = ("box", (20.0, 0.0))
    return goal, objects

#### MAIN FUNCTION ####
def main():
    #pdb.set_trace()
    goal, objects = init_objects()
    x,f,d = CIO(goal, objects)

if __name__ == '__main__':
    main()
