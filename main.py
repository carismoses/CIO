import numpy as np
import pdb
from world import Line, Rectangle
from CIO import CIO

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
    goal = ("box", (15.0, 0.0))
    return goal, objects

#### MAIN FUNCTION ####
def main():
    pdb.set_trace()
    goal, objects = init_objects()
    x,f,d = CIO(goal, objects)

if __name__ == '__main__':
    main()
