import numpy as np
from world import World, Line, Rectangle, Circle, Contact, Pose
from params import Params
from util import add_noise
from CIO import visualize_result, CIO

import pdb; pdb.set_trace()
# ground: origin is left
ground = Line(length=30.0, pose=Pose(0.0,0.0,0.0))

# circle: origin is center of the circle
rad = 5.0
box = Circle(radius=rad, pose=Pose(5.0,rad,0.0))
# box: origin is left bottom of box
#box = Rectangle((5.0, 0.0), np.pi/2, 10.0, 10.0, vel = (0.0, 0.0, 0.0))

# gripper1: origin is bottom of line
gripper1 = Line(length=2.0, pose=Pose(0.0, 20.0,np.pi/2))

# gripper2: origin is bottom of line
gripper2 = Line(length=2.0, pose=Pose(10.0, 20.0,np.pi/2))

# initial contact information (just in contact with the ground):
contact_state = {gripper1 : Contact(f=[0.0, 0.0], ro=[-5.0, 10.0], c=.5),
                 gripper2 : Contact(f=[0.0, 0.0], ro=[5.0, 10.0], c=.5),
                 ground : Contact(f=[0.0, 10.0], ro=[0.0, -5.0], c=.5)}

goal = ("box", (50.0, rad, np.pi/2))

world = World(ground=ground, manipulated_objects=[box], hands=[gripper1, gripper2], contact_state=contact_state)

'''
could replace the above line with
world = World(ground=.... , init_traj=some_other_function)
some_other_function would need to take in (goal, world, p) and return a list of worlds the length of the keyframes (p.K)
'''

p = Params(world)

visualize_result(world, goal, p, 'initial.gif')

phase_info = CIO(goal, world, p)

'''
show how to change Params
how to change init_Traj function
how to run a single cost calc
'''
