import numpy as np
from world import World, Line, Rectangle, Circle, ContactState, init_vars, Pose
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
con0 = ContactState(gripper1, box, f=[0.0, 0.0], ro=[-5.0, 10.0], c=.5)
con1 = ContactState(gripper2, box, f=[0.0, 0.0], ro=[5.0, 10.0], c=.5)
con2 = ContactState(ground, box, f=[0.0, 10.0], ro=[0.0, -5.0], c=.5)

world = World(ground=ground, manipulated_objects=[box], hands=[gripper1, gripper2], contact_state=[con0, con1, con2])

p = Params(world)
s0 = init_vars(world, p)

# initialize traj to all be the same as the starting state
S0 = np.zeros(p.len_S)
for k in range(p.K):
    S0[k*p.len_s:k*p.len_s+p.len_s] = s0

add_noise(S0);

goal = ("box", (50.0, rad, np.pi/2))


#visualize_result(S0, s0, world, goal, p, 'initial.gif')
open('initial.gif');

phase_info = CIO(goal, world, s0, S0, p)
