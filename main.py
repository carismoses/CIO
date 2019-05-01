import numpy as np
from world import Line, Rectangle, Circle

# ground: origin is left
ground = Line((0.0, 0.0), 0.0, 30.0, contact_index = 2)

# circle: origin is center of the circle
rad = 5.0
box = Circle(pose=(5.0, rad), angle=0.0, radius=rad, vel = (0.0, 0.0, 0.0), pose_index = 2)
# box: origin is left bottom of box
#box = Rectangle((5.0, 0.0), np.pi/2, 10.0, 10.0, vel = (0.0, 0.0, 0.0), pose_index = 2)

# gripper1: origin is bottom of line
gripper1 = Line((0.0, 20.0), np.pi/2, 2.0, pose_index = 0, contact_index = 0,\
                actuated = True)

# gripper2: origin is bottom of line
gripper2 = Line((10.0, 20.0), np.pi/2, 2.0, pose_index = 1, contact_index = 1,\
                actuated = True)

objects = [ground, box, gripper1, gripper2]

num_moveable_objects = len(objects) - 1 # don't count the ground
from params import Params
p = Params(num_moveable_objects)

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
con0 = [0.0, 0.0,  -5.0, 10.0, 1.0] # gripper1
con1 = [0.0, 0.0, 5.0, 10.0, 1.0] # gripper2
con2 = [0.0, 10.0,  0.0, -5.0, 1.0] # ground

s0[18:p.len_s] = (con0 + con1 + con2)

from util import add_noise

# initialize traj to all be the same as the starting state
S0 = np.zeros(p.len_S)
for k in range(p.K):
    S0[k*p.len_s:k*p.len_s+p.len_s] = s0

add_noise(S0);

goal = ("box", (50.0, rad, np.pi/2))

from CIO import visualize_result
visualize_result(S0, s0, objects, goal, p, 'initial.gif')
open('initial.gif');

from CIO import CIO
phase_info = CIO(goal, objects, s0, S0, p)
