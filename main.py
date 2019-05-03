import numpy as np
from world import World, Line, Rectangle, Circle, Contact, Pose
from params import Params, PhaseWeights
from CIO import visualize_result, CIO, L
from util import save_run

# circle: origin is center of the circle
rad = 5.0
obj = Circle(radius=rad, pose=Pose(5.0,rad,0.0))
# box: origin is left bottom of box
#box = Rectangle((5.0, 0.0), np.pi/2, 10.0, 10.0, vel = (0.0, 0.0, 0.0))

# gripper1: origin is bottom of line
gripper1 = Circle(radius=1.0, pose=Pose(0.0, 5.0,np.pi/2))

# gripper2: origin is bottom of line
gripper2 = Circle(radius=1.0, pose=Pose(10.0, 5.0,np.pi/2))

# initial contact information
contact_state = {gripper1 : Contact(f=[0.0, 0.0], ro=[-5.1, 0.1], c=.5),
                 gripper2 : Contact(f=[0.0, 0.0], ro=[5.1, 0.1], c=.5)}

goal = (30.0, rad, np.pi/2)

def straight_traj(world, goal, p):
    straight = np.linspace(5,50,p.K)
    S = np.zeros(p.len_S)
    for k in range(p.K):
        s = world.get_vars()
        s[6*2] = straight[k]
        S[k*p.len_s:k*p.len_s+p.len_s] = s
    return S

world = World(manip_obj=obj, hands=[gripper1, gripper2], contact_state=contact_state)#, traj_func=straight_traj)

phase_weights = [PhaseWeights(w_CI=0., w_physics=0., w_kinematics=0., w_task=1.),
                 PhaseWeights(w_CI=0., w_physics=1., w_kinematics=0., w_task=10.),
                 PhaseWeights(w_CI=1., w_physics=1., w_kinematics=0., w_task=10.)]
p = Params(world, K=1, delT=.1, phase_weights=phase_weights, lamb=10.e-5)

phase_info = CIO(goal, world, p, single=False)

save_run('good_run', p, world, phase_info)
