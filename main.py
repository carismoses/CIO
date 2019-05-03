import numpy as np
from world import World, Circle, Contact, Position
from params import Params, PhaseWeights
from CIO import visualize_result, CIO
from util import save_run
import argparse

def main(args):
    if args.debug:
        import pdb; pdb.set_trace()
    # objects
    radius = 5.0
    manip_obj = Circle(radius, Position(5.0,radius))
    finger0 = Circle(1.0, Position(-5.0, -5.0))
    finger1 = Circle(1.0, Position(15.0, -5.0))

    # initial contact information
    contact_state = {finger0 : Contact(f=(0.0, 0.0), ro=(-7., -7.), c=.5),
                     finger1 : Contact(f=(0.0, 0.0), ro=(7., -7.), c=.5)}
    goal = Position(5.0, 20.0)
    #goal = LinearVelocity(0.0, 0.0)

    world = World(manip_obj, [finger0, finger1], contact_state)
    phase_weights = [PhaseWeights(w_CI=1., w_physics=0., w_kinematics=0., w_task=0.)]
                     #PhaseWeights(w_CI=0., w_physics=1., w_kinematics=0., w_task=10.),
                     #PhaseWeights(w_CI=1., w_physics=1., w_kinematics=0., w_task=10.)]
    p = Params(world, K=10, delT=.1, phase_weights=phase_weights, lamb=10.e-3, mu=0.9)

    if args.single:
        phase_info = CIO(goal, world, p, single=True)
    else:
        phase_info = CIO(goal, world, p)

    if args.save:
        save_run(args.save, p, world, phase_info)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--single', action='store_true')
    parser.add_argument('--save', type=str)
    args = parser.parse_args()
    main(args)
