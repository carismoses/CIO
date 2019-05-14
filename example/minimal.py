import numpy as np
from cio.world import World, Circle, Contact, Position
from cio.params import Params, StageWeights
from cio.optimization import CIO
from cio.util import save_run, visualize_result
import argparse
from collections import OrderedDict

def main(args):
    if args.debug:
        import pdb; pdb.set_trace()

    # objects
    radius = 5.0
    manip_obj = Circle(radius, Position(5.0,radius))
    finger0 = Circle(1.0, Position(-5.0, -5.0))
    finger1 = Circle(1.0, Position(15.0, -5.0))

    # initial contact information
    contact_state = OrderedDict([(finger0, Contact(f=(0.0, 0.0), ro=(-7., -7.), c=.5)),
                                 (finger1, Contact(f=(0.0, 0.0), ro=(7., -7.), c=.5))])

    goals = [Position(5.0, 20.0)]

    world = World(manip_obj, [finger0, finger1], contact_state)

    # optimization parameters
    p = Params(world)

    # just get the cost of a single trajectory
    if args.single:
        stage_info = CIO(goals, world, p, single=True)
    # run the optimization
    else:
        stage_info = CIO(goals, world, p)

    # save the run to a .pickle file
    if args.save:
        save_run(args.save, p, world, stage_info)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--single', action='store_true')
    parser.add_argument('--save', type=str)
    args = parser.parse_args()
    main(args)
