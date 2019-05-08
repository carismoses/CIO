import numpy as np
from world import World, Circle, Contact, Position, WorldTraj, Pose
from params import Params, StageWeights
from CIO import visualize_result, CIO
from util import save_run, data_from_file
import argparse
from collections import OrderedDict

def traj_from_file(world, goal, p, traj_data):
    goals, world, p, stage_info = traj_data
    s0, S_final, final_cost, nit, all_final_costs = stage_info[0]

    # make a world traj from S_final
    world_traj = WorldTraj(S_final, world, p)
    for (t, world_t) in enumerate(world_traj.worlds):
        # phase, want to update
        if t % p.steps_per_phase:
            # if the object is moving
            if abs(world_t.manip_obj.vel.y) > 0.:
                for finger in world_t.fingers:
                    # make fingers touch
                    c = 1.0
                    # balance object acceleration with contact forces
                    force_mag = p.mass*world_t.manip_obj.accel.y
                    fy = force_mag/2
                    # move fingers and r's to be in contact with the obj
                    R = world_t.manip_obj.radius + finger.radius
                    if finger.pose.x > world_t.manip_obj.pose.x:
                        fpx = world_t.manip_obj.pose.x + R*np.sin(.5)
                        rox = world_t.manip_obj.radius*np.sin(.5)
                    else:
                        fpx = world_t.manip_obj.pose.x - R*np.sin(.5)
                        rox = -world_t.manip_obj.radius*np.sin(.5)
                    fpy = world_t.manip_obj.pose.y - R*np.cos(.5)
                    roy = -world_t.manip_obj.radius*np.cos(.5)
                    world_t.contact_state[finger] = Contact(c=c, ro=[rox,roy], f=[0.,fy])
                    finger.pose = Pose(x=fpx, y=fpy, theta=0.)

    # make S from the worlds
    S = np.zeros(p.len_S)
    k = 0
    for (t, world_t) in enumerate(world_traj.worlds):
        if t % p.steps_per_phase:
            S[k*p.len_s:k*p.len_s+p.len_s] = world_t.get_vars()
            k += 1
    return S

def main(args):
    if args.debug:
        import pdb; pdb.set_trace()

    if args.from_file:
        goals, world, p, stage_info = data_from_file(args.from_file)
        world.traj_func = traj_from_file
    else:
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

        stage_weights=[StageWeights(w_CI=0.1, w_physics=0.1, w_kinematics=0.0, w_task=1.0),
                        StageWeights(w_CI=10.**1, w_physics=10.**0, w_kinematics=0., w_task=10.**1)]
        p = Params(world, K=10, delT=0.05, delT_phase=0.5, mass=1.0,
                        mu=0.9, lamb=10**-3, stage_weights=stage_weights)

    traj_data = None
    if args.from_file:
        traj_data = goals, world, p, stage_info

    start_stage = 0
    if args.start_stage:
        start_stage = args.start_stage

    stage_info = CIO(goals, world, p, start_stage=start_stage, traj_data=traj_data, single=args.single)

    if args.save:
        save_run(args.save, goals, world, p, stage_info)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--single', action='store_true')
    parser.add_argument('--save', type=str)
    parser.add_argument('--from-file', type=str)
    parser.add_argument('--start-stage', type=int)
    args = parser.parse_args()
    main(args)
