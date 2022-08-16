# CIO implementation
from scipy.optimize import fmin_l_bfgs_b, minimize
import numpy as np
from cio.world import Position, LinearVelocity
from cio.util import print_final, visualize_result, get_bounds, add_noise, save_run, normalize, generate_world_traj

#### MAIN OBJECTIVE FUNCTION ####
def L(S, goals, world, p, stage=0):
    def L_CI(t, world_t):
        cost = 0
        for (ci, (cont_obj, cont)) in enumerate(world_t.contact_state.items()):
            cost += cont.c*(np.linalg.norm(world_t.e_O[ci])**2 +
                            np.linalg.norm(world_t.e_H[ci])**2 +
                            np.linalg.norm(world_t.e_dot_O[ci])**2 +
                            np.linalg.norm(world_t.e_dot_H[ci])**2)
        return cost

    # includes 1) limits on finger and arm joint angles (doesn't apply)
    #          2) distance from fingertips to palms limit (doesn't apply)
    #          3) collisions between fingers
    def L_kinematics(t, world_t):
        cost = 0
        # any overlap between objects is penalized
        all_objects = world_t.get_all_objects()
        obj_num = 0
        while obj_num < len(world_t.get_all_objects()):
            for col_object in all_objects[obj_num+1:]:
                col_dist = all_objects[obj_num].check_collisions(col_object)
                cost += col_dist
                obj_num += 1
        return cost

    def L_physics(t, world_t):
        # calculate sum of forces on object
        f_tot = [0.0, 0.0]
        for cont in world_t.contact_state.values():
            f_tot = np.add(f_tot, cont.c*cont.f)

        # calc change in linear momentum
        oa = world_t.manip_obj.accel
        p_dot = np.multiply(p.mass,[oa.x,oa.y])
        newton_cost = np.linalg.norm(f_tot - p_dot)**2

        force_reg_cost = 0
        for cont in world_t.contact_state.values():
            force_reg_cost += np.linalg.norm(cont.f)**2
        force_reg_cost = p.lamb*force_reg_cost

        # calc L_cone
        cone_cost = 0.0
        for (j,(cont_obj, cont)) in enumerate(world_t.contact_state.items()):
            n = cont_obj.get_surface_normal(world_t.pi_H[j])
            f_n = normalize(cont.f)
            angle = np.arccos(np.dot(f_n, n))
            cone_cost += max(angle - np.arctan(p.mu), 0)**2
        return force_reg_cost + newton_cost + cone_cost

    def L_task(t, world_t):
        # task constraint: get object to desired pos
        I = 1 if t == (p.T_steps-1) else 0
        task_cost = 0
        for goal in goals:
            if type(goal) == Position:
                obj_pos = [world_t.manip_obj.pose.x, world_t.manip_obj.pose.y]
                goal_pos = [goal.x, goal.y]
                task_cost += I*np.linalg.norm(np.subtract(obj_pos, goal_pos))**2
            elif type(goal) == LinearVelocity:
                obj_vel = [world_t.manip_obj.vel.x, world_t.manip_obj.vel.y]
                goal_vel = [goal.x, goal.y]
                task_cost += I*np.linalg.norm(np.subtract(obj_vel, goal_vel))**2

        # small acceleration constraint (supposed to keep hand accel small, but
        # don't have a central hand so use grippers individually)
        accel_cost = 0.0
        for obj in world_t.get_all_objects():
            accel_cost += np.linalg.norm([obj.accel.x, obj.accel.y])**2
        accel_cost = p.lamb*accel_cost

        return accel_cost + task_cost

    world_traj = generate_world_traj(S, world, p)
    total_cost, ci, phys, kinem, task = 0.0, 0.0, 0.0, 0.0, 0.0
    for (t, world_t) in enumerate(world_traj):
        ci += p.stage_weights[stage].w_CI*L_CI(t, world_t)
        phys += p.stage_weights[stage].w_physics*L_physics(t, world_t)
        kinem += 0.#p.stage_weights[stage].w_kinematics*L_kinematics(t, world_t)
        task += p.stage_weights[stage].w_task*L_task(t, world_t)
    total_cost = ci + phys + kinem + task

    global function_costs
    function_costs = [ci, phys, kinem, task]
    return total_cost

#### MAIN FUNCTION ####
from collections import namedtuple
StageInfo = namedtuple('StageInfo', 's0, x_final final_cost nit all_final_costs')
def CIO(goals, world, p, single=False, start_stage=0, traj_data=None, gif_tag=''):
    if single:
        # FOR TESTING A SINGLE traj
        S = world.traj_func(world, goals, p, traj_data)
        S = add_noise(S)
        visualize_result(world, goals, p, 'initial'+gif_tag+'.gif', S)
        tot_cost = L(S, goals, world, p, start_stage)
        print_final(*function_costs)
        return {}

    S = world.traj_func(world, goals, p, traj_data)
    if start_stage == 0:
        S = add_noise(S)
    visualize_result(world, goals, p, 'initial'+gif_tag+'.gif', S)
    tot_cost = L(S, goals, world, p)
    print_final(*function_costs)

    bounds = get_bounds(world, p)
    ret_info = {}
    x_init = S
    for stage in range(start_stage, len(p.stage_weights)):
        print('BEGINNING PHASE:', stage)
        p.print_stage_weights(stage)
        res = minimize(fun=L, x0=x_init, args=(goals, world, p, stage),
                method='L-BFGS-B', bounds=bounds, options={'eps': 10**-3, 'ftol': 1})
        x_final = res['x']
        nit = res['nit']
        final_cost = res['fun']

        visualize_result(world, goals, p, 'stage_{}'.format(stage)+gif_tag+'.gif', x_final)
        print_final(*function_costs)
        all_final_costs = function_costs
        ret_info[stage] = StageInfo(world.s0, x_final, final_cost, nit, all_final_costs)
        x_init = x_final
    return ret_info
