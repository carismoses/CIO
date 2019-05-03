# CIO implementation
from scipy.optimize import fmin_l_bfgs_b, minimize
import numpy as np
from world import WorldTraj
from util import print_final, visualize_result, get_bounds, add_noise, save_run, normalize

#### MAIN OBJECTIVE FUNCTION ####
def L(S, goal, world, p, phase=0):
    def L_CI(t, world_t):
        cost = 0
        for (ci, (cont_obj, cont)) in enumerate(world.contact_state.items()):
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
        obj_pos = [world_t.manip_obj.pose.x, world_t.manip_obj.pose.y]
        goal_pos = [goal.x, goal.y]
        task_cost = I*np.linalg.norm(np.subtract(obj_pos, goal_pos))**2

        # small acceleration constraint (supposed to keep hand accel small, but
        # don't have a central hand so use grippers individually)
        accel_cost = 0.0
        for obj in world_t.get_all_objects():
            accel_cost += np.linalg.norm([obj.accel.x, obj.accel.y])**2
        accel_cost = p.lamb*accel_cost

        return accel_cost + task_cost

    world_traj = WorldTraj(S, world, p)
    total_cost, ci, phys, kinem, task = 0.0, 0.0, 0.0, 0.0, 0.0
    for (t, world_t) in enumerate(world_traj.worlds):
        ci += p.phase_weights[phase].w_CI*L_CI(t, world_t)
        phys += p.phase_weights[phase].w_physics*L_physics(t, world_t)
        kinem += 0.#p.phase_weights[phase].w_kinematics*L_kinematics(t, world_t)
        task += p.phase_weights[phase].w_task*L_task(t, world_t)
    total_cost = ci + phys + kinem + task

    global function_costs
    function_costs = [ci, phys, kinem, task]
    return total_cost

#### MAIN FUNCTION ####
def CIO(goal, world, p, single=False, start_phase=0):
    if single:
        # FOR TESTING A SINGLE traj
        S = world.traj_func(world, goal, p)
        S_noise = add_noise(S)
        visualize_result(world, goal, p, 'initial.gif', S_noise)
        tot_cost = L(S, goal, world, p, start_phase)
        print_final(*function_costs)
        return {}

    print('INITIAL')
    S = world.traj_func(world, goal, p)
    S_noise = add_noise(S)
    visualize_result(world, goal, p, 'initial.gif', S_noise)
    tot_cost = L(S_noise, goal, world, p)
    print_final(*function_costs)

    bounds = get_bounds(world, p)
    ret_info = {}
    x_init = S_noise
    for phase in range(start_phase, len(p.phase_weights)):
        print('BEGINNING PHASE:', phase)
        p.print_phase_weights(phase)
        res = minimize(fun=L, x0=x_init, args=(goal, world, p, phase),
                method='L-BFGS-B', bounds=bounds, options={'eps': 10.e-3})
        x_final = res['x']
        nit = res['nit']
        final_cost = res['fun']

        visualize_result(world, goal, p, 'phase_{}.gif'.format(phase), x_final)
        print_final(*function_costs)
        all_final_costs = function_costs
        ret_info[phase] = world.s0, x_final, final_cost, nit, all_final_costs
        x_init = x_final
    return ret_info
