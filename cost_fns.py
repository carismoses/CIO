def L_CI(s, t, objects, world_traj):
    e_O, e_H = calc_e(s, objects)
    world_traj.e_Os[:,t,:], world_traj.e_Hs[:,t,:] = e_O, e_H
    _, _, cj = get_contact_info(s)
    e_O_tm1, e_H_tm1 = world_traj.e_Os[:,t-1,:], world_traj.e_Hs[:,t-1,:]

    # calculate the edots
    e_O_dot = (e_O - e_O_tm1)/delT
    e_H_dot = (e_H - e_H_tm1)/delT

    # calculate the contact invariance cost
    cost = 0
    for j in range(N_contacts):
        cost += cj[j]*(np.linalg.norm(e_O[j,:])**2 + np.linalg.norm(e_H[j,:])**2 \
                + np.linalg.norm(e_O_dot[j,:])**2 + np.linalg.norm(e_H_dot)**2)
    return cost

# includes 1) limits on finger and arm joint angles (doesn't apply)
#          2) distance from fingertips to palms limit (doesn't apply)
#          3) TODO: collisions between fingers
def L_kinematics(s, objects):
    cost = 0
    # penalize collisions between all objects
    obj_num = 0
    while obj_num < len(objects):
        for col_object in objects[obj_num+1:]:
            col_dist = objects[obj_num].check_collisions(col_object)
            cost += col_lamb*col_dist
            obj_num += 1
    return cost


def L_cone(s):
    # calc L_cone
    fj, roj, cj = get_contact_info(s)
    cost = 0.0
    # get contact surface angles
    angles = np.zeros((N_contacts))
    for j in range(N_contacts):
        angles[j] = contact_objects[j].angle
    # get unit normal to contact surfaces at pi_j using surface line
    nj = get_normals(angles)
    for j in range(N_contacts):
        if cj[j] > 0.0: # TODO: fix.. don't think it's working..
            cosangle_num = np.dot(fj[j], nj[j,:])
            cosangle_den = np.dot(np.linalg.norm(fj[j]), np.linalg.norm(nj[j,:]))
            if cosangle_den == 0.0: # TODO: is this correct?
                angle = 0.0
            else:
                angle = np.arccos(cosangle_num/cosangle_den)
            cost += max(angle - np.arctan(mu), 0)**2
    return cost

def L_contact(s):
    # discourage large contact forces
    fj, roj, cj = get_contact_info(s)
    cost = 0.
    for j in range(N_contacts):
        cost += np.linalg.norm(fj[j])**2
    cost = phys_lamb*term
    return cost

def L_vels(s, s_tm1):
    # penalize differences between poses and velocities
    cost = sum((get_object_vel(s) - (get_object_pos(s) - get_object_pos(s_tm1))/delT)**2)
    return cost
