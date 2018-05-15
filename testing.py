import numpy as np
import pdb
from CIO import L_CI, init_vars, calc_e
from util import interpolate_s, get_s_aug_t, get_contact_ind, get_roj_ind, len_s, T_final, N_contacts, get_object_pos_ind, get_gripper1_pos_ind, get_gripper2_pos_ind, get_s_t, get_contact_info
from main import init_objects
from world import WorldTraj

# run with T_final = 3 (in util.py)
def make_test_contact_traj(objects, S0):
    c_inds = get_contact_ind()
    roj_inds = get_roj_ind()
    o_pos_ind = get_object_pos_ind()
    g1_pos_ind = get_gripper1_pos_ind()
    g2_pos_ind = get_gripper2_pos_ind()

    for t in range(0,T_final-1):
        if t == 0:
            contacts = [0., 0., 1.]
            rojs = [[0.,10.],[10.,10.], [5.,0.]]
            for j in range(N_contacts):
                # cj
                S0[t*len_s+c_inds[j]] = contacts[j]
                # roj
                S0[t*len_s+roj_inds[j][0]:t*len_s+roj_inds[j][1]] = rojs[j]

        elif t == 1:
            contacts = [1., 1., 1.]
            rojs = [[0.,5.],[10.,5.],[5., 0.]]
            for j in range(N_contacts):
                # cj
                S0[t*len_s+c_inds[j]] = contacts[j]
                # roj
                S0[t*len_s+roj_inds[j][0]:t*len_s+roj_inds[j][1]] = rojs[j]
            # gripper poses
            S0[t*len_s+g1_pos_ind[0]:t*len_s+g1_pos_ind[1]] = (5., 5., np.pi/2)
            S0[t*len_s+g2_pos_ind[0]:t*len_s+g2_pos_ind[1]] = (15., 5., np.pi/2)

        elif t == 2:
            contacts = [1., 1., 0.]
            rojs = [[5.,0.],[5.,10.],[5.,0.]]
            for j in range(N_contacts):
                # cj
                S0[t*len_s+c_inds[j]] = contacts[j]
                # roj
                S0[t*len_s+roj_inds[j][0]:t*len_s+roj_inds[j][1]] = rojs[j]
            # object pose
            S0[t*len_s+o_pos_ind[0]:t*len_s+o_pos_ind[1]] = (5., 5., np.pi/2.)
            # gripper poses
            S0[t*len_s+g1_pos_ind[0]:t*len_s+g1_pos_ind[1]] = (10., 5., 0.)
            S0[t*len_s+g2_pos_ind[0]:t*len_s+g2_pos_ind[1]] = (10., 15., 0.)
    return S0

def main():
    pdb.set_trace()
    # initialize dec vars as usual
    goal, objects = init_objects()
    s0, S0 = init_vars(objects)

    # change some to get new contact states
    S_new = make_test_contact_traj(objects, S0)

    # continue as usual
    S_aug = interpolate_s(s0, S_new)
    world_traj = WorldTraj(s0, S_new, objects, N_contacts, T_final)
    world_traj.e_Os[:,0,:], world_traj.e_Hs[:,0,:] = calc_e(s0, objects)
    cost = 0
    for t in range(1,T_final):
        world_traj.step(t)
        s_aug_t = get_s_aug_t(S_aug,t)
        cost += L_CI(s_aug_t, t, objects, world_traj)

if __name__ == '__main__':
    main()
