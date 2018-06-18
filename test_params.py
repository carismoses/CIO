from main import main
import git
import csv
import os
from datetime import datetime, time
import params as p
import pdb
# outer loop to call CIO's main function with different params to test
accel_lamb_test_params = [1.e-15, 1.e-10, 1.e-5, 1.e-3, 1.e-1, 1., 1.e1]

date_time = datetime.now().strftime("%Y-%m-%d") + '_' + datetime.now().strftime("%H%M%S")
filename = '/Users/caris/CIO/output_files/' + date_time + '_accel_lambs.csv'

def test_params():
    for param_val in accel_lamb_test_params:
        s0, S0, S_final, final_cost, nit, all_final_costs = main({'accel_lamb':param_val})

        ## then save results ##
        # if file does not exist then make header
        if not os.path.isfile(filename):
            make_header()

        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha

        # write the following values to file: date, time, git hash, final cost, number of iterations,
        # s0, S0, S_final, p(arams)
        out = [sha, final_cost, nit, all_final_costs[0], all_final_costs[1]]
        out += p.get_global_params() + list(s0) + list(S0) + list(S_final)

        with open(filename, 'a') as f:
            writer  = csv.writer(f, lineterminator='\n')
            #for val in out:
            writer.writerow(out)
            f.close()

def make_header():
    s0_names = []
    x = 'x'
    y = 'y'
    po = '_pose_'
    v = '_vel_'
    f = '_f_'
    ro = '_ro_'
    c = '_c'
    th = 'th'

    for obj_num in range(p.num_objs):
        on = 'obj_' + str(obj_num)
        s0_names += [on+po+x, on+po+y, on+po+th, on+v+x, on+v+y, on+v+th]
    for j in range(p.N):
        cn = 'cont_' + str(j)
        s0_names += [cn+f+x, cn+f+y, cn+ro+x, cn+ro+y, cn+c]
    S0_names = []
    for i in range(p.K):
        S0_names += s0_names
    S_final_names = S0_names

    # make a dummy param dict just to get keys
    p_dummy = p.Params()
    param_names = list(p_dummy.default_params.keys())
    out_names = ['git hash', 'final cost', 'iterations', 'accel costs', 'task costs']
    out_names += param_names + s0_names + S0_names + S_final_names

    #pdb.set_trace()
    with open(filename, 'w') as f:
        writer  = csv.writer(f, lineterminator='\n')
        writer.writerow(out_names)
        f.close()

if __name__ == '__main__':
    test_params()
