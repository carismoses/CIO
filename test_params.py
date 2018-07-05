from main import main
import git
import csv
import os
from datetime import datetime, time
import params as p
import pdb
import sys
import time
from util import *

# outer loop to call CIO's main function with different params to test
accel_lamb_test_params = [1.e-3,]
# ci, phys, task
phase_weights_test = [[(0.,0.,1.), (0.,1.,1.)],]
cont_lamb_test_params = [1.e-7]
cone_lamb_test_params = [3.e-3]

# TODO: unhard code these vars
len_s = 33
num_ps = 22 # length of output vars to csv that are not part of s0 or Sfinal

fn_prefix = '/Users/caris/CIO/output_files/'
fn_suff = '.csv'

p_dummy = p.Params()

# this is for running an optimization starting from the results of a previous phase
# start_phase is the phase that you would like the start the optimization from
# file_name is the date-time stamp from the file to get initial vars from
# file_line_num is the line number you would like to load the initial vars from
def restart(old_file_name, file_line_num, single=False):
    old_filename = fn_prefix + old_file_name + fn_suff
    if not os.path.isfile(old_filename):
        print('This file does not exist!')
        return

    # read in old file to get starting information from
    with open(old_filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        reader = list(reader)

    # Run CIO from this point on
    start_vars = reader[file_line_num]
    s0, S0 = make_init_vars(start_vars)

    # check for file compatibility (TODO: this is before you change for test
    # params, so set params before this check, or check for compatibility after test params are set)
    if len(S0)/len_s != p_dummy.K:
        print('The file you are attempting to restart from has a different K value')
        print('than the K you are currently trying to test')

    # add the file the old file information
    date_time = datetime.now().strftime("%Y-%m-%d") + '_' + datetime.now().strftime("%H%M")
    filename = fn_prefix + date_time + fn_suff

    write(filename, ['Coming from file : ' + old_filename])
    write(filename, ['Coming from line number : ' + str(file_line_num)])

    # this 1 should not be hard coded but the phase number is currently the 2nd number in the vars
    start_phase = int(start_vars[1]) + 1
    test_params(s0, S0, start_phase, filename, single)

# TODO: should also return parameters used to ensure that the same parameters are used in the restart
def make_init_vars(init_vars):
    s0 = [float(i) for i in init_vars[num_ps+1:num_ps+1+len_s]]
    S0 = [float(i) for i in init_vars[num_ps+1+len_s:]]
    return s0, S0

def write_to_file(ret_info, filename):
    # write the following values to file: git hash, phase, final cost,
    # number of iterations, sub cost info,p(arams),  s0, S_final
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    for (phase, ret) in ret_info.items():
        s0, S_final, final_cost, nit, all_final_costs = ret
        out = [sha, phase, final_cost, nit] + all_final_costs + \
                p.get_global_params() + list(s0) + list(S_final)

        write(filename, out)

def test_params(s0=None, S0=None, start_phase=0, filename=None, single=False, traj=0):
    if filename == None:
        date_time = datetime.now().strftime("%Y-%m-%d") + '_' + datetime.now().strftime("%H%M")
        filename = fn_prefix + date_time + fn_suff

    # make header
    make_header(filename)

    for accel_val in accel_lamb_test_params:
        for cont_lamb in cont_lamb_test_params:
            for cone_lamb in cone_lamb_test_params:
                for phase_weights in phase_weights_test:
                    test_params = {'accel_lamb':accel_val, 'phase_weights':phase_weights, \
                    'start_phase': start_phase, 'cont_lamb': cont_lamb, 'cone_lamb': cone_lamb}
                    ret_info = main(test_params,s0,S0, single=single, traj=traj)
                    write_to_file(ret_info, filename)

def make_header(filename):
    x = 'x'
    y = 'y'
    po = '_pose_'
    v = '_vel_'
    f = '_f_'
    ro = '_ro_'
    c = '_c'
    th = 'th'

    s_names = []
    for obj_num in range(p_dummy.num_objs):
        on = 'obj_' + str(obj_num)
        s_names += [on+po+x, on+po+y, on+po+th, on+v+x, on+v+y, on+v+th]
    for j in range(p_dummy.N):
        cn = 'cont_' + str(j)
        s_names += [cn+f+x, cn+f+y, cn+ro+x, cn+ro+y, cn+c]

    s0_names = []
    for i in range(len(s_names)):
        s0_names += [s_names[i] + '_K=0']

    S_final_names = []
    for k in range(p_dummy.K):
        for j in range(len(s_names)):
            S_final_names += [s_names[j] + '_K=' + str(k+1)]

    # make a dummy param dict just to get keys
    param_names = list(p_dummy.default_params.keys())
    out_names = ['git hash', 'phase', 'final cost', 'iterations', 'ci_costs', 'kinem_costs', \
                    'phys_costs', 'cones_costs', 'cont_costs', 'task_costs', 'accel_costs']
    out_names += param_names + s0_names + S_final_names

    write(filename, out_names)

def write(filename, out):
    with open(filename, 'a') as f:
        writer  = csv.writer(f, lineterminator='\n')
        writer.writerow(out)
        f.close()

def pretty_print(file_name, row_num = None):
    filename = fn_prefix + str(file_name) + fn_suff
    if not os.path.isfile(filename):
        print('No file with this timestamp!')
        return

    def print_row(row, header):
        # this is SUPER hacky... fix
        paramClass = p.Params()
        p.set_global_params(paramClass)

        # first augment the output
        s0 = row[-p_dummy.len_S-p_dummy.len_s:-p_dummy.len_S]
        S = row[-p_dummy.len_S:]

        # convert to floats
        s0 = np.array([float(i) for i in s0])
        S = np.array([float(i) for i in S])

        # augment
        S_aug = augment_s(s0, S)

        # print
        print('---------------------------------------------------------')
        params = row[:num_ps]
        K = params[11]

        # print params used in this run
        for i in range(num_ps):
            print(header[i] + ': ' + str(params[i]))

        print()
        # print values with their corresponding heading
        header = header[num_ps+1:num_ps+1+p_dummy.len_s] + ['obj_0_accel_x',\
            'obj_0_accel_y', 'obj_0_accel_th', 'obj_1_accel_x', 'obj_1_accel_y',\
            'obj_1_accel_th', 'obj_2_accel_x', 'obj_2_accel_y', 'obj_2_accel_th']
        for v in range(p_dummy.len_s+9):
            if v < p_dummy.len_s:
                print(header[v][:-4] + ':')
            else:
                print(header[v] + ':')
            for k in range(p_dummy.T_steps+1):
                if k == 0:
                    if v < p_dummy.len_s:
                        print('   ' + str(s0[v]))
                else:
                    print('   ' + str(S_aug[(p_dummy.len_s_aug*(k-1)) + v]))

    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        reader = list(reader)

        # make header
        i = 0
        for row in reader:
            if row[0] == 'git hash':
                header = row
                break
            i += 1

        # pretty print all rows or just row_num
        if row_num != None:
            row = reader[int(row_num)]
            print_row(row, header)
        else:
            for row in reader:
                if row[0][:6] != 'Coming':
                    print_row(row, header)

if __name__ == '__main__':
    global start
    start = time.time()
    args = sys.argv
    if args[1] == 'test':
        if len(args) > 2:
            traj = args[2]
        else:
            traj = 0
        test_params(traj=traj)
    elif args[1] == 'restart':
        file_name, file_line_num = args[2:4]
        if len(args) == 5:
            single = True
            restart(file_name, int(file_line_num), single)
        else:
            restart(file_name, int(file_line_num))
    elif args[1] == 'pp':
        file_name = args[2]
        if len(args) == 4:
            print_line = args[3]
            pretty_print(file_name, print_line)
        else:
            pretty_print(file_name)
    else:
        print('Arguments are not valid')
