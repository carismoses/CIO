# Contact Invariant Optimization

Dependencies: 

To run:
1. in test_params.py change the fn_prefix to match where you would like to store output files
2. run test_params.py with one of the following command line args
    1. python test_params.py test traj_type - run CIO with all combinations of params specified at top of test_params (the rest of the params are specified as default_params in params.py)  
        1. traj_type (int, optional): if you don't enter a test_number the decision vars will be initialized to the initial state for all time steps, else enter the desired initial trajectory type from below
    2. python test_params.py restart file_name file_line_number - restart CIO from a solution saved to a file (when had a good phase and don't want to have to rerun the entire optimization just to test a later phase)
        1. file_name (str): just name, not complete path or .csv suffix
        2. file_line_number (int): the line number in the file you would like to start your optimization from
        3. single (bool, optional): True if just want to get the value of the objective function for a single trajectory from a file/just want to run one iteration of CIO for debugging
    3. python test_params.py pp file_name line_number - pretty print to the terminal what is in a file
        1. file_name (str): just name, not complete path or .csv suffix
        2. line_number (int, optional): prints just the line number specified, if no line_number then the entire file is pretty printed
----
traj_type options:
1. traj_type == 1: linearly interpolate box pose from initial to goal state and calc vel from that to get S0
2. traj_type == 2: just set the last point in S0 to the goal position of the box
3. traj_type == 3: for each time step gripper 1 applies a force in a different direction (was using to test friction cone calculation)
4. traj_type == 4: gripper 1 is applying a force in the +x direction with c0=1 and the ground is applying a force in the +y direction with c2=1
5. traj_type == 5: this is the solution to the objective function I think? can't remember how I found it
----
Not used in current obj fun (but implemented): L_kin, L_phys part with rotational stuff, L_ci

Possible issues: non-smoothness of box surface an issues?

Things to check: n_j in L_cone match pi_j from L_ci?
