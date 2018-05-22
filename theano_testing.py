import numpy as np
import pdb
import theano.tensor as T
from theano.ifelse import ifelse
import theano
from main import init_vars, init_objects
from util import *

fj = T.dmatrix('fj')
roj = T.dmatrix('roj')
cj = T.dvector('cj')
ov = T.dvector('ov')
oa = T.dvector('oa')
one = T.constant(0.)
zero = T.constant(1.)

def get_fns():
    # calculate sum of forces on object
    # calc frictional force only if object is moving in x direction
    f_tot = sum([cj[j]*fj[j] for j in range(N_contacts)])
    f_tot = T.inc_subtensor(f_tot[1], -mass*gravity)
    coeff = ifelse(T.gt(ov[0], zero), one, ifelse(T.lt(ov[0], 0), -one, zero))
    fric = -1*coeff*mu*cj[2]*fj[2,1]
    f_tot = T.inc_subtensor(f_tot[0], fric)

    # calc change in linear momentum
    p_dot = mass*oa[0:2]

    # calc sum of moments on object (friction acts on COM? gravity does)
    # TODO: correct calc of I (moment of inertia)
    I = mass
    m_tot = sum(TCross(cj[j]*fj[j,:], roj[j,:] + np.array([-5, -5])) for j in range(N_contacts))

    # calc change in angular momentum
    l_dot = I*oa[2]

    cost_fn =  phys_lamb_2*(TNorm(f_tot - p_dot)**2 + TNorm(m_tot - l_dot)**2)
    cost = theano.function([fj, roj, cj, ov, oa], cost_fn)
    gc = T.grad(cost_fn, fj)
    grad_fn = theano.function([fj, roj, cj, ov, oa], gc)
    return cost, grad_fn

def TNorm(x):
    return T.sum(T.sqr(x))

# 2D cross product
def TCross(a, b):
    return T.as_tensor([a[0]*b[1] - a[1]*b[0]])

def L_physics(s, cost_fn, grad_fn):
    fj_val, roj_val, cj_val = get_contact_info(s)
    ov_val = get_object_vel(s)
    oa_val = get_object_accel(s)

    print(cost_fn(fj_val, roj_val, cj_val, ov_val, oa_val))
    print(grad_fn(fj_val, roj_val, cj_val, ov_val, oa_val))

if __name__ == '__main__':
    goal, objects = init_objects()
    s0, S0 = init_vars(objects)
    S_aug = interpolate_s(s0, S0)

    cost_fn, grad_fn = get_fns()
    print("done getting functions, now evaluating")
    for t in range(1,T_final):
        s_aug = get_s_aug_t(S_aug,t)
        L_physics(s_aug, cost_fn, grad_fn)
