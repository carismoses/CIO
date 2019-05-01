import pdb
import numpy as np
from util import *
from collections import namedtuple

Pose = namedtuple('Pose', 'x y theta')
Velocity = namedtuple('Velocity', 'x y theta')

class WorldTraj(object):
    def __init__(self, S, world, p):
        # augment by calculating the accelerations from the velocities
        # interpolate all of the decision vars to get a finer trajectory disretization
        self.S = augment_s(world.s0, S, p)
        self.world = world
        self.p = p
        self.step(0)

        # initialize to zero and calulate e for t=0
        self.e_Os, self.e_Hs = np.zeros((self.p.N, self.p.T_steps, 2)), np.zeros((self.p.N, self.p.T_steps, 2))
        self.calc_e(world.s0, 0, world)

    def step(self, t):
        for object in self.world.get_contact_objects():
            if t == 0:
                object.step(self.world.s0,t,self.p)
            else:
                object.step(self.S,t,self.p)

    # e_O is the shortest distance between roj and the object
    # e_H is the shortest distance between roj and the contact surfaces
    def calc_e(self, s, t, world):
        box = world.manipulated_objects[0]
        o = np.array([box.pose.x, box.pose.y])

        # get ro: roj in world frame
        _, roj, _ = get_contact_info(s,self.p)
        rj = roj + np.tile(o, (3, 1))

        # get pi_j: project rj onto all contact surfaces
        pi_j = np.zeros((self.p.N, 2))
        for object in self.world.get_contact_objects():
            if object.contact_index != None:
                pi_j[object.contact_index,:] = object.project_point(rj[object.contact_index,:])

        # get pi_o: project rj onto object
        pi_o = np.zeros((self.p.N,2))
        for j in range(self.p.N):
            pi_o[j,:] = box.project_point(rj[j,:])

        e_O = pi_o - rj
        e_H = pi_j - rj
        self.e_Os[:,t,:], self.e_Hs[:,t,:] = e_O, e_H

        return e_O, e_H

class ContactState(object):
    def __init__(self, cont_object, manip_object, f=[0.0, 0.0], ro=[0.0, 0.0], c=0.5):
        self.cont_object = cont_object
        self.manip_object = manip_object
        self.f = f
        self.ro = ro
        self.c = c

def stationary_traj(world, goal, p):
    S = np.zeros(p.len_S)
    for k in range(p.K):
        S[k*p.len_s:k*p.len_s+p.len_s] = world.get_vars()
    S = add_noise(S)
    return S

class World(object):
    def __init__(self, ground=None, manipulated_objects=[], hands=[], contact_state=[], \
                    traj_func=stationary_traj):
        self.ground = ground
        self.manipulated_objects = manipulated_objects
        self.hands = hands
        self.contact_state = contact_state
        self.traj_func = traj_func
        self.s0 = self.get_vars()
        self.number_objects()

    def get_vars(self):
        s0 = np.array([])

        # fill in object poses and velocities
        for object in self.get_dynamic_objects():
            s0 = np.concatenate([s0,object.pose])
            s0 = np.concatenate([s0,object.vel])

        # fill in contact info
        for cont in self.contact_state:
            s0 = np.concatenate([s0,cont.f])
            s0 = np.concatenate([s0,cont.ro])
            s0 = np.concatenate([s0,[cont.c]])

        return s0

    # objects that can make contact need a contact index
    # objects that are dynamic need a pose index
    def number_objects(self):
        for (i,cont) in enumerate(self.contact_state):
            cont.cont_object.contact_index = i

        for (i,dyn_obj) in enumerate(self.get_dynamic_objects()):
            dyn_obj.pose_index = i

    def get_num_all_objects(self):
        return 1 + len(self.manipulated_objects) + len(self.hands)

    def get_all_objects(self):
        return [self.ground] + self.manipulated_objects + self.hands

    def get_num_dynamic_objects(self):
        return len(self.manipulated_objects) + len(self.hands)

    def get_dynamic_objects(self):
        return self.manipulated_objects + self.hands

    def get_num_contact_objects(self):
        return 1 + len(self.hands)

    def get_contact_objects(self):
        return self.hands + [self.ground]

# world origin is left bottom with 0 deg being along the x-axis (+ going ccw), all poses are in world frame
class Object(object):
    def __init__(self, pose = Pose(0.0,0.0,0.0), vel = Velocity(0.0, 0.0, 0.0), step_size = 0.5):
        self.pose = pose
        self.vel = vel
        self.step_size = step_size
        self.rad_bounds = 1e-1

        # set when world is initialized
        self.pose_index = None
        self.contact_index = None

    def step(self, S, t, p):
        if self.pose_index != None:
            if t == 0:
                self.pose = Pose(*S[6*self.pose_index:6*self.pose_index+3])
                self.vel = Velocity(*S[6*self.pose_index+3:6*self.pose_index+6])
            else:
                self.pose = Pose(*S[6*self.pose_index+(t-1)*p.len_s_aug:6*self.pose_index+(t-1)*p.len_s_aug+3])
                self.vel = Velocity(*S[6*self.pose_index+(t-1)*p.len_s_aug+3:6*self.pose_index+(t-1)*p.len_s_aug+6])

    def check_collisions(self, col_object):
        pts = self.discretize()
        max_col_dist = 0.0
        for pt in pts:
            col_dist = col_object.check_inside(pt)
            if col_dist > max_col_dist:
                max_col_dist = col_dist
        return max_col_dist

class Line(Object):
    def __init__(self, length = 10.0, pose = Pose(0.0,0.0,0.0), vel = Velocity(0.0, 0.0, 0.0), step_size = 0.5):
        self.length = length
        super(Line,self).__init__(pose, vel, step_size)

    def discretize(self):
        N_points = np.floor(self.length/self.step_size) + 1
        points = np.array((N_points,2))
        points[0,:], points[N_points-1,:] = self.get_endpoints()
        for i in range(1,N_points-1):
            points[i,:] = points[i-1,:] + self.step_size*np.array((np.cos(self.pose.theta), np.sin(self.angle)))
        return points

    def check_inside(self, point):
        pass #TODO

    def get_endpoints(self):
        p0 = np.array([self.pose.x, self.pose.y])
        endpoint0 = p0
        endpoint1 = p0 + self.length*np.array((np.cos(self.pose.theta), np.sin(self.pose.theta)))
        return (endpoint0, endpoint1)

    def line_eqn(self):
        # check for near straight lines
        # line close to horizontal
        if abs(self.pose.theta) < self.rad_bounds or abs(self.pose.theta - np.pi) < self.rad_bounds:
            a = 0.
            b = 1.
            c = -self.pose.y
        # line close to vertical
        elif abs(self.pose.theta - np.pi/2.) < self.rad_bounds or abs(self.pose.theta - 3.*np.pi/2.) < self.rad_bounds:
            a = 1.
            b = 0.
            c = -self.pose.x
        else:
            slope = np.sin(self.pose.theta)/np.cos(self.pose.theta)
            int = self.pose.y - slope*self.pose.x
            a = -slope
            b = 1
            c = -int
        return a,b,c

    # project a given point onto this line (or endpoint)
    # TODO: smooth out endpoints?
    def project_point(self, point):
        a,b,c = self.line_eqn()
        n = b*point[0] - a*point[1]
        d = a**2 + b**2
        proj_point = np.array([b*n - a*c, -1*a*n - b*c])/d

        # check if point is on line segment, otherwise return closest endpoint
        endpoints = self.get_endpoints()
        if not (proj_point[0] <= max(endpoints[0][0], endpoints[1][0]) and\
            proj_point[0] >= min(endpoints[0][0], endpoints[1][0]) and\
            proj_point[1] <= max(endpoints[0][1], endpoints[1][1]) and\
            proj_point[1] >= min(endpoints[0][1], endpoints[1][1])):

            dist0 = get_dist(proj_point, endpoints[0])
            dist1 = get_dist(proj_point, endpoints[1])
            if dist0 < dist1:
                proj_point = endpoints[0]
            else:
                proj_point = endpoints[1]
        return proj_point

class Rectangle(Object):
    def __init__(self, width = 10.0, height = 10.0, pose = Pose(0.0,0.0,0.0), \
                vel = Velocity(0.0, 0.0, 0.0), step_size = 0.5):
        self.width = width
        self.height = height
        super(Rectangle,self).__init__(pose, vel, step_size)
        self.lines = self.make_lines() # rectangles are made up of 4 line objects

    def discretize(self):
        pass#TODO

    def check_inside(self, point):
        pass#TODO

    # defines list of lines in clockwise starting from left line
    def make_lines(self):
        lines = []
        pose = self.pose
        angle = self.pose.theta
        length = self.height
        for i in range(4):
            line = Line(pose, angle, length)
            lines += [line]
            (_, pose) = line.get_endpoints()
            angle = angle - np.pi/2
            if length == self.height:
                length = self.width
            else:
                length = self.height
        return lines

    # return the closest projected point out of all rect surfaces
    # use a softmin instead of a hard min to make function smooth
    def project_point(self, point):
        k = 1.e4
        num_sides = len(self.lines)
        p_nearest = np.zeros((num_sides,2))
        for j in range(num_sides):
            p_nearest[j,:] = self.lines[j].project_point(point)
        p_mat = np.tile(point.T, (num_sides, 1))
        ones_vec = np.ones(num_sides)
        nu = np.divide(ones_vec, ones_vec + np.linalg.norm(p_mat-p_nearest,axis=1)**2*k)
        nu = nu/sum(nu)
        nu = np.tile(nu, (2,1)).T
        closest_point = sum(np.multiply(nu,p_nearest))
        return closest_point


class Circle(Object):
    def __init__(self, radius = 10.0, pose = Pose(0.0,0.0,0.0), vel = Velocity(0.0, 0.0, 0.0), \
                pose_index = None, contact_index = None, step_size = 0.5):
        self.radius = radius
        super(Circle,self).__init__(pose, vel, step_size)

    def discretize(self):
        pass#TODO

    def check_inside(self, point):
        pass#TODO

    # return the closest projected point out of all rect surfaces
    def project_point(self, point):

        origin_to_point = np.subtract(point[:2], np.array([self.pose.x,self.pose.y]))
        origin_to_point /= np.linalg.norm(origin_to_point)

        closest_point = np.array([self.pose.x, self.pose.x]) + (origin_to_point * self.radius)

        return closest_point

#### geometric helper functions ####
def get_dist(point0, point1):
    return np.linalg.norm(point1 - point0)**2
