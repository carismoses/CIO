import pdb
import numpy as np
from util import calc_obj_dynamics, get_contact_info, add_noise, calc_deriv, get_dist, normalize
from collections import namedtuple
from copy import deepcopy

Pose = namedtuple('Pose', 'x y theta')
Velocity = namedtuple('Velocity', 'x y theta')
Acceleration = namedtuple('Acceleration', 'x y theta')
Contact = namedtuple('Contact', 'f ro c')

class WorldTraj(object):
    def __init__(self, S, world, p):
        # get dyanamic and contact info from S
        dyn_info = {}
        for (i, dyn_obj) in enumerate(world.get_all_objects()):
            poses, vels, accels = calc_obj_dynamics(world.s0, S, p, i)
            dyn_info[i] = [poses, vels, accels]

        dyn_offset = len(world.get_all_objects())*6
        cont_info = {}
        for (ci, cont_obj) in enumerate(world.contact_state):
            fs, ros, cs = get_contact_info(world.s0, S, p, ci, dyn_offset)
            cont_info[ci] = [fs, ros, cs]

        # fill into list of new worlds
        self.worlds = []
        for t in range(p.T_steps+1):
            world_t = deepcopy(world)
            for i in range(len(world_t.get_all_objects())):
                world_t.set_dynamics(i,dyn_info[i][0][t], dyn_info[i][1][t], dyn_info[i][2][t])
            for ci in range(len(world_t.contact_state)):
                world_t.set_contact_state(ci, cont_info[ci][0][:,t], cont_info[ci][1][:,t], cont_info[ci][2][t])
            if t == 0:
                world_t.set_e_vars(None, p)
            else:
                world_t.set_e_vars(self.worlds[t-1], p)
            self.worlds += [world_t]

def stationary_traj(world, goal, p):
    S = np.zeros(p.len_S)
    for k in range(p.K):
        S[k*p.len_s:k*p.len_s+p.len_s] = world.get_vars()
    return S

class World(object):
    def __init__(self, manip_obj=None, hands=[], contact_state={}, \
                    traj_func=stationary_traj):
        self.manip_obj = manip_obj
        self.hands = hands
        self.contact_state = contact_state
        self.traj_func = traj_func
        self.s0 = self.get_vars()

        self.pi_O = {}
        self.pi_H = {}
        self.e_O = {}
        self.e_H = {}
        self.e_dot_O = {}
        self.e_dot_H = {}

    def set_dynamics(self, obj_index, pose, vel, accel):
        objs = self.get_all_objects()
        objs[obj_index].set_dynamics(pose, vel, accel)

    def set_contact_state(self, obj_index, f, ro, c):
        cont_objs = list(self.contact_state)
        self.contact_state[cont_objs[obj_index]] = Contact(f=f, ro=ro, c=c)

    def set_e_vars(self, world_tm1, p):
        object_pose = np.array([self.manip_obj.pose.x, self.manip_obj.pose.y])

        for (ci, (cont_obj, cont)) in enumerate(self.contact_state.items()):
            r = np.add(cont.ro, object_pose)
            self.pi_H[ci] = cont_obj.project_point(r)
            self.pi_O[ci] = self.manip_obj.project_point(r)
            self.e_H[ci] = np.subtract(self.pi_H[ci], r)
            self.e_O[ci] = np.subtract(self.pi_O[ci],r)

            if world_tm1 is None:
                self.e_dot_H[ci] = np.array([0., 0.])
                self.e_dot_O[ci] = np.array([0., 0.])
            else:
                self.e_dot_H[ci] = calc_deriv(self.e_H[ci],world_tm1.e_H[ci],p.delT)
                self.e_dot_O[ci] = calc_deriv(self.e_O[ci],world_tm1.e_O[ci],p.delT)

    def get_vars(self):
        s0 = np.array([])

        # fill in object poses and velocities
        for object in self.get_all_objects():
            s0 = np.concatenate([s0,object.pose])
            s0 = np.concatenate([s0,object.vel])

        # fill in contact info
        for cont_obj in self.contact_state:
            s0 = np.concatenate([s0,self.contact_state[cont_obj].f])
            s0 = np.concatenate([s0,self.contact_state[cont_obj].ro])
            s0 = np.concatenate([s0,[self.contact_state[cont_obj].c]])

        return s0

    def get_all_objects(self):
        return [self.manip_obj] + self.hands

# world origin is left bottom with 0 deg being along the x-axis (+ going ccw), all poses are in world frame
class Object(object):
    def __init__(self, pose = Pose(0.0,0.0,0.0), vel = Velocity(0.0, 0.0, 0.0),
                    step_size = 0.5):
        self.pose = pose
        self.vel = vel
        self.step_size = step_size
        self.rad_bounds = 1e-1

        self.accel = None

    def set_dynamics(self, pose, vel, accel):
        self.pose = Pose(*pose)
        self.vel = Velocity(*vel)
        self.accel = Acceleration(*accel)

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
                step_size = 0.5):
        self.radius = radius
        super(Circle,self).__init__(pose, vel, step_size)

    def discretize(self):
        pass#TODO

    def check_inside(self, point):
        pass#TODO

    # projects the given point onto the surface of this object
    def project_point(self, point):
        origin_to_point = np.subtract(point[:2], np.array([self.pose.x,self.pose.y]))
        origin_to_point /= np.linalg.norm(origin_to_point)
        closest_point = np.array([self.pose.x, self.pose.y]) + (origin_to_point * self.radius)
        return closest_point

    # get a normal vector in the world frame directed from the center of this object
    # to the given point
    def get_surface_normal(self, point):
        origin_to_point = np.subtract(point, [self.pose.x, self.pose.y])
        n = normalize(origin_to_point)
        return n
