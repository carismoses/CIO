import pdb
import numpy as np
from util import len_s

class WorldTraj(object):
    def __init__(self, s0, S, objects, Ncontacts, Tfinal):
        self.s0 = s0
        self.S = S
        self.objects = objects
        self.step(0)

        # initialize to zero
        self.e_Os, self.e_Hs = np.zeros((Ncontacts, Tfinal, 2)), np.zeros((Ncontacts, Tfinal, 2))

    def step(self, t):
        for object in self.objects:
            if t == 0:
                object.step(self.s0,t)
            else:
                object.step(self.S,t)

# world origin is left bottom with 0 deg being along the x-axis (+ going ccw), all poses are in world frame
class Object(object):
    def __init__(self, pose = (0.0,0.0), angle = 0.0, vel = (0.0, 0.0, 0.0), actuated = False, \
                pose_index = None, contact_index = None, step_size = 0.5):
        self.pose = np.array(pose)
        self.angle = np.array(angle)
        self.vel = np.array(vel)
        self.actuated = actuated
        self.pose_index = pose_index
        self.contact_index = contact_index
        self.step_size = step_size
        self.rad_bounds = 1e-1

    def step(self, S, t):
        if self.pose_index != None:
            if t == 0:
                self.pose = S[6*self.pose_index:6*self.pose_index+2]
                self.angle = S[6*self.pose_index+2]
                self.vel = S[6*self.pose_index+3:6*self.pose_index+7]
            else:
                self.pose = S[6*self.pose_index+(t-1)*len_s:6*self.pose_index+(t-1)*len_s+2]
                self.angle = S[6*self.pose_index+(t-1)*len_s+2]
                self.vel = S[6*self.pose_index+(t-1)*len_s+3:6*self.pose_index+(t-1)*len_s+7]

    def check_collisions(self, col_object):
        pts = self.discretize()
        max_col_dist = 0.0
        for pt in pts:
            col_dist = col_object.check_inside(pt)
            if col_dist > max_col_dist:
                max_col_dist = col_dist
        return max_col_dist

class Line(Object):
    def __init__(self, pose = (0.0,0.0), angle = 0.0, vel = (0.0, 0.0, 0.0), length = 10.0,\
                actuated = False, pose_index = None, contact_index = None, step_size = 0.5):
        self.length = length
        super(Line,self).__init__(pose, angle, vel, actuated, pose_index, contact_index,\
                                    step_size)

    def discretize(self):
        N_points = np.floor(self.length/self.step_size) + 1
        points = np.array((N_points,2))
        points[0,:], points[N_points-1,:] = self.get_endpoints()
        for i in range(1,N_points-1):
            points[i,:] = points[i-1,:] + self.step_size*np.array((np.cos(self.angle), np.sin(self.angle)))
        return points

    def check_inside(self, point):
        pass #TODO

    def get_endpoints(self):
        endpoint0 = self.pose
        endpoint1 = self.pose + self.length*np.array((np.cos(self.angle), np.sin(self.angle)))
        return (endpoint0, endpoint1)

    def line_eqn(self):
        # check for near straight lines
        # line close to horizontal
        if abs(self.angle) < self.rad_bounds or abs(self.angle - np.pi) < self.rad_bounds:
            a = 0.
            b = 1.
            c = -self.pose[1]
        # line close to vertical
        elif abs(self.angle - np.pi/2.) < self.rad_bounds or abs(self.angle - 3.*np.pi/2.) < self.rad_bounds:
            a = 1.
            b = 0.
            c = -self.pose[0]
        else:
            slope = np.sin(self.angle)/np.cos(self.angle)
            int = self.pose[1] - slope*self.pose[0]
            a = -slope
            b = 1
            c = -int
        return a,b,c

    # project a given point onto this line (or endpoint)
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
    def __init__(self, pose = (0.0,0.0), angle = 0.0, vel = (0.0, 0.0, 0.0), width = 10.0, height = 10.0, \
                actuated = False, pose_index = None, contact_index = None, step_size = 0.5):
        self.width = width
        self.height = height
        super(Rectangle,self).__init__(pose, angle, vel, actuated, pose_index, contact_index,\
                                        step_size)
        self.lines = self.make_lines() # rectangles are made up of 4 line objects


    def discretize(self):
        pass#TODO

    def check_inside(self, point):
        pass#TODO

    # defines list of lines in clockwise starting from left line
    def make_lines(self):
        lines = []
        pose = self.pose
        angle = self.angle
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

    # return the closest projected points out of all rect surfaces
    def project_point(self, point):
        shortest_dist = float("inf")
        closest_point = None
        for line in self.lines:
            proj_point = line.project_point(point)
            dist = get_dist(point, proj_point)
            if dist < shortest_dist:
                shortest_dist = dist
                closest_point = point
        return closest_point

#### geometric helper functions ####
def get_dist(point0, point1):
    return np.linalg.norm(point1 - point0)**2
