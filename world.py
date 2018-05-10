import pdb
import numpy as np

class WorldTraj(object):
    def __init__(self, s0, S, objects, Ncontacts, Tfinal):
        self.s0 = s0
        self.S = S
        self.objects = objects
        self.step(0)

        # initialize to zero
        self.e_O, self.e_H = np.zeros((Ncontacts, Tfinal+1, 2)), np.zeros((Ncontacts, Tfinal+1, 2))

    def step(self, t):
        for object in self.objects:
            object.step(self.S,t)

class Object(object):
    def __init__(self, index = None, pose = (0.0,0.0), angle = 0.0, \
                        actuated = False, contact = True, step_size = 0.5):
        self.index = index
        self.pose = np.array(pose)
        self.angle = np.array(angle)
        self.type = type
        self.contact = contact
        self.step_size = step_size
        self.rad_bounds = 1e-1

    def step(self, S, t):
        if self.index != None:
            self.pose = S[3*self.index*t:3*self.index*t+2]
            self.angle = S[3*self.index*t+2]

    def check_collisions(self, object):
        for object in world:
            if object != self.index:
                cost = object.check_point_collisions(self.discretize())

class Line(Object):
    def __init__(self, index = None, pose = (0.0,0.0), angle = 0.0, \
                actuated = False, contact = True, step_size = 0.5, length = 2.0):
        self.length = length
        super(Line,self).__init__(index, pose, angle, actuated, contact, \
                                    step_size)

    """
    def endpoints(self):

    def discretize(self):

    def check_point_collisions(self, points):
    """
    def line_eqn(self):
        # check for near straight lines
        # slope close to m = 0
        if abs(self.angle) < self.rad_bounds or abs(self.angle - np.pi) < self.rad_bounds:
            m = 0.
            b = self.pose[1]
        # slope close to m = inf
        elif abs(self.angle - np.pi/2.) < self.rad_bounds or abs(self.angle - 3.*np.pi/2.) < self.rad_bounds:
            m = 1.
            b = self.pose[0]
        else:
            m = np.sin(self.angle)/np.cos(self.angle)
            b = self.pos[1] - m*self.pos[0]
        return m, b

    def project_point(self, point):
        m, b = self.line_eqn()
        a = -m
        b = 1
        c = -b
        n = b*point[0] - a*point[1]
        d = a**2 + b**2
        proj_point = np.array([b*n - a*c, -1*a*n - b*c])/d
        return proj_point

class Rectangle(Object):
    def __init__(self, index = None, pose = (0.0,0.0), angle = 0.0, \
                actuated = False, contact = True, step_size = 0.5, width = 10.0, \
                height = 10.0):
        self.width = width
        self.height = height
        super(Rectangle,self).__init__(index, pose, angle, actuated, contact, \
                                    step_size)
    """
    def corners(self):

    def discretize(self):

    def check_point_collisions(self, points):
    """

    def project_point(self, point):
        pass
