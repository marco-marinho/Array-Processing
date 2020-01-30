import numpy as np

# Class containing a point in space
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def __str__(self):
        return "(" + str(self.x) + ", " + str(self.y) + ")"

    def __repr__(self):
        return "(" + str(self.x) + ", " + str(self.y) + ")"

#Class defining a line with a given slope m and intecerp b
#can be initalized with two points
class Line:
    def __init__(self, m, b):
        self.m = m
        self.b = b

    @classmethod
    def fromPoints(cls, point_one, point_two):
        m = (point_two.getY() - point_one.getY())/(point_two.getX() - point_one.getX())
        b = point_one.getY() - (m*point_one.getX())
        return cls(m, b)

    def getSlope(self):
        return self.m

    def getAngle(self):
        return np.arctan(self.m)

    def getIntercept(self):
        return self.b

    def getY(self, x):
        return self.m*x + self.b

#Function for calculating the intecerp point between two lines
def getLinesIntercept(line_1, line_2):
    x = (line_2.getIntercept() - line_1.getIntercept())/(line_1.getSlope() - line_2.getSlope())
    y = line_1.getSlope()*x + line_1.getIntercept()
    return Point(x, y)

#Function for calculating the angle formed between two lines
def getAngleLines(line_1, line_2, inDegree = False):
    angle = np.pi - np.abs(line_1.getAngle() - line_2.getAngle())
    if angle > np.pi/2:
        angle = np.pi - angle
    if inDegree:
        return angle*(180/np.pi)
    return angle

#Calculate distance between two points
def getPointDistance(point_1, point_2):
    return np.sqrt((point_1.getX()-point_2.getX())**2 + (point_1.getY()-point_2.getY())**2)

#Given a line and a point, find the two points in the same line a given distance from the given point
def getPointInLineGivenDistance(line, point_1, distance):
    if (line.getSlope()*point_1.getX() + line.getIntercept()) != point_1.getY():
        raise Exception("Point does not lie within given line.")

    x_1 = point_1.getX() + distance*np.sqrt(1/(1+line.getSlope()**2))
    y_1 = point_1.getY() + line.getSlope()*distance*np.sqrt(1/(1+line.getSlope()**2))

    x_2 = point_1.getX() - distance * np.sqrt(1 / (1 + line.getSlope() ** 2))
    y_2 = point_1.getY() - line.getSlope() * distance * np.sqrt(1 / (1 + line.getSlope() ** 2))

    return [Point(x_1, y_1), Point(x_2, y_2)]


def getPointGivenSlopeDistance(slope, point_1, distance):
    x_1 = point_1.getX() + distance*np.sqrt(1/(1+slope**2))
    y_1 = point_1.getY() + slope*distance*np.sqrt(1/(1+slope**2))

    x_2 = point_1.getX() - distance * np.sqrt(1 / (1 + slope ** 2))
    y_2 = point_1.getY() - slope * distance * np.sqrt(1 / (1 + slope ** 2))

    return [Point(x_1, y_1), Point(x_2, y_2)]

def isPointInLine(point, line):
    if (line.getSlope()*point.getX() + line.getIntercept()) != point.getY():
        return False
    else:
        return True

def getPointsPositiveY(points):
    result = []

    for point in points:
        if point.getY() >= 0:
            result.append(point)

    return result

def getIncidenceAngle(point_1, point_2, inDegree = False):
    line = Line.fromPoints(point_1, point_2)
    m = line.getSlope()
    if m > 0:
        if inDegree:
            return (np.pi/2 - np.arctan(m))*(180/np.pi)
        return np.pi/2 - np.arctan(m)
    else:
        if inDegree:
            return -(np.pi/2 + np.arctan(m))*(180/np.pi)
        return -(np.pi/2 + np.arctan(m))


def getReflectionAngle(transmitter, reflector, receiver, inDegree = False):
    line_1 = Line.fromPoints(transmitter, reflector)
    if inDegree:
        return line_1.getAngle()*180/np.pi
    return line_1.getAngle()

def reflectionToSlope(doa, reflection, inDegrees = False):
    if doa < 0:
        if inDegrees:
            return np.tan(reflection*np.pi/180)
        else:
            return np.tan(reflection)

    else:
        if inDegrees:
            return -1*np.tan(reflection*np.pi/180)
        else:
            return -1*np.tan(reflection)

def incidenceToSlope(angle, inDegres = False):
    if angle > 0:
        if inDegres:
            return np.tan((np.pi / 2) - angle*np.pi/180)
        return np.tan((np.pi/2) - angle)
    else:
        if inDegres:
            return -1*np.tan((np.pi / 2) + angle*np.pi/180)
        return -1*np.tan((np.pi/2) + angle)

def getReflectorLine(line_doa, angle_reflection):

    doa_slope = line_doa.getSlope()

    if doa_slope < 0:
        m_reflector = 180((angle_reflection*2)-np.arctan(doa_slope))

    line_reflector = Line(m_reflector, 0)

    return line_reflector