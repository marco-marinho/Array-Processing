import matplotlib.pyplot as plt
import numpy as np
import GeometryLibrary as geo

point_1 = geo.Point(0, 0)
point_2 = geo.Point(-2, 2)
point_3 = geo.Point(0, 3)
point_4 = geo.Point(3, 1.5)
line = geo.Line.fromPoints(point_1, point_2)
print(line.getSlope())
print(line.getIntercept())

line_1 = geo.Line.fromPoints(point_1, point_2)
line_2 = geo.Line.fromPoints(point_2, point_3)

print(geo.getLinesIntercept(line_1, line_2))
print(geo.getAngleLines(line_1, line_2, inDegree=True))

distance = geo.getPointDistance(point_1, point_2)+geo.getPointDistance(point_2, point_3)
print(distance)

points_1 = geo.getPointGivenSlopeDistance(line_2.getSlope(), point_1, distance)
print(points_1)
print(geo.getPointsPositiveY(points_1))

points_2 = geo.getPointInLineGivenDistance(line_1, point_1, distance)
print(points_2)
print(geo.getPointsPositiveY(points_2))

line_3 = geo.Line.fromPoints(geo.getPointsPositiveY(points_1)[0], geo.getPointsPositiveY(points_2)[0])

print(geo.isPointInLine(point_3, line_3))

line_4 = geo.Line.fromPoints(point_1, point_4)
line_5 = geo.Line.fromPoints(point_4, point_3)

distance_2 = geo.getPointDistance(point_1, point_4)+geo.getPointDistance(point_4, point_3)

points_3 = geo.getPointGivenSlopeDistance(line_5.getSlope(), point_1, distance_2)
points_4 = geo.getPointInLineGivenDistance(line_4, point_1, distance_2)

line_6 = geo.Line.fromPoints(geo.getPointsPositiveY(points_3)[0], geo.getPointsPositiveY(points_4)[0])

y = []
y_2 = []
y_3 = []
y_4 = []
y_5 = []
y_6 = []

x = np.linspace(0, -2)
x_2 = np.linspace(-2, 0)
x_3 = np.linspace(points_1[0].getX(), points_2[1].getX())
x_4 = np.linspace(0, 3)
x_5 = np.linspace(3, 0)

for x_i in x:
    y.append(line_1.getY(x_i))

for x_i in x_2:
    y_2.append(line_2.getY(x_i))

for x_i in x_3:
    y_3.append(line_3.getY(x_i))

for x_i in x_4:
    y_4.append(line_4.getY(x_i))

for x_i in x_5:
    y_5.append(line_5.getY(x_i))

for x_i in x_3:
    y_6.append(line_6.getY(x_i))

plt.plot(x, y)
plt.plot(x_2, y_2)
plt.plot(x_3, y_3)
plt.plot(x_4, y_4)
plt.plot(x_5, y_5)
plt.plot(x_3, y_6)
plt.show()