import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.widgets import *
import json
import math
from time import sleep


class PathFollower(object):

    def __init__(self, x, y, speed):
        self.position = (x, y)
        self.theta = np.pi
        self.wheelbase = 1
        self.history = [self.position]
        self.is_dead = False

    def steering_angle(self, goal):
        v = [math.cos(self.theta), math.sin(self.theta), 0]
        rg = [goal[0]-self.position[0], goal[1]-self.position[1], 0]
        prodNorm = np.linalg.norm(v)*np.linalg.norm(rg)
        cosPhi = np.dot(v, rg)/prodNorm
        sinPhi = np.cross(v, rg)[2]/prodNorm
        phi = math.atan2(sinPhi, cosPhi)

        return min(max(phi, -math.pi/4.1), math.pi/4.1)

    def move_torwards(self, x, y, lookahead, speed):
        offset = (x - self.position[0], y - self.position[1])
        dist = (offset[0] ** 2 + offset[1] ** 2) ** 0.5

        steering_angle = self.steering_angle((x, y))
        u = speed * 0.1 * np.cos(self.theta), speed * 0.1 * np.sin(self.theta)
        self.position = self.position[0] + u[0], self.position[1] + u[1]
        self.theta += speed * 0.1 * np.tan(steering_angle) / self.wheelbase
        # self.position = (self.position[0] + offset[0] / dist * self.speed,
        #                  self.position[1] + offset[1] / dist * self.speed)
        self.history.append(self.position)

    def get_pos(self):
        return self.position


class PurePursuit(object):

    def __init__(self, path, lookaheadDistance=2, lookaheadDistanceDelta=2.5, followerSpeed=1, followerStopDistance=1, ax=None):
        self.path = path
        self.lookaheadDistance = lookaheadDistance
        self.lookaheadDistanceDelta = lookaheadDistanceDelta
        self.followerSpeed = followerSpeed
        self.followerStopDistance = followerStopDistance
        self.ax = ax
        self.followers = []

    def draw(self):
        if self.ax != None:
            self.ax.clear()
            xs, ys = zip(*self.path)
            self.ax.plot(xs, ys)
            for f in self.followers:
                if f.is_dead:
                    continue
                pos = f.get_pos()
                lookahead = self.get_lookahead_pt(
                    pos[0], pos[1], self.lookaheadDistance)
                self.ax.scatter(pos[0], pos[1], c='b')
                xs, ys = zip(*f.history)
                self.ax.plot(xs, ys, linestyle=':')
                if lookahead != None:
                    self.ax.scatter(lookahead[0], lookahead[1], c='r')
                    delta = lookahead[0] - pos[0], lookahead[1] - pos[1]
                    dist = 2 * (delta[0] ** 2 + delta[1] ** 2) ** 0.5
                    self.ax.add_artist(plt.Circle(pos, dist/2, fill=False))
                    self.ax.plot((pos[0], lookahead[0]), (pos[1], lookahead[1]), linestyle='--')

                    if dist < self.followerStopDistance:
                        f.is_dead = True
                    else:
                        f.move_torwards(lookahead[0], lookahead[1], dist, self.followerSpeed)

    def add_follower(self, x, y):
        self.followers.append(PathFollower(
            x, y, self.followerSpeed))

    def sign(self, n):
        if n == 0:
            return 1
        else:
            return n/abs(n)

    def get_lookahead_pt(self, x, y, radius):
        look_pt = None
        # http://mathworld.wolfram.com/Circle-LineIntersection.html
        for i in range(len(path) - 1):
            seg_start = path[i]
            seg_end = path[i + 1]
            p1 = seg_start[0] - x, seg_start[1] - y
            p2 = seg_end[0] - x, seg_end[1] - y

            d = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
            D = p1[0] * p2[1] - p2[0] * p1[1]

            discriminant = radius ** 2 * d * d - D * D

            if discriminant < 0 or p1 == p2:
                continue

            x1 = (D * (p2[1] - p1[1]) + self.sign(p2[1] - p1[1])
                  * (p2[0] - p1[0]) * discriminant**0.5) / (d * d)
            y1 = (-D * (p2[0] - p1[0]) + abs(p2[1] - p1[1])
                  * discriminant ** 0.5) / (d * d)

            x2 = (D * (p2[1] - p1[1]) - self.sign(p2[1] - p1[1])
                  * (p2[0] - p1[0]) * discriminant ** 0.5) / (d * d)
            y2 = (-D * (p2[0] - p1[0]) - abs(p2[1] - p1[1])
                  * discriminant ** 0.5) / (d * d)

            intersection1 = min(p1[0], p2[0]) < x1 and x1 < max(
                p1[0], p2[0]) or min(p1[1], p2[1]) < y1 and y1 < max(p1[1], p2[1])
            intersection2 = min(p1[0], p2[0]) < x2 and x2 < max(
                p1[0], p2[0]) or min(p1[1], p2[1]) < y2 and y2 < max(p1[1], p2[1])

            if intersection1 or intersection2:
                look_pt = None

            if intersection1:
                look_pt = x1 + x, y1 + y

            if intersection2 and (look_pt == None or abs(x1 - p2[0]) > abs(x2 - p2[0]) or abs(y1 - p2[1]) > abs(y2 - p2[1])):
                look_pt = x2 + x, y2 + y

        if len(self.path) > 0:
            if ((self.path[-1][0] - x)**2 + (self.path[-1][1] - y)**2)**0.5 <= radius:
                return self.path[-1][0], self.path[-1][1]
        return look_pt


with open('path.json', 'r') as f:
    path = json.load(f)


fig, ax = plt.subplots()
ax.set_aspect(1)

plt.ion()
plt.show()

pp = []
for i in range(len(path)):
    if i % 100 == 0:
        pp.append(path[i])
pure_pursuit = PurePursuit(pp, ax=ax, followerSpeed=2, lookaheadDistance=5)


def add_car(x):
    pure_pursuit.add_follsower(random.uniform(0, 3), random.uniform(2, 20))

def set_lookahead(x):
    pure_pursuit.lookaheadDistance = x

def set_speed(x):
    pure_pursuit.followerSpeed = x


add_btn = Button(plt.axes([0.75, 0.9, 0.15, 0.075]), 'Add car')
add_btn.on_clicked(add_car)
lookahead_distance = Slider(plt.axes(
    [0.8, 0.7, 0.15, 0.05]), 'Lookahead', 1, 30, valinit=2.5, valstep=0.1)
lookahead_distance.on_changed(set_lookahead)
speed = Slider(plt.axes(
    [0.8, 0.5, 0.15, 0.05]), 'Speed', 0.1, 30, valinit=1, valstep=0.1)
speed.on_changed(set_speed)

for i in range(1000):
    pure_pursuit.draw()
    plt.pause(0.1)
    if not plt.fignum_exists(1):
        break
