import torch


class Bezier:
    def TwoPoints(t, P1, P2):
        Q1 = (1 - t) * P1 + t * P2
        return Q1

    def Points(t, points):
        newpoints = []
        for i1 in range(0, len(points) - 1):
            newpoints += [Bezier.TwoPoints(t, points[i1], points[i1 + 1])]
        return newpoints

    def Point(t, points):
        newpoints = points
        while len(newpoints) > 1:
            newpoints = Bezier.Points(t, newpoints)
        return newpoints[0]
