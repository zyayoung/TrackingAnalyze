import numpy as np
from libs.read_record import Record

def point_in_box(point, box):
    return box[0] < point[0] < box[2] and box[1] < point[1] < box[3]

def points_in_box(points, box):
    return (box[0] < points[..., 0]) & (points[..., 0] < box[2]) & (box[1] < points[..., 1]) & (points[..., 1] < box[3])

class RoI:
    def __init__(self, roi):
        rect = roi.boundingRect()
        x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
        self.rect = np.array([x, y, x+w, y+h])
    
    def enterCount(self, record: Record):
        counts = []
        for id in map(int, record.unique_ids):
            points = record.get_centers(id=id)
            inmask = points_in_box(points, self.rect)
            inmask = (inmask[1:] == 1) & (inmask[:-1] == 0)
            counts.append(inmask.sum())
        return counts

    
    def inTime(self, record: Record):
        times = []
        for id in map(int, record.unique_ids):
            points = record.get_centers(id=id)
            inmask = points_in_box(points, self.rect)
            times.append(record.get_intervals(id)[inmask].sum())
        return times
