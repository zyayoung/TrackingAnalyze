import cv2
import numpy as np
from PIL import Image
import io

from libs.sort import Sort


class Record:
    def __init__(self, ann_filename):
        self.ann_filename = ann_filename.replace(".mjpeg", ".txt")
        ann_raw = open(self.ann_filename).read().split("T")
        self.ann = []
        self.p = []
        self.times = []
        self.det_counts = []
        self.dets = []
        self.trks = []
        self.unique_ids = []
        for i in range(len(ann_raw)-1):
            a = ann_raw[i].split()
            if a:
                time = int(a[0])
                if len(a)>2:
                    ann_tmp = np.array(a[2:], dtype=float).reshape(-1, 5)
                    self.ann.append((time, int(a[1]), int(ann_raw[i+1].split()[1]), np.array(np.int32(ann_tmp))))
                    self.dets.append(ann_tmp)
                else:
                    self.ann.append((time, int(a[1]), int(ann_raw[i+1].split()[1]), np.empty((0,5))))
                    self.dets.append(np.empty((0,5)))
                self.times.append(time)
                self.det_counts.append(self.ann[-1][-1].shape[0])
        self.frame_count = len(self.ann)
        self.vid_file = open(self.ann_filename[:-4]+".mjpeg", "rb")

        # Gen interval
        self.times = np.array(self.times)
        self.intervals = np.zeros_like(self.times)
        self.intervals[:-1] = self.times[1:] - self.times[:-1]
        self.intervals[-1] = 1000/self.get_mean_fps()

        self.cur_pos = 0
        self.threshold = 0.0

        # Sort
        self.tracker = None
        self.sorted = False
        self.run_sort()
    
    def run_sort(self):
        self.tracked_id = []
        self.tracker = Sort(max_age=12, min_hits=6)
        for i, dets in enumerate(self.dets):
            trks = self.tracker.update(dets)
            self.trks.append(trks)
            for trk in trks:
                self.tracked_id.append([i, *trk])
        self.tracked_id = np.array(self.tracked_id, dtype=int)
        ids, count = np.unique(self.tracked_id[..., -1], return_counts=True)
        ids = ids[count >= 6]
        self.unique_ids = list(map(str, ids))
        mask = np.isin(self.tracked_id[..., -1], ids)
        self.tracked_id = self.tracked_id[mask]
        for i, id in enumerate(ids):
            self.tracked_id[self.tracked_id[..., -1]==id][..., -1] = i
        self.sorted = True
        print("Sort Done! Found {} ids.".format(len(ids)))

    def get_frame(self, index=None):
        if index is None:
            index = self.cur_pos
        time, p, pend, dets = self.ann[index]
        self.vid_file.seek(p)
        im_b = self.vid_file.read(pend-p+4)
        im = cv2.imdecode(np.asarray(bytearray(im_b), dtype="uint8"), cv2.IMREAD_COLOR)
        return im
    
    def get_dets(self, index=None):
        if index is None:
            index = self.cur_pos
        dets = self.dets[index]
        keep = dets[..., 4] > self.threshold
        return dets[keep]
    
    def get_trks(self, index=None):
        if index is None:
            index = self.cur_pos
        if self.sorted:
            trks = self.tracked_id[self.tracked_id[..., 0]==index][..., 1:]
        else:
            trks = self.dets[index]
            trks[..., -1] = -1
        return trks
    
    def get_id_first_occur(self, id):
        return self.tracked_id[self.tracked_id[..., -1] == id][..., 0]
    
    def get_time(self, index=None):
        if index is None:
            index = self.cur_pos
        return self.times[index]
    
    def get_interval(self, index=None):
        if index is None:
            index = self.cur_pos
        return self.intervals[index]
    
    def get_mean_fps(self):
        return len(self.times) * 1000 / (self.times[-1] - self.times[0])
    
    def scanAllImages(self):
        return list(map(str, range(self.frame_count)))

    def get_shapes(self, index):
        if type(index) is str:
            index = int(index)
        shapes = []  # [labbel, [(x1,y1), (x2,y2), (x3,y3), (x4,y4)], color, color, difficult]
        for dets in self.get_trks(index):
            xmin, ymin, xmax, ymax, id = tuple(dets)
            shapes.append([
                f"{id}",
                [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)],
                None,
                None,
                0,
            ])
        return shapes
    
if __name__ == "__main__":
    rec = Record("25.txt")
    ann = rec.ann
    for i in range(rec.frame_count):
        im = rec.get_frame(i)
        dets = rec.get_dets(i)
        time = rec.get_time(i)
        
        cv2.putText(im, f"{(time//60000)%60}:{(time//1000)%60}", (20, 20), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.6, (0,255,0))
        for det in dets:
            x_min, y_min, x_max, y_max = np.int32(det[:-1])
            cv2.rectangle(im, (x_min, y_min), (x_max, y_max), (255, 0, 0))
        cv2.imshow("im", im)
        cv2.waitKey(1)
