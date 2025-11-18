import cv2
import torch
import numpy as np


class VideoRead(object):
    """
    Class representing frame by frame video reading 
    """
    def __init__(self, video_path=None):
        self.video_path = video_path
        self.height = None
        self.width = None
        self.fps = None
    
    def get_frames(self):
        """
        Description
        -----------
        Generator function for yielding video frames at each step
        """
        video_capture = cv2.VideoCapture(self.video_path)
        #saving image characteristics
        self.height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.fps = video_capture.get(cv2.CAP_PROP_FPS)
        #creating frames generator
        while True:
            ret, frame = video_capture.read()
            #process picture frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor_img = torch.from_numpy(rgb_frame).permute(2,0,1).float() / 255.0
            #stop yielding when video ends
            if not ret:
                raise StopIteration
            else:
                yield tensor_img





if __name__ == '__main__':
    some_ar = np.array([k for k in range(1,11)])
    print(some_ar)
