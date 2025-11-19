import cv2
import numpy as np

from tqdm import tqdm

import torch
from torchvision import transforms

from ultralytics import YOLO


class VideoReader:
    """
    Frame by frame video reading

    Reads a video and implements a generator for video frames

    Parameters
    ----------
    video_path : str, default='./crowd.mp4'
        System path for video reading

    img_resize : int, default=1280
        Size to resize image

    Attributes
    ----------
    video_capture : cv2.VideoCapture
        Object capturing video frames

    width : int
        Width of video frames
    
    height : int
        Height of video frames
    
    fps : int
        Video fps
    
    transforms : torchvision.transforms.Resize
        Transform for image resizing
    """
    def __init__(self, video_path: str = 'crowd.mp4', img_resize: int = 1088):
        self.video_capture = cv2.VideoCapture(video_path)
        # Saving image characteristics to use for later video writing
        self.height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        # Applying transformations to fit model input size
        self.transforms = transforms.Resize((img_resize, img_resize))
    
    def generate_frames(self):
        """
        Generator function for yielding video frames at each time step

        Yields
        ------
        ret : bool
            Frame presence flag
        
        tensor_img : torch.Tensor
            Original Numpy image
            Tensor image
        """
        # Creating generator to return video frames
        while True:
            ret, frame = self.video_capture.read()
            # Checking if frame is present
            if not ret:
                break
            # Image processing to fit the frame to model input
            tensor_img = torch.from_numpy(frame).permute(2, 0, 1)
            tensor_img = tensor_img.unsqueeze(dim=0).float() / 255.0
            tensor_img = self.transforms(tensor_img)

            yield frame, tensor_img
    
    def release(self) -> None:
        """Releases the video after it is read"""
        self.video_capture.release()


class VideoSaver:
    """
    Frame by frame video writing

    Writes frames to video with specified width, height, and fps

    Parameters
    ----------
    fps : float
        Video fps

    width : int
        Video frame width

    height : int
        Video frame height

    path : str
        Path to write a video into
    
    Attributes
    ----------
    writer : cv2.VideoWriter
        Object for writing videos

    fps : float
        Video fps

    width : int
        Video frame width

    height : int
        Video frame height

    save_path : str
        Path to write a video into
    """
    def __init__(self, fps: float, width: int, height: int,
                 path: str = 'videoWithBoundingBoxes.mp4'):
        self.fps = fps
        self.width = width
        self.height = height
        self.save_path = path
        self.writer = cv2.VideoWriter(self.save_path, 
                                      cv2.VideoWriter_fourcc(*'mp4v'), 
                                      self.fps, (self.width, self.height))
    
    def add_frame(self, frame: np.ndarray) -> None:
        """Adds a frame to video"""
        self.writer.write(frame)
    
    def release(self) -> None:
        """Releases object resources after the video is written"""
        self.writer.release()


class ImageRecDrawing:
    """
    Prediction objects drawing

    Draws rectangles, labels, and model confidence on video frames

    Parameters
    ----------
    picture_size : tuple[int, int]
        Video writing image (width, height) for final video frame output
    
    model_picture_size : tuple[int, int]
        Model image (width, height) to scale coordinates

    Attributes
    ----------
    picture_size : tuple[int, int]
        Image (width, height) to resize final video frame output to 
    
    model_picture_size : tuple[int, int]
        Model image (width, height) to scale coordinates
    """
    def __init__(self, picture_size: tuple[int, int], model_picture_size: tuple[int, int]):
        self.picture_size = picture_size
        self.model_picture_size = model_picture_size
    
    def draw(self, picture, prediction) -> np.ndarray:
        """
        Given a picture and prediction draws prediction on picture inplace

        Parameters
        ----------
        picture : np.ndarray
            Array picture in cv2 format (i.e. uint8)

        prediction : tuple[tuple[int, int, int, int], str, float]
            Model predctions to draw on picture

        Returns
        -------
        picture : np.ndarray
            Array picture ready to be written to RGB video
        """
        scale_x = self.picture_size[0] / self.model_picture_size[0]
        scale_y = self.picture_size[1] / self.model_picture_size[1]

        label = prediction[1]
        confidence = prediction[2]
        bounding_box = prediction[0]

        pt1 = int(bounding_box[0]*scale_x), int(bounding_box[1]*scale_y)
        pt2 = int(bounding_box[2]*scale_x), int(bounding_box[3]*scale_y)

        cv2.rectangle(picture, pt1, pt2, (128, 255, 52), 1)
        cv2.putText(picture, f"{label} {confidence}",
                    (int(bounding_box[0]*scale_x), int(bounding_box[1]*scale_y)-5),
                    cv2.FONT_ITALIC, 0.75, (15, 15, 97), 2)
            
        return picture
    

class ModelLoad:
    """
    Detection model with pretrained weights

    Loads a pretrained object detection model for inference

    Parameters
    ----------
    confidence : float
        Model makes prediction if its assurance is above this number
    
    save_img_flag : bool
        Flag to save inference image in current directory
    
    include_classes : list
        A list of classes to include in predictions result
    
    verbose : bool
        Flag to print model inference information

    imgsz : int
        A size to square a frame to before passing to model
    
    Attributes
    ----------
    model : YOLO
        Model used for inference
    
    confidence : float
        Model confidence
    
    save_img_flag : bool
        Flag to save inference image in current directory

    verbose : bool
        Flag to print model inference information
    
    include_classes : list
        A list of classes to include in predictions result

    imgsz : int
        A size to square a frame to before passing to model
    """
    def __init__(self, confidence: float = 0.5, save_img_flag: bool = False,
                 include_classes: list = ['person'], verbose: bool = False, imgsz=640):
        self.model = YOLO("yolo11l.pt")
        self.confidence = confidence
        self.save_img_flag = save_img_flag
        self.include_classes = include_classes
        self.verbose = verbose
        self.imgsz = imgsz
    
    def predict(self, picture: torch.Tensor) -> tuple[
        np.ndarray,
        list[tuple[tuple[int, int, int, int], str, float]]
    ]:
        """
        Makes object predictions based on picture

        Parameters
        ----------
        picture : torch.Tensor
            Processed tensor picture for model input

        Returns
        -------
        return_picture : np.ndarray
            Numpy array picture ready to be written to another video
        
        detected_objects : list[tuple[tuple[int, int, int, int], str, float]]
            Objects included in the classes list and found on picture 
        """
        prediction = self.model(picture, 
                                save=self.save_img_flag, 
                                conf=self.confidence, 
                                verbose=self.verbose,
                                imgsz=self.imgsz)
        # Peparing variables to store data
        detected_objects = []

        for box in prediction[0].boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].to(dtype=torch.int).tolist()
            label = self.model.names[int(box.cls[0])]

            # Adding only predictions included in list classes to exclude others
            if label in self.include_classes:
                detection_tuple = ((x1, y1, x2, y2), label, round(conf, 2))
                detected_objects.append(detection_tuple)

        return detected_objects


def make_video() -> None:
    """
    Integrates all the components of a program. 
    Creates all the steps: model, reading video, drawing, writing video.
    Saves results.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    model = ModelLoad(imgsz=640)
    videoReader = VideoReader(video_path='crowd.mp4', img_resize=640)
    videoSaver = VideoSaver(videoReader.fps, videoReader.width, videoReader.height)
    imgDraw = ImageRecDrawing(picture_size=(videoReader.width, videoReader.height), model_picture_size=(640, 640))
    video_frames = videoReader.generate_frames()
    
    for numpy_frame, tensor_frame in tqdm(video_frames, desc='Writing the video'):
        detected_objects = model.predict(tensor_frame)
        
        for obj in detected_objects:
            numpy_frame = imgDraw.draw(numpy_frame, obj)
        
        videoSaver.add_frame(numpy_frame)
    

    videoReader.release()
    videoSaver.release()

    print("Done!")



if __name__ == '__main__':
    make_video()
