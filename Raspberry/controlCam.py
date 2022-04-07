import time
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
from threading import Thread

class Camera:
    def __init__(self, resolution=(640, 480), framerate=24, hflip=False, vflip=False):
        self.camera = PiCamera()
        self.camera.resolution = resolution
        self.camera.framerate = framerate
        self.camera.hflip = hflip
        self.camera.vflip = vflip
        self.rawCapture = PiRGBArray(self.camera, size=resolution)
        self.capture = self.camera.capture_continuous(self.rawCapture, format="bgr", use_video_port=True)
        
        self.frame = None
        self.stop_streaming = False

    def start(self):
        self.stop_streaming = False
        t = Thread(target=self.stream, args=())
        t.daemon = True
        t.start()
        return self
    
    def stream(self):
        print("Streaming image")
        for f in self.capture:
            self.frame = f.array
            self.rawCapture.truncate(0)

            if self.stop_streaming:
                self.capture.close()
                self.rawCapture.close()
                self.camera.close()
                return
            
    def get_frame(self):
        return self.frame
        
    def horizontal_flip(self):
        self.camera.hflip = True
        
    def vertical_flip(self):
        self.camera.vflip = True
        
    def horizontal_flip_undo(self):
        self.camera.hflip = False
        
    def vertical_flip_undo(self):
        self.camera.vflip = False

    def stop(self):
        self.stop_streaming = True
            
            