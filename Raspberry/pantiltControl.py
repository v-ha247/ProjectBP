import math
import time
from gpiozero import AngularServo
from gpiozero.pins.pigpio import PiGPIOFactory

class Pantilt:
    def __init__(self, pan_pin, tilt_pin, cam_width, cam_height, pan_angle, tilt_angle):
        self.pan_pin = pan_pin
        self.tilt_pin = tilt_pin
        
        self.pan_angle = pan_angle
        self.tilt_angle = tilt_angle
        
        self.mid_x = pan_angle / 2
        self.mid_y = tilt_angle / 2
                
        self.min_x = 0 - self.mid_x
        self.max_x = pan_angle - self.mid_x
        self.min_y = 0 - self.mid_y
        self.max_y = tilt_angle - self.mid_y
        
        self.cam_cx = cam_width / 2
        self.cam_cy = cam_height / 2
        
        # run command sudo pigpiod if error
        self.factory = PiGPIOFactory()
        self.pan = AngularServo(self.pan_pin, min_angle=self.min_x, max_angle=self.max_x, pin_factory=self.factory)
        self.tilt = AngularServo(self.tilt_pin, min_angle=self.min_y, max_angle=self.max_y, pin_factory=self.factory)
        
        print(f"dim {self.min_x} {self.max_x} {self.min_y} {self.max_y}")
        
        self.angle_x = self.mid_x
        self.angle_y = self.mid_y
        self.servo_delay = 0.15
    
        
    def move(self, x, y, w, h):
        print("\nmoving")
        cx = int(x + w / 2)
        cy = int(y + h / 2)
        move_horizontally = int((self.cam_cx - cx) / 7)
        move_vertically = int((self.cam_cy - cy) / 6)
        self.angle_x = self.angle_x - move_horizontally
        self.angle_y = self.angle_y - move_vertically
        
        print(f"Pan To {move_horizontally} {move_vertically} cx={cx} cy={cy}")
        print(f"Cam mid {self.cam_cx} {self.cam_cy}")
        
        if self.angle_x <  0:
            self.angle_x = 0
        if self.angle_x > self.pan_angle:
            self.angle_x = self.pan_angle
            
        print(f"self.angle_x - {self.angle_x}")

        if self.angle_y < 0:
            self.angle_y = 0
        if self.angle_y > self.tilt_angle:
            self.angle_y = self.tilt_angle
            
        print(f"self.angle_y - {self.angle_y}")

        pan_servo = self.angle_x - self.mid_x
        if pan_servo > self.max_x:
            pan_servo = self.max_x
        if pan_servo < self.min_x:
            pan_servo = self.min_x
        
        print(f"pan_servo - {pan_servo}")
        self.pan.angle = pan_servo

        tilt_servo = self.angle_y - self.mid_y
        if tilt_servo > self.max_y:
            servo_y = self.max_y
        if tilt_servo < self.min_y:
            tilt_servo = self.min_y
            
        print(f"tilt_servo - {tilt_servo}")    
        self.tilt.angle = tilt_servo
        
        time.sleep(self.servo_delay)
