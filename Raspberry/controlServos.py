import math
import time
from threading import Thread
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
        
        self.angle_x = self.mid_x
        self.angle_y = self.mid_y
        self.servo_delay = 0.005
        
        self.min_movement = 4
        self.max_movement = 15
        self.moving_x = False
        self.moving_y = False
    
        
    def move(self, x, y, w, h):
        cx = int(x + w / 2)
        cy = int(y + h / 2)
        move_horizontally = int((self.cam_cx - cx) / 7)
        move_vertically = int((self.cam_cy - cy) / 8)
              
        if((move_horizontally > self.min_movement or move_horizontally < 0-self.min_movement) and
           (move_horizontally < self.max_movement or move_horizontally > 0-self.max_movement)):
            self.moving_x = True
            old_angle_x = self.angle_x - self.mid_x
            self.angle_x = self.angle_x - move_horizontally
            
        if((move_vertically > self.min_movement or move_vertically < 0-self.min_movement) and
           (move_vertically < self.max_movement or move_vertically > 0-self.max_movement)):
            self.moving_y = True
            old_angle_y = self.angle_y - self.mid_y
            self.angle_y = self.angle_y - move_vertically
        

        if((move_horizontally > self.min_movement or move_horizontally < 0-self.min_movement) and
           (move_horizontally < self.max_movement or move_horizontally > 0-self.max_movement)):
            if self.angle_x <  0:
                self.angle_x = 0
            if self.angle_x > self.pan_angle:
                self.angle_x = self.pan_angle
            
        if((move_vertically > self.min_movement or move_vertically < 0-self.min_movement) and
           (move_vertically < self.max_movement or move_vertically > 0-self.max_movement)):
            if self.angle_y < 0:
                self.angle_y = 0
            if self.angle_y > self.tilt_angle:
                self.angle_y = self.tilt_angle

        if((move_horizontally > self.min_movement or move_horizontally < 0-self.min_movement) and
           (move_horizontally < self.max_movement or move_horizontally > 0-self.max_movement)):
            pan_servo = self.angle_x - self.mid_x
            if pan_servo > self.max_x:
                pan_servo = self.max_x
            if pan_servo < self.min_x:
                pan_servo = self.min_x
        
            print(f"pan_servo - {pan_servo}")
            t_pan = Thread(target=self.smooth, args=('pan',old_angle_x,pan_servo))
            t_pan.daemon = True
            t_pan.start()
        
        if((move_vertically > self.min_movement or move_vertically < 0-self.min_movement) and
           (move_vertically < self.max_movement or move_vertically > 0-self.max_movement)):
            tilt_servo = self.angle_y - self.mid_y
            if tilt_servo > self.max_y:
                servo_y = self.max_y
            if tilt_servo < self.min_y:
                tilt_servo = self.min_y
            
            print(f"tilt_servo - {tilt_servo}")
            t_tilt = Thread(target=self.smooth, args=('tilt',old_angle_y,tilt_servo))
            t_tilt.daemon = True
            t_tilt.start()
        
    def smooth(self, servo, angle, move_to):
        print(f'angle - {angle}, servo - {servo}, move_to - {move_to}')
        if angle < move_to:
            while angle < move_to:
                angle += 0.1
                if servo == 'pan' and angle >= self.min_x and angle <= self.max_x:
                    self.pan.angle = angle
                if servo == 'tilt' and angle >= self.min_y and angle <= self.max_y:
                    self.tilt.angle = angle
                time.sleep(self.servo_delay)
        else:
            while angle > move_to:
                angle -= 0.1
                if servo == 'pan' and angle >= self.min_x and angle <= self.max_x:
                    self.pan.angle = angle
                if servo == 'tilt' and angle >= self.min_y and angle <= self.max_y:
                    self.tilt.angle = angle
                time.sleep(self.servo_delay)
        if servo == 'pan':
            self.moving_x = False
        if servo == 'tilt':
            self.moving_y = False
        
    def move_default(self):
        self.moving = True       
        t_pan = Thread(target=self.smooth, args=('pan', self.angle_x - self.mid_x, 0))
        t_pan.daemon = True
        t_pan.start()
        t_tilt = Thread(target=self.smooth, args=('tilt', self.angle_y - self.mid_y, 0))
        t_tilt.daemon = True
        t_tilt.start()
        self.angle_x = self.mid_x
        self.angle_y = self.mid_y
        self.pan.angle = 0
        self.tilt.angle = 0
        
    def get_status(self):
        if not self.moving_x and not self.moving_y:
            return False
        return True
