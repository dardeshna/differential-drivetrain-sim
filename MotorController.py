from enum import Enum
import numpy as np

class MotorControllerMode(Enum):
    Disabled = -1
    PercentOutput = 0
    Position = 1
    Velocity = 2

class MotorController():

    def __init__(self):
        self.measurement_window = 100
        self.rolling_avg_period = 64

        self.mode = MotorControllerMode.Disabled

        self.P = 0
        self.I = 0
        self.D = 0
        self.F = 0
        self.izone = 0

        self.sensor_readings = np.zeros(self.measurement_window+self.rolling_avg_period)
        self.prev_sensor_rate = 0
        self.sensor_rate = 0
        self.setpoint = 0
        self.error = 0
        self.i_accum = 0
        self.reset = 0
        self.output = 0
        self.prev_output = 0
        self.forwards_peak_output = 1
        self.reverse_peak_output = 1
        self.ramp_rate = 0
        self.deadband = 0.04
        self.voltage_comp_saturation = 12.0
    
        self.input_voltage = 12.0

        self.dt = 0.001
        self.cpr = 4096
    
    def set(self, mode, setpoint):

        if self.mode != mode:
            self.reset = True
        self.mode = mode
        self.setpoint = setpoint
    
    def update(self):

        self.prev_output = self.output

        if self.mode == MotorControllerMode.PercentOutput:
            self.output = int(self.setpoint * 1023)
            self.error = 0
        elif self.mode == MotorControllerMode.Position:
            self._pid(self.sensor_readings[0], self.sensor_readings[1])
        elif self.mode == MotorControllerMode.Velocity:
            self._pid(self.sensor_rate, self.prev_sensor_rate)
        else:
            self.output = 0
            self.error = 0
                
        if self.ramp_rate != 0:
            if np.sign(self.output)*(self.output - self.prev_output) > 1023/self.ramp_rate * self.dt:
                self.output = int(self.prev_output + np.sign(self.output - self.prev_output) * 1023/self.ramp_rate * self.dt)
    
    def _pid(self, pos, prev_pos):

        self.error = self.setpoint - pos
        if self.reset:
            self.i_accum = 0
        
        if self.izone == 0 or abs(self.error) < self.izone:
            self.i_accum += self.error
        else:
            self.i_accum = 0
        
        dErr = prev_pos - pos

        if self.reset:
            dErr = 0
            self.reset = False
         
        self.output = int(max(min(1023*self.forwards_peak_output, self.error*self.P + dErr*self.D + self.i_accum*self.I + self.setpoint*self.F), -1023*self.reverse_peak_output))

    def push_reading(self, reading):
        
        temp = np.zeros(self.measurement_window+self.rolling_avg_period)
        temp[1:] = self.sensor_readings[:-1]
        temp[0] = int(reading)
        self.sensor_readings = temp

        avg = (self.sensor_readings[:self.rolling_avg_period] - self.sensor_readings[self.measurement_window:self.measurement_window+self.rolling_avg_period]).sum() / self.rolling_avg_period
        
        self.prev_sensor_rate = self.sensor_rate
        self.sensor_rate = avg
    
    def reset_sensor(self):
        self.sensor_readings = self.sensor_readings - self.sensor_readings[0]
    
    def get_reading(self):
        return self.sensor_readings[0]

    def get_motor_output_voltage(self):

        if abs(self.output) <= self.deadband*1023:
            return 0
        else:
            return max(-self.input_voltage, min(self.input_voltage, self.output/1023.0 * self.voltage_comp_saturation))

    def get_motor_output_percent(self):
        return self.get_motor_output_voltage()/self.input_voltage
    
    def get_conversions(self, r_wheel):

        ticks_per_meter = self.cpr / (r_wheel * 2 * np.pi)
        ticks_per_100ms_per_meter_per_second = self.cpr / (r_wheel * 2 * np.pi * 10)

        return ticks_per_meter, ticks_per_100ms_per_meter_per_second
