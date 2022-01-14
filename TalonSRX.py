from enum import Enum
import numpy as np

class TalonSRXControlMode(Enum):
    PercentOutput = 0
    Position = 1
    Velocity = 2
    # Current = 3
    # Follower = 5
    # MotionProfile = 6
    # MotionMagic = 7
    # MotionProfileArc = 10
    Disabled = 15

class TalonSRX():

    def __init__(self):
        self.measurementWindow = 100
        self.rollingAvgPeriod = 64

        self.mode = TalonSRXControlMode.Disabled

        self.P = 0
        self.I = 0
        self.D = 0
        self.F = 0
        self.izone = 0

        self.sensorReadings = np.zeros(self.measurementWindow+self.rollingAvgPeriod)
        self.prevSensorRate = 0
        self.sensorRate = 0
        self.setpoint = 0
        self.error = 0
        self.iAccum = 0
        self.reset = 0
        self.output = 0
        self.prevOutput = 0
        self.forwardsPeakOutput = 1
        self.reversePeakOutput = 1
        self.rampRate = 0
        self.deadband = 0.04
        self.voltageCompSaturation = 12.0

        self.dt = 0.001
        self.cpr = 4096
    
    def set(self, mode, setpoint):

        # print(f"Control Mode:{mode}\nSetpoint: {setpoint}")

        if self.mode != mode:
            self.reset = True
        self.mode = mode
        self.setpoint = setpoint
    
    def update(self):

        self.prevOutput = self.output

        if self.mode == TalonSRXControlMode.PercentOutput:
            self.output = int(self.setpoint * 1023)
            self.error = 0
        elif self.mode == TalonSRXControlMode.Position:
            self.pid(self.sensorReadings[0], self.sensorReadings[1])
        elif self.mode == TalonSRXControlMode.Velocity:
            self.pid(self.sensorRate, self.prevSensorRate)
        else:
            self.output = 0
            self.error = 0

        if self.rampRate != 0:
            if np.sign(self.output)*(self.output - self.prevOutput) > 1023/self.rampRate * self.dt:
                self.output = int(self.prevOutput + np.sign(self.output - self.prevOutput) * 1023/self.rampRate * self.dt)
    
    def pid(self, pos, prevPos):

        self.error = self.setpoint - pos
        if self.reset:
            self.iAccum = 0
        
        if self.izone == 0 or abs(self.error) < self.izone:
            self.iAccum += self.error
        else:
            self.iAccum = 0
        
        dErr = prevPos - pos

        if self.reset:
            dErr = 0
            self.reset = False
         
        self.output = int(max(min(1023*self.forwardsPeakOutput, self.error*self.P + dErr*self.D + self.iAccum*self.I + self.setpoint*self.F), -1023*self.reversePeakOutput))

    def pushReading(self, reading):
        
        temp = np.zeros(self.measurementWindow+self.rollingAvgPeriod)
        temp[1:] = self.sensorReadings[:-1]
        temp[0] = int(reading)
        self.sensorReadings = temp

        avg = (self.sensorReadings[:self.rollingAvgPeriod] - self.sensorReadings[self.measurementWindow:self.measurementWindow+self.rollingAvgPeriod]).sum() / self.rollingAvgPeriod
        
        self.prevSensorRate = self.sensorRate
        self.sensorRate = avg
    
    def resetSensor(self):
        self.sensorReadings = self.sensorReadings - self.sensorReadings[0]
    
    def getReading(self):
        return self.sensorReadings[0]
    
    def getMotorOutputVoltage(self, batteryVoltage=12.0):

        if abs(self.output) <= self.deadband*1023:
            return 0
        else:
            return max(-batteryVoltage, min(batteryVoltage, self.output/1023.0 * self.voltageCompSaturation))

    def getMotorOutputPercent(self, batteryVoltage=12.0):
        return self.getMotorOutputVoltage(batteryVoltage)/batteryVoltage
    
    def get_conversions(self, r_wheel):

        ticks_per_meter = self.cpr / (r_wheel * 2 * np.pi)
        ticks_per_100ms_per_meter_per_second = self.cpr / (r_wheel * 2 * np.pi * 10)

        return ticks_per_meter, ticks_per_100ms_per_meter_per_second