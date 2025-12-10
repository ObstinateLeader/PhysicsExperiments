import nidaqmx as dx
import matplotlib.pyplot as plt
import numpy as np
import time as timemod

class myDAQ():
    def __init__(self):
        '''
        sets initial variables which are empty at this time
        but will be filled in while using the functions
        '''
        self.data = None
        self.time = None
        self.sin = None
        pass

    def write(self, array, rate):
        '''
        Writes to the mydaq
        array: list/array, containing the voltages
        rate: integer, rate at which the voltages will be sent to the mydaq
        '''
        if np.max(array) >= 10:
            array=[0]
       
        with dx.Task() as writeTask:

            writeTask.ao_channels.add_ao_voltage_chan('myDAQ1/ao0')

            writeTask.timing.cfg_samp_clk_timing(rate, sample_mode=dx.constants.AcquisitionType.FINITE, samps_per_chan=len(array)) # number of samples

            writeTask.write(array, auto_start=True)

            timemod.sleep(len(array)/rate+0.001)

            writeTask.stop()

    def read(self, rate, time):
        '''
        Reads from the mydaq
        rate: integer, rate at which the voltages will be read from the mydaq
        '''
        with dx.Task() as readTask:
            samps_per_chan = rate*time
            readTask.ai_channels.add_ai_voltage_chan('myDAQ1/ai0')

            readTask.timing.cfg_samp_clk_timing(rate, sample_mode=dx.constants.AcquisitionType.FINITE, samps_per_chan=samps_per_chan)

            data = readTask.read(number_of_samples_per_channel=samps_per_chan)
           
            # save data
            np.savetxt("mydaq_data.csv", data)
           
            self.data = data

    def readwrite(self, array, rate, time):
        '''
        Writes to the mydaq while also reading back from the mydaq

        array: list/array, containing the voltages
        rate: integer, rate at which the voltages will be sent to/read from the mydaq
        '''
        if np.max(array) >= 10:
            array=[0]
        with dx.Task('AOTask') as writeTask, dx.Task('AITask') as readTask:
            # samps_per_chan = len(array)
            samps_per_chan = round(rate*time)
            writeTask.ao_channels.add_ao_voltage_chan('myDAQ1/ao0')

            writeTask.timing.cfg_samp_clk_timing(rate, sample_mode=dx.constants.AcquisitionType.FINITE,samps_per_chan=samps_per_chan)
            readTask.ai_channels.add_ai_voltage_chan('myDAQ1/ai0')

            readTask.timing.cfg_samp_clk_timing(rate, sample_mode=dx.constants.AcquisitionType.FINITE,samps_per_chan=samps_per_chan)

            writeTask.write(array, auto_start=True)
            data = readTask.read(number_of_samples_per_channel=samps_per_chan)
           
            # save data
            np.savetxt("mydaq_data.csv", data)
           
            timemod.sleep(time+0.00000001)
            self.data = data
            time = samps_per_chan/rate
            self.time = time
            writeTask.stop()

    def getData(self):
        '''
        function to return the data that was read from the mydaq
        '''
        return self.data

    def getTime(self):
        '''
        function to return the time that the data was read from the mydaq
        '''
        return self.time

    def sine(self, num_points, num_periods, f=1, A=1, phi=0, C=0):
        '''
        function to generate a sinewave

        rate: integer, rate at which the voltages will be sent to the mydaq
        f: float, frequency of the sinewave
        A: float, amplitude of the sinewave
        phi: float, phase of the sinewave
        C: float, offset of the sinewave
        '''
        x = np.linspace(0, num_periods, num_points)*2*3.14159265358
        y = A * np.sin(f*x+phi) + C
        self.sin = y
        return self.sin # to use the sine as a function gen you must manually fill it into either write or readwrite.
   
mydaq = myDAQ()
f = 50000
T_wave = 1/f
print(T_wave)
rate = f * 3#int(1e4)
num_points = f * 3#int(1e4)
num_periods = 1

sine = mydaq.sine(num_points, num_periods, f)


# writing

#plt.figure()
#plt.plot(np.linspace(0, num_points/rate, num_points) , sine)
#plt.xlabel("time (seconds)")
#plt.ylabel("voltage (V)")
#plt.title("Sine wave sent to myDAQ")
#plt.show()

# #mydaq.write(sine, rate)


# reading

time = (len(sine) / rate)
# mydaq.read(rate, time)
# data = mydaq.getData()


# plt.plot(np.linspace(0, time, len(data)), data)
# plt.xlabel("time (seconds)")
# plt.ylabel("voltage (V)")
# plt.title("Sine wave received from myDAQ")
# plt.show()

# reading and writing

mydaq.readwrite(sine, rate, time)
data = mydaq.getData()
plt.plot(np.linspace(0, time, len(data)), data)
plt.plot(np.linspace(0, time, len(data)), sine)
plt.xlabel("time (seconds)")
plt.ylabel("voltage (V)")
# plt.xlim(0, time/len(data)*175/f)
plt.xlim(0, time/num_periods/f)
plt.title("Sine wave received from myDAQ")
plt.show()