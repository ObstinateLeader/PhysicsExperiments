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
        self.fft = None
        self.freq = None
        self.time = None
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
            self.time = len(array)/rate
            writeTask.stop()

    def read(self, rate, time):
        '''
        Reads from the mydaq
        rate: integer, rate at which the voltages will be read from the mydaq
        '''
        with dx.Task() as readTask:
            # self.time = time
            samps_per_chan = rate*time
            readTask.ai_channels.add_ai_voltage_chan('myDAQ1/ai0')


            readTask.timing.cfg_samp_clk_timing(rate, sample_mode=dx.constants.AcquisitionType.FINITE, samps_per_chan=samps_per_chan)

            data = readTask.read(number_of_samples_per_channel=samps_per_chan)
           
            # save data
            np.savetxt("mydaq_data.csv", data)
            self.time = time
            self.data = data

    def readwrite(self, array, rate):
        '''
        Writes to the mydaq while also reading back from the mydaq

        array: list/array, containing the voltages
        rate: integer, rate at which the voltages will be sent to/read from the mydaq
        '''
        if np.max(array) >= 10:
            array=[0]
        with dx.Task('AOTask') as writeTask, dx.Task('AITask') as readTask:
            # samps_per_chan = len(array)
            self.time = len(array)/rate
            samps_per_chan = round(rate*self.time)
            writeTask.ao_channels.add_ao_voltage_chan('myDAQ1/ao0')

            writeTask.timing.cfg_samp_clk_timing(rate, sample_mode=dx.constants.AcquisitionType.FINITE,samps_per_chan=samps_per_chan)
            readTask.ai_channels.add_ai_voltage_chan('myDAQ1/ai0')

            readTask.timing.cfg_samp_clk_timing(rate, sample_mode=dx.constants.AcquisitionType.FINITE,samps_per_chan=samps_per_chan)

            writeTask.write(array, auto_start=True)
            data = readTask.read(number_of_samples_per_channel=samps_per_chan)
           
            # save data
            np.savetxt("mydaq_data.csv", data)
           
            timemod.sleep(self.time+0.00000001)
            self.data = data
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
   
    def setData(self, data):
        self.data = data
       
    def setTime(self, time):
        self.time = time

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

    def fft_func(self, rate):
        self.fft = np.fft.fft(self.data)
        self.freq = np.fft.fftfreq(len(self.data), 1/rate)
        return self.fft, self.freq

    def ifft_func(self):
        self.ifft = np.fft.ifft(self.fft)
        return np.real(self.ifft)

    def bodeplots(self):
        freqs = self.freq[:int(len(self.freq)/2)]
        fft = self.fft[:int(len(self.fft)/2)]
        #Plot the magnitude and frequency on a logarithmic scale
        plt.semilogx(freqs, 20*np.log10(np.abs(fft)))
        plt.xlabel("Frequency")
        plt.ylabel("Magnitude (dB)")
        plt.title("Magnitude vs frequency (Bode plot)")
        plt.show()
       
        #We will do the same with the phase information
        plt.semilogx(freqs, np.mod(np.angle(fft), 2*np.pi), '.')
        plt.xlabel("Frequency")
        plt.ylabel("Phase")
        plt.title("Phase vs frequency")
        plt.show()

    def ifft_plot(self):
        plt.plot(np.linspace(0, self.time, len(self.ifft)), self.ifft)
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.title('Amplitude vs time')
        plt.show()
   
mydaq = myDAQ()
# f = 50000
# T_wave = 1/f
# print(T_wave)
# rate = f * 3#int(1e4)
# num_points = f * 3#int(1e4)
# num_periods = 1

# sine = mydaq.sine(num_points, num_periods, f)


# writing

#plt.figure()
#plt.plot(np.linspace(0, num_points/rate, num_points) , sine)
#plt.xlabel("time (seconds)")
#plt.ylabel("voltage (V)")
#plt.title("Sine wave sent to myDAQ")
#plt.show()

# #mydaq.write(sine, rate)


# reading

# mydaq.read(rate, time)
# data = mydaq.getData()


# plt.plot(np.linspace(0, time, len(data)), data)
# plt.xlabel("time (seconds)")
# plt.ylabel("voltage (V)")
# plt.title("Sine wave received from myDAQ")
# plt.show()

# reading and writing

# mydaq.readwrite(sine, rate)
# data = mydaq.getData()
# time = mydaq.getTime()
# plt.plot(np.linspace(0, time, len(data)), data)
# plt.plot(np.linspace(0, time, len(data)), sine)
# plt.xlabel("time (seconds)")
# plt.ylabel("voltage (V)")
# # plt.xlim(0, time/len(data)*175/f)
# plt.xlim(0, time/num_periods/f)
# plt.title("Sine wave received from myDAQ")
# plt.show()


rate = 46000
samples = 5000

#Create the time data array
# t = np.linspace(0, samples/rate, samples)

# #Artifically create a signal, in this case a sine wave
# frequency = 5
# signal = np.sin(2*np.pi * frequency * t)
# signal2 = 0.01*np.sin(2*np.pi*10*t) + 0.1*np.sin(2*np.pi*100*t + 3) + 10*np.cos(2*np.pi*50*t)

# mydaq.setData(signal2)
# mydaq.setTime(t)
# mydaq.fft_func(rate)
# mydaq.bodeplots()
# mydaq.ifft_func()
# mydaq.ifft_plot()

mydaq.read(rate, 5)

#%%
data = mydaq.getData()
plt.plot(np.linspace(0, 2, len(data)), data)
# plt.xlim(800, 1000)
plt.xlabel('time (s)')
plt.ylabel('Signal (V)')
plt.title('Voice measured')
plt.savefig('signal.png')
plt.show()


#%%
mydaq.write(np.array(data)/3, rate)
print(len(data))
#%%
# normal
mydaq.fft_func(rate)
mydaq.bodeplots()
mydaq.ifft_func()
# mydaq.ifft_plot()
plt.plot(np.linspace(0, 2, len(mydaq.ifft)), np.array(mydaq.ifft))
plt.xlabel('time (s)')
plt.ylabel('Amplitude (V)')
plt.title('inverse fourier transform')
plt.savefig('ifft.png')
plt.show()
#%%
mydaq.write(np.array(mydaq.ifft)/5, rate)
#%%
# isolate magnitudes
mydaq.fft_func(rate)
mydaq.fft= np.abs(mydaq.fft)
mydaq.bodeplots()
mydaq.ifft_func()
plt.plot(np.array(mydaq.ifft))
mydaq.write(mydaq.ifft/3, rate)

#%%
# isolate phases
mydaq.fft_func(rate)
fft = mydaq.fft
# mydaq.bodeplots()
expon = np.exp(1j * np.angle(fft))
mydaq.fft = expon
mydaq.bodeplots()
mydaq.ifft_func()
mydaq.ifft_plot()
mydaq.write(np.array(data)/3, rate)
mydaq.write(mydaq.ifft/3, rate)