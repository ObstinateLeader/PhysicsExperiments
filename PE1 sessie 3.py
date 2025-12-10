import nidaqmx as dx
import matplotlib.pyplot as plt
import numpy as np
import time as timemod
from scipy import integrate
integrate.trapz = integrate.trapezoid

def integral(x,y):
    #integral functions from scipy will be used to integrate over the datapoints
    return integrate.trapz(y, x)

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
        'Function to set the data manually'
        self.data = data
       
    def setTime(self, time):
        'Function to set the time manually'
        self.time = time

    def sine(self, t, f=1, A=1, phi=0, C=0):
        '''
        function to generate a sinewave

        rate: integer, rate at which the voltages will be sent to the mydaq
        f: float, frequency of the sinewave
        A: float, amplitude of the sinewave
        phi: float, phase of the sinewave
        C: float, offset of the sinewave
        '''
        y = A * np.sin(2*np.pi*f*t+phi) + C
        plt.plot(t, y)
        plt.xlim(0, 1/f)
        plt.show()
        self.sin = y
        return self.sin # to use the sine as a function gen you must manually fill it into either write or readwrite.

    def fft_func(self, rate):
        '''
        function thats performs a fast fourier transform on the saved data
        rate: integer, rate at which the voltages were read from the mydaq
        '''
        self.fft = np.fft.fft(self.data)
        self.freq = np.fft.fftfreq(len(self.data), 1/rate)
        return self.fft, self.freq

    def ifft_func(self):
        '''
        function that performs an inverse fast fourier transform on the saved fft data
        '''
        self.ifft = np.fft.ifft(self.fft)
        return np.real(self.ifft)

    def bodeplots(self):
        '''
        Function that creates bodeplots with the performed fourier transform
        1. Magnitude vs frequency (logarithmic scale)
        2. Phase vs frequency (logarithmic scale)
        '''
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
        '''
        Function that creates a plot of the inverse fourier transform.
        '''
        plt.plot(np.linspace(0, self.time, len(self.ifft)), self.ifft)
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.title('Amplitude vs time')
        plt.show()

    def power(self, f, freqs=None, fft=None, delta_f=1):
        '''
        Function to get the power of specific frequencies.
        freqs: array, frequencies from the fft
        fft: array, fft data
        f: float, frequency of interest
        delta_f: float, width of the interval over which you want to integrate
        returns: power in the interval
        '''
        print(fft)
        if freqs == None:
            freqs = self.freq
        if np.any(fft) == None:
            fft = self.fft
        interval = (freqs > f - delta_f) & (freqs < f + delta_f)
        print(fft)
        power = integral(freqs[interval], np.abs(fft[interval])**2)
        return power
   
    def calc_H(self, U_out, U_in):
        H = np.abs(U_out)/np.abs(U_in)
        return H
   
    def savedata(self, data, name="mydaq_data.csv"):
        np.savetxt(name, data)

    def singlemeasurement(self, f, A, rate, samples):
        '''
        Function to perform a single measurement with a sine wave as input.
        f: float, frequency of the sine wave
        A: float, amplitude of the sine wave
        rate: integer, rate at which the voltages will be sent to/read from the mydaq
        samples: integer, number of samples to be sent/read
        returns: U_in, U_out, fft_in, fft_out, freqs
        '''
        t = np.linspace(0, samples/rate, samples)
        self.sine(t, f=f, A=A)
        self.readwrite(self.sin, rate)
        U_out = self.getData()
        self.fft_func(rate)
        fft_out = self.fft
        freqs = self.freq
        # plt.plot(t, U_out)
        # plt.xlim(0, 1/f)
        # plt.show()
        U_in = self.sin
        fft_in = np.fft.fft(U_in)
        print(self.fft)
        U_out_power = self.power(f, fft=self.fft)
        U_in_power = self.power(f, fft=fft_in)

        H = self.calc_H(U_out_power, U_in_power)
        return H
   
    def multimeasure(self, freqs, Amplitudes, rate, samples):
        '''
        Function to perform multiple measurements with a sine wave as input.
        freqs: array, frequencies of the sine wave
        A: float, amplitude of the sine wave
        rate: integer, rate at which the voltages will be sent to/read from the mydaq
        samples: integer, number of samples to be sent/read
        returns: Hs, freqs
        '''
        Hs = []
        for f, A in zip(freqs, Amplitudes):
            H = self.singlemeasurement(f, A, rate, samples)
            Hs.append(H)
        return Hs, freqs
       
mydaq = myDAQ()
rate = 200000
samples = rate*5
t = np.linspace(0, samples/rate, samples)
# mydaq.singlemeasurement(30, 5, rate, samples)
freqs = np.logspace(np.log10(20), np.log10(2000), 5)
print(freqs)
Amplitudes= np.zeros(10) + 5
print(Amplitudes)
H, fs = mydaq.multimeasure(freqs, Amplitudes, rate, samples)

np.savetxt('Datasession3.txt', [H, fs])

plt.scatter(fs, H)
plt.loglog()
plt.xlabel(r'$\omega$')
plt.ylabel('H')
plt.savefig('goodplot.png')
plt.show()