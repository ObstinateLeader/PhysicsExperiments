import nidaqmx as dx
import matplotlib.pyplot as plt
import numpy as np
import time as timemod
from scipy import integrate
integrate.trapz = integrate.trapezoid
from scipy.optimize import curve_fit
from datetime import datetime

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

    def read(self, rate, time, channel="myDAQ1/ai0"):
        '''
        Reads from the mydaq
        rate: integer, rate at which the voltages will be read from the mydaq
        '''
        with dx.Task() as readTask:
            # self.time = time
            samps_per_chan = rate*time
            readTask.ai_channels.add_ai_voltage_chan(channel)


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
            readTask.ai_channels.add_ai_voltage_chan('myDAQ1/ai1')
            readTask.timing.cfg_samp_clk_timing(rate, sample_mode=dx.constants.AcquisitionType.FINITE,samps_per_chan=samps_per_chan)
           
            # extra
           

            # readTask2.timing.cfg_samp_clk_timing(rate, sample_mode=dx.constants.AcquisitionType.FINITE,samps_per_chan=samps_per_chan)
           
           
            writeTask.write(array, auto_start=True)
            data = readTask.read(number_of_samples_per_channel=samps_per_chan)
            # data2 = readTask2.read(number_of_samples_per_channel=samps_per_chan)
            # save data
            np.savetxt("mydaq_data.csv", data)
           
            timemod.sleep(self.time+0.00000001)
            # print(type(data))
            self.data = data[0]
            self.data2 = data[1]
            writeTask.stop()

    def getData(self):
        '''
        function to return the data that was read from the mydaq
        '''
        return self.data
   
    def getData2(self):
        '''
        function to return the data that was read from the mydaq
        '''
        return self.data2

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
        # print(fft)
        if freqs == None:
            freqs = self.freq
        if np.any(fft) == None:
            fft = self.fft
        interval = (freqs > f - delta_f) & (freqs < f + delta_f)
        # print(fft)
        power = integral(freqs[interval], np.abs(fft[interval])**2)
        return power
   
    def calc_H(self, U_out, U_in):
        H = np.abs(U_out)/np.abs(U_in)
        return H
   
    def calc_phase_H(self, U_out, U_in, f):
        print(f"U_out is {U_out}")
        print(f"U_in is {U_in}")
        print([np.argmin(np.abs(f-self.freq))], len(U_out))
        arg_H = np.angle(U_out[np.argmin(np.abs(f-self.freq))], deg=True) - np.angle(U_in[np.argmin(np.abs(f-self.freq))], deg=True)
        return arg_H
   
    def savedata(self, data, name="mydaq_data.csv"):
        np.savetxt(name, data)
       
    def fft_func2(self, rate):
        '''
        function thats performs a fast fourier transform on the saved data
        rate: integer, rate at which the voltages were read from the mydaq
        '''
        self.fft2 = np.fft.fft(self.data2)
        self.freq2 = np.fft.fftfreq(len(self.data2), 1/rate)
        return self.fft2, self.freq2

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
        U_out2 = self.getData2()
       
        plt.plot(t, U_out2, color="red")
        plt.xlim(0, 1/f)
        plt.show()
        # save data
        data_to_save = np.array([U_out, U_out2])
        base_name = "experiment_rawdata"
        extension = ".txt"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{base_name}_{timestamp}{extension}"
        np.savetxt(filename, data_to_save, fmt='%.4f', delimiter=',')
       
       
        self.fft_func(rate)
        fft_out = self.fft
        freqs = self.freq
        self.fft_func2(rate)
        fft_out2 = self.fft2
        freqs2 = self.freq2
       
        # plt.plot(t, U_out)
        # plt.xlim(0, 1/f)
        # plt.show()
        U_in = self.sin
        fft_in = np.fft.fft(U_in)
        # H = fft_out / fft_in
        H = fft_out / fft_out2
        H_frequency = H[np.argmin(np.abs(f-self.freq))]
        # print(self.freq[np.argmin(np.abs(f-self.freq))], f)
        arg_H = np.angle(H_frequency, deg=True)
        # print(self.fft)
        U_out_power = self.power(f, fft=self.fft)
        U_in_power = self.power(f, fft=fft_in)

        H = self.calc_H(U_out_power, U_in_power)
        # arg_H = self.calc_phase_H(fft_out, fft_in, f)
        return H, arg_H
   
    def create_plots(self, Hs, arg_Hs, freqs, sigma_H=0.01, sigma_arg_H=0.01):
        def H_magnitude(omega, omega_0, Q):
            return 20 * np.log10(np.abs(1/(1 - (omega/omega_0)**2 + 1j * (omega/(omega_0 * Q)))))
       
        initial_guess = [1500, 3] # deze aanpassen met de gevonden waarden van methode 1
        popt, pcov = curve_fit(H_magnitude, freqs, Hs, p0=initial_guess) # is freqs = omega?
       
        perr = np.sqrt(np.diag(pcov))
        a_fit, b_fit = popt
        print(f"Fitted parameters: a = {a_fit:.3f} +/- {perr[0]}, b = {b_fit:.3f} +/- {perr[1]}")
       
       
        plt.plot(freqs, H_magnitude(freqs, *popt), 'r-', label='Fit')
       
        plt.errorbar(
            freqs, 20 * np.log10(Hs), yerr= np.abs(20 / (Hs * np.log(10)) * sigma_H),
            fmt='o',                # circle markers
            ecolor='gray',          # error bar color
            elinewidth=1.2,         # error bar line width
            capsize=4,              # end cap size
            capthick=1.2,           # end cap thickness
            color='#1f77b4',        # main color (blue)
            markersize=6,           # marker size
            markeredgecolor='black',
            markerfacecolor='#1f77b4',
            label="data"
        )
        # plt.loglog()
        plt.xscale("log")
        plt.xlabel(r'$\omega$ (rad s$^{-1}$)')
        plt.title('Magnitude Bode Plot')
        plt.ylabel(r'20 $\log_{10}$|H($\omega$)|(dB)')
        plt.savefig('Bodeplot magnitude H')
        plt.legend()
        plt.show()
       
        def H_arg(omega, omega_0, Q):
            return np.angle(1/(1 - (omega/omega_0)**2 + 1j * (omega/(omega_0 * Q))))
       
        initial_guess = [1500, 3] # deze aanpassen met de gevonden waarden van methode 1
        popt, pcov = curve_fit(H_arg, freqs, Hs, p0=initial_guess) # is freqs = omega?
       
        perr2 = np.sqrt(np.diag(pcov))
        a_fit2, b_fit2 = popt
        print(f"Fitted parameters: a = {a_fit2:.3f} +/- {perr2[0]}, b = {b_fit2:.3f} +/- {perr2[1]}")
       
       
        plt.plot(freqs, H_arg(freqs, *popt), 'r-', label='Fit')
       
       
        plt.errorbar(
            freqs, arg_Hs, yerr=sigma_arg_H,
            fmt='o',                # circle markers
            ecolor='gray',          # error bar color
            elinewidth=1.2,         # error bar line width
            capsize=4,              # end cap size
            capthick=1.2,           # end cap thickness
            color='#1f77b4',        # main color (blue)
            markersize=6,           # marker size
            markeredgecolor='black',
            markerfacecolor='#1f77b4',
            label="data"
        )
        # plt.loglog()
        plt.xscale("log")
        plt.xlabel(r'$\omega$ (rad s$^{-1}$)')
        plt.title('Phase Bode Plot')
        plt.ylabel(r'phase H($\omega$)')
        plt.savefig('Bodeplot magnitude H')
        plt.legend()
        plt.show()
       
        fit_results = [a_fit, b_fit, a_fit2, b_fit2]
        return fit_results
   
    def fits(self, omega, Hs, arg_Hs):
        try:
            print(f"The magnitude is {Hs}")
            print(f"The angle is {arg_Hs}")
            H = Hs * np.exp(1j * (arg_Hs / 180 * np.pi))
            print(f"The transfer function is {H}")
            print(f"Omega is {omega}")
            # print(f"")
            def H_func(omega, omega_0, Q):
                return 1 / (1 - (omega/omega_0)**2 + 1j * (omega/(omega_0 * Q)))
           
            initial_guess = [1/(100e3 * 10e-9), 1] # deze aanpassen met de gevonden waarden van methode 1
            popt, pcov = curve_fit(H_func, omega, H, p0=initial_guess)
           
            a_fit, b_fit = popt
            print(f"Fitted parameters: a = {a_fit:.3f}, b = {b_fit:.3f}")
           
            plt.scatter(np.real(H), np.imag(H))
            plt.plot(omega, H_func(omega, *popt), 'r-', label='Fit')
            plt.xlabel('Re[H]')
            plt.ylabel('Im[H]')
            plt.legend()
            plt.show()
        except:
            plt.scatter(np.real(H), np.imag(H))
            # plt.plot(omega, H_func(omega, *popt), 'r-', label='Fit')
            plt.xlabel('Re[H]')
            plt.ylabel('Im[H]')
            plt.legend()
            plt.title("Polar Plot")
            plt.show()
            return
       
    def fwhm_frequency(self, freq, amp):    
        # Peak
        peak_idx = np.argmax(amp)
        f0 = freq[peak_idx]
        Amax = amp[peak_idx]
   
        # Half-maximum amplitude
        Ah = Amax / np.sqrt(2)      
   
        # Find where amplitude crosses Ah
        # Left side
        left_indices = np.where(amp[:peak_idx] < Ah)[0]
        if len(left_indices) == 0:
            raise ValueError("No left half-maximum crossing found.")
        i1 = left_indices[-1]
       
        # Linear interpolation for more accurate location
        f1 = np.interp(Ah, [amp[i1], amp[i1+1]], [freq[i1], freq[i1+1]])
   
        # Right side
        right_indices = np.where(amp[peak_idx:] < Ah)[0]
        if len(right_indices) == 0:
            raise ValueError("No right half-maximum crossing found.")
        i2 = right_indices[0] + peak_idx
   
        f2 = np.interp(Ah, [amp[i2-1], amp[i2]], [freq[i2-1], freq[i2]])
   
        # FWHM
        fwhm = f2 - f1
   
        return f0, f1, f2, fwhm
       
   
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
        arg_Hs = []
        std_Hs = []
        std_arg_Hs = []
        n_measurements = 3
        for f, A in zip(freqs, Amplitudes):
            print(f"The frequency is {f}")
            H_list, arg_list = [], []
            for _ in range(n_measurements):
                print(f"{_}")
                H, arg_H = self.singlemeasurement(f, A, rate, samples)
               
                H_list.append(H)
                arg_list.append(arg_H)
            avg_H = np.mean(H_list)
            avg_angle = np.mean(arg_list)
            std_H = np.std(H_list)
            std_angle = np.std(arg_list)
            Hs.append(np.abs(avg_H))
            arg_Hs.append(avg_angle)
            std_Hs.append(std_H)
            std_arg_Hs.append(std_angle)
       
        Hs = np.array(Hs)
        arg_Hs = np.array(arg_Hs)
        self.create_plots(Hs, arg_Hs, freqs, std_Hs, std_arg_Hs)
        self.fits(freqs, Hs, arg_Hs)
       
        # fwhm
        f0, f1, f2, fwhm = self.fwhm_frequency(freqs, Hs)

        print("Peak frequency f0 =", f0)
        print("Left half-max f1 =", f1)
        print("Right half-max f2 =", f2)
        print("FWHM =", fwhm)
        print("Q =", f0 / fwhm)
       
        return Hs, arg_Hs, freqs
   
   
       
mydaq = myDAQ()
#%%
rate = 200000
# rate = 20000
samples = rate*2
t = np.linspace(0, samples/rate, samples)
# mydaq.singlemeasurement(30, 5, rate, samples)
# freqs = np.logspace(np.log10(20), np.log10(2000), 10)
n_datapoints = 20
freqs = np.logspace(np.log10(100), np.log10(10000), n_datapoints, dtype=int)

# freqs = [1700]
# print(freqs)
Amplitudes= np.zeros(n_datapoints) + 5
# print(Amplitudes)
mag_H, arg_H, fs = mydaq.multimeasure(freqs, Amplitudes, rate, samples)

#%%
np.savetxt('Datasession3.txt', [mag_H, arg_H, fs])

plt.scatter(fs, mag_H)
plt.loglog()
plt.xlabel(r'$\omega$')
plt.ylabel('H')
plt.savefig('goodplot.png')
plt.show()

#%%
rate = 200000
samples = rate*2
mydaq.singlemeasurement(1100, 5, rate, samples)
freq1, fft1 = mydaq.freq2, np.abs(mydaq.fft2/mydaq.fft)

#%%
mydaq.singlemeasurement(1100, 5, rate, samples)
freq2, fft2 = mydaq.freq2, np.abs(mydaq.fft2/mydaq.fft)

#%%
plt.plot(freq1, fft1, label='Without Apertures', alpha=0.5)
plt.plot(freq2, fft2, label='With Apertures', alpha=0.5)
plt.xlabel('Frequency (Hz)')
plt.xlim(0, 2000)
plt.ylim(0, 200)
plt.ylabel('Intensity')
plt.legend()
plt.show()
#%%
def H_magnitude(omega, omega_0, Q):
    return 20 * np.log10(np.abs(1/(1 - (omega/omega_0)**2 + 1j * (omega/(omega_0 * Q)))))

frequencies = np.logspace(np.log10(2000), np.log10(20000), 100, dtype=int)
H = H_magnitude(frequencies, omega_0=10000, Q=3)

plt.errorbar(
    frequencies, H, yerr= 0,
    fmt='o',                # circle markers
    ecolor='gray',          # error bar color
    elinewidth=1.2,         # error bar line width
    capsize=4,              # end cap size
    capthick=1.2,           # end cap thickness
    color='#1f77b4',        # main color (blue)
    markersize=6,           # marker size
    markeredgecolor='black',
    markerfacecolor='#1f77b4',
    label="data"
)
# plt.loglog()
plt.title("Theoretical bodeplot for the magnitude")
plt.xscale("log")
plt.xlabel(r'$\omega$ (rad s$^{-1}$)')
plt.ylabel(r'20 $\log_{10}$|H($\omega$)|(dB)')
plt.savefig('Bodeplot magnitude H')
plt.legend()
plt.show()


def H_arg(omega, omega_0, Q):
    return np.angle(1/(1 - (omega/omega_0)**2 + 1j * (omega/(omega_0 * Q))))

arg_Hs = H_arg(frequencies, omega_0=10000, Q=3)

plt.errorbar(
    frequencies, arg_Hs, yerr=0,
    fmt='o',                # circle markers
    ecolor='gray',          # error bar color
    elinewidth=1.2,         # error bar line width
    capsize=4,              # end cap size
    capthick=1.2,           # end cap thickness
    color='#1f77b4',        # main color (blue)
    markersize=6,           # marker size
    markeredgecolor='black',
    markerfacecolor='#1f77b4',
    label="data"
)
# plt.loglog()
plt.title("Theoretical bodeplot for the argument")
plt.xscale("log")
plt.xlabel(r'$\omega$ (rad s$^{-1}$)')
plt.ylabel(r'phase H($\omega$)')
plt.savefig('Bodeplot magnitude H')
plt.legend()
plt.show()

#%%
def H_func(omega, omega_0, Q):
    return 1 / (1 - (omega/omega_0)**2 + 1j * (omega/(omega_0 * Q)))


frequencies = np.logspace(np.log10(2000), np.log10(20000), 1000, dtype=int)
H = H_func(frequencies, omega_0 = 10000, Q=10)


plt.scatter(np.real(H), np.imag(H))
# plt.plot(omega, H_func(omega, *popt), 'r-', label='Fit')
plt.xlabel('Re[H]')
plt.ylabel('Im[H]')
plt.legend()
plt.show()
#%%
print(len(mag_H), len(fs))
Hs = mag_H
freqs = fs

def H_magnitude(omega, omega_0, Q):
    return 20 * np.log10(np.abs(1/(1 - (omega/omega_0)**2 + 1j * (omega/(omega_0 * Q)))))

initial_guess = [1703, 2.5] # deze aanpassen met de gevonden waarden van methode 1
popt, pcov = curve_fit(H_magnitude, freqs, Hs, p0=initial_guess) # is freqs = omega?

perr = np.sqrt(np.diag(pcov))
a_fit, b_fit = popt
print(f"Fitted parameters: a = {a_fit:.3f} +/- {perr[0]}, b = {b_fit:.3f} +/- {perr[1]}")


plt.plot(freqs, H_magnitude(freqs, *popt), 'r-', label='Fit')

plt.scatter(freqs, 20 * np.log10(Hs), label='Data')
plt.plot(freqs, H_magnitude(freqs, 1700, 5), label='FWHM approx.')
# plt.loglog()
plt.xscale("log")
plt.xlabel(r'$\omega$ (rad s$^{-1}$)')
plt.title('Magnitude Bode Plot')
plt.ylabel(r'20 $\log_{10}$|H($\omega$)|(dB)')
plt.savefig('Bodeplot magnitude H')
plt.legend()
plt.show()

def H_arg(omega, omega_0, Q):
    return np.angle(1/(1 - (omega/omega_0)**2 + 1j * (omega/(omega_0 * Q))))

initial_guess = [1500, 3] # deze aanpassen met de gevonden waarden van methode 1
popt, pcov = curve_fit(H_arg, freqs, Hs, p0=initial_guess) # is freqs = omega?

perr2 = np.sqrt(np.diag(pcov))
a_fit2, b_fit2 = popt
print(f"Fitted parameters: a = {a_fit2:.3f} +/- {perr2[0]}, b = {b_fit2:.3f} +/- {perr2[1]}")


plt.plot(freqs, H_arg(freqs, *popt), 'r-', label='Fit')


plt.errorbar(freqs, arg_Hs, label="data")
# plt.loglog()
plt.xscale("log")
plt.xlabel(r'$\omega$ (rad s$^{-1}$)')
plt.title('Phase Bode Plot')
plt.ylabel(r'phase H($\omega$)')
plt.savefig('Bodeplot magnitude H')
plt.legend()
plt.show()