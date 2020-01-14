import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import optimize
from scipy import fftpack

class LineEmission():
    """Helper functions for plotting and analyzing Hydrogen/Deuterium Emission spectra.
    
    """
    def __init__(self, filename):
        #load the data
        data = np.genfromtxt(filename, skip_header = 1).T
        self.absorb = data[1]
        self.freq = data[0]
        
        #find the peaks
        self.peaks, self.peakprop = signal.find_peaks(self.absorb, height = 0.01, distance = 200)
        self.peakfreq = self.freq[self.peaks]
        self.peakheight = self.peakprop["peak_heights"]
        
    def plot_spectrum(self, label = True):
        """Plots the entire Hydrogen/Deuterium emission spectrum.
        
        Optionally labels peaks based on their balmer series transitions.
        
        """
        
        #plot the spectrum
        plt.plot(self.freq, self.absorb)
        if label == True:
            #label teh peaks with their respective transistions
            for i in range(0, len(self.peakfreq)):
                plt.annotate("n = "+str(9-i), (self.peakfreq[i], self.peakheight[i]), horizontalalignment = "left", 
                     verticalalignment = "bottom")
        #label axis
        plt.xlabel("$\lambda$ (Angstroms)" )
        plt.ylabel("Relative Intensity")
        
        
    def plot_peak(self, number):
        """Plots individual peaks based on the number provided, from right to left.
        
        This uses a fourier smoothing technique to accurately detect peaks in relatively noisy data.
        This generally happens with peaks that have a smaller population associated with them.
        
        """
        
        #the first peak may not have 100 points to the left of it
        if self.peaks[number-1] < 100:
            x = self.freq[:2*self.peaks[number-1]]
            y = self.absorb[:2*self.peaks[number-1]]
        else:
            x = self.freq[self.peaks[number-1]-100:self.peaks[number-1]+100]
            y = self.absorb[self.peaks[number-1]-100:self.peaks[number-1]+100]
        
        #plot the peak
        plt.plot(x, y, "b.")
        
        #perform fourier smoothing to eliminate noise
        dct = fftpack.dct(y, norm = "ortho")
        dct[20:] = 0
        smoothed = fftpack.idct(dct, norm = "ortho")
        
        #plot fourier smoothed data
        plt.plot(x, smoothed, "r-")
        
        #I found that this was a good way to automatically find the peak prominances
        prom = 10**(number - 6)
        peak, properties = signal.find_peaks(smoothed, height = prom)
        
        #label the axis
        plt.xlabel("$\lambda$ (Angstroms)")
        plt.ylabel("Relative Intensity")

        #label the peak with a vertical line
        plt.axvline(x[peak[0]])
        plt.axvline(x[peak[1]])

        #annotate the peaks with wavelength, and the element.
        heights = properties.get("peak_heights")
        
        plt.annotate(str(x[peak[0]])+" $^2$H ", 
                     (x[peak[0]], smoothed[peak[0]]+0.005), 
                     horizontalalignment = "right", 
                     verticalalignment = "bottom")
        plt.annotate("   "+str(x[peak[1]]) + " $^1$H", 
                 (x[peak[1]], smoothed[peak[1]]), 
                 horizontalalignment = "left", 
                 verticalalignment = "top")
        
class Radiation():
    """Helper functions for plotting and analyzing different blackbody spectra.
    """

    def __init__(self, filename):
        #declare some constants.
        #wien constant
        self.b = 2.89e6
        #planks constant
        self.h = 6.626e-34
        #speed of light
        self.c = 2.998e8
        #boltzmann constant
        self.k = 1.381e-23
        
        self.readFileInfo(filename)
        self.wiensLaw()
        #self.wienTemp()
        
        
    def readFileInfo(self, filename):
            #read the info from the file
            data = np.genfromtxt(filename, delimiter = ",", skip_header = 1)
            data = data.T
            self.absorb = data[1]
            self.wave = data[0]
        
    def wiensLaw(self):
            #finds the wien temperature based on peak absorption wavelength
                    
            peaks, _ = signal.find_peaks(self.absorb, height = max(self.absorb)-min(self.absorb), distance = 100)
            mWave = self.wave[peaks[0]]
            Twien = self.b / mWave
            self.wien_temp = Twien
        
    def blackbody(self):
        
        def scaleFact(wave, absorb, Twien):
            #provide some initial guess for a scaling factor
            h = 6.626e-34
            c = 2.998e8
            k = 1.381e-23
            wave = wave*1e-9
            theory = (8 * np.pi * self.h * self.c) / (wave**5*(np.exp((self.h*self.c)/(wave*self.k*Twien))-1))
            scale = max(self.absorb)/max(theory)
            return scale
        
        def f(x, T, a):
            wave = x*1e-9
            h = 6.626e-34
            c = 2.998e8
            k = 1.381e-23
            return (a * 8 * np.pi * self.h * self.c) / (wave**5*(np.exp((self.h*self.c)/(wave*self.k*T))-1))
        
        
        scale = scaleFact(self.wave, self.absorb, self.wien_temp)
        #fit the a curve to the raw data
        popt, _ = optimize.curve_fit(f, self.wave, self.absorb, p0 = (self.wien_temp, scale))
        T = popt[0]
        a = popt[1]
        theory = f(self.wave, T, a)
        plt.plot(self.wave, theory/max(theory), label = "Theory, T = " + str(round(T, 0)))
        plt.plot(self.wave, self.absorb/max(self.absorb), label = "Experiment")
        plt.ylabel("Intensity")
        plt.xlabel("Wavelength (nm)")
        plt.legend()
        
    def wienTemp(self):
        T = self.wien_temp
        print(T)