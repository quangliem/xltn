from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import rfft
from numpy import argmax, mean, diff, log
from matplotlib.mlab import find
from scipy.signal import fftconvolve
from numpy import polyfit
from autocorrelation import compute_arcf_list, compute_arcf
from average_magnitude_difference import compute_dk_list
import librosa
from scikits.talkbox.linpred.levinson_lpc import lpc
from scipy import signal

class SnaptoCursor(object):
    """
    Like Cursor but the crosshair snaps to the nearest x,y point
    For simplicity, I'm assuming x is sorted
    """

    def __init__(self, ax1, ax2, ax3, ax4, ax5, ax_lpc_fft, y_or, sr, window_len = 0.02):
        self.ax1 = ax1
        self.ax2 = ax2
        self.ax3 = ax3
        self.ax4 = ax4
        self.ax5 = ax5
        self.ax_lpc_fft = ax_lpc_fft
        self.ly1 = ax1.axvline(color='k')  # the vert line
        self.ly2 = ax1.axvline(color='k')  # the vert line
        self.ax2ly3 = ax2.axvline(color='k')  # the vert line
        self.ax4_ly1 = ax4.axvline(color='k')
        self.ax4_ly2 = ax4.axvline(color='k')
        self.x = 0
        self.y = 0
        self.y_or = y_or
        self.sr = sr
        # text location in axes coords
        self.txt = ax1.text(0.7, 0.9, '', transform=ax1.transAxes)
        self.window_len = window_len
        self.bien_thien_f0 = []
        self.draw_f0_tu_tuong_quan(y_or=self.y_or, sr=self.sr, window_len=self.window_len)
        self.draw_f0_amdf(y_or=self.y_or, sr=self.sr, window_len=self.window_len)

    def mouse_click(self, event):

        if not event.inaxes:
            return
        x, y = event.xdata, event.ydata

        self.ly1.set_xdata(x)
        self.ly2.set_xdata(x+self.window_len)
        self.x = x
        self.y = y

        ###############draw ax2##########################
        self.ax2.clear()
        self.ax2.set_title('tu tuong quan')
        y_windows = self.y_or[int(x * self.sr):int((x + self.window_len) * self.sr)]
        R_list = compute_arcf_list(y_windows, start=0, stop=301)
        #####################################################################
        p = 14
        A = [1.0]
        B=[1.0, 0.95]
        signal_emph = signal.lfilter(B, A, y_windows, axis=-1, zi=None)
        a, e, k = lpc(signal_emph, p, -1)
        # a = np.append(a, np.zeros(self.nfft - p))
        a = np.concatenate((a, np.zeros(512-len(a))))
        A_fft = np.fft.fft(a)
        afft_log = -10*np.log(np.abs(A_fft[:len(A_fft)/2]))
        self.ax_lpc_fft.clear()
        self.ax_lpc_fft.plot(range(len(afft_log)), afft_log)
        # print("lpca: ", a)
        # print("lpce: ", e)
        # print("lpck: ", k)
        ###########################################################################
        # Find the first low point
        d = diff(R_list)
        start = find(d > 0)[0]

        # Find the next peak after the low point (other than 0 lag).  This bit is
        # not reliable for long signals, due to the desired peak occurring between
        # samples, and other peaks appearing higher.
        # Should use a weighting function to de-emphasize the peaks at longer lags.
        # Also could zero-pad before doing circular autocorrelation.
        max_arcf_loc = argmax(R_list[start:]) + start
        self.ax2.plot(range(len(R_list)), R_list)
        R_second_list = R_list[(max_arcf_loc+1):]
        d = diff(R_second_list)
        start = find(d > 0)[0]
        second_arcf_loc = argmax(R_second_list[start:]) + start
        print("f0_auto_corelation: ", float(self.sr)/(second_arcf_loc+1))
        self.ax2.axvline(max_arcf_loc, color='y')
        self.ax2.axvline(max_arcf_loc+second_arcf_loc+1, color='r')
        ###############draw ax4 tinh tu tuong quan bang amdf#############
        self.ax4.clear()
        dk_list = compute_dk_list(y_or=y_windows)
        d = diff(dk_list)
        start = find(d > 0)[0]
        min_index1 = np.argmin(dk_list[start:]) + start
        dk_second_list = dk_list[(min_index1+1):]
        d = diff(dk_second_list)
        start = find(d > 0)[0]
        min_index2 = np.argmin(dk_second_list[start:]) + start
        self.ax4.axvline(min_index1, color='y')
        self.ax4.axvline(min_index2+min_index1+1, color='r')
        self.ax4.plot(range(len(dk_list)), dk_list)
        print("start:", start, "minindex1:", dk_list[min_index2])
        plt.draw()

    def draw_f0_tu_tuong_quan(self, y_or, sr, window_len):
        duration = float(len(self.y_or)) / sr
        print("duration:", duration)
        print("yor", y_or)
        x=window_len
        time_ptr = []
        f0_list = []
        print("start draw ...")
        while x<duration-window_len*1.01:
            # print("x: ", x)
            try:
                y_windows = y_or[int((x - window_len/2) *sr):int((x + window_len/2) * sr)]
                z = librosa.zero_crossings(y_windows)

                # print("zcr:", len(np.nonzero(z)[0]))
                N = len(y_windows)
                Xk = np.fft.fft(y_windows)
                E_fft = np.sum(np.abs(Xk) ** 2) / N
                # print("Efft: ",E_fft)
                if(len(np.nonzero(z)[0])>110) or (E_fft<5):

                    f0_list.append(0)
                    time_ptr.append(x)
                    x += window_len / 2
                    continue
                    # return 0
                R_list = compute_arcf_list(y_windows, start=0, stop=301)
                d = diff(R_list)
                start = find(d > 0)[0]
                max_arcf_loc = argmax(R_list[start:]) + start
                R_second_list = R_list[(max_arcf_loc + 1):]
                d = diff(R_second_list)
                start = find(d > 0)[0]
                second_arcf_loc = argmax(R_second_list[start:]) + start
                # print(float(self.sr) / (second_arcf_loc))
                f0_list.append(float(self.sr) / (second_arcf_loc+1))
                time_ptr.append(x)
                x += window_len / 2
            except Exception as e:
                print("loi ham draw_f0_tu_tuong_quan::: ", e)
                time_ptr.append(x)
                f0_list.append(0)
                x += window_len / 2
        average = np.average(f0_list)
        print(average, "; ", np.median(f0_list))
        self.ax3.set_ylim([0, 400])
        self.ax3.scatter(time_ptr, f0_list, s=2)
        pass
    def draw_f0_amdf(self, y_or, sr, window_len):
        duration = float(len(y_or)) / sr
        # print("duration:", duration)
        x=window_len
        time_ptr = []
        f0_list = []
        while x<duration-window_len*1.01:
            try:
                y_windows = y_or[int((x - window_len/2) *sr):int((x + window_len/2) * sr)]
                z = librosa.zero_crossings(y_windows)
                N = len(y_windows)
                Xk = np.fft.fft(y_windows)
                E_fft = np.sum(np.abs(Xk) ** 2) / N
                if (len(np.nonzero(z)[0]) > 110) or (E_fft < 5):
                    f0_list.append(0)
                    time_ptr.append(x)
                    x += window_len / 2
                    continue
                dk_list = compute_dk_list(y_or=y_windows)
                d = diff(dk_list)
                start = find(d > 0)[0]
                min_index1 = np.argmin(dk_list[start:]) + start
                dk_second_list = dk_list[(min_index1 + 5):]
                d = diff(dk_second_list)
                start = find(d > 0)[0]
                min_index2 = np.argmin(dk_second_list[start:]) + start
                f0_list.append(float(self.sr) / (min_index2+1))
            except Exception as e:
                print("loi draw_f0_amdf: ", e)
                f0_list.append(0)
            time_ptr.append(x)
            x += window_len / 2
        average = np.average(f0_list)
        print(average, "; ", np.median(f0_list))
        self.ax5.set_ylim([0, 400])
        self.ax5.scatter(time_ptr, f0_list, s=2)
        pass
