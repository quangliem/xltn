import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
import librosa
from librosa import display, stft
from slider_demo import SnaptoCursor


def main():
    # y_or, sr = librosa.load("khoosoothunhus.wav", duration=10)
    y_or, sr = librosa.load("Xe.wav", duration=10)
    # print(y_or)
    fig1 = plt.figure()
    fig1.suptitle('mouse hover over figure or axes to trigger events')
    ax1 = fig1.add_subplot(311)
    ax1.set_title("do thi am")
    display.waveplot(y_or, sr=sr)
    ax2 = fig1.add_subplot(312)
    ax2.set_title('tu tuong quan')
    fig2 = plt.figure()
    ax3 = fig2.add_subplot(211)
    ax3.set_title("bien thien f0 tu_tuong_quan")
    ax4= fig1.add_subplot(313)
    ax5 =fig2.add_subplot(212)
    ax5.set_title("bien thien f0 amdf")
    print("sample rate: ", sr)
    ax_lpc_fft = fig1.add_subplot(414)
    cursor = SnaptoCursor(ax1, ax2, ax3, ax4, ax5, ax_lpc_fft, y_or, sr)
    # plt.connect('motion_notify_event', cursor.mouse_move)
    fig1.canvas.mpl_connect('button_press_event', cursor.mouse_click)
    # plt.connect('button_press_event', cursor.mouse_click)
    plt.show()
if __name__ == '__main__':
    main()


