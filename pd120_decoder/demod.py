#!/usr/bin/env python3

from scipy.io import wavfile
import numpy as np
import scipy.signal

import matplotlib.pyplot as plt
import pylab
import math
from utils import escribir_pixel, filtrar
from PIL import Image
import sys

''' constantes '''
PORCH_TIME = 0.00208
SYNC_TIME = 0.02
LINE_COMP_TIME = 0.1216

# fs, data = wavfile.read('./iss.wav')
fs, data = wavfile.read(sys.argv[1])

img = Image.new('YCbCr', (640, 496), "white")
#t = np.arange(len(data))/fs


def crear_hilbert(atten, delta):
    ''' crea el filtro de hilbert enventanado por kaiser '''

    if atten < 21:
        beta = 0
    elif atten > 21 and atten < 50:
        beta = 0.5842*(atten-21)**(2/5) + 0.07886*(atten-21)
    else:
        beta = 0.1102*(atten-8.7)
    m = 2*((atten-8)/(4.57*delta))
    if int(m) % 2 == 0:
        m = int(m+1)
    else:
        m = int(m+2)
    window = np.kaiser(m, beta)
    filter = []
    for n in range((-m+1)//2, (m-1)//2+1):
        if n % 2 != 0:
            filter.append(2/(np.pi*n))
        else:
            filter.append(0)
    hilbert = filter * window

    return(hilbert)


def crear_analitica(datos, filtro):
    '''devuelve la senal analitica de la forma x + y*j'''

    zeros = np.zeros((len(filtro)-1)//2)
    datareal = np.concatenate([zeros, datos, zeros])
    datacompleja = np.convolve(datos, filtro)*1j
    senal = datareal + datacompleja

    return(senal)


def boundary(value):
    '''checkea los valores maximos y minimos de la frecuencia instantanea'''

    value = min(valor, 2300)
    value = max(1500, valor)

    return(value)


def hpf(data, fs):
    firw = scipy.signal.firwin(101, cutoff=800, fs=fs, pass_zero=False)
    return scipy.signal.lfilter(firw, [1.0], data)


signal = crear_analitica(hpf(data, fs), crear_hilbert(
    40, np.pi/1200))  # frecuencia normalizada

# xunwrap deja a la fase de forma lineal en vez de rampa
inst_ph = np.unwrap(np.angle(signal))
# diff toma el valor de x(n+1) y lo resta con x(n)
inst_fr = (np.diff(inst_ph) / (2.0*np.pi) * fs)

# toma senal, frec corte, banda de trans, y caida en dB
inst_fr = list(filtrar(inst_fr, 0.2, 0.2, 40))

muestras = 0
cont_linea = -1
i = 0

while i < len(inst_fr):

    if 900 <= inst_fr[i] <= 1300:
        muestras += 1  # contador de muestras, si muchas se encuentran en el rango era cambio de linea

    if muestras > int((SYNC_TIME-0.002)*fs):
        cont_linea += 2  # casi seguro indico la siguiente
        muestras = 0  # resetear muestras para la proxima iteracion
        # encajar i para comenzar justo en luminancia
        i = i-int((SYNC_TIME-0.002)*fs)+int((SYNC_TIME+PORCH_TIME)*fs)
        desfase = 1200 - \
            np.mean(inst_fr[i-int((SYNC_TIME+PORCH_TIME)*fs): i-int(PORCH_TIME*fs)])

        try:
            y_resampleados = scipy.signal.resample(
                inst_fr[i:i+int(LINE_COMP_TIME*fs)], 640)
            for columna, valor in enumerate(y_resampleados):
                escribir_pixel(img, columna, cont_linea,
                               "lum", boundary(valor+desfase))

            cr_resampleados = scipy.signal.resample(
                inst_fr[i+int(LINE_COMP_TIME*fs):i+int(LINE_COMP_TIME*2*fs)], 640)
            for columna, valor in enumerate(cr_resampleados):
                escribir_pixel(img, columna, cont_linea,
                               "cr", boundary(valor+desfase))

            cb_resampleados = scipy.signal.resample(
                inst_fr[i+int(LINE_COMP_TIME*2*fs):i+int(LINE_COMP_TIME*3*fs)], 640)
            for columna, valor in enumerate(cb_resampleados):
                escribir_pixel(img, columna, cont_linea,
                               "cb", boundary(valor+desfase))

            ny_resampleados = scipy.signal.resample(
                inst_fr[i+int(LINE_COMP_TIME*3*fs):i+int(LINE_COMP_TIME*4*fs)], 640)
            for columna, valor in enumerate(ny_resampleados):
                escribir_pixel(img, columna, cont_linea,
                               "nxt_lum", boundary(valor+desfase))
        except:
            break

        i += int(LINE_COMP_TIME*2*fs)

    i += 1

imgrgb = img.convert("RGB")
imgrgb.save("%s.png" % sys.argv[1], "PNG")
