import collections
import math
import numpy as np

def mapeadora(value):
    valor_mapeado = int((value-1500)/800*255)
    return valor_mapeado

def escribir_pixel(img, columna, linea, canal, valor):
    '''funcion encargada de escribir pixel por pixel la imagen'''

    if linea >= img.height:
        return
    if canal == "lum":
        prev = img.getpixel((columna,linea-1))
        datapixel = (mapeadora(valor), prev[1], prev[2])
        img.putpixel((columna,linea-1), datapixel)
    if canal == "cr":
        prev = img.getpixel((columna,linea-1))
        nxt_prev = img.getpixel((columna,linea))
        datapixel = (prev[0], prev[1], mapeadora(valor))
        nxt_datapixel = (nxt_prev[0], nxt_prev[1], mapeadora(valor))
        img.putpixel((columna,linea-1), datapixel)
        img.putpixel((columna,linea), nxt_datapixel)
    if canal == "cb":
        prev = img.getpixel((columna,linea-1))
        nxt_prev = img.getpixel((columna,linea))
        datapixel = (prev[0], mapeadora(valor), prev[2])
        nxt_datapixel = (nxt_prev[0], mapeadora(valor), nxt_prev[2])
        img.putpixel((columna,linea-1), datapixel)
        img.putpixel((columna,linea), nxt_datapixel)
    if canal == "nxt_lum":
        prev = img.getpixel((columna,linea))
        datapixel = (mapeadora(valor), prev[1], prev[2])
        img.putpixel((columna,linea), datapixel)


def lowpass(cutout, delta_w, atten):
    '''
    cutout y delta_w en fracciones de pi radianes por segundo.

    atten en decibeles positivos.
    '''

    beta = 0
    if atten > 50:
        beta = 0.1102 * (atten - 8.7)
    elif atten < 21:
        beta = 0
    else:
        beta = 0.5842 * (atten - 21)**0.4 + 0.07886 * (atten - 21)

    length = math.ceil((atten - 8) / (2.285 * delta_w * math.pi)) + 1;
    if length % 2 == 0:
        length += 1

    coeffs = np.kaiser(length, beta)

    # i es el indice en el vector, n es el eje con el cero centrado en el medio
    # del filtro
    for i, n in enumerate(range(
            int(-(length - 1) / 2),
            int((length - 1) / 2)+1)):

        if n == 0:
            coeffs[i] *= cutout
        else:
            coeffs[i] *= math.sin(n * math.pi * cutout) / (n * math.pi)

    return coeffs

def filtrar(input, cutout, delta_w, atten):
    '''
    La derecha del buff tiene la muestra mas reciente y tiene el indice mas alto
    '''

    coeffs = lowpass(cutout, delta_w, atten)

    #  plot(coeffs, numpy.abs(numpy.fft.fft(coeffs)))

    buf = collections.deque([0] * len(coeffs))

    for s in input:
        buf.popleft()
        buf.append(s)
        sum = 0
        for j in range(len(coeffs)):
            sum += buf[-j - 1] * coeffs[j]

        yield sum
