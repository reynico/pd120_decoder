#!/usr/bin/env python3
from scipy.io import wavfile
import numpy as np
import scipy.signal
from utils import write_px, filt
from PIL import Image
import os
import argparse

''' constants '''
PORCH_TIME = 0.00208
SYNC_TIME = 0.02
LINE_COMP_TIME = 0.1216


def create_hilbert(atten, delta):
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
    return hilbert


def create_analytica(datos, filtro):
    zeros = np.zeros((len(filtro)-1)//2)
    realdata = np.concatenate([zeros, datos, zeros])
    complexdata = np.convolve(datos, filtro)*1j
    return realdata + complexdata


def boundary(val):
    value = min(val, 2300)
    value = max(1500, val)
    return value


def hpf(data, fs):
    firw = scipy.signal.firwin(201, cutoff=1000, fs=fs, pass_zero=False)
    return scipy.signal.lfilter(firw, [1.0], data)


def decode(start, samples_list, fs, output_path, img_filename):
    img = Image.new('YCbCr', (640, 496), "white")
    samples = 0
    cont_line = -1
    i = 0
    while i < len(samples_list):
        if 900 <= samples_list[i] <= 1300:
            samples += 1
        if samples > int((SYNC_TIME-0.002)*fs):
            cont_line += 2
            samples = 0
            i = i-int((SYNC_TIME-0.002)*fs)+int((SYNC_TIME+PORCH_TIME)*fs)
            gap = 1200 - np.mean(samples_list[i-int((SYNC_TIME+PORCH_TIME)*fs): i-int(PORCH_TIME*fs)])

            try:
                y_resampled = scipy.signal.resample(
                    samples_list[i:i+int(LINE_COMP_TIME*fs)], 640)
                for col, val in enumerate(y_resampled):
                    write_px(img, col, cont_line, "lum", boundary(val+gap))

                cr_resampled = scipy.signal.resample(
                    samples_list[i+int(LINE_COMP_TIME*fs):i+int(LINE_COMP_TIME*2*fs)], 640)
                for col, val in enumerate(cr_resampled):
                    write_px(img, col, cont_line, "cr", boundary(val+gap))

                cb_resampled = scipy.signal.resample(
                    samples_list[i+int(LINE_COMP_TIME*2*fs):i+int(LINE_COMP_TIME*3*fs)], 640)
                for col, val in enumerate(cb_resampled):
                    write_px(img, col, cont_line, "cb", boundary(val+gap))

                ny_resampled = scipy.signal.resample(
                    samples_list[i+int(LINE_COMP_TIME*3*fs):i+int(LINE_COMP_TIME*4*fs)], 640)
                for col, val in enumerate(ny_resampled):
                    write_px(img, col, cont_line, "nxt_lum", boundary(val+gap))
            except Exception:
                break
            i += int(LINE_COMP_TIME*2*fs)
        i += 1
    imgrgb = img.convert("RGB")
    imgrgb.save(f"{output_path}/{img_filename}-{start}.png", "PNG")
    return img


def process_audio(audio_file, output_folder, threshold=700):
    fs, data = wavfile.read(audio_file)
    img_filename = os.path.splitext(os.path.basename(audio_file))[0]
    carrier_found = False

    # Carrier detection parameters
    freq = 2270
    items = 700
    pass_len = 1425000
    wait_time = pass_len
    header_end = []
    idx = 0

    signal = create_analytica(hpf(data, fs), create_hilbert(40, np.pi/1200))
    inst_ph = np.unwrap(np.angle(signal))
    inst_fr = (np.diff(inst_ph) / (2.0 * np.pi) * fs)
    inst_fr = scipy.ndimage.gaussian_filter1d(inst_fr, sigma=1.0)
    inst_fr = scipy.signal.medfilt(inst_fr, kernel_size=5)
    inst_fr = list(filt(inst_fr, 0.2, 0.2, 40))

    while idx < len(inst_fr):
        if all((freq - threshold) < x < (freq + threshold) for x in inst_fr[idx:idx+items:1]):
            header_end.append(idx)
            print(f"found carrier at sample {idx}")
            idx += wait_time
            carrier_found = True
        else:
            idx += 1

    results = []
    for idx, padding in enumerate(header_end):
        print(f"start: {padding} \t end: {padding+pass_len}")
        img = decode(padding, inst_fr[padding:padding+pass_len], fs, output_folder, img_filename)
        results.append(img)

    if not carrier_found:
        print(f"no carrier found in {audio_file}, try adjusting the detection threshold.")
    return results


def main():
    parser = argparse.ArgumentParser(description="Decode SSTV image from audio file")
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Path to the audio file (.wav)")
    parser.add_argument("-o", "--output_folder", type=str, required=True,
                        help="Folder to save the decoded image")
    parser.add_argument("-t", "--threshold", type=int, default=700,
                        help="Threshold for carrier detection (default: 700)")
    args = parser.parse_args()

    audio_file = args.input
    output_folder = args.output_folder
    threshold = args.threshold

    process_audio(audio_file, output_folder, threshold)


if __name__ == "__main__":
    main()
