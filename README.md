# PD120 decoder
This is a fork from [Martinber's rtlsdr_sstv project](https://github.com/martinber/rtlsdr_sstv). Most of the code and the most important decoding stuff was taken from Martin's project.

### Usage
```bash
cd pd120_decoder/
source venv/bin/activate
pip install -r requirements.txt
./demod.py <path to wav file>
```

### Useful information
This software is intended to be used (and integrated) with [raspberry-noaa](https://github.com/reynico/raspberry-noaa).

Some of the minor modifications I done to Martin's work are about decoding existing ISS recordings that I've done with [raspberry-noaa](https://github.com/reynico/raspberry-noaa). Those modifications helps the software recognize multiple SSTV transmissions on a single audio file and also adjusts the DC offset caused by doppler effect drift on frequency.

Decoding times for a 10 minute recording with two or three SSTV transmissions is around 2 minutes on my MacBook Pro 2019 and around 8 minutes on a Raspberry PI 4.
