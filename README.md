# PD120 decoder
This is a fork from [Martinber's rtlsdr_sstv project](https://github.com/martinber/rtlsdr_sstv). Most of the code and the most important decoding stuff was taken from Martin's project.

### Usage
```bash
cd pd120_decoder/
source venv/bin/activate
pip install -r requirements.txt
./demod.py <path to wav file>
```

This software is intended to be used (and integrated) with [raspberry-noaa](https://github.com/reynico/raspberry-noaa).
