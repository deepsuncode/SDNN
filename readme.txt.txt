To get the pre-trained model, named pretrained_model.h5, go to https://web.njit.edu/~wangj/deepsuncode/SDNN/

Requirements:

python=2.7
tensorflow=1.11.0
keras=2.2.4
astropy
matplotlib

Usage:

The source code package contains the following folders/directories.

The “inputs” folder contains NIRIS Stokes profiles samples from BBSO/GST.
The “outputs” folder contains inverted results: bx, by, bz, Doppler width and LOS velocity.

Run

python SDNN.py

Notes: The code will also save b_total, inclination and azimuth if save_mag_field_o = True; 
otherwise the code will only save bx, by, bz, Doppler width and LOS velocity.