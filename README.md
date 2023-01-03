# Inferring Line-of-sight Velocities and Doppler Widths from Stokes Profiles of GST/NIRIS Using Stacked Deep Neural Networks
Haodi Jiang, Qin Li, Yan Xu, Wynne Hsu, Kwangsu Ahn, Wenda Cao, Jason T. L. Wang, Haimin Wang
Institute for Space Weather Sciences, New Jersey Institute of Technology

Institute of Data Science, National University of Singapore

Obtaining high-quality magnetic and velocity fields through Stokes inversion is crucial in solar physics. 
In this paper, we present a new deep learning method, named Stacked Deep Neural Networks (SDNN), 
for inferring line-of-sight (LOS) velocities and Doppler widths 
from Stokes profiles collected by the Near InfraRed Imaging Spectropolarimeter (NIRIS) 
on the 1.6 m Goode Solar Telescope (GST) at the Big Bear Solar Observatory (BBSO). 
The training data for SDNN are prepared by a Milne-Eddington (ME) inversion code used by BBSO. We quantitatively assess SDNN, 
comparing its inversion results with those obtained by the ME inversion code and related machine-learning (ML) algorithms 
such as multiple support vector regression, multilayer perceptrons, and a pixel-level convolutional neural network. 
Major findings from our experimental study are summarized as follows. 
First, the SDNN-inferred LOS velocities are highly correlated to the ME-calculated ones with the Pearson product-moment correlation coefficient 
being close to 0.9 on average. Second, SDNN is faster, while producing smoother and cleaner LOS velocity and Doppler width maps, 
than the ME inversion code. 
Third, the maps produced by SDNN are closer to ME's maps than those from the related ML algorithms, 
demonstrating that the learning capability of SDNN is better than those of the ML algorithms. 
Finally, a comparison between the inversion results of ME and SDNN based on GST/NIRIS and 
those from the Helioseismic and Magnetic Imager on board the Solar Dynamics Observatory 
in flare-prolific active region NOAA 12673 is presented. We also discuss extensions of SDNN 
for inferring vector magnetic fields with empirical evaluation.

References:

Inferring Line-of-sight Velocities and Doppler Widths from Stokes Profiles of GST/NIRIS Using Stacked Deep Neural Networks. 
Haodi Jiang, Qin Li, Yan Xu, Wynne Hsu, Kwangsu Ahn, Wenda Cao, Jason T. L. Wang, Haimin Wang, 
The Astrophysical Journal, Volume 939, Issue 2, id.66, 12 pp., November 2022.

https://iopscience.iop.org/article/10.3847/1538-4357/ac927e

https://arxiv.org/abs/2210.04122
