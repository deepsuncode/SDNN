Changes made:

Add imports:

from keras import backend as K
import numpy as np

Line 50:
tf.compat.v1.disable_eager_execution() 

Fixes issue with lambda layer/expand_dims type error (line 99)


Warning:

Be sure to remove readme and any other files (e.g., .DS_Store) from the inputs folder before running


Ran with: 

tensorflow 2.9.0
keras 2.9.0
numpy 1.23.2
matplotlib 3.6.2
astropy 5.1