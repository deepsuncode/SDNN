Changes made:

Drop the .decode('utf8') part

Error solved:
AttributeError: 'str' object has no attribute 'decode'

Warning:
Be sure to remove readme and any other files (e.g., .DS_Store) from the inputs folder before running

Ran with: 

python 3.6
tensorflow 1.11.0
keras 2.2.4
astropy 4.1
matplotlib 3.3.4