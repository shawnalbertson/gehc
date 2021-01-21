Last updated 11 December 2020 by Shawn Albertson

## Overview
This project contains files used to run a vibration testing setup developed in the fall semester 2020, as part of a capstone engineering project at Olin College in collaboration with GE Healthcare. 

The files contained in this directory were used to 
- communicate with an Arduino over UART, sending commands which included the collection of data and operation of a motor
- load and process data from both an accelerometer and a microphone

## Arduino interface
The following modules depend on communication with an Arduino which had been previously programmed. The code code for the arduino can be found at `../vibration_test/vibration_test.ino`.  

## Description of modules
`record.py` - This module is used to communicate with an Arduino which has been programmed based on the file mentioned above. Many of the functions are called in other modules which are specifically used to save and process the data which results from this communication.

`save.py` - This module is used to save data directly from the accelerometer. It calls functions from `record.py` . To use `save.py`, open the file in a text editor and uncomment one of the lines at the end. A directory should already exist which matches the file destination specified by `'prefix'`. 

`analyze.py` - This module contains a host of analysis tools for parsing data from the accelerometer. A large portion of it is dedicated to calculating RMS acceleration in specified frequency bins, including a host of functions catered to third octave analysis. Finally, the results of the analysis can be plotted with specific plotting functions. No functions are called directly within the script. 

`test.py` - This module calls functions from both `analyze.py` and `save.py`. It is used to quickly take a sample from the accelerometer through the Arduino, and plot it to visualy compare third octave RMS values with the values recorded by GE. Before running, the file destination in the `write_file` function should be changed depending on the test being conducted. Similarly, the `power` and `freq` values should be changed to match the new file destination so that the right file is subsequently visualized.

`microphone_test.py` - This module loads data recorded from microphone on an Analog Discovery device. To run it, uncomment the sections at the bottom of the function definitions as specified within the script. 
