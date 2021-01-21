# Data collection libraries
import serial
import time
import numpy as np

# File writing libraries
import csv
from pathlib import Path
import os
global home

# Change depending on your computer / Arduino connection
home = 'C:/users/salbertson/Documents/+Current/SCOPE/vibration_test'
ser = serial.Serial('COM6', baudrate = 115200, timeout = 10)


def sample_rate(length = 500):
    """
    Determine sample rate. Useful to quickly determine sample rate directly from setup.
    In:
        length - how many samples to take
    Out:
        sample rate
    """
# Get data
    data = retrieve(length, motor = False)

# Calculate sample rate
    fs = len(data) * 1000 / (data[-1][0] - data[0][0])
    print(fs)
    return fs


def retrieve(length, motor = True):
    """
    In: 
        length - how many samples to take
        motor - whether or not to turn the motor on
    Out:
        python list containing time, acceleration in x, y, z
    """
# Give time to establish serial connection
    time.sleep(2)

# Get rid of previous data builtup
    ser.flushInput()

    if motor: # Turn on motor
        print("Motor on")
        ser.write(b'm')

    counter = 0
    data = []

# Turn on data stream
    ser.write(b'd')

    time.sleep(1)

    # for i in range(10):
    #     print(ser.readline().decode("ascii"))

    while counter < length:
        try:
            line_str = ser.readline().decode("ascii").split(",") 

        except UnicodeDecodeError:
            print("decoding error!!")
            continue

        try:
            line_float = [float(i) for i in line_str]
            data.append(line_float)
            # print(line_float)

        except ValueError:
            print("value error - skip!!")
            continue

        counter += 1

    print("Data collected successfully")

# Turn off data stream - wait for this to take effect
    ser.write(b'd')
    time.sleep(1)

    if motor: # Turn off motor
        print("Motor off")
        ser.write(b',')

    if len(data[0]) != 4:
        data = data[1:]

    return data


def control_motor():
    """
        Turn motor on/off by writing to arduino with input from terminal. Quit when 'q' is pressed
    """
    mod = input()
    if mod != "q":
        ser.write(bytes(mod, 'ascii'))
        control_motor()
    else:
        return


def write_file(data, file_path, prefix = ''):
    """
    Used to save output from record() directly as csv
    In:
        data - output from record()
        file_path - base directory to save file to
        prefix - specific file destination
    Out: 
        saved .csv file at the specified filepath
    """

    if not os.path.exists(file_path):
        os.mkdir(file_path)

    index = len(os.listdir(file_path))

    csv_file = file_path + prefix + '.csv'
    # csvFile = filePath + prefix + 'take' + '_' + str(index) + '_' + str(length) + '.csv'

    file = open(csv_file, 'w+', newline = '')

    with file:     
        write = csv.writer(file) 
        write.writerows(data) 









