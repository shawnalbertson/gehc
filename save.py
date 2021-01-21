from record import *

def get_norm():
	length = 500
	data = retrieve(length, motor = False)
	write_file(data, "../data/", prefix = "base/accel/take0")

def get_sample():
	length = 8192
	data = retrieve(length)
	prefix_ex0 = "control/fast/not/take0"
	prefix_ex1 = "sorbothane/fast/accel/take0"
	write_file(data, "../data/", prefix = prefix_ex0)


# # Uncomment the following line to save a reference file with accelerometer data in the system's resting state
# get_norm()

# Uncomment the following line to save a data file with accelerometer data while the system is running
# get_sample()