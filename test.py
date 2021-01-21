from record import *
from analyze import *

def get_sample():
	"""
	Function to wrap the data retrieval and file saving data process together
	"""
	length = 8192
	data = retrieve(length)
	write_file(data, "../data/", prefix = "control/fast/accel/200x5")

# Collect data - this can be commented if the data already exists
get_sample()

# Change the following two values depending on the file destination in `write_file`
power = "200"
freq = "5"

# Load data
accel_file = "../data/control/fast/accel/" + power + "x" + freq + ".csv"
norm_file = "../data/base/accel/take1.csv"

# Visualize data
d = get_octave(norm_file, accel_file, 'z')
ax = plot_bar(d, "freq", "rms")
ax.set_title('Test RMS Acceleration in 1/3 Octave Bins\nPWM Frequency = %sHz, PWM Power = %s' % (power, freq))
plt.show()



#---------------------------------------------------------------------#
#-------------------------Visualize GEHC Data-------------------------#
#---------------------------------------------------------------------#
# ge_fast = load_ge()
# ge_slow = load_ge(test="slow")
# plot_both(ge_fast, d_fast)
# plot_bar(d_fast, "freq", "rms")
# ge = plot_bar(ge_fast, "freqs", "rms_fast")
# ge = plot_bar(ge_slow, "freqs", "rms_slow")
# ge.set_title("GEHC Gantry RMS Acceleration in 1/3 Octave Bins at 0.28 s/rev")
# plt.show()