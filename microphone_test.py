# Skip 9 rows with 2 channel Analog Discovery data to bypass column titles

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from analyze import *


def load_data(filename):
	"""
	Function to load two channel output from Analog Discovery

	In: 
		config - a specification of which test the data corresponds to
		speed - a specification of which test the data corresponds to
	Out:
		dataframe containing correct data
	"""
	df = pd.read_csv(filename, skiprows = 9, names=["time", "fix", "vib"], index_col = 0)
	return df

def resample(df, fs_new, start = 0):
	"""
	Function to take microphone data sampled at a high rate and resample at a rate
	more consistent with expected audio sampling rates

	In: 
		df - a dataframe with data corresponding to microphone output
		freq - a desired resample frequency
		start - the index at which to start the resample sequence
	Out:
		dataframe sampled at updated frequency
	"""
	fs_old = int(sample_rate(df))
	interval = int(fs_old / fs_new)
	df_new = df.iloc[start::interval, :]
	return(df_new)


def plot_mics(df0, title, label1, label2):
	"""
	Special plotting function which takes a dataframe and makes a plot to 
	compare the vibrating and stationary microphones

	In:
		df0 - dataframe with 'fix' and 'vib' columns containing psd of fixed and vibrating microphone
		title - title for plot
		label1 - legend label for fixed microphone
		label2 - legend label for vibrating microphone
	Out:
		matplotlib.pyplot visual comparing fixed and vibrating microphone in a single experiment
	"""
	nfft = 2048
	fs = sample_rate(df0)

	plt.subplot(211)
	plt.xscale('log')
	plt.psd(df0.fix, NFFT = nfft, Fs = fs, label = label1)
	plt.psd(df0.vib, NFFT = nfft, Fs = fs, label = label2)
	plt.xlim(0, 1000)
	plt.ylim(-75, -10)
	plt.yticks([-70, -50, -30, -10])
	plt.title(title + ' // 0 Hz - 1 kHz')
	plt.ylabel('Volume Density (db/Hz)')
	plt.legend()
	plt.xlabel('')

	plt.subplot(212)
	plt.xscale('log')
	plt.psd(df0.fix, NFFT = nfft, Fs = fs, label = label1)
	plt.psd(df0.vib, NFFT = nfft, Fs = fs, label = label2)
	plt.xlim(1000, 10000)
	plt.ylim(-100, -50)
	plt.yticks([-90, -70, -50])
	plt.title(title + ' // 1 kHz - 10 kHz')
	plt.ylabel('Volume Density (db/Hz)')
	plt.xlabel('Frequency (Hz)')
	plt.legend()

	plt.show()


def across_mics(df0, df1, title, label0, label1):
	"""
	Special plotting function which takes two dataframes and makes a plot to 
	compare the vibrating microphone results from each

	In:
		df0 - dataframe with 'fix' and 'vib' columns containing time series from fixed and vibrating microphone
		df1 - dataframe with 'fix' and 'vib' columns containing time series from fixed and vibrating microphone
		title - title for plot
		label1 - legend label for fixed microphone
		label2 - legend label for vibrating microphone
	Out:
		matplotlib.pyplot visual comparing vibrating microphone across experiments
	"""
	nfft = 2048
	fs0 = sample_rate(df0)
	fs1 = sample_rate(df1)

	plt.subplot(211)
	plt.xscale('log')
	plt.psd(df0.vib, NFFT = nfft, Fs = fs0, label = label0)
	plt.psd(df1.vib, NFFT = nfft, Fs = fs1, label = label1)
	plt.xlim(0, 1000)
	plt.ylim(-75, -10)
	plt.yticks([-70, -50, -30, -10])
	plt.title(title + ' // 0 Hz - 1 kHz')
	plt.ylabel('Volume Density (db/Hz)')
	plt.legend()
	plt.xlabel('')

	plt.subplot(212)
	plt.xscale('log')
	plt.psd(df0.vib, NFFT = nfft, Fs = fs0, label = label0)
	plt.psd(df1.vib, NFFT = nfft, Fs = fs1, label = label1)
	plt.xlim(1000, 10000)
	plt.ylim(-100, -50)
	plt.yticks([-90, -70, -50])
	plt.title(title + ' // 1 kHz - 10 kHz')
	plt.ylabel('Volume Density (db/Hz)')
	plt.xlabel('Frequency (Hz)')
	plt.legend()

	plt.show()

# # Uncomment the following four lines to find load base (motor off) data and make visual
# base = "../data/base/mic/take0.csv"
# dfbase = load_data(base)
# df_base = resample(df0, 44100)
# plot_mics(df_0, 'Microphone Comparison with No Motor', 'mic1', 'mic2')

# # Uncomment the following four lines to find load control (motor on, no foam) data and make visual
# control = "../data/control/fast/mic/take0.csv"
# dfcontrol = load_data(control)
# df_control = resample(df0, 44100)
# plot_mics(df_0, 'Microphone Comparison with Motor Fast', 'fixed', 'vibrating')

# # Uncomment the following four lines to find load foam (motor on, with sorbothane) data and make visual
# foam = "../data/sorbothane/fast/mic/take0.csv"
# dffoam = load_data(foam)
# df_foam = resample(df1, 44100)
# plot_mics(df_0, 'Microphone Comparison with Motor Fast and 0.080" Sorbothane', 'fixed', 'vibrating')

# # Uncomment the following seven lines to find load control and foam data and make visual to compare
# control = "../data/control/fast/mic/take0.csv"
# dfcontrol = load_data(control)
# df_control = resample(df0, 44100)
# foam = "../data/sorbothane/fast/mic/take0.csv"
# dffoam = load_data(foam)
# df_foam = resample(df1, 44100)
# across_mics(df_foam, df_control, 'Comparison of No Foam and With Foam Scenarios', 'control', 'sorbothane')


#---------------------------------------------------------------------#
#---------------------Combine Multiple Trials-------------------------#
#---------------------------------------------------------------------#
def combine_trials(resample_rate, *dfs):
	"""
	In: 
		resample_rate - rate at which to resample acceleration data
		*dfs - any number of pandas dataframes with microphone data
	Out: 
		dataframe with mean psd outputs for fixed and vibrating microphone across several trials
	"""
	pass
	d = {}
	for index, df in enumerate(dfs):
		df = resample(df, resample_rate)
		fs = sample_rate(df)
		nfft = 2048

		pxx_fix, freqs = mlab.psd(df.fix, NFFT = nfft, Fs = fs)
		pxx_vib, freqs = mlab.psd(df.vib, NFFT = nfft, Fs = fs)
		if len(d) == 0:
			d["freqs"] = freqs
		d["pxx_fix" + str(index)] = pxx_fix
		d["pxx_vib" + str(index)] = pxx_vib

	df = pd.DataFrame(d)

	df = df.set_index('freqs')
	
# Return the mean of each pxx in decibels
	df['fix'] = 10*np.log10((df.filter(regex=("fix")).mean(axis = 1)))
	df['vib'] = 10*np.log10((df.filter(regex=("vib")).mean(axis = 1)))

	return df


def plot_mean(df0, title, label1, label2):
	"""
	Special plotting function for result of `combine_trials()` 
	function which returns a dataframe where psd is already calculated

	In:
		df0 - dataframe with 'fix' and 'vib' columns containing psd of fixed and vibrating microphone
		title - title for plot
		label1 - legend label for fixed microphone
		label2 - legend label for vibrating microphone
	Out:
		matplotlib.pyplot visual comparing fixed and vibrating microphone in a single experiment
	"""
	plt.subplot(211)
	plt.grid()
	plt.xscale('log')
	plt.plot(df0.fix, label = label1)
	plt.plot(df0.vib, label = label2)
	plt.xlim(0, 1000)
	plt.ylim(-75, -10)
	plt.yticks([-70, -50, -30, -10])
	plt.title(title + ' // 0 Hz - 1 kHz')
	plt.ylabel('Volume Density (db/Hz)')
	plt.legend()
	plt.xlabel('')

	plt.subplot(212)
	plt.grid()
	plt.xscale('log')
	plt.plot(df0.fix, label = label1)
	plt.plot(df0.vib, label = label2)
	plt.xlim(1000, 10000)
	plt.ylim(-100, -50)
	plt.yticks([-90, -70, -50])
	plt.title(title + ' // 1 kHz - 10 kHz')
	plt.ylabel('Volume Density (db/Hz)')
	plt.xlabel('Frequency (Hz)')
	plt.legend()

	plt.show()

def across_mean(df0, df1, title, label0, label1):
	"""
	Special plotting function for results of `combine_trials()` 
	function which returns a dataframe where psd is already calculated

	In:
		df0 - dataframe with 'fix' and 'vib' columns containing psd of fixed and vibrating microphone
		df1 - dataframe with 'fix' and 'vib' columns containing psd of fixed and vibrating microphone
		title - title for plot
		label1 - legend label for fixed microphone
		label2 - legend label for vibrating microphone
	Out:
		matplotlib.pyplot visual comparing vibrating microphone across experiments
	"""
	plt.subplot(211)
	plt.xscale('log')
	plt.plot(df0.vib, label = label0)
	plt.plot(df1.vib, label = label1)
	plt.xlim(0, 1000)
	plt.ylim(-75, -10)
	plt.yticks([-70, -50, -30, -10])
	plt.title(title + ' // 0 Hz - 1 kHz')
	plt.ylabel('Volume Density (db/Hz)')
	plt.legend()
	plt.xlabel('')

	plt.subplot(212)
	plt.xscale('log')
	plt.plot(df0.vib, label = label0)
	plt.plot(df1.vib, label = label1)
	plt.xlim(1000, 10000)
	plt.ylim(-100, -50)
	plt.yticks([-90, -70, -50])
	plt.title(title + ' // 1 kHz - 10 kHz')
	plt.ylabel('Volume Density (db/Hz)')
	plt.xlabel('Frequency (Hz)')
	plt.legend()

	plt.show()

# Load several trials worth of control data (vibrating, no foam)
control0 = "../data/control/fast/mic/take0.csv"
control1 = "../data/control/fast/mic/take1.csv"
control2 = "../data/control/fast/mic/take2.csv"
control3 = "../data/control/fast/mic/take3.csv"
control4 = "../data/control/fast/mic/take4.csv"
control5 = "../data/control/fast/mic/take5.csv"
control6 = "../data/control/fast/mic/take6.csv"

control_0 = load_data(control0)
control_1 = load_data(control1)
control_2 = load_data(control2)
control_3 = load_data(control3)
control_4 = load_data(control4)
control_5 = load_data(control5)
control_6 = load_data(control6)

# # Uncomment the following two lines to find average across trials and make visual
# df_control = combine_trials(44100, control_0, control_1, control_2, control_3, control_4, control_5)
# plot_mean(df_control, 'Microphone Comparison with Motor Fast', 'fixed', 'vibrating')


foam0 = "../data/sorbothane/fast/mic/take0.csv"
foam1 = "../data/sorbothane/fast/mic/take1.csv"
foam2 = "../data/sorbothane/fast/mic/take2.csv"
foam3 = "../data/sorbothane/fast/mic/take3.csv"
foam4 = "../data/sorbothane/fast/mic/take4.csv"
foam5 = "../data/sorbothane/fast/mic/take5.csv"
foam6 = "../data/sorbothane/fast/mic/take6.csv"

foam_0 = load_data(foam0)
foam_1 = load_data(foam1)
foam_2 = load_data(foam2)
foam_3 = load_data(foam3)
foam_4 = load_data(foam4)
foam_5 = load_data(foam5)
foam_6 = load_data(foam6)

# # Uncomment the following two lines to find average across foam trials and make visual
# df_foam = combine_trials(44100, foam_0, foam_1, foam_2, foam_3, foam_4, foam_5)
# plot_mean(df_foam, 'Microphone Comparison with Motor Fast and 0.080" Sorbothane', 'fixed', 'vibrating')

# # Uncomment the following three lines to find average across foam and control trials and make visual
# df_control = combine_trials(44100, control_0, control_1, control_2, control_3, control_4, control_5)
# df_foam = combine_trials(44100, foam_0, foam_1, foam_2, foam_3, foam_4, foam_5)
# across_mean(df_foam, df_control, 'Comparison of No Foam and With Foam Scenarios', 'control', 'sorbothane')