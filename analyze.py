import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import matplotlib.mlab as mlab
import numpy as np


def load_data(norm_file, df_file):
    """
    Function to load data from .csv files and get pandas dataframes back

    In:
        norm_file - csv file from sample which is not shaking
        df_file - csv file from sample which is shaking
    Out: 
        norm - pandas dataframe containing data in norm_file
        df - pandas dataframe containing data in df
    """
    norm = pd.read_csv(norm_file, names=["x", "y", "z"], index_col=0)
    df = pd.read_csv(df_file, names=["x", "y", "z"], index_col=0)

    return norm, df


def load_ge(test = "fast", max_freq=250):
    """
    Function to load the GEHC data containing RMS acceleration data for both cycle speeds
    In:
        test - specify which operation cycle speed to select for data
        max_freq - cutoff frequency to use while visualizing data later on
    Out: 
        pandas dataframe with desired frequency bounds and cycle speed
    """
    file = "../ge_octave.csv"
    df = pd.read_csv(file)

    if test == "fast":
        df = df.drop(["rms_slow"], axis=1)
    elif test == "slow":
        df = df.drop(["rms_fast"], axis=1)

    if max_freq:
        # Get names of indexes for which freqs has value greater than in my data
        index_names = df[df['freqs'] > max_freq].index
    # Drop these row indexes from dataFrame
        df.drop(index_names, inplace=True)

    return df


def scale_df(norm, df):
    """
    In:
        norm - sample which is not shaking
        df - sample which is shaking
    Out: 
        single dataframe which is scaled based on normalization data
    """
# Scaling factor determined by accelerometer
    scaling_factor = 9.81/2048

# Convert voltages to m/s^2
    norm *= scaling_factor
    df *= scaling_factor

# Subtract resting values to only consider movement from shaking
    df = df - norm.mean()
    return df


def sample_rate(df):
    """
    Determine sample rate for an already loaded dataframe
    In:
        pandas dataframe
    Out: 
        sampling rate
    """
    fs = len(df) / (df.index[-1] - df.index[0])
    return fs

def calc_psd(df, axis):
    """
    Function to calculate PSD of time signal on a specified axes
    In:
            df - a pandas dataframe
            axis - choose an axis on which to evaluate the PSD - z axis is normal to microphone in my experiment
    Out:
            pxx - acceleration density [rms**2 / Hz]
            freqs - frequencies for which pxx is evaluated


    nfft - number of points to use in each fft - more leads to more `freqs` outputs
    """
# Scale sample rate by 1000 because time is output in ms
    fs = 1000 * sample_rate(df)
# Choose number of points to use in each FFT of PSD calculation
    nfft = 1024

    pxx, freqs = mlab.psd(df[axis], NFFT=nfft, Fs=fs)

    return(pxx, freqs)


def third_octave_bins():
    """
    Helper function for rms_third_octave()

    Out: 
       bin_edges - limits for each bin
    """

    lobin = -18

# This may need to be adjusted (set to -5) if acceleration data does not include values above 280.6Hz (from 1/3 octave bin definitions)
    hibin = -4

# Find the centers of the 1/3 octave bins
    centers = 10 ** 3 * (2**(np.arange(lobin, hibin)/3))
    fd = 2 ** (1/6)

    bin_edges = []
    for index, el in enumerate(centers):
        if index == 0:
            bin_edges.append(el/fd)
            bin_edges.append(el*fd)
        else:
            bin_edges.append(el*fd)
    return bin_edges, centers

def log_auc(pxx, freqs):
	"""
	Calculate area under log curve defined by pxx on vertical axis and freqs on horizontal axis
    In:
        pxx - list of energy densities provided by PSD calculation
        freqs - list of frequencies corresponding to pxx values
    Out:
        sum of the areas calculated for every interval defined by pxx and freqs
	"""
# Merge lists into a dataframe for easier comprehension
	df = pd.DataFrame({'pxx': pxx, 'freqs': freqs})

# Initialize list to contain sum for every interval
	auc_int = []

# Iterate across rows of dataframe
	for index, row in df.iterrows():
		if index == len(df) - 1:
# If this is the last element in the list, return the sum of the elements in the list
			return(sum(auc_int))
# Use method for area under a log curve outlined by https://femci.gsfc.nasa.gov/random/randomgrms.html
# This method calculates the area between the selected location and the next location
		asd_lo = row.pxx
		asd_hi = df['pxx'].iloc[index + 1]
		f_lo = row.freqs
		f_hi = df['freqs'].iloc[index + 1]

		dB = 10*np.log10(asd_hi / asd_lo)
		octaves = np.log10(f_hi/f_lo) / np.log10(2)
		m = dB / octaves
		AUC = 10*np.log10(2) * asd_hi / (10*np.log10(2) + m) * (f_hi - (f_lo * (f_lo / f_hi) ** (m / (10*np.log10(2)))))

# Add the calculated area for these points to auc_int 
		auc_int.append(AUC)


def rms_log(df):
    """
    Calculates g RMS for previously specified bins
    In:
        df - a dataframe with a 'cut' column specifying a bin division for every row
    Out: 
        list of RMS values for every bin 
    """
    this_bin = df['cut'].iloc[0]
    pxx_temp = []
    freqs_temp = []
    rms_list = []
    counter = 0
# Iterate over every value in density
    for index in range(len(df)):
# Check to see if you're getting to the end of the list
        if index == (len(df) - 1):
            freqs_temp.append(df['freqs'].iloc[index])
            pxx_temp.append(df['pxx'].iloc[index])            
            sums = log_auc(pxx_temp, freqs_temp)
            rms = np.sqrt(sums)
            rms_list.append(rms)
# Check to see if the next bin is the same as this one
        elif df['cut'].iloc[index] == df['cut'].iloc[index + 1]:
            freqs_temp.append(df['freqs'].iloc[index])
            pxx_temp.append(df['pxx'].iloc[index])
# Check to see if the next bin is the same as this one
# If it's not, append these values, calculate and append rms value, reset temporary lists
        elif df['cut'].iloc[index] != df['cut'].iloc[index + 1]:  
            freqs_temp.append(df['freqs'].iloc[index])
            pxx_temp.append(df['pxx'].iloc[index])

            sums = log_auc(pxx_temp, freqs_temp)
            rms = np.sqrt(sums)
            rms_list.append(rms)
            pxx_temp = []
            freqs_temp = []

    return rms_list





def rms_third_octave(pxx, freqs):
    """
    In:
            pxx - list, acceleration density [rms**2 / Hz]
            freqs - list, frequencies for which pxx is evaluated
                    (pxx and freqs should be the same length, come from mlab.psd calculation)
    Out:
            rms for every bin - square root of numeric integral for this range
    """
    merged = {"pxx": pxx, "freqs": freqs}
    density = pd.DataFrame(merged)

# Get 1/3 octave frequency bin centers from third_octave_bins()
    bins, centers = third_octave_bins()

# Get names of indexes for which freqs has value less than the smallest bin
    index_names = density[density['freqs'] <= bins[0]].index

# Drop these row indexes from dataFrame
    density.drop(index_names, inplace=True)

# Use pd.cut to add column which defines which bin the row is in
    density['cut'], bin_edges = pd.cut(density['freqs'], bins=bins, retbins=True)

# Find the rms for every bin specified
    rms_found = rms_log(density)
# Combine rms values and bins into dictionary, create DataFrame
    rms_dict = {"rms": rms_found, "freq": centers}

    rms_df = pd.DataFrame(rms_dict)

    return rms_df


def get_octave(norm_file, data_file, axis):
    """
    In: 
            norm_file - csv containing accelerometer data at rest
            data_file - csv containing accelerometer data while shaking
            axis - choose the accelerometer axis on which to do analysis
    Out: 
            dataframe with RMS values in third octave bands, specified by the band's center
    """
# Read CSVs as pandas dataframes
    norm, df = load_data(norm_file, data_file)
# Adjust for base offset
    df = scale_df(norm, df)
# Evaluate psd for adjusted values
    pxx, freqs = calc_psd(df, axis)
# Calculate RMS for octave bands
    d = rms_third_octave(pxx, freqs)

    return d


#---------------------------------------------------------------------#
#-------------------------------Plots---------------------------------#
#---------------------------------------------------------------------#


def plot_bar(df, xlab, ylab):
    """
    In: 
            dd - a dataframe
    Out: 
            plot
    """
    nominal_centers = [16, 20, 25, 31.5, 40,
                       50, 63, 80, 100, 125, 160, 200, 250]
    ylo = 10**-2
    yhi = 10**0

    ax = df.plot.bar(x=xlab, y=ylab, color="red", width=1.0, legend = False)

    ax.set_yscale('log')
    ax.set_xticklabels(nominal_centers, rotation=0)
    ax.yaxis.grid(which='major', color='black', linestyle='dashed')
    ax.yaxis.grid(which='minor', color='gray', linestyle='dashed', alpha=.5)
    ax.set_axisbelow(True)
    ax.set_ylim(ylo, yhi)
# Annotate plot
    # ax.set_title("Shaker Test 1 - RMS Acceleration in 1/3 Octave Bins")
    ax.set_xlabel("Frequency Bin Centers (Hz)")
    ax.set_ylabel("RMS Acceleration (m/s²)")

# Add bar labels
    for p in ax.patches:
    	ax.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    
    return ax

def plot_both(ge, dd):
    """
    In:
            ge - dataframe of ge gantry data
            dd - dataframe of my accel data

    displays matplotlib plot of two dataframes
    """
    nominal_centers = [16, 20, 25, 31.5, 40,
                       50, 63, 80, 100, 125, 160, 200, 250]
    ylo = 10**-2
    yhi = 10**0

    fig, axes = plt.subplots(nrows=2, ncols=1)

    ax = dd.plot.bar(x="freq", y="rms", color="red", width=1.0, ax=axes[0])
    plt.yscale('log')
    ax.set_yscale('log')
    ax.set_xticklabels(nominal_centers, rotation=0)
    ax.yaxis.grid(which='major', color='black', linestyle='dashed')
    ax.yaxis.grid(which='minor', color='gray', linestyle='dashed', alpha=.5)
    ax.set_axisbelow(True)
    ax.set_ylim(ylo, yhi)

    ax1 = ge.plot.bar(x="freqs", y="rms_fast",
                      color="red", width=1.0, ax=axes[1])
    # ax.set_yscale('log')
    ax1.set_xticklabels(nominal_centers, rotation=0)
    ax1.yaxis.grid(which='major', color='black', linestyle='dashed')
    ax1.yaxis.grid(which='minor', color='gray', linestyle='dashed', alpha=.5)
    ax1.set_axisbelow(True)
    ax1.set_ylim(ylo, yhi)

    plt.show()


#---------------------------------------------------------------------#
#-------------------------RMS in narrow bins--------------------------#
#---------------------------------------------------------------------#

def rms_list(df):
    """
    In:
            df - a dataframe with a column called 'cut' which sorts each row into bins
    Out: 
            rms_list - a list containing the rms value for every bin specified
    """

    # Initialize data for rms by creating empty lists and establishing the first bin comparison
    this_bin = df['cut'].iloc[0]
    pxx_temp = []
    freqs_temp = []
    rms_list = []

# Iterate over every value in density
    for index in range(len(df)):

        # Compare this bin to the previous bin
        if index == (len(df) - 1):
            sums = np.trapz(pxx_temp, freqs_temp)
            rms = np.sqrt(sums)
            rms_list.append(rms)
        # If it's the same, append the value to `temp` lists
        elif df['cut'].iloc[index] == this_bin:
            freqs_temp.append(df['freqs'].iloc[index])
            pxx_temp.append(df['pxx'].iloc[index])
        else:  # if it's not, append the rms value, reset temp lists, change `this_bin`
            sums = np.trapz(pxx_temp, freqs_temp)
            rms = np.sqrt(sums)
            rms_list.append(rms)
            pxx_temp = []
            freqs_temp = []
            this_bin = df['cut'].iloc[index]

    return rms_list


def rms_linear(pxx, freqs, bin_width):
    """
    In:
            pxx - list, acceleration density [rms**2 / Hz]
            freqs - list, frequencies for which pxx is evaluated
                    (pxx and freqs should be the same length, come from mlab.psd calculation)
            bin_width - band width for RMS calculation
    Out:
            rms for every bin - square root of numeric integral for this range
    """
    merged = {"pxx": pxx, "freqs": freqs}
    density = pd.DataFrame(merged)

    bins = np.arange(-.0001, freqs[-1] + bin_width, bin_width)

# Use pd.cut to add column which defines which bin the row is in
    density['cut'], bin_edges = pd.cut(
        density['freqs'], bins=bins, retbins=True)

# Find the rms for every bin specified
    rms_found = rms_list(density)

# Combine rms values and bins into dictionary, create DataFrame
    rms_dict = {"rms": rms_found, "freq": bin_edges[1:-1]}
    rms_df = pd.DataFrame(rms_dict)

    return rms_df
