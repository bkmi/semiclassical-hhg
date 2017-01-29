
# coding: utf-8

# In[ ]:

# This program gathers information from modules within semiclassicalhhg and finds saddle points using methods in
# Milosevic, et al. for two color elliptical field mixing with a sin2 envelope. The parameters of the envelope as well
# as the fields are controlled from within this program.

# When the entire program is run it will output 3 set of data in the form of numpy.savetxt files.
# params.npy is the set of parameters defined below. (i.e. pulse length, field frequencies)
# intensity_coef.npy is the set of relative field strengths. (Ref Milosevic: same idea as i/6 + j/6 = 1 with i,j = int)
# saddle_pts.npy is the set of saddle points for the parameters and for every relative field strength combination.

# The user inputs data until the cell that shows where calculations begin!
# The program requires a confirmation if you are about to overwrite old data.


#we want to used pulsed supplimentary equations so...
supp_eqns = 'pulsed_supp_eqns'

import numpy as np
import scipy.optimize
from math import isnan
import matplotlib.pyplot as plt
import multiprocessing as mp
import os


# In[ ]:

#Prepare intensity coefficients


#Method 1


# i/6 and j/6 method like in milosevic:
#   Input the number of relative intensity combinations to search through as count_intensity_steps.
#   The program is setup to start with all intensity distributed to the "int1*w" frequency field.
#   Then it searches through relative intensity combinations in a number of steps equal to count_intensity_steps + 1.
#   You end up with count_intensity_steps + 1 total intensity options to search through because the program includes
#   all intensity given to int1*w field, and all intensity given to int2*w field.
#   
#   For example: count_intensity_steps = 2
#   The program generates these field combinations in the format (i/2 + j/2 = 1):
#               2/2 + 0/2 = 1
#               1/2 + 1/2 = 1
#               0/2 + 2/2 = 1

#"""
count_intensity_steps = 4
intensity_coef = (np.transpose(np.array([np.arange(0,count_intensity_steps+1,1),
                                         np.arange(count_intensity_steps,-1,-1)]))
                  /count_intensity_steps)
intensity_coef = intensity_coef[::-1]
#This line removes the fields that are not mixed, but rather just one field. Comment it to search with just one field.
intensity_coef = intensity_coef[1:(len(intensity_coef)-1)]

#"""


#Method 2

#With this method, simply select a single relative field strength combination. The program normalizes the coefficients.

"""
int1_w_rel_intensity = 1
int2_w_rel_intensity = 1

normalize_intensity = int1_w_rel_intensity + int2_w_rel_intensity
intensity_coef = np.array([[int1_w_rel_intensity, int2_w_rel_intensity]])/normalize_intensity
"""
intensity_coef


# In[ ]:

#prepare params 

# This section defines the parameters of the light that the system is exposed to. There are ten parameters that define
# the system. These parameters are:
#             omega - Frequency omega
#             potential_well - Depth of the potential well
#             electric_field1 - Electric field strength of the int1 field (array with len(intensity_coef))
#             int1 - Integer multiplied by omega defining the int1 field frequency
#             electric_field2 - Electric field strength of the int2 field (array with len(intensity_coef))
#             int2 - Integer multiplied by omega defining the int2 field frequency
#             epsilon1 - Ellipticity of the int1 field
#             epsilon2 - Ellipticity of the int2 field
#             mod_omega - Frequency of the sin2 envelope
#             mod_phase - Phase of the envelope relative to the phase of the elliptical em waves.
# params is a list of sets of 10 parameters. Each set of parameters is for a particular relative intensity combination.


# User defined params:
wavelength = 800 # nm
potential_well = -0.5 # atomic units
intensity1 = intensity_coef[:,0]*1e14 # W/cm^2
frequency_integer1 = 1.0 # no units
intensity2 = intensity_coef[:,1]*1e14 # W/cm^2
frequency_integer2 = 2.0 # no units
                    # epsilon1 = epsilon2 = 1 = corotating circular
                    # epsilon1 = -epsilon2 = 1 = counterrotating circular
epsilon1 = 1  # no units
epsilon2 = -1 # no units

#Determine the width of the pulse
cycles_per_pulse = 4.0 #The number of times omega oscillates in a single sin2 oscillation.
mod_phase = 0




# Params that are not defined by the user:
omega = 45.56/wavelength
period = 2*np.pi/omega

electric_field1 = np.sqrt(intensity1/3.509e16)
omega1 = frequency_integer1 * omega
period1 = 2*np.pi/omega1

electric_field2 = np.sqrt(intensity2/3.509e16)
omega2 = frequency_integer1 * omega
period2 = 2*np.pi/omega2

mod_omega = (1 / cycles_per_pulse) * omega

params = np.zeros((len(intensity_coef), 10))
for i in np.arange(len(intensity_coef)):
    params[i] = (omega, potential_well, electric_field1[i], frequency_integer1,
                 electric_field2[i], frequency_integer2, epsilon1, epsilon2, mod_omega, mod_phase)


# In[ ]:

#Generate and organize the time values used in guessing. Recall: The program kills saddle pts with Re[tau] > Period.
#The calculation time scales exponentially with the number of guesses on the grid so use caution with large numbers.

#Enter below the grid size you would like to use:
recombination_grid_size = 20  #Number of guesses for tf
real_tau_grid_size =      10  #Number of guesses for Re[tau]
imag_tau_grid_size =      4   #Number of guesses for Im[tau]


#--


#recombTimes = bounds for program to consider HHG, this program considers a single pulse for recomb (tf)
recombTimes = [0.0, cycles_per_pulse]  #given in field cycles
recomb_grid = np.array(np.linspace( (recombTimes[0]) * period, (recombTimes[1])
                                   * period, recombination_grid_size) )

#The tau guesses are limited to 1 period or less because anything longer gets removed later anyway.
#Generate real tau grid
real_tau_grid = np.array(np.linspace( 0.0, period, real_tau_grid_size), dtype = np.complex128 )
#Generate imag tau grid
imag_tau_grid = np.array(np.linspace( 0.0, period, imag_tau_grid_size), dtype = np.complex128 )

#times for plotting/later analysis
dt = 0.01
all_real_times = np.arange( float(np.real(min(recomb_grid) - max(real_tau_grid))),
                           float(np.real(max(recomb_grid))), dt)


# In[ ]:

# Name the output data. Files will be placed in a directory: data_ellip_sin2_pulse_(filename)

filename = "sample"


# In[ ]:

# User data is all entered ABOVE


# --


# Processing is done BELOW


# In[ ]:

# This just makes sure that you don't accidently overwrite something you weren't expecting to.
# The exception is raised if the directory ./data_ellip_sin2_pulse_(filename) already exists, no matter what is inside.

import sys

def query_yes_no(question, default="no"):
    """Ask a yes/no question via input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")



try:
    os.mkdir("./data_ellip_sin2_pulse_"+str(filename))
except OSError:
    print("The directory " + "./data_ellip_sin2_pulse_" + str(filename) + " already exists. \n")
    if query_yes_no("Would you like to continue and overwrite the data in the directory?"):
        print("\nOkay, I will overwrite the data in " + "./data_ellip_sin2_pulse_" + str(filename))
        pass
    else:
        print("\nYou have chosen to stop the program. \nRename the variable 'filename' such that "
              + "./data_ellip_sin2_pulse_(filename) does not already exist. \nThe filename is currently: "
              + str(filename) + "\nThe directory that already exists is called: " + "./data_ellip_sin2_pulse_"
              + str(filename))
        raise


# In[ ]:

from semiclassicalhhg.pulsed_supp_eqns import nineteen_cplx_tf, nineteen_cplx_tf_p, position, velocity

from semiclassicalhhg.saddle_pts import *


# In[ ]:

def efieldvec( t, args ):
    '''
    Efield as a function of time. Elliptical polarization given in Milosevic.
    
    Input time (t) and args.
    args = frequency (w), ground state energy (gse), electric field for int1*w (EF1), integer1 multiplied by w (int1),
        electric field for int2*w (EF2), integer1 multiplied by w (int2), ellipticity1 (epsilon1),
        ellipticity2 (epsilon2), frequency of pulse modulation (modw), and phase of pulse modulation (modphase).
    
    Returns electric field vector at a given time
    '''
    
    from numpy import sqrt
    w, gse, EF1, int1, EF2, int2, epsilon1, epsilon2, modw, modphase = args
    
    xdir = (((1.0j/2)*EF1*sin(modphase + modw*t)**2)/(e**(1.0j*int1*t*w)*sqrt(1 + epsilon1**2)) 
            - ((1.0j/2)*e**(1.0j*int1*t*w)*EF1*sin(modphase + modw*t)**2)/sqrt(1 + epsilon1**2) 
            + ((1.0j/2)*EF2*sin(modphase + modw*t)**2)/(e**(1.0j*int2*t*w)*sqrt(1 + epsilon2**2)) 
            - ((1.0j/2)*e**(1.0j*int2*t*w)*EF2*sin(modphase + modw*t)**2)/sqrt(1 + epsilon2**2))
    ydir = (-(EF1*epsilon1*sin(modphase + modw*t)**2)/(2*e**(1.0j*int1*t*w)*sqrt(1 + epsilon1**2))
            - (e**(1.0j*int1*t*w)*EF1*epsilon1*sin(modphase + modw*t)**2)/(2*sqrt(1 + epsilon1**2))
            - (EF2*epsilon2*sin(modphase + modw*t)**2)/(2*e**(1.0j*int2*t*w)*sqrt(1 + epsilon2**2))
            - (e**(1.0j*int2*t*w)*EF2*epsilon2*sin(modphase + modw*t)**2)/(2*sqrt(1 + epsilon2**2)))
    
    return [xdir, ydir]


# In[ ]:

'''
This is the final stage.

Input tau grid guesses real, imag, tf grid, args, supp_eqns.
(supp_eqns is a string directing to the file containing the kind of supplementary equations you are using)

Args comes as a list of paramters lists. Each one will be considered for the output.

Output sorted "good tau" list of lists.
'''

def last_step(dressed_input):
    
    real_tau_grid = dressed_input[0]
    imag_tau_grid = dressed_input[1]
    recomb_grid = dressed_input[2]
    params = dressed_input[3]
    supp_eqns = dressed_input[4]
    
    data = []
    data.append(good_taus( nineteen_scan_for_tau(real_tau_grid, imag_tau_grid, recomb_grid, params, supp_eqns)[0],
                          period)[0]
               )
    
    return data


# --


dressed_input = []
for i in np.arange(len(params)):
    current = (real_tau_grid, imag_tau_grid, recomb_grid, params[i], supp_eqns)
    dressed_input.append(current)

# --


between = []

if __name__ == '__main__':
    with mp.Pool() as p:
        between.append(p.map(last_step, dressed_input))

        
quick_pos_uniq_round_vals = []

#the process adds some weird index stuff. To fix this problem undo it with this step:
for i, v in enumerate(between[0]):
    quick_pos_uniq_round_vals.append(between[0][i][0])


# In[ ]:

what_to_say = "The parameters for this dataset are as follows: \n"
what_to_say = what_to_say + "\n"

what_to_say = what_to_say + "Omega = " + str(omega) + "\n"
what_to_say = what_to_say + "Potential Well = " + str(potential_well) + "\n"
what_to_say = what_to_say + "1st Electric Field Amplitudes = " + str(electric_field1) + "\n"
what_to_say = what_to_say + "1st Frequency Integer = " + str(frequency_integer1) + "\n"
what_to_say = what_to_say + "1st epsilon = " + str(epsilon1) + "\n"
what_to_say = what_to_say + "2nd Electric Field Amplitudes = " + str(electric_field2) + "\n"
what_to_say = what_to_say + "2nd Frequency Integer = " + str(frequency_integer2) + "\n"
what_to_say = what_to_say + "2nd epsilon = " + str(epsilon2) + "\n"
what_to_say = what_to_say + "Sin2 Modulation Omega = " + str(mod_omega) + "\n"
what_to_say = what_to_say + "Sin2 Modulation Phase = " + str(mod_phase) + "\n"
what_to_say = what_to_say + "\n"

what_to_say = what_to_say + "The number of field combinations tested was: " + str(len(intensity_coef)) + "\n"
what_to_say = what_to_say + "With the following relative intensities:\n" +str(intensity_coef) + "\n"


with open('./'+"data_ellip_sin2_pulse_"+str(filename)+'/information', 'w') as info:
    info.write(what_to_say)


# In[ ]:

np.save('./'+"data_ellip_sin2_pulse_"+str(filename)+'/params', np.array(params))


# In[ ]:

np.save('./'+"data_ellip_sin2_pulse_"+str(filename)+'/intensity_coef', np.array(intensity_coef))


# In[ ]:

np.save('./'+"data_ellip_sin2_pulse_"+str(filename)+'/saddle_pts', np.array(quick_pos_uniq_round_vals))


# In[ ]:

# Now I will save each intensity ratio test as its own array for easy access.

try:
    os.mkdir("./data_ellip_sin2_pulse_"+str(filename)+"/data_by_intensity_ratio")
except:
    pass

for i, v in enumerate(params):
    if i<9:
        np.savetxt('./'+"data_ellip_sin2_pulse_"+str(filename)+"/data_by_intensity_ratio/0"+str(i)+"_params",
                  np.array(params)[i])
        np.savetxt('./'+"data_ellip_sin2_pulse_"+str(filename)+"/data_by_intensity_ratio/0"+str(i)+"_intensity_coef",
                  np.array(intensity_coef)[i])
        np.savetxt('./'+"data_ellip_sin2_pulse_"+str(filename)+"/data_by_intensity_ratio/0"+str(i)+"_saddle_pts",
                  np.array(quick_pos_uniq_round_vals)[i])
    else:
        np.savetxt('./'+"data_ellip_sin2_pulse_"+str(filename)+"/data_by_intensity_ratio/"+str(i)+"_params",
                  np.array(params)[i])
        np.savetxt('./'+"data_ellip_sin2_pulse_"+str(filename)+"/data_by_intensity_ratio/"+str(i)+"_intensity_coef",
                  np.array(intensity_coef)[i])
        np.savetxt('./'+"data_ellip_sin2_pulse_"+str(filename)+"/data_by_intensity_ratio/"+str(i)+"_saddle_pts",
                  np.array(quick_pos_uniq_round_vals)[i])


# In[ ]:



