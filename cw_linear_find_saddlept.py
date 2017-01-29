
# coding: utf-8

# In[1]:

# This program gathers information from modules within semiclassicalhhg and finds saddle points using methods in
# Milosevic, et al. for linear polarization. The parameters of the envelope as well
# as the fields are controlled from within this program.

# When the entire program is run it will output 3 set of data in the form of numpy.savetxt files.
# params.npy is the set of parameters defined below. (i.e. field frequencies)
# intensity_coef.npy is the set of relative field strengths. (Ref Milosevic: same idea as i/6 + j/6 = 1 with i,j = int)
# saddle_pts.npy is the set of saddle points for the parameters and for every relative field strength combination.

# The user inputs data until the cell that shows where calculations begin!
# The program requires a confirmation if you are about to overwrite old data.


#we want to used constant wave supplimentary equations so...
supp_eqns = 'linear_supp_eqns'

import numpy as np
import scipy.optimize
from math import isnan
import matplotlib.pyplot as plt
import multiprocessing as mp
import os


# In[2]:

#Prepare intensity coefficients

# This section is not really important for linear polarization because there is no field mixing.
# The only intensity that matters is the intensity of the single linear polarization. (1 in this section.)

int1_w_rel_intensity = 1
int2_w_rel_intensity = 0

normalize_intensity = int1_w_rel_intensity + int2_w_rel_intensity
intensity_coef = np.array([[int1_w_rel_intensity, int2_w_rel_intensity]])/normalize_intensity

intensity_coef


# In[3]:

#prepare params 

# This section defines the parameters of the light that the system is exposed to. There are ten parameters that define
# the system. These parameters are:
#             omega - Frequency omega
#             potential_well - Depth of the potential well
#             electric_field1 - Electric field strength of the int1 field (array with len(intensity_coef))
#             int1 - Integer multiplied by omega defining the int1 field frequency - N/A
#             electric_field2 - Electric field strength of the int2 field (array with len(intensity_coef)) - N/A
#             int2 - Integer multiplied by omega defining the int2 field frequency - N/A
#             epsilon1 - Ellipticity of the int1 field - N/A
#             epsilon2 - Ellipticity of the int2 field - N/A
#             mod_omega - Frequency of the sin2 envelope - N/A
#             mod_phase - Phase of the envelope relative to the phase of the elliptical em waves. - N/A
# params is a list of sets of 10 parameters. Each set of parameters is for a particular relative intensity combination.
# There is only one intensity combination in the linear case 1.0 * intensity.

# User defined params:
wavelength = 900 # nm
potential_well = -0.5 # atomic units
intensity1 = intensity_coef[:,0] * 1e14 # W/cm^2




# Params that are not defined by the user:
omega = 45.56/wavelength
period = 2*np.pi/omega

frequency_integer1 = "nan"
frequency_integer2 = "nan"

electric_field1 = np.sqrt(intensity1/3.509e16)

intensity2 = intensity_coef[:,1]*0 # W/cm^2
electric_field2 = np.sqrt(intensity2/3.509e16)

epsilon1 = "nan" 
epsilon2 = "nan" 

mod_omega = 'nan'
mod_phase = 'nan'

params = np.zeros((len(intensity_coef), 10))
for i in np.arange(len(intensity_coef)):
    params[i] = (omega, potential_well, electric_field1[i], frequency_integer1,
                 electric_field2[i], frequency_integer2, epsilon1, epsilon2, mod_omega, mod_phase)
    
params


# In[4]:

#Generate and organize the time values used in guessing. Recall: The program kills saddle pts with Re[tau] > Period.
#The calculation time scales exponentially with the number of guesses on the grid so use caution with large numbers.

#Enter below the grid size you would like to use:
recombination_grid_size = 30  #Number of guesses for tf
real_tau_grid_size =      15  #Number of guesses for Re[tau]
imag_tau_grid_size =      4   #Number of guesses for Im[tau]


#--


#recombTimes = bounds for program to consider HHG, this program considers a 2*period for recomb (tf)
recombTimes = [0.0, 2.0 * period]  #given in field cycles
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


# In[5]:

# Name the output data. Files will be placed in a directory: data_cw_linear_(filename)

filename = "900nm"


# In[6]:

# User data is all entered ABOVE


# --


# Processing is done BELOW


# In[7]:

# This just makes sure that you don't accidently overwrite something you weren't expecting to.
# The exception is raised if the directory ./data_cw_linear_(filename) already exists, no matter what is inside.

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
    os.mkdir("./data_cw_linear_"+str(filename))
except OSError:
    print("The directory " + "./data_cw_linear_" + str(filename) + " already exists. \n")
    if query_yes_no("Would you like to continue and overwrite the data in the directory?"):
        print("\nOkay, I will overwrite the data in " + "./data_cw_linear_" + str(filename))
        pass
    else:
        print("\nYou have chosen to stop the program. \nRename the variable 'filename' such that "
              + "./data_cw_linear_(filename) does not already exist. \nThe filename is currently: "
              + str(filename) + "\nThe directory that already exists is called: " + "./data_cw_linear_"
              + str(filename))
        raise


# In[9]:

from semiclassicalhhg.linear_supp_eqns import nineteen_cplx_tf, nineteen_cplx_tf_p, position, velocity

from semiclassicalhhg.saddle_pts import *


# In[10]:

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
    
    
    
    xdir = EF1*sin(t*w)
    
    ydir = 0
    
    return [xdir, ydir]


# In[11]:

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


# In[12]:

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


with open('./'+"data_cw_linear_"+str(filename)+'/information', 'w') as info:
    info.write(what_to_say)


# In[13]:

np.save('./'+"data_cw_linear_"+str(filename)+'/params', np.array(params))


# In[14]:

np.save('./'+"data_cw_linear_"+str(filename)+'/intensity_coef', np.array(intensity_coef))


# In[15]:

np.save('./'+"data_cw_linear_"+str(filename)+'/saddle_pts', np.array(quick_pos_uniq_round_vals))


# In[16]:

# Now I will save each intensity ratio test as its own array for easy access.

try:
    os.mkdir("./data_cw_linear_"+str(filename)+"/data_by_intensity_ratio")
except:
    pass

for i, v in enumerate(params):
    if i<9:
        np.savetxt('./'+"data_cw_linear_"+str(filename)+"/data_by_intensity_ratio/0"+str(i)+"_params",
                  np.array(params)[i])
        np.savetxt('./'+"data_cw_linear_"+str(filename)+"/data_by_intensity_ratio/0"+str(i)+"_intensity_coef",
                  np.array(intensity_coef)[i])
        np.savetxt('./'+"data_cw_linear_"+str(filename)+"/data_by_intensity_ratio/0"+str(i)+"_saddle_pts",
                  np.array(quick_pos_uniq_round_vals)[i])
    else:
        np.savetxt('./'+"data_cw_linear_"+str(filename)+"/data_by_intensity_ratio/"+str(i)+"_params",
                  np.array(params)[i])
        np.savetxt('./'+"data_cw_linear_"+str(filename)+"/data_by_intensity_ratio/"+str(i)+"_intensity_coef",
                  np.array(intensity_coef)[i])
        np.savetxt('./'+"data_cw_linear_"+str(filename)+"/data_by_intensity_ratio/"+str(i)+"_saddle_pts",
                  np.array(quick_pos_uniq_round_vals)[i])


# In[ ]:



