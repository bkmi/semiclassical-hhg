These scripts were designed to be run with python version:
Python 3.4.3 :: Anaconda 2.1.0 (x86_64)

They were written in Jupyter iPython Notebook and converted to .py script files after debugging. To run these programs I would recommend installing the anaconda package and creating a python3 environment. Information about how to do this can be found:

https://www.continuum.io/sites/default/files/Anaconda-Quickstart.pdf
https://www.continuum.io/content/python-3-support-anaconda


Installing the most recent version of anaconda created the python3 environment for me with the command python3.

Run python3 with the command:
python3

Run a script in python3 with the command:
python3 script.py




-



The general idea of this program is that you modify the relevant .py script such that it searches for the kinds of saddle points for are looking for. You can do this by selecting the properly named script for what kind of polarization combination you would like. The program searches for saddle points given linear polarization, 2 color elliptical field mixing, and pulsed 2 color elliptical field mixing. Open the script and modify its parameters so you are studying the field combination you are interested in. You can also edit the number of saddle points to look for. Note that looking for more saddle points causes exponential growth of calculation time.

One important variable to consider is the “intensity_coef” variable. Just like in milosevic, this program will combine two color fields with differing intensity ratios. The program is designed to test a variety of intensity ratios with a single run. If you look for the section of the program that contains the definition of the “intensity_coef” variable you will learn more about how to do this. An example is given within the program.

After you have run the script a folder will be generated containing the data you’re interested in. The outer folder contains a document called information which will tell you about the trial. The data is contained as .npy binaries. Within another folder called ./(data_folder)/data_by_intensity_ratio the saved arrays are listed as .txt files. The .txt files are named with this in mind:
params contains the variables that defined the trial.
intensity_coef tells you about which intensity ratio was tested.
saddle_pts shows the saddle points found for a particular parameter set and intensity ratio.



—




access_data_sample.py

This program shows how to access a particular dataset from within python. The data can be saved in another format, or plotted, from there. Simply run the program and it will give you information about the data you have collected and where to find it! If you create new data by running one of the scripts you must modify “access_data_sample.py” such that it looks for the directory you’ve created with the new data in it. (modify the directory variable!)





Sin^2 Pulsed Elliptical Field Mixing:

ellip_sin2_pulse_find_saddlept.py

This is a script that finds saddle points given two color field mixing with a sin^2 envelope. All user data is entered above line 173.

The script considers the parameters entered by the user then creates a folder called:
./data_ellip_sin2_pulse_(filename)
where filename is determined by the user by altering the filename variable.

Five files/dir are generated inside the folder called:
information		(Summary of the test that produced these data.)
intensity_coef		(List of each relative intensity for each run.)
params			(List of each parameter set for each run.)
saddle_pts		(List of sets of saddle points.)
data_by_intensity_ratio (Lists of data for each intensity ratio test that was run in .txt)

There is a sample program that accesses these data in the package. It should explain how the data is organized more clearly.

The data can be accessed by .npy binary files from within python, or it can be accessed by intensity ratio in the folder “./data_ellip_sin2_pulse_(filename)/data_by_intensity_ratio”. If you would like to see which intensity ratios have been tested the data is saved as “##_intensity_coef” or can be accessed by using the “access_data_sample.py”. If you create new data by running one of the scripts you must modify “access_data_sample.py” such that it looks for the directory you’ve created with the new data in it. (modify the directory variable!)

The .txt files within /data_by_intensity_ratio were arranged with this in mind:
##_ prefix relates a number to an intensity ratio. 00 was the first intensity ratio tested, 01 was the second, and so on…
params contains the variables that defined the trial.
intensity_coef tells you about which intensity ratio was tested.
saddle_pts shows the saddle points found for a particular parameter set and intensity ratio.



Pulsed Elliptical Field Functions (Mathematica Notebook)

The relevant notebook where I calculated the functions used in
./semiclassicalhhg/pulsed_supp_eqns.pyc
Everything should be fine, but just in case you can see how I got the equations.





Constant Wave Elliptical Field Mixing:


cw_ellip_find_saddlept.py

This is a script that finds saddle points given two color field mixing without an envelope. All user data is entered above line ~175.

The script considers the parameters entered by the user then creates a folder called:
./data_cw_ellip_(filename)
where filename is determined by the user by altering the filename variable.

Five files/dir are generated inside the folder called:
information		(Summary of the test that produced these data.)
intensity_coef		(List of each relative intensity for each run.)
params			(List of each parameter set for each run.)
saddle_pts		(List of sets of saddle points.)
data_by_intensity_ratio (Lists of data for each intensity ratio test that was run in .txt)

There is a sample program that accesses these data in the package. It should explain how the data is organized more clearly.

The data can be accessed by .npy binary files from within python, or it can be accessed by intensity ratio in the folder ./data_cw_ellip_(filename)/data_by_intensity_ratio . If you would like to see which intensity ratios have been tested the data is saved as “##_intensity_coef” or can be accessed by using the “access_data_sample.py”. If you create new data by running one of the scripts you must modify “access_data_sample.py” such that it looks for the directory you’ve created with the new data in it. (modify the directory variable!)

The .txt files within /data_by_intensity_ratio were arranged with this in mind:
##_ prefix relates a number to an intensity ratio. 00 was the first intensity ratio tested, 01 was the second, and so on…
params contains the variables that defined the trial.
intensity_coef tells you about which intensity ratio was tested.
saddle_pts shows the saddle points found for a particular parameter set and intensity ratio.


CW Elliptical Field Functions  (Mathematica Notebook)

The relevant notebook where I calculated the functions used in
./semiclassicalhhg/cw_ellip_supp_eqns.pyc
Everything should be fine, but just in case you can see how I got the equations.





Constant Wave Linear Polarization:

cw_linear_find_saddlept.py

This is a script that finds saddle points given linear polarization (in the x direction). All user data is entered above line ~175.

The script considers the parameters entered by the user then creates a folder called:
./data_cw_linear_(filename)
where filename is determined by the user by altering the filename variable.

Five files/dir are generated inside the folder called:
information		(Summary of the test that produced these data.)
intensity_coef		(List of each relative intensity for each run.) (This isn’t relevant for linear polarization because the clef is always [1, 0])
params			(List of each parameter set for each run.)
saddle_pts		(List of sets of saddle points.)
data_by_intensity_ratio (Lists of data for each intensity ratio test that was run in .txt)

The data can be accessed by .npy binary files from within python, or it can be accessed by intensity ratio in the folder ./data_cw_linear_(filename)/data_by_intensity_ratio . If you would like to see which intensity ratios have been tested the data is saved as “##_intensity_coef” or can be accessed by using the “access_data_sample.py”. If you create new data by running one of the scripts you must modify “access_data_sample.py” such that it looks for the directory you’ve created with the new data in it. (modify the directory variable!) Recall that the with linear polarization the intensity ratio doesn’t really apply.

The .txt files within /data_by_intensity_ratio were arranged with this in mind:
##_ prefix relates a number to an intensity ratio. 00 was the first intensity ratio tested, 01 was the second, and so on… (Only 00_ for linear)
params contains the variables that defined the trial.
intensity_coef tells you about which intensity ratio was tested.
saddle_pts shows the saddle points found for a particular parameter set and intensity ratio.


CW Linear Field Functions  (Mathematica Notebook)

The relevant notebook where I calculated the functions used in
./semiclassicalhhg/linear_supp_eqns.py
Everything should be fine, but just in case you can see how I got the equations.





Pulsed Linear Polarization: