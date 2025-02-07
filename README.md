# noise_dMRI_ML
Code for the paper "The gain of adding noise: Reduced bias and improved tissue microstructure estimation using supervised machine learning"

Includes the following files for the proposed noise injection method specifically to fit a cylinder and zeppelin model:

**llsFitSH.py** - Fit the spherical harmonics (SH) to the real data using linear least squares.

**normalize_noisemap.py** - Normalize the estimated noisemaps using the average b=0 data. 

**RiceMean.py** - Given a signal and a noise sigma, calculate the mean of the Rician distribution.

**SphericalMeanFromSH.py** - Estimate the spherical mean of the real data and the median absolute deviation of the residuals of the SH fit to the real data.

**simulate_data_and_train_BR.py** - Performs the proposed noise injection method, trains the bootstrap-aggregating regressor and performs inference on the real data. The training dataset was built with [DiffSimGen](https://github.com/Bradley-Karat/DiffSimGen)

**add_BR_debias.py** - Perform a linear regression of the estimated parameters and the simulated parameters from the training set. Use the coefficients to debias the estimation from the inference step

For a current practical implementation of the proposed noise injection method, see the [MATLAB](https://github.com/palombom/SANDI-Matlab-Toolbox-Latest-Release) or [Python](https://github.com/Bradley-Karat/SANDI_python_toolbox) implementation of the SANDI toolbox
