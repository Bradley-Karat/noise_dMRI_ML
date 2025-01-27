import numpy as np
import nibabel as nib
import sys
import diffsimrun
from diffsimgen.scripts import models
from diffsimgen.scripts import simulate_signal
from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import LinearRegression
import pickle

data_dir = '/path/to/data'
sim_dir = '/path/to/simulations'

# Load in brain mask, MPPCA sigma, and real data
tmp = nib.load(f'{data_dir}/derivatives/ses-1p5mm/masks/nodif_brain_mask.nii.gz')
masknoflat = np.double(tmp.get_fdata())
mask = np.double(tmp.get_fdata()).flatten(order='F')

tmp = nib.load(f'{sim_dir}/sigma_MPPCA.nii.gz') # This is the estimate of the noise sigma from MPPCA
sigma_mppca_noflat = np.double(tmp.get_fdata())
tmp_img = np.double(tmp.get_fdata()).flatten(order='F')
sigma_mppca = np.transpose(tmp_img[mask==1]) # Only want sigma within the brain mask

tmp = nib.load(f'{sim_dir}/magnitude_noisy_signal.nii.gz') # Load in real magnitude signal
affine = tmp.affine
I = np.double(np.abs(tmp.get_fdata()))

#Load in bvals and bvecs
bvals = np.loadtxt(f'{data_dir}/dwi/sub-01_ses-1p5mm_dir-AP_dwi.bval')
bvecs = np.loadtxt(f'{data_dir}/dwi/sub-01_ses-1p5mm_dir-AP_dwi.bvec')

[sx, sy, sz, vols] = I.shape # Dimensions of real data

FWHM = 0.001
sigma = FWHM/(2*np.sqrt(2*np.log(2)))

I[np.isnan(I)] = 0 # Change all nans to 0 which will get smoothed out with small gaussian filter

print("Loaded data")

dwinorm = np.zeros((sx, sy, sz, vols))
S0mean = np.nanmean(np.double(I[:,:,:,bvals<=100]),axis=3) # The mean of the b0 data

# Normalize dwi data
for mm in range(vols):
    dwinorm[:,:,:,mm] = np.divide(I[:,:,:,mm],S0mean)

# Open simulated training set to get parameters
with open(f'{sim_dir}/{Nsim}_zeppelin_stick_simulations.pkl', 'rb') as f:
    zep_stick_sim = pickle.load(f)

parameters = zep_stick_sim['parameters']
parameter_names = zep_stick_sim['parameter_names']
noiseless_signal = zep_stick_sim['signal_noiseless']

avg_parameters = parameters[:,2:4] # Only want Dpar [2] and stick frac [3], mu gets averaged out via SM
avg_parameter_names = ['parallel_diffusivity','stick_signal_fraction']

bunique = np.unique(bvals)

# Load in noisy signal
with open(f'{sim_dir}/{Nsim}_zeppelin_stick_simulations_signal_with_rician_and_SH.pkl', 'rb') as f:
    signal_with_rician_bias_SH = pickle.load(f)

# Load in trained BR
with open(f'{sim_dir}/{Nsim}_SM_zeppelin_stick_simulations_trained_BR.pkl', 'rb') as f:
    Mdl = pickle.load(f)

# Take spherical mean of noisy simulated signal
bunique = np.unique(bvals)
signal_with_rician_bias_SH_avg = np.empty((Nsim,len(bunique)))
for ii in range(len(bunique)): # Spherical mean across each b-value
    ind = bvals == bunique[ii]
    signal_with_rician_bias_SH_avg[:,ii] = np.mean(signal_with_rician_bias_SH[:,ind],axis=1)

MLprediction = np.zeros((avg_parameters.shape))
# Fitting the slope and intercept between true simulated parameters and parameters estimated with the noisy signal
Slope = np.zeros((avg_parameters.shape[1],1)) 
Intercept = np.zeros((avg_parameters.shape[1],1))

for i in range(avg_parameters.shape[1]): # Each parameter seperately

    # The prediction of the trained BR on the noisy signal
    MLprediction[:,i] = Mdl[i].predict(signal_with_rician_bias_SH_avg)

    # Linear regression between the parameters in the training set and the estimated parameters
    linmodel = LinearRegression()
    X = np.transpose(avg_parameters[:,i]).reshape(-1,1)
    y = np.transpose(MLprediction[:,i]).reshape(-1,1)
    linmodel.fit(X,y)

    Slope[i] = linmodel.coef_
    Intercept[i] = linmodel.intercept_

# Take SM of the normalized real data
dwinorm_avg = np.empty((sx,sy,sz,len(bunique)))
for ii in range(len(bunique)): #spherical mean across each shell
    ind = bvals == bunique[ii]
    dwinorm_avg[:,:,:,ii] = np.mean(dwinorm[:,:,:,ind],axis=-1)

ROI = np.reshape(dwinorm_avg, [sx*sy*sz,len(bunique)])
m = np.reshape(masknoflat, [sx*sy*sz])
signal = (ROI[m==1,:]) # Only voxels in the brain mask

realpredhold = np.zeros((sx*sy*sz))

for i in range(avg_parameters.shape[1]): # For each parameter seperately

    realpred = Mdl[i].predict(signal) # Model prediction on real data
    realpred = (realpred - Intercept[i])/Slope[i] # Debias the prediction - measured y and we want X

    np.place(realpredhold,mask,realpred)
    realpredhold = np.reshape(realpredhold,[sx,sy,sz])

    # Save the debiased inference
    tmpsave = nib.Nifti1Image(realpredhold,affine) 
    nib.save(tmpsave,f'{sim_dir}/with_debias/magnitude_noisy_BR_debiased_SM_prediction_{avg_parameter_names[i]}.nii.gz')