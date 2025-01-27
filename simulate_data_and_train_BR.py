import numpy as np
import nibabel as nib
import sys
import diffsimrun
from diffsimgen.scripts import models
from diffsimgen.scripts import simulate_signal
from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues
from scipy.ndimage import gaussian_filter
from SphericalMeanFromSH import SphericalMeanFromSH
from normalize_noisemap import normalize_noisemap
from RiceMean import RiceMean
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
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

for i in range(vols):
    I[:,:,:,i] = gaussian_filter(np.squeeze(I[:,:,:,i]), sigma, mode='nearest')


bunique = np.unique(bvals)

dwinorm = np.zeros((sx, sy, sz, vols))
S0mean = np.nanmean(np.double(I[:,:,:,bvals<=100]),axis=3) # The mean of the b0 data

# Normalize dwi data
for mm in range(vols):
    dwinorm[:,:,:,mm] = np.divide(I[:,:,:,mm],S0mean)

# Identify b-shells and direction-average per shell
estimated_SH_sigma = np.empty((len(bunique),int(np.sum(mask)))) # Averaging all directions within each shell so at each voxel in brain we get len(bunique) direction averaged data
estimated_SH_sigma[:,:] = np.nan
Ndirs_per_shell = np.zeros((1, len(bunique)))
Save = np.empty((sx,sy,sz,len(bunique)))

for i in range(len(bunique)):
    
    Ndirs_per_shell[0,i] = np.sum(bvals==bunique[i])
        
    if i>0: # Not at b=0
        
        dirs = np.transpose(bvecs[:,bvals==bunique[i]])
        Ndir = np.sum(bvals==bunique[i])
        
        # Guidelines from MRtrix for SH order based on # of bvecs
        if Ndir<=15:             
            order = 2
        elif Ndir>15 and Ndir<=28:
            order = 4
        elif Ndir>28 and Ndir<=45:  
            order = 6
        elif Ndir>45:              
            order = 8
            
        print(f"\nFitting SH to the per-shell data with order l= {str(order)} -shell {str(i-1)} of {str(len(bunique)-1)} -directions = {str(np.sum(bvals==bunique[i]))}")

        y_tmp = I[:,:,:,bvals==bunique[i]]
        y_tmp = np.reshape(y_tmp,[sx*sy*sz, np.shape(y_tmp)[-1]],order='F')
        y_tmp = np.transpose(y_tmp)
        indhold = mask==1
        y = y_tmp[:,indhold]
        [Save_tmp, sigma_tmp] = SphericalMeanFromSH(dirs, y, order) # Save_tmp is spherical mean, sigma_tmp is the median absolute deviation of the SH residuals per b-value
        
        estimated_SH_sigma[i,:] = sigma_tmp                         

sigma_SHresiduals = np.nanmean(estimated_SH_sigma,axis=0) # Mean SH residual across all b-values
noisemap = np.zeros((sx*sy*sz, 1))
np.place(noisemap, mask, sigma_SHresiduals) # Place the SH residuals only within the brain mask
sigma_SH_noisemap = np.reshape(noisemap,[sx,sy,sz],order='F')

# Only want sigmas that are greater then 0 and less than 1
ind = np.logical_and(noisemap_SHresiduals>0,noisemap_SHresiduals<1)
noisemap_SHresiduals = noisemap_SHresiduals[ind]

# Saving the brain mask of the SH residuals
tmpsave = nib.Nifti1Image(sigma_SH_noisemap,affine)
nib.save(tmpsave,f'{sim_dir}/SH_MAD_residuals.nii.gz')

noisemap_mppca = normalize_noisemap(sigma_mppca_noflat, I, masknoflat, bvals) # Take the noisemap estimated by MP-PCA denoising and normalize it by dividing it by the mean b=0

# Saving the brain mask of the normalized MPPCA sigma
tmpsave = nib.Nifti1Image(noisemap_mppca,affine)
nib.save(tmpsave,f'{sim_dir}/MPPCA_eroded-mask_normalized_sigma.nii.gz')

# Sample sigma from the distribution of SH residuals
sigma_SHresiduals = noisemap_SHresiduals.flatten()
Nsamples = len(sigma_SHresiduals)
Nsim = 100000 # Size of the training set
sigma_SHresiduals = sigma_SHresiduals[np.random.randint(0, int(Nsamples), [int(Nsim),1], dtype=int)] # Randomly getting Nsim samples of the SH residuals with replacement

# Sample sigma from the distribution of MPPCA sigma
sigma_mppca = noisemap_mppca.flatten()
Nsamples = len(sigma_mppca)
sigma_mppca_sampled = sigma_mppca[np.random.randint(0, int(Nsamples), [int(Nsim),1], dtype=int)] # Randomly getting Nsim samples of the MPPCA sigma with replacement
sigma_mppca = sigma_mppca_sampled

print('Simulating zeppelin stick data')
# Use diffsimgen to get noiseless training set of a directional zeppelin stick model
signal, noiseless_signal, parameters, parameter_names, SNRarr = diffsimrun.diffsimrun(model='zeppelin_stick',bval=bvals,bvec=bvecs,SNR=[50,100],output=f'{sim_dir}/simulate_two_compartment_signal/{Nsim}_zeppelin_stick_simulations.pkl',numofsim=Nsim)

avg_parameters = parameters[:,2:4] # Only want Dpar [2] and stick frac [3] in the training set, that is we mu or the stick-zeppelin orientation gets averaged out before training
avg_parameters[:,0] = avg_parameters[:,0] * 1e9 # Before training, want Dpar on a um2/ms scale
avg_parameter_names = ['parallel_diffusivity','stick_signal_fraction']

# Take noiseless signal and add Rician bias and SH residual
signal_with_rician_bias = np.zeros((Nsim, len(bvals)))
signal_with_rician_bias_SH = np.zeros((Nsim, len(bvals)))

# Adding noise to orientation signal
for i in range(Nsim):
    signal_with_rician_bias[i,:] = RiceMean(noiseless_signal[i,:], sigma_mppca[i]) # Take the mean of a Rician distribution given MPPCA sigma and the directional simulated data
    signal_with_rician_bias_SH[i,:] =  signal_with_rician_bias[i,:] + np.random.normal(0,sigma_SHresiduals[i],size=len(signal_with_rician_bias[i,:])) # To the Rician mean add in the SH residuals sampled from a Gaussian

# Save the noisy simulated signal
with open(f'{sim_dir}/{Nsim}_zeppelin_stick_simulations_signal_with_rician_and_SH.pkl', 'wb') as fp:
    pickle.dump(signal_with_rician_bias_SH, fp)

# Take spherical mean of noisy simulated signal
signal_with_rician_bias_SH_avg = np.empty((Nsim,len(bunique)))
for ii in range(len(bunique)): # Spherical mean across each shell
    ind = bvals == bunique[ii]
    signal_with_rician_bias_SH_avg[:,ii] = np.mean(signal_with_rician_bias_SH[:,ind],axis=1)

n_trees = 200 # Number of trees for the BR
Mdl = []
MLprediction = np.zeros((parameters.shape))

# Training BR to go from spherical mean noisy simulated signal to parameters
print('Training BR')
for i in range(avg_parameters.shape[1]): # Training each parameter seperately

    print(f'BR model {i} of {avg_parameters.shape[1]}')
    BRregress = BaggingRegressor(estimator=DecisionTreeRegressor(),n_estimators=n_trees,bootstrap=True,oob_score=True,n_jobs=-1) # BR from sklearn with a Decision Tree estimator
    Mdl.append(BRregress.fit(signal_with_rician_bias_SH_avg,avg_parameters[:,i])) # Append the fitted BR to Mdl

# Save the trained BR
with open(f'{sim_dir}/{Nsim}_SM_zeppelin_stick_simulations_trained_BR.pkl', 'wb') as fp:
    pickle.dump(Mdl, fp)

# Run inference on real data using the trained models
# First take the SM of the the normalized real data
dwinorm_avg = np.empty((sx,sy,sz,len(bunique)))
for ii in range(len(bunique)):
    ind = bvals == bunique[ii]
    dwinorm_avg[:,:,:,ii] = np.mean(dwinorm[:,:,:,ind],axis=-1)

ROI = np.reshape(dwinorm_avg, [sx*sy*sz,len(bunique)]) # Flatten
m = np.reshape(masknoflat, [sx*sy*sz])
signal = (ROI[m==1,:]) # Only considering voxels within the brain mask

realpredhold = np.zeros((sx*sy*sz))

# Run inference for each parameter seperately
for i in range(avg_parameters.shape[1]):
    realpred = Mdl[i].predict(signal) # BR prediction on real data
    np.place(realpredhold,mask,realpred) # Place prediction within brain mask
    realpredhold = np.reshape(realpredhold,[sx,sy,sz])

    # Save parameter inference
    tmpsave = nib.Nifti1Image(realpredhold,affine) 
    nib.save(tmpsave,f'{sim_dir}/magnitude_noisy_BR_SM_prediction_{avg_parameter_names[i]}.nii.gz')