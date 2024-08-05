#!/usr/bin/env python
# coding: utf-8

# In[1]:


######################################################################################################
##                                                                                                  ##
##            Code to convert ShapeFit compressed parameters to cosmological parameters             ##
##                                                                                                  ##
##                                  Author: Hernan E. Noriega                                       ##                                  
##                                                                                                  ##
######################################################################################################

import sys, os
import numpy as np
from scipy.interpolate import CubicSpline
from EH98_funcs import*
from classy import Class
import json
import emcee
from schwimmbad import MPIPool
import scipy.constants as conts


# In[2]:


# Inputs:

# Paths for data
# means and cov should be ordered as: a_perp, a_par, f*sigma_8, m

means_paths = {
    'z_0p8': 'means_SF_LRGz0p8_Cov25_MinF_kmax018_l02.dat'
}

cov_paths = {
    'z_0p8': 'covparams_SF_LRGz0p8_Cov25_MinF_kmax018_l02.dat'
}


# Redshifts
z_pk = [0.8]

shift_template = 1.00# 0%

# Priors
prior_file = 'priors.json'


# Output files
output_directory = 'Outputs'
os.makedirs(output_directory, exist_ok=True)
filename = f'{output_directory}/Chain_CosmoParams_SF_LRGz0p8_Cov25_MinF_kmax018_l02.h5'


# In[3]:


class LoadData:
    """
    A class for loading and managing data and covariance matrices from multiple paths.
    
    Args:
        means_paths (dict, optional): A dictionary where keys are names and values are file paths for means data.
        cov_paths (dict, optional): A dictionary where keys are names and values are file paths for covariance matrices.
    """
    
    def __init__(self, means_paths=None, cov_paths=None):
        self.means = {}
        self.cov_inverse = {}
        
        if means_paths:
            for name, path in means_paths.items():
                self.load_means(name, path)
        
        if cov_paths:
            for name, path in cov_paths.items():
                self.load_covariance(name, path)
    
    def load_means(self, name, path):
        try:
            self.means[name] = np.loadtxt(path, unpack=True)
        except FileNotFoundError:
            print(f'ERROR: File not found for {name}')
    
    def load_covariance(self, name, path):
        try:
            cov_matrix = np.loadtxt(path, unpack=True)
            cov_inverse = np.linalg.inv(cov_matrix)
            self.cov_inverse[name] = cov_inverse
        except FileNotFoundError:
            print(f'ERROR: File not found for {name}')
    
    def get_means(self, name):
        return self.means.get(name, None)
    
    def get_covariance_inverse(self, name):
        return self.cov_inverse.get(name, None)
    
    def calculate_residuals(self, param_array):
        stacked_means = []  
        for name, path in means_paths.items():
            means = self.get_means(name)
            if means is not None:
                stacked_means.append(means)
        residuals = param_array - np.vstack(stacked_means)
        return residuals


# In[4]:


class SetCosmologies:
    """
    A class to set cosmological parameters and compute observables using fiducial and reference cosmologies.

    Fiducial cosmology: Used to convert redshift into physical distances.
    Reference cosmology: Used during the compressed step.

    Args:
        params_fid (dict, optional): A dictionary of fiducial parameter values.
        params_ref (dict, optional): A dictionary of reference parameter values.
            Both dictionaries should have the same keys as the default fiducial parameters.
            Defaults to None for both.

    Attributes:
        default_params (dict): The default fiducial cosmological parameters (AbacusSummit c000).
        fiducial_cosmo (Class): An instance of the Class cosmology calculator with fiducial parameters.
        reference_cosmo (Class): An instance of the Class cosmology calculator with reference parameters.

    Methods:
        calculate_fiducial(z_pk=None):
            Calculate cosmological observables using the fiducial parameter values.

        calculate_reference(z_pk=None):
            Calculate cosmological observables using the reference parameter values.
    """

    def __init__(self, params_fid=None, params_ref=None):
        # Default fiducial parameters: AbacusSummit c000
        self.default_params = {
            'output': 'mPk',
            'omega_b': 0.02237,
            'omega_cdm': 0.1200,
            'omega_ncdm': 0.00064420,
            'h': 0.6736,
            'A_s': 2.0830e-9,
            'n_s': 0.9649,
            'P_k_max_1/Mpc': 1.0,
            'z_max_pk': 10.0,
            'N_ur': 2.0328,
            'N_ncdm': 1
        }

        # If custom parameters are provided, update the default values
        if params_fid:
            self.default_params.update(params_fid)

        # Initialize the fiducial cosmology calculator and compute
        self.cosmo_fid = Class()
        self.cosmo_fid.set(self.default_params)
        self.cosmo_fid.compute()

        # Default reference parameters: You can modify these as needed
        self.reference_params = self.default_params.copy()

        # If custom parameters are provided, update the default values
        if params_ref:
            self.reference_params.update(params_ref)

        # Initialize the reference cosmology calculator and compute
        self.cosmo_ref = Class()
        self.cosmo_ref.set(self.reference_params)
        self.cosmo_ref.compute()
        
        
        self.rs_ref = self.cosmo_ref.rs_drag()
        
        self.krange = np.logspace(-2.0, 0.0, 300) #h/Mpc
        self.k_pivot = 0.03                       #h/Mpc 
        
            
    def calculate(self, z_pk=None):
        results = []

        if z_pk:
            for z in z_pk:
                DA_fid = self.cosmo_fid.angular_distance(z) * (conts.c / 100000.0 / self.cosmo_fid.h())**(-1)
                H_fid = self.cosmo_fid.Hubble(z) * conts.c / 100000.0 / self.cosmo_fid.h()
                
                sigma8_ref = self.cosmo_ref.sigma(R = 8.0/self.cosmo_ref.h(), z = z)
                Amp_ref = np.sqrt(self.cosmo_ref.pk_cb(self.k_pivot*self.cosmo_ref.h(), z)*self.cosmo_ref.h()**3)
                transfer_ref = EH98(kvector=self.krange, redshift=z, scaling_factor=1.0, cosmo=self.cosmo_ref)*self.cosmo_ref.h()**3

                results.append({'z_' + str(z): z, 'DA_fid': DA_fid, 'H_fid': H_fid,
                                'rs_ref': self.rs_ref,
                                'sigma8_ref':sigma8_ref, 'Amp_ref':Amp_ref,
                                'transfer_ref':transfer_ref})
        else:
            print('ERROR: Please provide redshift(s) (z_pk)')
            
        return results


# In[5]:


def interp(k, x, y):
    '''Cubic spline interpolation.
    
    Args:
        k: coordinates at which to evaluate the interpolated values.
        x: x-coordinates of the data points.
        y: y-coordinates of the data points.
    Returns:
        Cubic interpolation of ‘y’ evaluated at ‘k’.
    '''
    inter = CubicSpline(x, y)
    return inter(k) 


# In[6]:


def slope_at_x(xvector,yvector):
    #find the slope
    diff = np.diff(yvector)/np.diff(xvector)
    diff = np.append(diff,diff[-1])
    return diff


# In[7]:


class ShapeFitConversion:
    """
    A class for converting ShapeFit compressed parameters {a_perp, a_par, fsigma8, m} to cosmological parameters.

    Args:
        fiducial_cosmology : An instance of the SetCosmologies class.
        reference_cosmology : An instance of the SetCosmologies class.

    Methods:
        cosmofit(self, h, omega_cdm, omega_b, logAs, z_pk=[0.8, 1.1]):
            Convert compressed parameters to cosmological parameters and calculate observables.

            Args:
                h (float): The Hubble parameter.
                omega_cdm (float): Density parameter for cold dark matter.
                omega_b (float): Density parameter for baryonic matter.
                logAs (float): Logarithm of the amplitude of primordial fluctuations.
                z_pk (list, optional): A list of redshift values at which to calculate observables.

            Returns:
                list: A list of dictionaries containing observables for each redshift value.
    """

    def __init__(self, cosmologies):
        self.cosmologies = cosmologies
        self.cosmo_ref = cosmologies.cosmo_ref
        self.cosmo_fid = cosmologies.cosmo_fid
    
        self.krange = np.logspace(-2.0, 0.0, 300) #h/Mpc
        self.k_pivot = 0.03                       #h/Mpc 
        

    def cosmofit(self, h, omega_cdm, omega_b, logAs, z_pk=z_pk):
        # Massive neutrinos and spectral index from reference cosmology
        omega_ncdm = self.cosmo_ref.Omega_nu * self.cosmo_ref.h()**2     
        n_s = self.cosmo_ref.n_s()  
                
        cosmo_fid = self.cosmo_fid
        cosmo_ref = self.cosmo_ref
        
        
        # Initialize the cosmology and compute
        params = {
            'output': 'mPk',
            'omega_b': omega_b,
            'omega_cdm': omega_cdm,
            'omega_ncdm': omega_ncdm,
            'h': h,
            'A_s': np.exp(logAs) / 10**10,
            'n_s': n_s,
            'P_k_max_1/Mpc': self.cosmologies.reference_params['P_k_max_1/Mpc'],
            'z_max_pk': self.cosmologies.reference_params['z_max_pk'],
            'N_ur': self.cosmologies.reference_params['N_ur'],
            'N_ncdm': self.cosmologies.reference_params['N_ncdm']
        }

        # Initialize the cosmology and compute
        cosmology = Class()
        cosmology.set(params)
        cosmology.compute()

        rs = cosmology.rs_drag()

        results_exp = []
        
        #print(params)
        

        for z in z_pk:
            # Angular distance, Hubble & drag at new cosmo
            DA = cosmology.angular_distance(z) * (conts.c / 100000.0 / cosmology.h())**(-1)
            H = cosmology.Hubble(z) * conts.c / 100000.0 / cosmology.h()
            
            #fiducial (DA, H) & reference (rs) for a_perp & a_par
            DA_fid = cosmo_fid.angular_distance(z) * (conts.c / 100000.0 / cosmo_fid.h())**(-1)
            H_fid = cosmo_fid.Hubble(z) * conts.c / 100000.0 / cosmo_fid.h()
            rs_ref = self.cosmologies.rs_ref

            a_perp = (DA * rs_ref * cosmo_ref.h()) / (DA_fid * rs * cosmology.h())
            a_par = (H_fid * rs_ref * cosmo_ref.h()) / (H * rs *cosmology.h())

            #reference for fs8 and m
            if cosmo_fid.rs_drag() == cosmo_ref.rs_drag():
                sigma8_ref = cosmo_ref.sigma(R = (8.0/cosmo_ref.h()), z = z)
            else:
                sigma8_ref = cosmo_ref.sigma(R = (8.0/cosmo_ref.h()) * (cosmology.rs_drag()/cosmo_ref.rs_drag()), z = z)
                
            Amp_ref = np.sqrt(cosmo_ref.pk_cb(self.k_pivot*cosmo_ref.h(), z)*cosmo_ref.h()**3)
            transfer_ref = EH98(kvector=self.krange, redshift=z, scaling_factor=1.0, cosmo=cosmo_ref)*cosmo_ref.h()**3
            
            #cosmo
            f_cosmo = cosmology.scale_independent_growth_factor_f(z)
            
            Amp_cosmo = np.sqrt(cosmology.pk_cb(self.k_pivot * cosmo_ref.h() * (cosmo_ref.rs_drag()/cosmology.rs_drag()), z) * (cosmo_ref.h()*cosmo_ref.rs_drag()/cosmology.rs_drag())**3)

            fsigma8 = f_cosmo * sigma8_ref * (Amp_cosmo/Amp_ref)
            
            transfer_cosmo = EH98(kvector=self.krange*cosmo_ref.h()*rs_ref/(rs*cosmology.h()), redshift=z , scaling_factor=1.0, cosmo=cosmology)*(rs_ref/rs)**3
            ratio_Pk_diff = slope_at_x(np.log(self.krange), np.log(transfer_cosmo/transfer_ref))
            m_slope = interp(self.k_pivot, self.krange, ratio_Pk_diff)
                        
            results_exp.append({'z': z, 'a_perp': a_perp, 'a_par': a_par, 
                                'fsigma8':fsigma8, 
                                'm':m_slope})

        return results_exp


# In[8]:


def log_prior(theta):
    ''' The natural logarithm of the prior probability. '''
    
    # Define priors_data as a class attribute
    with open(prior_file, 'r') as json_file:
        priors_data = json.load(json_file)
    
    # Check if parameters are within defined ranges
    for i, param_range in enumerate(priors_data.values()):
        if not (param_range["min"] < theta[i] < param_range["max"]):
            return -np.inf  # Outside the prior range

    # Gaussian prior on omega_b
    omega_b_mean = priors_data["omega_b"]["gaussian_mean"]
    omega_b_std_dev = priors_data["omega_b"]["gaussian_std_dev"]
    lp = -0.5 * ((theta[2] - omega_b_mean) / omega_b_std_dev) ** 2  # theta[2]:omega_b

    return lp


# In[9]:


def log_likelihood(theta):
    '''The natural logarithm of the likelihood.'''
    
    # unpack the model parameters
    h, omega_cdm, omega_b, logAs = theta
    
    # Define priors_data as a class attribute
    with open(prior_file, 'r') as json_file:
        priors_data = json.load(json_file)
    
    # Check if parameters are within defined ranges: prevent CLASS from crashing
    for i, param_range in enumerate(priors_data.values()):
        if not (param_range["min"] < theta[i] < param_range["max"]):
            return -10e10  # Outside the prior range
 
                
    # Calculate StandardConversion for the given parameter values
    param_results = shapefit_conversion.cosmofit(h, omega_cdm, omega_b, logAs)
        
    # array in the order: a_perp, a_par, fsigma8
    param_array = np.array([[result['a_perp'], result['a_par'], result['fsigma8'], result['m']] for result in param_results])
                
    # compute the residuals: model - data
    residuals = data_loader.calculate_residuals(param_array)
        
    log_like = 0.0
        
    for name in data_loader.cov_inverse.keys():
        cov_inv = data_loader.get_covariance_inverse(name)
        #if cov_inv is None:
        #    continue
                
        index = list(data_loader.cov_inverse.keys()).index(name)
        residual_for_name = residuals[index]
         
        log_like += -0.5 * (residual_for_name.T @ cov_inv @ residual_for_name)
        
    return log_like


# In[10]:


def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)


# In[11]:


def Sampler(filename = None, start0=None, nsteps = 500000):
    if start0 is None:
        # Default value for start0
        start0 = np.array([0.6760,   #h
                           0.1186,   #ocdm
                           0.0223,   #ob 
                           3.0322    #logAs
                           ])

    ndim = len(start0)
    nwalkers = 2 * ndim

    start = np.array([start0 + 1e-3 * np.random.rand(ndim) for i in range(nwalkers)])

    # Set up the backend and convergence parameters
    backend = emcee.backends.HDFBackend(filename)
    max_n = nsteps
    
    # We'll track how the average autocorrelation time estimate changes
    index = 0
    autocorr = np.empty(max_n)

    # This will be useful to testing convergence
    old_tau = np.inf

    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, backend=backend, pool=pool)
    
        # Now we'll sample for up to max_n steps
        for sample in sampler.sample(start, iterations=max_n, progress=True):
            # Only check convergence every 100 steps
            if sampler.iteration % 100:
                continue
            
            
            # Compute the autocorrelation time so far
            # Using tol=0 means that we'll always get an estimate even
            # if it isn't trustworthy
            tau = sampler.get_autocorr_time(tol=0)
            autocorr[index] = np.mean(tau)
            index += 1
        
            # Check convergence
            converged = np.all(tau * 100 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            if converged:
                break
            old_tau = tau


    np.savetxt('autocorr_index'+str(index)+'.dat', np.transpose([autocorr[:index]]), 
           header = 'index ='+str(index)+',  mean_autocorr')



    print(
    "Mean acceptance fraction: {0:.3f}".format(
        np.mean(sampler.acceptance_fraction)
        )
    )

    print('All computations have been completed successfully')


# In[12]:


data_loader = LoadData(means_paths=means_paths, cov_paths=cov_paths) 

cosmologies = SetCosmologies(params_ref={'h':0.6736 * shift_template,
                                         'omega_cdm': 0.1200 * shift_template,
                                         'A_s': 2.0830e-9 * shift_template})

shapefit_conversion = ShapeFitConversion(cosmologies=cosmologies)

if __name__ == "__main__":
    Sampler(filename=filename)  # Calling Sampler with default start0 value


# In[ ]:




