import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.integrate import simpson

class DetectorFold:
    """
    Fold a 2D specific luminosity array through a detector's effective area curve.
    Calculate the expected count rates and Poisson errors.
    """
    def __init__(self, arf_file_path: str, e_grid_kev: np.ndarray):
        """
        Args:
            arf_file_path (str): Path to the detector's standard .arf FITS file.
            e_grid_kev (np.ndarray): The 1D array of energies (in keV) used to 
                                     generate your L_E_2d array.
        """
        self.e_grid_kev = e_grid_kev
        self.kev_to_ergs = 1.60218e-9
        self.e_grid_ergs = self.e_grid_kev * self.kev_to_ergs

        # 1. Read the effective area curve from Ancillary Response Files
        with fits.open(arf_file_path) as hdul:
            data = hdul[1].data
            e_lo = data["ENERG_LO"]
            e_hi = data["ENERG_HI"]
            a_eff_raw = data["SPECRESP"] #Effective area in cm2

        # 2. Define the ARF energies at the bin midpoints
        e_mid = (e_lo + e_hi) / 2.0

        # 3. Interpolate the ARF effective area onto your theoretical energy grid
        # bounds_error=False and fill_value=0.0 ensures that if your grid extends 
        # beyond the detector's sensitivity, the area drops cleanly to 0.
        interp_func = interp1d(e_mid, a_eff_raw, bounds_error=False, fill_value=0.0)
        self.a_eff_grid = interp_func(self.e_grid_kev)

    def calculate_rates_and_errors(self, L_E_2d: np.ndarray, distance_cm: float,
                                   dt: float, bg_rate: float=0.0):
        """
        Args:
            L_E_2d (np.ndarray): 2D array of shape (nE, nT) in ergs/s/keV.
            distance_cm (float): Luminosity distance to the source D_L in cm.
            dt (float): The time step size of your simulation in seconds.
            bkg_rate (float): Optional background count rate of the detector (counts/s).
            
        Returns:
            count_rate (np.ndarray): 1D array of source count rates (counts/s).
            rate_error (np.ndarray): 1D array of 1-sigma errors on the count rate.
        """
        nT = L_E_2d.shape[1]
        count_rate = np.zeros(nT)

        # Geometric factor cm^-2
        geometric_factor = 1.0 / (4.0 * np.pi * distance_cm**2)

        # loop over time steps to integrate the spectrum
        for i in range(nT):

            #specific luminosity at this time-step
            L_E = L_E_2d[:, i]

            # convert specific luminosity (erg/s/keV) to photon flux density
            # (photons/s/cm^2/keV) by dividng by the photon energy in ergs
            photon_flux_density = geometric_factor * (L_E / self.e_grid_ergs)

            # Multiply by the energy-dependent effective area (cm^2)
            integrand = photon_flux_density * self.a_eff_grid

            # Integrate over the energy band using Simpson's rule
            count_rate[i] = simpson(y=integrand, x=self.e_grid_kev)

        # Calculate Poisson Errors
        total_variance = (count_rate * dt) + (bg_rate * dt)
        rate_error = np.sqrt(total_variance) / dt

        return count_rate, rate_error