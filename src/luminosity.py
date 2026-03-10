import numpy as np
from math import cos, pi
import kerrgeopy as kg
from kerrgeopy.units import time_in_seconds
from scipy.optimize import brentq

class Luminosity:
    """
    Compute the dynamic specific luminosity L_E(E, t) of a quasi-periodic eruption
    modeled as an EMRI in an accretion disk in units of ergs/s/keV.
    We use the conversion E = h * nu where nu is the photon frequency.
    """
    # Physical Constants (cgs units)
    G = 6.67430e-8
    Msol = 1.989e33
    c = 2.99792458e10
    mp = 1.67262192e-24
    sigmaSB = 5.670374419e-5
    kb = 1.380649e-16
    h = 6.62607015e-27
    kev_to_ergs = 1.60218e-9

    def __init__(self, tmax: float, dt: float, e_min_kev: float=0.2, e_max_kev: float=3.0, nE: int=200):
        """
        Initializes the grid resolution for the luminosity calculation.
        
        Args:
            tmax (float): Total duration of the simulated time series in seconds.
            dt (float): Time step size in seconds.
            e_min_kev (float): Minimum energy of the observation band in keV.
            e_max_kev (float): Maximum energy of the observation band in keV.
            nE (int): Number of energy bins.
        """
        self.tmax = tmax
        self.dt = dt
        self.nE = nE

        self.t_grid = np.arange(0, tmax, dt, dtype=float)
        self.nT = len(self.t_grid) #number of time points
        self.e_grid_kev = np.linspace(e_min_kev, e_max_kev, nE)
        self.e_grid_ergs = self.e_grid_kev * self.kev_to_ergs

    def __call__(self, a: float, p: float, e: float, inc_deg: float,
                 m1: float, m2: float, m1dot: float, f: float, 
                 Delta_t: float = 3600.0, pli: float = 1.75, eta: float = 0.1, 
                 Rout: float = 300.0) -> np.ndarray:
        """
        Calculates the specific luminosity L_E(E, t) for the given EMRI parameters.
        
        Args:
            a (float): Primary black hole spin parameter.
            p (float): Semi-latus rectum of the EMRI orbit.
            e (float): Orbital eccentricity.
            inc_deg (float): Orbital inclination in degrees.
            m1 (float): Mass of the primary supermassive black hole in solar masses.
            m2 (float): Mass of the secondary orbiting black hole in solar masses.
            m1dot (float): Eddington-scaled mass accretion rate of the primary.
            f (float): Disk mass in solar masses.
            Delta_t (float): Flare expansion timescale in seconds.
            pli (float): Surface density power-law index.
            eta (float): Accretion efficiency.
            Rout (float): outer disk radius in units of M.
            
        Returns:
            np.ndarray: A 2D array of shape (nE, nT) representing the specific 
                        luminosity in ergs/s/keV at each energy and time step.
        """

        # 1. Compute orbits and crossings
        t_crossings, r_crossings, v_rel_mag, c_s, rho = self._compute_orbit_dynamics(
            a, p, e, inc_deg, m1, m2, m1dot, f, pli, eta, Rout
        )

        if len(t_crossings) == 0:
            return np.zeros((self.nE, self.nT))
        
        # 2. Compute initial properties of the cloud
        Rin_list = (self.G * m2 * self.Msol) / (c_s**2 + v_rel_mag**2)
        T1_list = np.array([self._safe_T1_root(r, c) for r, c in zip(rho, c_s)])

        Xlist = ((v_rel_mag/c_s)**2) * ((rho * self.kb * T1_list)/self.mp)/((4.0/3.0)*self.sigmaSB * T1_list**4 / self.c)
        Xlist = np.maximum(Xlist, 1.0)
        T2_list = T1_list * np.power(1.0 + (8.0/7.0)*Xlist - 8.0/7.0, 0.25)
        
        return self._evaluate_luminosity_grid(t_crossings, Rin_list, T2_list, Delta_t)
    
    def _evaluate_luminosity_grid(self, t_crossings, Rin_list, T2_list, Delta_t):
        """
        Vectorized evaluation of the specific luminosity over the (E, t) grid. 
        """

        L_E_2d = np.zeros((self.nE, self.nT)) #iniitalize a grid array

        # Pre-compute energy term for specific luminosity (ergs/s/erg). We substitute E = h * nu, dE = h * dnu
        # L_E = 4*pi*R^2 * (2*h*E^3 / (h^3*c^2)) / (exp(E/kT) - 1) dE/h
        #     = 4*pi*R^2 * (2*E^3 / (h^3*c^2)) / (exp(E/kT)-1) dE
        planck_prefactor = (2.0 * self.e_grid_ergs**3) / ((self.h**3)*(self.c**2))

        for t0, Rin, T2 in zip(t_crossings, Rin_list, T2_list):
            #Only calculate for times during and after the crossing
            mask = self.t_grid >= t0
            t_active = self.t_grid[mask]

            #adiabatic expansion factors
            tau = (t_active-t0)/Delta_t
            R_t = Rin + 2.0* Rin * tau #Eq. 22 in https://arxiv.org/pdf/2304.00775
            T_t = T2 * Rin / R_t #paragraph below Eq. 22 in https://arxiv.org/pdf/2304.00775

            #calculate exponent matrix: shape (nE, n_active_t)
            exponent = self.e_grid_ergs[:, None] / (self.kb * T_t[None,:])
            exponent = np.clip(exponent, None, 700) #prevent overflow

            #specific luminosity in ergs/s/erg
            L_E_burst_ergs = 4.0 * np.pi * (R_t[None,:]**2) * planck_prefactor[:, None] / (np.exp(exponent)-1.0) #Eq. (23) in https://arxiv.org/pdf/2304.00775

            #Convert to ergs/s/keV
            L_E_burst_keV = L_E_burst_ergs * self.kev_to_ergs #* 1e7 #TODO: ask Khurshid why there is a factor of 1e7 here.

            L_E_2d[:, mask] += L_E_burst_keV

        return L_E_2d
        
    def _compute_orbit_dynamics(self, a, p, e, inc_deg, m1, m2, m1dot, f, pli, eta, Rout):
        """
        kerrgeopy orbital integration and identification of disk crossing events
        Returns:
            tcross_sec (np.1darray): crossing times in seconds
            rcross (np.1darray): radius at crossings
            vrelmag (np.1darray): relative velocitity magnitude at crossings
            cs (np.1darray): speed of sound at crossings
            rho (np.1darray): disk midplane volume density Sigma / (2 * pi * H) at crossings
        """
        x_val = cos(np.deg2rad(inc_deg))
        x_val = np.sign(x_val) * max(abs(x_val), 1e-6) #minimum x_val
        orbit = kg.StableOrbit(a, p, e, x_val, M=m1, mu=m2)

        toflam, roflam, thoflam, phioflam = orbit.trajectory()

        #Mino time samples
        lam0 = 10.0 # initial Mino time
        bound = 3.6e4 #upper bound on Mino time TODO: how was this bound selected?
        while True: #loop to get all Mino time steps
            t_geom = toflam(lam0)
            t_sec = time_in_seconds(t_geom, m1)
            if (t_sec >= self.tmax) or (lam0 >= bound):
                break
            lam0 *= 2 #TODO: Ask Khurshid why is it being multiplied by 2?

        lam_samples = np.linspace(0, lam0, 10_000+1)
        tlist_sec = time_in_seconds(toflam(lam_samples), m1)

        mask = tlist_sec <= self.tmax
        lam_samples = lam_samples[mask]

        if lam_samples.size == 0:
            return [], [], [], [], []
        
        if lam_samples.size >= 2:
            lam_samples = np.append(lam_samples, lam_samples[-1] + (lam_samples[-1] - lam_samples[-2]))

        #Find crossings
        lam = np.linspace(lam_samples[0], lam_samples[-1], 200_000+1)
        th = thoflam(lam)
        g = th - np.pi/2 #crossings restricted to the equatorial plane for now. 
        s = np.sign(g)
        idx = np.where(s[:-1]*s[1:] < 0)[0] #identify the turning point by the sign of the angle
        
        lam_cross = []
        for i in idx:
            if not (np.isfinite(g[i]) and np.isfinite(g[i+1])): continue
            if g[i] == 0.0: lam_cross.append(lam[i]); continue
            if g[i+1] == 0.0: lam_cross.append(lam[i+1]); continue

            a_idx, b_idx = lam[i], lam[i+1] #rootfinding bounds
            try:
                r = brentq(lambda L:thoflam(L)-(np.pi/2), a_idx, b_idx, maxiter=100)
                lam_cross.append(r)
            except:
                r = a_idx - g[i] * (b_idx - a_idx) / (g[i+1] - g[i])
                lam_cross.append(r)

        lam_cross = np.array(lam_cross)
        if len(lam_cross) == 0:
            return [], [], [], [], []
        
        tcross_sec = time_in_seconds(toflam(lam_cross), m1)
        rcross = roflam(lam_cross)
        phicross = phioflam(lam_cross)

        #get velocities
        utoflam, uroflam, uthoflam, uphioflam = orbit.four_velocity()
        ut, ur, uth, uph = utoflam(lam_cross), uroflam(lam_cross), uthoflam(lam_cross), uphioflam(lam_cross)

        cosphi, sinphi = np.cos(phicross), np.sin(phicross)
        vx = (cosphi*(ur/ut)-rcross*sinphi*(uph/ut)) * self.c
        vy = (sinphi*(ur/ut)+rcross*cosphi*(uph/ut)) * self.c
        vz = -(rcross*(uth/ut))*self.c

        Rg = (self.G * m1 * self.Msol) / (self.c**2)
        Rcross = rcross * Rg #Rcross in metric units. 
        vgasmod = np.sqrt(self.G * m1 * self.Msol / Rcross)

        vrelx = (-vgasmod*sinphi) - vx
        vrely = (vgasmod*cosphi) - vy
        vrelz = 0.0 - vz
        vrelmag = np.sqrt(vrelx**2 + vrely**2 + vrelz**2)

        #get disk parameters at crossing
        Md = f * self.Msol #disk mass in metric units
        Risco = self._r_ISCO(a, prograde=(True if np.sign(a) >= 0 else False))
        Kr = 0.5 + np.sqrt(0.25 + 6.0 * (m1dot/eta)**2 * rcross**-2)
        
        # !!! IMPORTANT: Eq. (2) in https://arxiv.org/pdf/2304.00775 has a typo: the radial scaling term should be in the denominator. 
        # Md = \int_Risco^Rout \Sigma(R) 2\pi R dR where we have \Sigma(R) = \Sigma0(R/Rg)^-p. Substituting,
        # Md = 2\pi\Sigma0 \int_Risco^Rout (R/Rg)^-p R dR
        # Md = 2\pi\Sigma0 Rg^p [R^(2-p)/(2-p)]_Risco^Rout
        # Md = 2\pi\Sigma0 Rg^2 / (2-p) [(Rout/Rg)^(2-p) - (Risco/Rg)^(2-p)]
        # If we now invert this equation in terms of \Sigma0, we will get
        # \Sigma0 = Md(2-p)/((2\pi Rg^2)[(Rout/Rg)^(2-p) - (Risco/Rg)^(2-p)])
        Sigma0 = (Md * (2-pli)) / (2*np.pi * Rg**2) / (Rout**(2-pli) - Risco**(2-pli)) 
        Sigma = Sigma0 * rcross ** (-pli) #power-law model for the disk profile.

        H = Rcross * (3/2) * np.sqrt(2*np.pi) * eta**-1 * Kr**-1 * m1dot * rcross**-1 #disk height at crossings
        rho = Sigma / (np.sqrt(2*np.pi)*H) #midplane volume density
        cs = (3/2) * np.sqrt(2*np.pi) * eta**-1 * self.c * Kr**-1 * m1dot * rcross**(-1.5)

        return tcross_sec, rcross, vrelmag, cs, rho


    def _r_ISCO(self, a, prograde=True):
        """
        Radius of the innermost stable orbit for a given dimensionless spin a.
        """
        z1 = 1 + (1 - a*a)**(1/3) * ((1 + a)**(1/3) + (1 - a)**(1/3))
        z2 = (3*a*a + z1*z1)**0.5
        if prograde:
            return 3 + z2 - ((3 - z1)*(3 + z1 + 2*z2)**0.5)
        else:
            return 3 + z2 + ((3 - z1)*(3 + z1 + 2*z2)**0.5)
        
    def _T1_scalar(self, T, rho_i, cs_i):
        term1 = (3.0 * rho_i * cs_i**2) / 4.0
        term2 = (4.0/3.0) * (self.sigmaSB * T**4) / self.c
        term3 = (rho_i * self.kb * T) / self.mp
        return term1 - term2 - term3
        
    def _safe_T1_root(self, rho_i, cs_i, T_lo=1e3, T_hi=1e9, maxiter=200):
        f_lo = self._T1_scalar(T_lo, rho_i, cs_i)
        f_hi = self._T1_scalar(T_hi, rho_i, cs_i)
        if np.isfinite(f_lo) and np.isfinite(f_hi) and f_lo*f_hi < 0:
            return brentq(lambda T: self._T1_scalar(T, rho_i, cs_i), T_lo, T_hi, maxiter=maxiter)
        for k in range(6):
            T_hi2 = T_hi * (10**(k+1))
            f_hi2 = self._T1_scalar(T_hi2, rho_i, cs_i)
            if np.isfinite(f_hi2) and f_lo*f_hi2 < 0:
                return brentq(lambda T: self._T1_scalar(T, rho_i, cs_i), T_lo, T_hi2, maxiter=maxiter)
        return 1.0e6





