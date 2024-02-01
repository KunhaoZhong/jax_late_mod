import jax.numpy as np
from jax.experimental.ode import odeint
from jax.tree_util import register_pytree_node_class

import jax_cosmo.constants as const
from jax_cosmo.utils import a2z
from jax_cosmo.utils import z2a

__all__ = ["Cosmology"]


@register_pytree_node_class
class Cosmology:
    def __init__(self, Omega_c, Omega_b, h, n_s, sigma8, Omega_k, w0, wa, gamma=None, a_late=None, b_late=None, z_mod_form=None, k_mod_form=None, z_bin=None, k_bin=None): #KZ
        """
        Cosmology object, stores primary and derived cosmological parameters.

        Parameters:
        -----------
        Omega_c, float
          Cold dark matter density fraction.
        Omega_b, float
          Baryonic matter density fraction.
        h, float
          Hubble constant divided by 100 km/s/Mpc; unitless.
        n_s, float
          Primordial scalar perturbation spectral index.
        sigma8, float
          Variance of matter density perturbations at an 8 Mpc/h scale
        Omega_k, float
          Curvature density fraction.
        w0, float
          First order term of dark energy equation
        wa, float
          Second order term of dark energy equation of state
        gamma: float
          Index of the growth rate (optional)
        
        ------
        Late-time modification parameters (optional) #KZ
        a_late
            z dependent change parameters, array-like
        b_late
            k dependent change parameters, array-like

        Notes:
        ------

        If `gamma` is specified, the emprical characterisation of growth in
        terms of  dlnD/dlna = \omega^\gamma will be used to define growth throughout.
        Otherwise the linear growth factor and growth rate will be solved by ODE.

        """
        # Store primary parameters
        self._Omega_c = Omega_c
        self._Omega_b = Omega_b
        self._h = h
        self._n_s = n_s
        self._sigma8 = sigma8
        self._Omega_k = Omega_k
        self._w0 = w0
        self._wa = wa

        self._flags = {}

        # Secondary optional parameters
        self._gamma = gamma
        self._flags["gamma_growth"] = gamma is not None

        # KZ start
        self._a_late = a_late
        self._b_late = b_late 
        ## flags for whether z or k are modified
        self._flags["late_time_z_mod"] = a_late is not None
        self._flags["late_time_k_mod"] = b_late is not None
        ## flag for which modification is used; should be a string
        self._flags["z_mod_form"] = z_mod_form
        self._flags["k_mod_form"] = k_mod_form
        self._z_bin = z_bin
        self._k_bin = k_bin
        assert self._flags["late_time_z_mod"] == (z_mod_form is not None), "modification parameters set, but the functionals not specified"
        assert self._flags["late_time_k_mod"] == (k_mod_form is not None), "modification parameters set, but the functionals not specified"
        if self._flags["z_mod_form"] == 'bin_custom':
            assert len(z_bin)==3, "please specify bins = [start, end, N_bin]"
        if self._flags["k_mod_form"] == 'bin_custom':
            assert len(k_bin)==3, "please specify bins = [start, end, N_bin]" 
        # KZ end

        # Create a workspace where functions can store some precomputed
        # results
        self._workspace = {}

    def __str__(self):
        return (
            "Cosmological parameters: \n"
            + "    h:        "
            + str(self.h)
            + " \n"
            + "    Omega_b:  "
            + str(self.Omega_b)
            + " \n"
            + "    Omega_c:  "
            + str(self.Omega_c)
            + " \n"
            + "    Omega_k:  "
            + str(self.Omega_k)
            + " \n"
            + "    w0:       "
            + str(self.w0)
            + " \n"
            + "    wa:       "
            + str(self.wa)
            + " \n"
            + "    n:        "
            + str(self.n_s)
            + " \n"
            + "    sigma8:   "
            + str(self.sigma8)
        )

    def __repr__(self):
        return self.__str__()

    # Operations for flattening/unflattening representation
    def tree_flatten(self):
        params = (
            self._Omega_c,
            self._Omega_b,
            self._h,
            self._n_s,
            self._sigma8,
            self._Omega_k,
            self._w0,
            self._wa,
        )

        if self._flags["gamma_growth"]:
            params += (self._gamma,)

        # KZ start
        if self._flags["late_time_z_mod"]:
            params += (self._a_late,)
        if self._flags["late_time_k_mod"]:
            params += (self._b_late,)
        if self._flags["z_mod_form"] == 'bin_custom':
            params += (self._z_bin,)
        if self._flags["k_mod_form"] == 'bin_custom':
            params += (self._k_bin,)
        # KZ end

        return (
            params,
            self._flags,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # Retrieve base parameters
        Omega_c, Omega_b, h, n_s, sigma8, Omega_k, w0, wa = children[:8]
        children = list(children[8:]).reverse()

        #We extract the remaining parameters in reverse order from how they were inserted

        #KZ start: There seems to be a bug, even the gamma modification doesn't work, children is always None
        # if aux_data["late_time_k_mod"]:
        #     b_late = children.pop()
        # else:
        #     b_late = None

        # if aux_data["late_time_z_mod"]:
        #     print("KZ testing", type(children), children)
        #     a_late = children.pop()
        # else:
        #     a_late = None

        # if aux_data["gamma_growth"]:
        #     print("KZ testing", type(children), children)
        #     gamma = children.pop()
        # else:
        #     gamma = None

        #For now:
        gamma = None
        a_late = None
        b_late = None
        # KZ end

        return cls(
            Omega_c=Omega_c,
            Omega_b=Omega_b,
            h=h,
            n_s=n_s,
            sigma8=sigma8,
            Omega_k=Omega_k,
            w0=w0,
            wa=wa,
            gamma=gamma,
            a_late=a_late, #KZ
            b_late=b_late  #KZ
        )

    # Cosmological parameters, base and derived
    @property
    def Omega(self):
        return 1.0 - self._Omega_k

    @property
    def Omega_b(self):
        return self._Omega_b

    @property
    def Omega_c(self):
        return self._Omega_c

    @property
    def Omega_m(self):
        return self._Omega_b + self._Omega_c

    @property
    def Omega_de(self):
        return self.Omega - self.Omega_m

    @property
    def Omega_k(self):
        return self._Omega_k

    @property
    def k(self):
        return -np.sign(self._Omega_k).astype(np.int8)

    @property
    def sqrtk(self):
        return np.sqrt(np.abs(self._Omega_k))

    @property
    def h(self):
        return self._h

    @property
    def w0(self):
        return self._w0

    @property
    def wa(self):
        return self._wa

    @property
    def n_s(self):
        return self._n_s

    @property
    def sigma8(self):
        return self._sigma8

    @property
    def gamma(self):
        return self._gamma

    #KZ start
    @property
    def a_late(self):
        return self._a_late

    @property
    def b_late(self):
        return self._b_late
    
    @property
    def z_bin(self):
        return self._z_bin

    @property
    def k_bin(self):
        return self._k_bin
    #KZ end
