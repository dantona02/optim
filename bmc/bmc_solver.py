"""
bmc_solver.py
    Definition of BlochMcConnellSolver class.
"""
import math

import numpy as np

from bmc.params import Params

class BlochMcConnellSolver:
    """
    Solver class for Bloch-McConnell equations.
    """

    def __init__(self, params: Params, n_offsets: int, z_positions: np.ndarray) -> None:
        """
        __init__ Initialize BlochMcConnellSolver class.

        Parameters
        ----------
        params : Params
            Parameters object containing all required parameters.
        n_offsets : int
            Number of frequency offsets.
        """
        self.params: Params = params
        self.n_offsets: int = n_offsets
        self.z_positions: np.ndarray = z_positions
        self.n_isochromats: int = len(z_positions)

        self.par_calc: bool = params.options["par_calc"]
        self.first_dim: int = 1
        self.n_pools: int = len(params.cest_pools)
        self.is_mt_active = bool(params.mt_pool)
        self.size: int = params.m_vec.size
        self.arr_a: np.ndarray = None
        self.arr_c: np.ndarray = None
        self.w0: float = None
        self.dw0: float = None
        self.mean_ppm: float = 0

        self.update_params(params)

    def _init_matrix_a(self) -> None:
        """
        Initialize self.arr_a with all parameters from self.params.
        """
        n_p = self.n_pools
        self.arr_a = np.zeros([self.size, self.size], dtype=float)

        # set mt_pool parameters
        k_ac = 0.0
        if self.is_mt_active:
            k_ca = self.params.mt_pool["k"]
            k_ac = k_ca * self.params.mt_pool["f"]
            self.arr_a[2 * (n_p + 1), 3 * (n_p + 1)] = k_ca
            self.arr_a[3 * (n_p + 1), 2 * (n_p + 1)] = k_ac

        # set water_pool parameters
        k1a = self.params.water_pool["r1"] + k_ac
        k2a = self.params.water_pool["r2"]
        for pool in self.params.cest_pools:
            k_ai = pool["f"] * pool["k"]
            k1a += k_ai
            k2a += k_ai

        self.arr_a[0, 0] = -k2a
        self.arr_a[1 + n_p, 1 + n_p] = -k2a
        self.arr_a[2 + 2 * n_p, 2 + 2 * n_p] = -k1a

        # set cest_pools parameters
        for i, pool in enumerate(self.params.cest_pools):
            k_ia = pool["k"]
            k_ai = k_ia * pool["f"]
            k_1i = k_ia + pool["r1"]
            k_2i = k_ia + pool["r2"]

            self.arr_a[0, i + 1] = k_ia
            self.arr_a[i + 1, 0] = k_ai
            self.arr_a[i + 1, i + 1] = -k_2i

            self.arr_a[1 + n_p, i + 2 + n_p] = k_ia
            self.arr_a[i + 2 + n_p, 1 + n_p] = k_ai
            self.arr_a[i + 2 + n_p, i + 2 + n_p] = -k_2i

            self.arr_a[2 * (n_p + 1), i + 1 + 2 * (n_p + 1)] = k_ia
            self.arr_a[i + 1 + 2 * (n_p + 1), 2 * (n_p + 1)] = k_ai
            self.arr_a[i + 1 + 2 * (n_p + 1), i + 1 + 2 * (n_p + 1)] = -k_1i

        # always expand to 4 dimensions
        self.arr_a = self.arr_a[
            np.newaxis,
            np.newaxis,
        ]
        self.arr_a = np.repeat(self.arr_a, self.n_isochromats, axis=0) #array shape (n_isochromats, 1, size, size)


        # if parallel computation is activated, repeat matrix A n_offsets times along a new axis
        if self.par_calc:
            self.arr_a = np.repeat(self.arr_a, self.n_offsets, axis=1)
            self.first_dim = self.n_offsets

    def _init_vector_c(self) -> None:
        """
        Initialize vector self.C with all parameters from self.params.
        """
        n_p = self.n_pools
        self.arr_c = np.zeros([self.size], dtype=float)
        self.arr_c[(n_p + 1) * 2] = self.params.water_pool["f"] * self.params.water_pool["r1"]
        for i, pool in enumerate(self.params.cest_pools):
            self.arr_c[(n_p + 1) * 2 + (i + 1)] = pool["f"] * pool["r1"]

        if self.is_mt_active:
            self.arr_c[3 * (n_p + 1)] = self.params.mt_pool["f"] * self.params.mt_pool["r1"]

        # always expand to 3 dimensions (independent of sequential or parallel computation)
        self.arr_c = self.arr_c[np.newaxis, np.newaxis, :, np.newaxis]
        self.arr_c = np.repeat(self.arr_c, self.n_isochromats, axis=0)

        # if parallel computation is activated, repeat matrix C n_offsets times along axis 0
        if self.par_calc:
            self.arr_c = np.repeat(self.arr_c, self.n_offsets, axis=0)

    def update_params(self, params: Params) -> None:
        """
        Updates matrix self.A according to given Params object.
        """
        self.params = params
        self.w0 = params.scanner["b0"] * params.scanner["gamma"]
        # np.random.seed(42)  # Fester Seed für Reproduzierbarkeit
        # self.dw0 = self.w0 * np.random.normal(self.mean_ppm, params.scanner["b0_inhomogeneity"], self.n_isochromats)
        # self.dw0 = self.w0 * (-1 * np.sort(np.random.normal(self.mean_ppm, params.scanner["b0_inhomogeneity"], self.n_isochromats)))
        # self.dw0 = self.w0 * np.random.normal(self.mean_ppm, params.scanner["b0_inhomogeneity"], self.n_isochromats)
        # self.dw0 = self.w0 * np.linspace(params.scanner["b0_inhomogeneity"], -params.scanner["b0_inhomogeneity"], self.n_isochromats)
        # self.dw0 = self.w0 * np.random.uniform(-params.scanner["b0_inhomogeneity"], params.scanner["b0_inhomogeneity"], self.n_isochromats)
        self.dw0 = self.w0 * self.generate_gaussian_random_field_ppm(1, params.scanner["b0_inhomogeneity"], 40)
        self._init_matrix_a()
        self._init_vector_c()

    def update_matrix(self, rf_amp: float, rf_phase: float, rf_freq: float, grad_amp: float = 0) -> None:
        """
        rf_phase and rf_freq might be floats not arrays
        Updates matrix self.A according to given parameters.
        :param rf_amp: amplitude of current step (e.g. pulse fragment)
        :param rf_phase: phase of current step (e.g. pulse fragment)
        :param rf_freq: frequency value of current step (e.g. pulse fragment)
        """
        j = self.first_dim  # size of first dimension (=1 for sequential, n_offsets for parallel)
        n_p = self.n_pools
        grad_term = 2 * np.pi * grad_amp * self.z_positions.reshape(self.n_isochromats, self.n_offsets)
       

        # set dw0 due to b0_inhomogeneity
        self.arr_a[:, :, 0, 1 + n_p] = -self.dw0[:, np.newaxis] * j
        self.arr_a[:, :, 1 + n_p, 0] = self.dw0[:, np.newaxis] * j

        #set dw_grad for water pool
        self.arr_a[:, :, 0, 1 + n_p] += grad_term
        self.arr_a[:, :, 1 + n_p, 0] -= grad_term 

        # calculate omega_1
        rf_amp_2pi = rf_amp * 2 * np.pi * self.params.scanner["rel_b1"]
        rf_amp_2pi_sin = rf_amp_2pi * np.sin(rf_phase)
        rf_amp_2pi_cos = rf_amp_2pi * np.cos(rf_phase)

        # set omega_1 for water_pool
        self.arr_a[:, :, 0, 2 * (n_p + 1)] = -rf_amp_2pi_sin
        self.arr_a[:, :, 2 * (n_p + 1), 0] = rf_amp_2pi_sin

        self.arr_a[:, :, n_p + 1, 2 * (n_p + 1)] = rf_amp_2pi_cos
        self.arr_a[:, :, 2 * (n_p + 1), n_p + 1] = -rf_amp_2pi_cos

        # set omega_1 for cest pools
        for i in range(1, n_p + 1):
            self.arr_a[:, :, i, i + 2 * (n_p + 1)] = -rf_amp_2pi_sin
            self.arr_a[:, :, i + 2 * (n_p + 1), i] = rf_amp_2pi_sin

            self.arr_a[:, :, n_p + 1 + i, i + 2 * (n_p + 1)] = rf_amp_2pi_cos
            self.arr_a[:, :, i + 2 * (n_p + 1), n_p + 1 + i] = -rf_amp_2pi_cos

        # set off-resonance terms for water pool, rf_freq is frequency offset from water
        rf_freq_2pi = rf_freq * 2 * np.pi
        self.arr_a[:, :, 0, 1 + n_p] += rf_freq_2pi
        self.arr_a[:, :, 1 + n_p, 0] -= rf_freq_2pi

        

        # set off-resonance terms for cest pools
        for i in range(1, n_p + 1):
            dwi = self.params.cest_pools[i - 1]["dw"] * self.w0 - (rf_freq_2pi + self.dw0)
            self.arr_a[:, :, i, i + n_p + 1] = -dwi[:, np.newaxis]
            self.arr_a[:, :, i + n_p + 1, i] = dwi[:, np.newaxis]

            self.arr_a[:, :, i, i + n_p + 1] += grad_term
            self.arr_a[:, :, i + n_p + 1, i] -= grad_term

        # mt_pool
        if self.is_mt_active:
            self.arr_a[:, :, 3 * (n_p + 1), 3 * (n_p + 1)] = (
                -self.params.mt_pool["r1"]
                - self.params.mt_pool["k"]
                - rf_amp_2pi**2 * self.get_mt_shape_at_offset(rf_freq_2pi + self.dw0, self.w0) # implementation of gradient needs to be implemented
            )
    
    def solve_equation(self, mag: np.ndarray, dtp: float) -> np.ndarray:
        """
        Solves one step of BMC equations for multiple Isochromaten using the Padé approximation.
        :param mag: magnetization vector before current step (shape: [n_isochromats, size, 1])
        :param dtp: duration of current step
        :return: magnetization vector after current step (shape: [n_isochromats, size, 1])
        """
        n_iter = 6  # number of iterations
        arr_a = self.arr_a
        arr_c = self.arr_c
        a_inv_t = np.matmul(np.linalg.pinv(arr_a), arr_c)  # Shape: [n_isochromats, n_offsets, size, 1]

        # Compute `a_t` for all Isochromaten
        a_t = arr_a * dtp  # Shape: [n_isochromats, n_offsets, size, size]

        # Normalize `a_t` to avoid numerical instability
        _, inf_exp = np.frexp(np.linalg.norm(a_t, ord=np.inf, axis=(2, 3)))
        j = np.maximum(0, inf_exp)
        a_t /= np.power(2, j[:, :, np.newaxis, np.newaxis])  # Normalize across batch and offset

        # Initialize Padé approximation
        identity = np.eye(arr_a.shape[-1])[np.newaxis, np.newaxis, :, :]
        identity = np.repeat(identity, arr_a.shape[0], axis=0)
        x = a_t.copy()
        c = 0.5
        n = identity + c * a_t
        d = identity - c * a_t

        p = True
        for k in range(2, n_iter + 1):
            c = c * (n_iter - k + 1) / (k * (2 * n_iter - k + 1))
            x = np.matmul(a_t, x)
            c_x = c * x
            n += c_x
            d += c_x if p else -c_x
            p = not p

        # Solve for the matrix exponential
        f = np.matmul(np.linalg.pinv(d), n)
        for _ in range(j.max()):
            f = np.matmul(f, f)

        # Compute the final magnetization
        mag = np.matmul(f, mag + a_inv_t) - a_inv_t
        return mag
    

    # def solve_equation_expm(self, mag: np.ndarray, dtp: float) -> np.ndarray:
    #     """
    #     Solves one step of BMC equations using the eigenwert ansatz.
    #     :param mag: magnetization vector before current step
    #     :param dtp: duration of current step
    #     :return: magnetization vector after current step
    #     """
    #     arr_a = self.arr_a
    #     arr_c = self.arr_c

    #     if not arr_a.ndim == arr_c.ndim == mag.ndim:
    #         raise Exception("Matrix dimensions don't match. That's not gonna work.")

    #     # solve matrix exponential for current timestep
    #     ex = self._solve_expm(arr_a, dtp)

    #     # because np.linalg.lstsq(A_,b_) doesn't work for stacked arrays, it is calculated as np.linalg.solve(
    #     # A_.T.dot(A_), A_.T.dot(b_)). For speed reasons, the transpose of A_ (A_.T) is pre-calculated and the
    #     # .dot notation is replaced by the Einstein summation (np.einsum).
    #     arr_at = arr_a.T
    #     tmps = np.linalg.solve(np.einsum("kji,ikl->ijl", arr_at, arr_a), np.einsum("kji,ikl->ijl", arr_at, arr_c))

    #     # solve equation for magnetization M: np.einsum('ijk,ikl->ijl') is used to calculate the matrix
    #     # multiplication for each element along the first (=offset) axis.
    #     mag = np.real(np.einsum("ijk,ikl->ijl", ex, mag + tmps) - tmps)
    #     return mag

    @staticmethod
    def _solve_expm(matrix: np.ndarray, dtp: float) -> np.ndarray:
        """
        Solve the matrix exponential. This version is faster than scipy expm for typical BMC matrices.
        :param matrix: matrix representation of Bloch-McConnell equations
        :param dtp: duration of current step
        :return: solution of matrix exponential
        """
        vals, vects = np.linalg.eig(matrix * dtp)
        tmp = np.einsum("ijk,ikl->ijl", vects, np.apply_along_axis(np.diag, -1, np.exp(vals)))
        inv = np.linalg.inv(vects)
        return np.einsum("ijk,ikl->ijl", tmp, inv)

    def get_mt_shape_at_offset(self, offsets: np.ndarray, w0: float) -> np.ndarray:
        """
        Calculates the lineshape of the MT pool at the given offset(s).
        :param offsets: frequency offset(s)
        :param w0: Larmor frequency of simulated system
        :return: lineshape of mt pool at given offset(s)
        """
        ls = self.params.mt_pool["lineshape"].lower()
        dw = self.params.mt_pool["dw"]
        t2 = 1 / self.params.mt_pool["r2"]
        if ls == "lorentzian":
            mt_line = t2 / (1 + pow((offsets - dw * w0) * t2, 2.0))
        elif ls == "superlorentzian":
            dw_pool = offsets - dw * w0
            if self.par_calc:
                mt_line = np.zeros(offsets.size)
                for i, dw_ in enumerate(dw_pool):
                    if abs(dw_) >= w0:
                        mt_line[i] = self.interpolate_sl(dw_)
                    else:
                        mt_line[i] = self.interpolate_chs(dw_, w0)
            else:
                if abs(dw_pool) >= w0:
                    mt_line = self.interpolate_sl(dw_pool)
                else:
                    mt_line = self.interpolate_chs(dw_pool, w0)
        else:
            mt_line = np.zeros(offsets.size)
        return mt_line

    def interpolate_sl(self, dw: float) -> float:
        """
        Interpolates MT profile for SuperLorentzian lineshape.
        :param dw: relative frequency offset
        :return: MT profile at given relative frequency offset
        """
        mt_line = 0
        t2 = 1 / self.params.mt_pool["r2"]
        n_samples = 101
        step_size = 0.01
        sqrt_2pi = np.sqrt(2 / np.pi)
        for i in range(n_samples):
            powcu2 = abs(3 * pow(step_size * i, 2) - 1)
            mt_line += sqrt_2pi * t2 / powcu2 * np.exp(-2 * pow(dw * t2 / powcu2, 2))
        return mt_line * np.pi * step_size

    def interpolate_chs(self, dw_pool: float, w0: float) -> np.ndarray:
        """
        Cubic Hermite Spline Interpolation
        """
        mt_line = 0
        px = np.array([-300 - w0, -100 - w0, 100 + w0, 300 + w0])
        py = np.zeros(px.size)
        for i in range(px.size):
            py[i] = self.interpolate_sl(px[i])
        if px.size != 4 or py.size != 4:
            return mt_line
        else:
            tan_weight = 30
            d0y = tan_weight * (py[1] - py[0])
            d1y = tan_weight * (py[3] - py[2])
            c_step = abs((dw_pool - px[1] + 1) / (px[2] - px[1] + 1))
            h0 = 2 * (c_step**3) - 3 * (c_step**2) + 1
            h1 = -2 * (c_step**3) + 3 * (c_step**2)
            h2 = (c_step**3) - 2 * (c_step**2) + c_step
            h3 = (c_step**3) - (c_step**2)

            mt_line = h0 * py[1] + h1 * py[2] + h2 * d0y + h3 * d1y
            return mt_line


    def generate_gaussian_random_field_ppm(self, xi, sigma_ppm, L):
        """
        Generiert ein Gaussian Random Field mit räumlicher Korrelation in ppm.

        Parameter:
        - N: Anzahl der Isochromaten
        - xi: Korrelationslänge (in der gleichen räumlichen Einheit wie L, z.B. µm)
        - sigma_ppm: Standardabweichung der Inhomogenität in ppm
        - L: Gesamtlänge der z-Achse (z.B. in µm)

        Rückgabe:
        - field_ppm: Array mit Inhomogenitäten in ppm
        - z: Positionen der Isochromaten entlang der z-Achse
        """
        # Positionen der Isochromaten gleichmäßig verteilt
        np.random.seed(42)
        z = np.linspace(-L/2, L/2, self.n_isochromats)

        # Räumliche Auflösung
        dx = L / self.n_isochromats  # Abstand zwischen Isochromaten

        # Wellenzahlen berechnen
        k = np.fft.fftfreq(self.n_isochromats, d=dx)  # in 1/µm

        # Gaußsche Power Spectral Density (PSD)
        S_k = np.exp(-(k**2) * (xi**2))

        # Rausch im Fourier-Raum
        noise = np.random.normal(scale=1.0, size=self.n_isochromats) + 1j * np.random.normal(scale=1.0, size=self.n_isochromats)
        field_k = noise * np.sqrt(S_k)

        # Inverse FFT, um das räumliche Feld zu erhalten
        field_ppm = np.fft.ifft(field_k).real

        # Normalisieren auf die gewünschte Standardabweichung in ppm
        field_ppm = field_ppm / np.std(field_ppm) * sigma_ppm

        return field_ppm