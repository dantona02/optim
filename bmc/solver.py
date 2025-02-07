"""
bmc_solver.py
    Definition of BlochMcConnellSolver class.
"""
import numpy as np
from bmc.params import Params
import torch
from bmc.utils.global_device import GLOBAL_DEVICE
from torch.amp import autocast

from torch.utils.cpp_extension import load

# Für maximale GPU-Auslastung
torch.backends.cuda.matmul.allow_tf32 = False  # Deaktivieren für volle FP64-Genauigkeit
torch.backends.cudnn.allow_tf32 = False
# torch.backends.cuda.matmul.allow_fp64_reduced_precision_reduction = True



class BlochMcConnellSolver:
    """
    Solver class for Bloch-McConnell equations.
    """
    def __init__(self, params: Params, n_offsets: int, z_positions: torch.Tensor) -> None:
        """
        Initialize BlochMcConnellSolver class.

        Parameters:
            params : Params
                Parameters object containing all required parameters.
            n_offsets : int
                Number of frequency offsets.
            z_positions : torch.Tensor
                Tensor containing z-positions for isochromats.
        """
        self.params: Params = params
        self.n_offsets: int = n_offsets
        self.z_positions: torch.Tensor = z_positions.to(GLOBAL_DEVICE)
        self.n_isochromats: int = len(z_positions)

        self.par_calc: bool = params.options["par_calc"]
        self.first_dim: int = 1
        self.n_pools: int = len(params.cest_pools)
        self.is_mt_active = bool(params.mt_pool)
        self.size: int = params.m_vec.size
        self.arr_a: torch.Tensor = None
        self.arr_c: torch.Tensor = None
        self.w0: float = None
        self.dw0: torch.Tensor = None
        self.mean_ppm: float = 0

        self.dw_tensors = torch.tensor(
            [pool["dw"] for pool in params.cest_pools],
            dtype=torch.float64,
            device=GLOBAL_DEVICE
        )

        self.update_params(params)

    def _init_matrix_a(self) -> None:
        """
        Initialize self.arr_a with all parameters from self.params.
        """
        n_p = self.n_pools
        self.arr_a = torch.zeros([self.size, self.size], dtype=torch.float64, device=GLOBAL_DEVICE)

        # Set MT pool parameters
        k_ac = 0.0
        if self.is_mt_active:
            k_ca = self.params.mt_pool["k"]
            k_ac = k_ca * self.params.mt_pool["f"]
            self.arr_a[2 * (n_p + 1), 3 * (n_p + 1)] = k_ca
            self.arr_a[3 * (n_p + 1), 2 * (n_p + 1)] = k_ac

        # Set water pool parameters
        k1a = self.params.water_pool["r1"] + k_ac
        k2a = self.params.water_pool["r2"]
        for pool in self.params.cest_pools:
            k_ai = pool["f"] * pool["k"]
            k1a += k_ai
            k2a += k_ai

        self.arr_a[0, 0] = -k2a
        self.arr_a[1 + n_p, 1 + n_p] = -k2a
        self.arr_a[2 + 2 * n_p, 2 + 2 * n_p] = -k1a

        # Set CEST pools parameters
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

        # Always expand to 4 dimensions
        self.arr_a = self.arr_a.unsqueeze(0).unsqueeze(0)  # Add batch and offset dimensions
        self.arr_a = self.arr_a.repeat(self.n_isochromats, 1, 1, 1)  # Repeat for n_isochromats

        # If parallel computation is activated, repeat matrix A n_offsets times along a new axis
        if self.par_calc:
            self.arr_a = self.arr_a.repeat(1, self.n_offsets, 1, 1)
            self.first_dim = self.n_offsets

    def _init_vector_c(self) -> None:
        """
        Initialize vector self.C with all parameters from self.params.
        """
        n_p = self.n_pools
        self.arr_c = torch.zeros([self.size], dtype=torch.float64, device=GLOBAL_DEVICE)

        # Set water pool parameters
        self.arr_c[(n_p + 1) * 2] = self.params.water_pool["f"] * self.params.water_pool["r1"]

        # Set CEST pools parameters
        for i, pool in enumerate(self.params.cest_pools):
            self.arr_c[(n_p + 1) * 2 + (i + 1)] = pool["f"] * pool["r1"]

        # Set MT pool parameters if active
        if self.is_mt_active:
            self.arr_c[3 * (n_p + 1)] = self.params.mt_pool["f"] * self.params.mt_pool["r1"]

        # Expand to 3 dimensions: [n_isochromats, 1, size, 1]
        self.arr_c = self.arr_c.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        self.arr_c = self.arr_c.repeat(self.n_isochromats, 1, 1, 1)

        # If parallel computation is activated, repeat vector C n_offsets times along the batch dimension
        if self.par_calc:
            self.arr_c = self.arr_c.repeat(1, self.n_offsets, 1, 1)
        
    def update_params(self, params: Params) -> None:
        """
        Updates matrix self.A according to given Params object.
        """
        self.params = params
        self.w0 = params.scanner["b0"] * params.scanner["gamma"]

        np.random.seed(42)  # Fester Seed für Reproduzierbarkeit
        self.dw0 = self.w0 * np.random.normal(self.mean_ppm, params.scanner["b0_inhomogeneity"], self.n_isochromats)
        self.dw0 = torch.tensor(self.dw0, dtype=torch.float64, device=GLOBAL_DEVICE)
        self._init_matrix_a()
        self._init_vector_c()

    def update_matrix(self, rf_amp: torch.Tensor, rf_phase: torch.Tensor, rf_freq: float, grad_amp: torch.Tensor = torch.tensor(0, dtype=torch.float64, device=GLOBAL_DEVICE)) -> None:
        """
        Updates matrix self.A according to given parameters.

        Parameters:
            rf_amp : torch.Tensor
                Amplitude of current step (e.g. pulse fragment).
            rf_phase : torch.Tensor
                Phase of current step (e.g. pulse fragment).
            rf_freq : float
                Frequency value of current step (e.g. pulse fragment).
            grad_amp : torch.Tensor, optional
                Gradient amplitude for dephasing. Default is 0.
        """
        j = self.first_dim  # size of first dimension (=1 for sequential, n_offsets for parallel)
        n_p = self.n_pools
        # Erzeuge grad_term als neuen Tensor
        grad_term = 2 * torch.pi * grad_amp * self.z_positions.reshape(self.n_isochromats, self.n_offsets)
        
        # Erstelle einen neuen Tensor für arr_a, um in-place Modifikationen zu vermeiden
        new_arr_a = self.arr_a.clone()

        # Set dw0 due to B0 inhomogeneity
        new_arr_a[:, :, 0, 1 + n_p] = -self.dw0.unsqueeze(1) * j
        new_arr_a[:, :, 1 + n_p, 0] = self.dw0.unsqueeze(1) * j

        # Set gradient terms for water pool
        new_arr_a[:, :, 0, 1 + n_p] = new_arr_a[:, :, 0, 1 + n_p] + grad_term
        new_arr_a[:, :, 1 + n_p, 0] = new_arr_a[:, :, 1 + n_p, 0] - grad_term

        # Sicherstellen, dass rf_phase ein Tensor ist
        if not isinstance(rf_phase, torch.Tensor):
            rf_phase = torch.tensor(rf_phase, dtype=torch.float64, device=GLOBAL_DEVICE)
        rf_amp_2pi = rf_amp * 2 * torch.pi * self.params.scanner["rel_b1"]
        # Wir vermeiden hier in-place Operationen: 
        rf_amp_2pi_sin = rf_amp_2pi * torch.sin(rf_phase)
        rf_amp_2pi_cos = rf_amp_2pi * torch.cos(rf_phase)

        # Set omega_1 for water pool
        new_arr_a[:, :, 0, 2 * (n_p + 1)] = -rf_amp_2pi_sin
        new_arr_a[:, :, 2 * (n_p + 1), 0] = rf_amp_2pi_sin
        new_arr_a[:, :, n_p + 1, 2 * (n_p + 1)] = rf_amp_2pi_cos
        new_arr_a[:, :, 2 * (n_p + 1), n_p + 1] = -rf_amp_2pi_cos

        # Set omega_1 for CEST pools
        for i in range(1, n_p + 1):
            new_arr_a[:, :, i, i + 2 * (n_p + 1)] = -rf_amp_2pi_sin
            new_arr_a[:, :, i + 2 * (n_p + 1), i] = rf_amp_2pi_sin
            new_arr_a[:, :, n_p + 1 + i, i + 2 * (n_p + 1)] = rf_amp_2pi_cos
            new_arr_a[:, :, i + 2 * (n_p + 1), n_p + 1 + i] = -rf_amp_2pi_cos

        # Set off-resonance terms for water pool
        rf_freq_2pi = rf_freq * 2 * torch.pi
        new_arr_a[:, :, 0, 1 + n_p] = new_arr_a[:, :, 0, 1 + n_p] + rf_freq_2pi
        new_arr_a[:, :, 1 + n_p, 0] = new_arr_a[:, :, 1 + n_p, 0] - rf_freq_2pi

        # Set off-resonance terms for CEST pools
        for i in range(1, n_p + 1):
            # dwi: Differenz zwischen dem pool-spezifischen offset und dem Wasseroffset plus B0-Inhomogenität
            dwi = self.dw_tensors[i - 1] * self.w0 - (rf_freq_2pi + self.dw0)
            new_arr_a[:, :, i, i + n_p + 1] = -dwi.unsqueeze(1)
            new_arr_a[:, :, i + n_p + 1, i] = dwi.unsqueeze(1)

            # Add gradient terms for CEST pools
            new_arr_a[:, :, i, i + n_p + 1] = new_arr_a[:, :, i, i + n_p + 1] + grad_term
            new_arr_a[:, :, i + n_p + 1, i] = new_arr_a[:, :, i + n_p + 1, i] - grad_term

        # MT pool (if active)
        if self.is_mt_active:
            new_arr_a[:, :, 3 * (n_p + 1), 3 * (n_p + 1)] = (
                -self.params.mt_pool["r1"]
                - self.params.mt_pool["k"]
                - rf_amp_2pi**2 * self.get_mt_shape_at_offset(rf_freq_2pi + self.dw0, self.w0)
            )

        # Set the new arr_a
        self.arr_a = new_arr_a

    # def solve_equation(self, mag: torch.Tensor, dtp: float) -> torch.Tensor:
    #     """
    #     Solves one step of BMC equations for multiple Isochromats using the Padé approximation.
    #     :param mag: magnetization vector before current step (shape: [n_isochromats, size, 1])
    #     :param dtp: duration of current step
    #     :return: magnetization vector after current step (shape: [n_isochromats, size, 1])
    #     """
    #     n_iter = 6  # number of iterations
    #     arr_a = self.arr_a.to(dtype=torch.float64)
    #     arr_c = self.arr_c.to(dtype=torch.float64)

    #     # Compute a_inv_t
    #     a_inv_t = torch.matmul(torch.linalg.pinv(arr_a), arr_c)  # Shape: [n_isochromats, n_offsets, size, 1]

    #     # Compute a_t
    #     a_t = arr_a * dtp  # Shape: [n_isochromats, n_offsets, size, size]

    #     # Normalize a_t to avoid numerical instability
    #     max_norm = torch.linalg.norm(a_t, ord=float('inf'), dim=(2, 3))
    #     _, exp_shift = torch.frexp(max_norm)
    #     exp_shift = torch.clamp(exp_shift, min=0)
    #     a_t = a_t / (2.0 ** exp_shift.view(-1, 1, 1, 1))

    #     # Initialize Padé approximation
    #     identity = torch.eye(arr_a.shape[-1], dtype=torch.float64, device=GLOBAL_DEVICE).unsqueeze(0).unsqueeze(0)
    #     identity = identity.expand_as(arr_a)
    #     x = a_t.clone()
    #     c = 0.5
    #     n = identity + c * a_t
    #     d = identity - c * a_t

    #     p = True
    #     for k in range(2, n_iter + 1):
    #         c = c * (n_iter - k + 1) / (k * (2 * n_iter - k + 1))
    #         x = torch.matmul(a_t, x)
    #         c_x = c * x
    #         n = n + c_x
    #         if p:
    #             d = d + c_x
    #         else:
    #             d = d - c_x
    #         p = not p

    #     # Solve for matrix exponential
    #     f = torch.matmul(torch.linalg.pinv(d), n)
    #     for _ in range(int(exp_shift.max())):
    #         f = torch.matmul(f, f)

    #     # Compute the final magnetization
    #     mag = torch.matmul(f, mag + a_inv_t) - a_inv_t
    #     return mag.to(dtype=torch.float64)

    # def solve_equation(self, mag: torch.Tensor, dtp: float) -> torch.Tensor:
    #     """
    #     Variante des Solvers mit Mixed Precision (float32) + optionaler Verfeinerung in float64.
    #     """
    #     n_iter = 6

    #     # 1) Input in float32
    #     arr_a_32 = self.arr_a.to(dtype=torch.float32)
    #     arr_c_32 = self.arr_c.to(dtype=torch.float32)
    #     mag_32   = mag.to(dtype=torch.float32)

    #     # Optional: Kritische Inversion in float64
    #     #    => Hier machen wir 'arr_a_64 -> pinv -> a_inv_t_64 -> back to float32'
    #     arr_a_64  = arr_a_32.to(dtype=torch.float64)
    #     arr_c_64  = arr_c_32.to(dtype=torch.float64)
    #     a_inv_t_64 = torch.matmul(torch.linalg.pinv(arr_a_64), arr_c_64)  # shape: [n_iso, n_off, size, 1]
    #     a_inv_t_32 = a_inv_t_64.to(dtype=torch.float32)

    #     # 2) A * dt -> a_t
    #     a_t_32 = arr_a_32 * dtp

    #     # 3) Scaling: wie in deinem Code
    #     max_norm_32 = torch.linalg.norm(a_t_32, ord=float('inf'), dim=(2, 3))
    #     _, exp_shift_32 = torch.frexp(max_norm_32)
    #     exp_shift_32 = torch.clamp(exp_shift_32, min=0)
    #     a_t_32 = a_t_32 / (2.0 ** exp_shift_32.view(-1, 1, 1, 1))

    #     # 4) Pade-Approx
    #     size = arr_a_32.shape[-1]
    #     identity_32 = torch.eye(size, dtype=torch.float32, device=arr_a_32.device)
    #     identity_32 = identity_32.unsqueeze(0).unsqueeze(0).expand_as(arr_a_32)

    #     x_32 = a_t_32.clone()
    #     c_32 = torch.tensor(0.5, dtype=torch.float32, device=arr_a_32.device)
    #     n_32 = identity_32 + c_32 * a_t_32
    #     d_32 = identity_32 - c_32 * a_t_32

    #     p = True
    #     for k in range(2, n_iter + 1):
    #         c_32 = c_32 * (n_iter - k + 1) / (k * (2 * n_iter - k + 1))
    #         x_32 = torch.matmul(a_t_32, x_32)
    #         c_x_32 = c_32 * x_32
    #         n_32 = n_32 + c_x_32
    #         if p:
    #             d_32 = d_32 + c_x_32
    #         else:
    #             d_32 = d_32 - c_x_32
    #         p = not p

    #     # 5) F = D^-1 * N (Inversion in float32, optional Verfeinerung in float64)
    #     #    => again, you could do float64 pinv if you want
    #     f_32 = torch.matmul(torch.linalg.pinv(d_32), n_32)

    #     # 6) Exponent-Shifts ausführen
    #     max_shift = int(exp_shift_32.max().item())
    #     for _ in range(max_shift):
    #         f_32 = torch.matmul(f_32, f_32)

    #     # 7) Vorläufiges Ergebnis in float32
    #     #    mag + a_inv_t in float32, dann matrix-mult in float32
    #     tmp_32 = mag_32 + a_inv_t_32
    #     mag_approx_32 = torch.matmul(f_32, tmp_32) - a_inv_t_32

    #     # ===========================================
    #     # 8) OPTIONAL: Iterative Verfeinerung
    #     #    => Wir berechnen Residuum (z.B. in float64) und korrigieren
    #     # ===========================================
    #     do_refinement = True
    #     if do_refinement:
    #         with torch.no_grad():
    #             # Cast alles nötige in float64
    #             f_64    = f_32.to(dtype=torch.float64)
    #             mag_in_64  = mag.to(dtype=torch.float64)
    #             a_inv_t_64 = a_inv_t_32.to(dtype=torch.float64)

    #             mag_approx_64 = mag_approx_32.to(dtype=torch.float64)

    #             # Residuum: r = exp(A dt)*(mag_in + a_inv_t) - a_inv_t - mag_approx
    #             #   => r = f_64 @ (mag_in_64 + a_inv_t_64) - a_inv_t_64 - mag_approx_64
    #             rhs_64 = torch.matmul(f_64, mag_in_64 + a_inv_t_64) - a_inv_t_64
    #             r_64 = rhs_64 - mag_approx_64

    #             # Minimale Korrektur: mag_approx_64 += r_64
    #             #  (Man könnte hier theoretisch noch ein lineares Glssystem lösen,
    #             #   aber oft reicht Addieren bereits, wenn f_64 * ... "genau" war.)
    #             mag_approx_64 = mag_approx_64 + r_64

    #         mag_approx_32 = mag_approx_64.to(dtype=torch.float32)

    #     # Rückgabe
    #     return mag_approx_32

    
    # Batch-Optimierte Version mit Tensor Core-optimierten Operationen
    import torch

    def solve_equation(self, mag: torch.Tensor, dtp: float) -> torch.Tensor:
        n_iter = 6
        
        def align_matrix(tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
            """Pad both dimensions to make matrix multiplication compatible"""
            pad_rows = (target_dim - tensor.size(-2) % target_dim) % target_dim
            pad_cols = (target_dim - tensor.size(-1) % target_dim) % target_dim
            return torch.nn.functional.pad(tensor, (0, pad_cols, 0, pad_rows))
        
        # Dynamische Zielgröße basierend auf arr_a
        original_dim = self.arr_a.size(-1)  # Ursprüngliche Größe vor Padding
        arr_a = align_matrix(self.arr_a, original_dim).to(torch.float64)
        arr_c = align_matrix(self.arr_c, original_dim).to(torch.float64)

        # Debug-Ausgaben
        # print(f"arr_a shape after padding: {arr_a.shape}")
        # print(f"arr_c shape after padding: {arr_c.shape}")
        
        # Erweiterter Konsistenzcheck
        assert arr_a.size(-1) == arr_c.size(-2), (
            f"Matrix-Dimensionskonflikt: arr_a Spalten ({arr_a.size(-1)}) "
            f"≠ arr_c Zeilen ({arr_c.size(-2)})"
        )
        
        # Rest der ursprünglichen Implementierung
        a_inv_t = torch.linalg.pinv(arr_a) @ arr_c
        a_t = arr_a * dtp
        
        max_norm = torch.linalg.norm(a_t, ord=float('inf'), dim=(2,3))
        exp_shift = torch.clamp(max_norm.log2().ceil(), min=0)
        a_t_scaled = a_t / (2.0 ** exp_shift.view(-1, 1, 1, 1))
        
        identity = torch.eye(a_t_scaled.size(-1), 
                            dtype=torch.float64,
                            device=a_t_scaled.device).expand_as(a_t_scaled)
        
        # Padé-Approximation mit optimierten Operationen
        x = a_t_scaled.clone()
        c_factor = 0.5
        numerator = identity + c_factor * x
        denominator = identity - c_factor * x
        
        for k in range(2, n_iter + 1):
            c_factor *= (n_iter - k + 1) / (k * (2 * n_iter - k + 1))
            x = torch.matmul(a_t_scaled, x)
            cx = c_factor * x
            numerator += cx
            denominator += cx if k % 2 == 0 else -cx
        
        f_matrix = torch.linalg.pinv(denominator) @ numerator
        for _ in range(int(exp_shift.max())):
            f_matrix = torch.matmul(f_matrix, f_matrix)
        
        # Finale Berechnung mit Trimmen der Padding-Dimensionen
        result = (f_matrix @ (mag + a_inv_t)) - a_inv_t
        return result[..., :mag.size(-1)]  # Originaldimension wiederherstellen








    def solve_equation_expm(self, mag: np.ndarray, dtp: float) -> np.ndarray:
        """
        Solves one step of BMC equations using the eigenvalue approach.
        """
        arr_a = self.arr_a
        arr_c = self.arr_c

        if not arr_a.ndim == arr_c.ndim == mag.ndim:
            raise Exception("Matrix dimensions don't match. That's not gonna work.")

        ex = self._solve_expm(arr_a, dtp)
        arr_at = arr_a.T
        tmps = np.linalg.solve(np.einsum("kji,ikl->ijl", arr_at, arr_a),
                               np.einsum("kji,ikl->ijl", arr_at, arr_c))
        mag = np.real(np.einsum("ijk,ikl->ijl", ex, mag + tmps) - tmps)
        return mag

    @staticmethod
    def _solve_expm(matrix: np.ndarray, dtp: float) -> np.ndarray:
        vals, vects = np.linalg.eig(matrix * dtp)
        tmp = np.einsum("ijk,ikl->ijl", vects, np.apply_along_axis(np.diag, -1, np.exp(vals)))
        inv = np.linalg.inv(vects)
        return np.einsum("ijk,ikl->ijl", tmp, inv)

    def get_mt_shape_at_offset(self, offsets: np.ndarray, w0: float) -> np.ndarray:
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