"""
OPES (On-the-fly Probability Enhanced Sampling) for OpenMM
Based on PLUMED implementation: https://github.com/plumed/plumed2/blob/master/src/opes/OPESmetad.cpp
Reference: Invernizzi & Parrinello, J. Phys. Chem. Lett. 2020, 11, 2731-2736
"""

import math
import numpy as np
import os
from openmm import CustomCVForce, Continuous2DFunction
from openmm import unit


class OPES:
    """
    On-the-fly Probability Enhanced Sampling (OPES) method.

    The bias is: V(s) = (1-1/γ) * kT * log(P(s)/Z + ε)
    where P(s) is estimated via weighted kernel density estimation.

    Parameters
    ----------
    system : openmm.System
        The system to which the bias will be applied
    variables : list of openmm.Force
        The collective variables to bias
    temperature : openmm.unit.Quantity
        The temperature at which the simulation is run
    barrier : openmm.unit.Quantity
        The free energy barrier to overcome (typically 30-50 kJ/mol)
    sigma : list of openmm.unit.Quantity or None
        Initial bandwidth for each CV (if None, uses adaptive)
    stride : int
        Deposition frequency in time steps (default: 500)
    compression_threshold : float
        Merge kernels closer than this threshold (default: 1.0)
    saveFrequency : int
        Frequency to save kernels (default: 10000 steps)
    biasDir : str
        Directory to save bias information
    periodic : list of (min,max) tuples or None
        Periodic ranges for each CV
    bias_factor : float or None
        Well-tempered bias factor γ. If None, derived from BARRIER
    adaptive_sigma : bool
        Whether to adapt sigma during simulation (default: True)
    """

    def __init__(
        self,
        system,
        variables,
        temperature,
        barrier,
        sigma=None,
        stride=500,
        compression_threshold=1.0,
        saveFrequency=10000,
        biasDir=None,
        periodic=None,
        bias_factor=None,
        adaptive_sigma=True,
    ):
        self.system = system
        self.variables = variables
        self.num_cvs = len(variables)

        # Periodic boundaries
        if periodic is None:
            self.periodic: list[tuple[float, float] | None] = [None] * self.num_cvs
        else:
            if len(periodic) != self.num_cvs:
                raise ValueError("`periodic` must have the same length as `variables`")
            self.periodic = list(periodic)

        # Temperature
        kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
        kT_quantity = kB * temperature
        self.kT = kT_quantity.value_in_unit(unit.kilojoules_per_mole)
        self.beta = 1.0 / self.kT

        # BARRIER and derived parameters
        self.barrier = barrier.value_in_unit(unit.kilojoules_per_mole)

        # Derive bias_factor from BARRIER if not provided
        # PLUMED uses: biasfactor = 1 + barrier/(kT)
        if bias_factor is None:
            self.bias_factor = 1.0 + self.barrier / self.kT
        else:
            self.bias_factor = float(bias_factor)

        # Epsilon: regularization constant
        # PLUMED: epsilon = exp(-barrier/kT) / (avgN + exp(-barrier/kT))
        # Simplified: epsilon ≈ exp(-barrier/kT)
        self.epsilon = math.exp(-self.barrier / self.kT)

        # C++ OPESmetad.cpp truncates each Gaussian kernel at a finite cutoff
        # and subtracts its value at the cutoff to make the bias smooth.
        # This is crucial for reproducing the correct FES shape.
        self.bias_prefactor = 1.0 - 1.0 / self.bias_factor
        self.kernel_cutoff = math.sqrt(2.0 * self.barrier / (self.bias_prefactor * self.kT))
        self.kernel_cutoff2 = self.kernel_cutoff ** 2
        self.val_at_cutoff = math.exp(-0.5 * self.kernel_cutoff2)

        print(f"OPES parameters:")
        print(f"  BARRIER = {self.barrier:.2f} kJ/mol")
        print(f"  kT = {self.kT:.2f} kJ/mol")
        print(f"  bias_factor (γ) = {self.bias_factor:.2f}")
        print(f"  epsilon = {self.epsilon:.6f}")

        self.stride = int(stride)
        self.compression_threshold = compression_threshold
        self.compression_threshold2 = compression_threshold ** 2
        self.saveFrequency = int(saveFrequency)
        self.biasDir = biasDir
        self.adaptive_sigma = adaptive_sigma
        self.use_tabulated_bias = self.num_cvs == 2 and all(p is not None for p in self.periodic)
        self.bias_grid_points = 96
        self.bias_update_stride = 1
        self.bias_function = None
        self.bias_grid = None

        # Initial bandwidth (sigma)
        if sigma is None:
            sigma = [0.1 * unit.radian] * self.num_cvs
        if not hasattr(sigma, '__iter__') or isinstance(sigma, unit.Quantity):
            sigma = [sigma] * self.num_cvs
        if len(sigma) != self.num_cvs:
            raise ValueError("`sigma` must have the same length as `variables`")

        self.sigma = list(sigma)
        self.sigma0_vals = []  # Initial sigma values
        self.sigma_vals = []    # Current sigma values

        for i, s in enumerate(self.sigma):
            if hasattr(s, 'unit'):
                # This OPES implementation is used for torsion CVs in this repo.
                # Keep the conversion simple and consistent across periodic CVs.
                val = s.value_in_unit(unit.radian)
            else:
                val = float(s)
            self.sigma0_vals.append(float(np.abs(val)))
            self.sigma_vals.append(float(np.abs(val)))

        # Storage for kernels: [cv0, cv1, ..., sigma0, sigma1, ..., weight, height]
        # weight carries the kernel amplitude (PLUMED's "height" after sigma
        # rescaling); height is kept at 1.0 so that amplitude = weight * height.
        self.kernels = []
        self.kernel_counter = 0

        # Z_n / Zed: normalization over explored CV space
        self.Zed = 1.0
        self.Zn = 1.0

        # Create the bias force
        self._createBiasForce()

        # Statistics — sum_weights tracks the raw reweighting factors
        # exp(V/kT) BEFORE sigma rescaling, matching PLUMED's sum_weights_.
        self.step_count = 0
        self.sum_weights = 0.0
        self.sum_weights_sq = 0.0

        # PLUMED skips the very first update() call (useful for restarts).
        self._is_first_step = True
        # Counter of actual deposition calls (including the skipped first one)
        self._counter = 0

        # --- Performance: delta-kernel tracking for incremental Zed update ---
        # Inspired by OPESmetad.cpp's delta_kernels_ mechanism which avoids
        # the O(N²) full Zed recomputation on every step.
        self._delta_kernels = []   # list of (height, center_list, sigma_list)
        self._old_KDEnorm = 0.0
        self._old_nker = 0

        # --- Performance: numpy array cache for vectorized operations ---
        self._np_centers = np.empty((0, self.num_cvs))
        self._np_sigmas = np.empty((0, self.num_cvs))
        self._np_amps = np.empty(0)

    # ------------------------------------------------------------------
    # Periodic helpers
    # ------------------------------------------------------------------

    def _periodic_difference(self, value, center, cv_index):
        """Calculate periodic difference for a CV."""

        if self.periodic[cv_index] is None:
            delta = value - center
            return float(delta) if np.isscalar(delta) else delta

        period_min, period_max = self.periodic[cv_index]
        period = period_max - period_min

        diff = value - center
        diff = diff - period * np.round(diff / period)
        return float(diff) if np.isscalar(diff) else diff

    def _periodic_range(self, cv_index):
        periodic = self.periodic[cv_index]
        if periodic is None:
            raise ValueError("Requested periodic range for a non-periodic CV")
        return periodic

    # ------------------------------------------------------------------
    # Kernel accessors
    # ------------------------------------------------------------------

    def _kernel_sigma(self, kernel):
        """Return the per-kernel sigma vector from a stored kernel row."""
        if len(kernel) >= 2 * self.num_cvs + 2:
            return [float(kernel[self.num_cvs + i]) for i in range(self.num_cvs)]
        return list(self.sigma_vals)

    def _kernel_weight(self, kernel):
        return float(kernel[2 * self.num_cvs]) if len(kernel) >= 2 * self.num_cvs + 2 else float(kernel[self.num_cvs])

    def _kernel_height(self, kernel):
        return float(kernel[2 * self.num_cvs + 1]) if len(kernel) >= 2 * self.num_cvs + 2 else float(kernel[self.num_cvs + 1])

    def _kernel_center(self, kernel):
        return [float(kernel[i]) for i in range(self.num_cvs)]

    def _kernel_periodic_difference(self, value, center, cv_index):
        return self._periodic_difference(value, center, cv_index)

    def _kernel_row(self, cv_values, sigma_vals, weight, height):
        return [float(v) for v in cv_values] + [float(s) for s in sigma_vals] + [float(weight), float(height)]

    # ------------------------------------------------------------------
    # Kernel amplitude helpers
    # ------------------------------------------------------------------

    def _kernel_amplitude(self, kernel):
        """Effective amplitude of a stored kernel (weight × height)."""
        return self._kernel_weight(kernel) * self._kernel_height(kernel)

    # ------------------------------------------------------------------
    # Numpy cache for vectorized operations
    # ------------------------------------------------------------------

    def _build_numpy_cache(self):
        """Rebuild numpy arrays from the kernel list.

        This is O(N) and called once per deposition step — negligible
        compared to the kernel evaluations it accelerates.
        """
        n = len(self.kernels)
        if n == 0:
            self._np_centers = np.empty((0, self.num_cvs))
            self._np_sigmas = np.empty((0, self.num_cvs))
            self._np_amps = np.empty(0)
            return
        arr = np.array(self.kernels, dtype=np.float64)
        self._np_centers = arr[:, :self.num_cvs]
        self._np_sigmas = arr[:, self.num_cvs:2 * self.num_cvs]
        self._np_amps = arr[:, 2 * self.num_cvs] * arr[:, 2 * self.num_cvs + 1]

    # ------------------------------------------------------------------
    # Kernel evaluation (vectorized)
    # ------------------------------------------------------------------

    def _evaluate_probability_kernel_set(self, cv_values, kernels=None):
        """Evaluate the KDE probability using vectorized numpy operations.

        Returns  sum(amplitude_k * G_k(s)) / sum_weights.
        """
        if kernels is not None:
            return self._evaluate_probability_kernel_set_list(cv_values, kernels)
        if len(self.kernels) == 0 or self.sum_weights <= 0:
            return 0.0

        cv = np.asarray(cv_values, dtype=np.float64)
        diff = cv[np.newaxis, :] - self._np_centers          # (N, ncv)
        for j in range(self.num_cvs):
            if self.periodic[j] is not None:
                period = self.periodic[j][1] - self.periodic[j][0]
                diff[:, j] -= period * np.round(diff[:, j] / period)
        norm2 = np.sum((diff / self._np_sigmas) ** 2, axis=1)  # (N,)
        mask = norm2 < self.kernel_cutoff2
        gaussian = np.where(mask, np.exp(-0.5 * norm2) - self.val_at_cutoff, 0.0)
        return float(np.sum(self._np_amps * gaussian)) / self.sum_weights

    def _evaluate_probability_kernel_set_list(self, cv_values, kernels):
        """Fallback list-based probability evaluation for custom kernel sets."""
        if len(kernels) == 0:
            return 0.0
        weighted_sum = 0.0
        for kernel in kernels:
            centers = self._kernel_center(kernel)
            sigma_vals = self._kernel_sigma(kernel)
            norm2 = 0.0
            for j in range(self.num_cvs):
                diff = self._periodic_difference(cv_values[j], centers[j], j)
                norm2 += (diff / sigma_vals[j]) ** 2
            if norm2 >= self.kernel_cutoff2:
                gaussian = 0.0
            else:
                gaussian = math.exp(-0.5 * norm2) - self.val_at_cutoff
            weighted_sum += self._kernel_amplitude(kernel) * gaussian
        return weighted_sum / self.sum_weights if self.sum_weights > 0 else 0.0

    def _evaluateKernel(self, kernel, cv_values):
        """Evaluate a single kernel at a point, matching OPESmetad.cpp."""
        centers = self._kernel_center(kernel)
        sigma_vals = self._kernel_sigma(kernel)
        norm2 = 0.0
        for i in range(self.num_cvs):
            diff = self._periodic_difference(cv_values[i], centers[i], i)
            norm2 += (diff / sigma_vals[i]) ** 2
            if norm2 >= self.kernel_cutoff2:
                return 0.0
        return self._kernel_amplitude(kernel) * (math.exp(-0.5 * norm2) - self.val_at_cutoff)

    def _build_bias_table(self):
        """Build a 2D tabulated bias from the current kernel set.

        Uses chunked kernel processing to keep memory bounded at O(C*Ny*Nx)
        instead of O(N*Ny*Nx), where C is a small constant chunk size.
        Inspired by OPESmetad.cpp which processes kernels with cache-friendly
        access patterns.
        """
        if not self.use_tabulated_bias:
            return None

        periodic0 = self._periodic_range(0)
        periodic1 = self._periodic_range(1)
        x_min = periodic0[0]
        x_max = periodic0[1]
        y_min = periodic1[0]
        y_max = periodic1[1]

        if self.bias_grid is None:
            self.bias_grid = (
                np.linspace(x_min, x_max, self.bias_grid_points),
                np.linspace(y_min, y_max, self.bias_grid_points),
            )
            # Cache meshgrid — these never change
            x_grid, y_grid = self.bias_grid
            self._bias_X, self._bias_Y = np.meshgrid(x_grid, y_grid)

        X = self._bias_X  # (Ny, Nx)
        Y = self._bias_Y

        n = len(self.kernels)
        ng = self.bias_grid_points
        prob = np.zeros(ng * ng, dtype=np.float64).reshape(ng, ng)

        if n > 0:
            centers = self._np_centers   # (N, 2)
            sigmas = self._np_sigmas     # (N, 2)
            amps = self._np_amps         # (N,)
            p0 = (self.periodic[0][1] - self.periodic[0][0]) if self.periodic[0] is not None else 0.0
            p1 = (self.periodic[1][1] - self.periodic[1][0]) if self.periodic[1] is not None else 0.0
            cutoff2 = self.kernel_cutoff2
            vac = self.val_at_cutoff

            # Process kernels in small chunks to bound peak memory at
            # chunk_size * Ny * Nx * 8 bytes  (~2.4 MB for chunk=32, 96×96)
            # instead of N * Ny * Nx * 8 bytes (~37 MB at N=500).
            chunk_size = 32
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                c = centers[start:end]      # (C, 2)
                s = sigmas[start:end]       # (C, 2)
                a = amps[start:end]         # (C,)

                dx = X[np.newaxis, :, :] - c[:, 0, np.newaxis, np.newaxis]
                dy = Y[np.newaxis, :, :] - c[:, 1, np.newaxis, np.newaxis]

                if p0 != 0.0:
                    dx -= p0 * np.round(dx / p0)
                if p1 != 0.0:
                    dy -= p1 * np.round(dy / p1)

                norm2 = (dx / s[:, 0, np.newaxis, np.newaxis]) ** 2 + \
                         (dy / s[:, 1, np.newaxis, np.newaxis]) ** 2
                mask = norm2 < cutoff2
                # Only compute exp where mask is True to save FLOPs
                gaussian = np.where(mask, np.exp(-0.5 * norm2) - vac, 0.0)
                prob += np.einsum('k,kij->ij', a, gaussian)

        if self.sum_weights > 0:
            prob *= (1.0 / self.sum_weights)

        regularized = prob / self.Zed + self.epsilon
        np.clip(regularized, self.epsilon, None, out=regularized)
        coeff = (1.0 - 1.0 / self.bias_factor) * self.kT
        bias = coeff * np.log(regularized)
        bias -= np.min(bias)

        # Continuous2DFunction(periodic=True) requires the first and last
        # tabulated points along each axis to match exactly. Enforce that here
        # so the spline constructor does not fail near the periodic boundary.
        bias[0, :] = bias[-1, :] = 0.5 * (bias[0, :] + bias[-1, :])
        bias[:, 0] = bias[:, -1] = 0.5 * (bias[:, 0] + bias[:, -1])
        corner = float(np.mean([bias[0, 0], bias[0, -1], bias[-1, 0], bias[-1, -1]]))
        bias[0, 0] = bias[0, -1] = bias[-1, 0] = bias[-1, -1] = corner

        # Continuous2DFunction expects a flat list of z values.
        return bias.ravel(order='C')

    def _createBiasForce(self):
        """Create the CustomCVForce for OPES bias."""

        if self.use_tabulated_bias:
            periodic0 = self._periodic_range(0)
            periodic1 = self._periodic_range(1)
            x_min = periodic0[0]
            x_max = periodic0[1]
            y_min = periodic1[0]
            y_max = periodic1[1]
            self.bias_function = Continuous2DFunction(
                self.bias_grid_points,
                self.bias_grid_points,
                np.zeros(self.bias_grid_points * self.bias_grid_points),
                x_min,
                x_max,
                y_min,
                y_max,
                True,
            )
            self.force = CustomCVForce("bias(cv0,cv1)")
            self.force.addTabulatedFunction("bias", self.bias_function)
        else:
            self.force = CustomCVForce("0")  # Start with zero bias

        for i, var in enumerate(self.variables):
            self.force.addCollectiveVariable(f"cv{i}", var)

        self.force.setForceGroup(15)
        self.system.addForce(self.force)

    def _evaluateBias(self, cv_values):
        """
        Evaluate bias at given CV values.

        V(s) = (1 - 1/γ) * kT * log(P(s)/Z_n + ε)
        """

        prob = self._evaluate_probability_kernel_set(cv_values)

        # Regularized probability: P/Z + ε
        regularized_prob = prob / self.Zed + self.epsilon

        # Avoid log of non-positive values
        if regularized_prob <= 0 or not np.isfinite(regularized_prob):
            regularized_prob = self.epsilon

        # Well-tempered bias
        bias = (1.0 - 1.0/self.bias_factor) * self.kT * math.log(regularized_prob)

        return float(bias)

    # ------------------------------------------------------------------
    # Zed normalisation — incremental update (OPESmetad.cpp pattern)
    # ------------------------------------------------------------------

    def _updateZn(self):
        """Update Zed using incremental delta-kernel approach.

        Instead of the O(N²) full double-sum over all kernel pairs,
        this tracks which kernels were added/removed (delta_kernels)
        and computes only the O(N·D) correction, where D is typically 2-3.

        Falls back to the full (but numpy-vectorized) computation when
        the kernel set is very small or no delta information is available.
        """
        if len(self.kernels) == 0 or self.sum_weights <= 0:
            self.Zed = 1.0
            self.Zn = 1.0
            return

        n_kernels = len(self.kernels)
        n_delta = len(self._delta_kernels)
        old_KDEnorm = self._old_KDEnorm
        old_nker = self._old_nker

        # Heuristic from OPESmetad.cpp: full recomputation is cheaper when
        # N² < 3·N·D + 2·D² (+ some overhead constant).
        few_kernels = (n_kernels * n_kernels
                       < 3 * n_kernels * n_delta + 2 * n_delta * n_delta + 100)

        if old_nker == 0 or old_KDEnorm <= 0 or n_delta == 0 or few_kernels:
            self._updateZn_full()
            return

        # --- Incremental update ---
        delta_sum = 0.0

        for d in range(n_delta):
            d_height, d_center, d_sigma = self._delta_kernels[d]
            d_center_arr = np.asarray(d_center, dtype=np.float64)
            d_sigma_arr = np.asarray(d_sigma, dtype=np.float64)
            d_sign = 1.0 if d_height >= 0 else -1.0

            # Part A: evaluateKernel(delta_d, center_k) for all k
            diff1 = self._np_centers - d_center_arr[np.newaxis, :]   # (N, ncv)
            for j in range(self.num_cvs):
                if self.periodic[j] is not None:
                    p = self.periodic[j][1] - self.periodic[j][0]
                    diff1[:, j] -= p * np.round(diff1[:, j] / p)
            norm2_1 = np.sum((diff1 / d_sigma_arr[np.newaxis, :]) ** 2, axis=1)
            mask1 = norm2_1 < self.kernel_cutoff2
            vals1 = np.where(mask1,
                             d_height * (np.exp(-0.5 * norm2_1) - self.val_at_cutoff),
                             0.0)

            # Part B: sign * evaluateKernel(kernel_k, center_d) for all k
            diff2 = d_center_arr[np.newaxis, :] - self._np_centers   # (N, ncv)
            for j in range(self.num_cvs):
                if self.periodic[j] is not None:
                    p = self.periodic[j][1] - self.periodic[j][0]
                    diff2[:, j] -= p * np.round(diff2[:, j] / p)
            norm2_2 = np.sum((diff2 / self._np_sigmas) ** 2, axis=1)
            mask2 = norm2_2 < self.kernel_cutoff2
            vals2 = np.where(mask2,
                             d_sign * self._np_amps * (np.exp(-0.5 * norm2_2)
                                                       - self.val_at_cutoff),
                             0.0)

            delta_sum += float(np.sum(vals1) + np.sum(vals2))

        # Part C: subtract delta–delta overcounting (D² terms, D is tiny)
        for d in range(n_delta):
            d_h, d_c, d_s = self._delta_kernels[d]
            d_sign = 1.0 if d_h >= 0 else -1.0
            d_c_arr = np.asarray(d_c, dtype=np.float64)
            for dd in range(n_delta):
                dd_h, dd_c, dd_s = self._delta_kernels[dd]
                dd_c_arr = np.asarray(dd_c, dtype=np.float64)
                dd_s_arr = np.asarray(dd_s, dtype=np.float64)
                diff = d_c_arr - dd_c_arr
                for j in range(self.num_cvs):
                    if self.periodic[j] is not None:
                        p = self.periodic[j][1] - self.periodic[j][0]
                        diff[j] -= p * round(float(diff[j]) / p)
                n2 = float(np.sum((diff / dd_s_arr) ** 2))
                if n2 < self.kernel_cutoff2:
                    delta_sum -= d_sign * dd_h * (math.exp(-0.5 * n2)
                                                  - self.val_at_cutoff)

        sum_uprob = self.Zed * old_KDEnorm * old_nker + delta_sum
        self.Zed = sum_uprob / self.sum_weights / n_kernels
        self.Zn = self.Zed

    def _updateZn_full(self):
        """Full Zed computation, fully vectorized with numpy.

        O(N²) in kernel evaluations, processed in chunks to bound memory
        at O(C*N) instead of O(N²).  Eliminates the Python for-loop that
        previously iterated over each kernel individually.
        """
        if len(self.kernels) == 0 or self.sum_weights <= 0:
            self.Zed = 1.0
            self.Zn = 1.0
            return

        n = len(self.kernels)
        centers = self._np_centers   # (N, ncv)
        sigmas = self._np_sigmas     # (N, ncv)
        amps = self._np_amps         # (N,)
        sum_uprob = 0.0

        # Process outer dimension in chunks to cap memory at chunk*N*ncv
        chunk = min(n, 64)
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            # (C, 1, ncv) - (1, N, ncv) → (C, N, ncv)
            diff = (centers[start:end, np.newaxis, :]
                    - centers[np.newaxis, :, :])
            for j in range(self.num_cvs):
                if self.periodic[j] is not None:
                    p = self.periodic[j][1] - self.periodic[j][0]
                    diff[:, :, j] -= p * np.round(diff[:, :, j] / p)
            norm2 = np.sum((diff / sigmas[np.newaxis, :, :]) ** 2, axis=2)  # (C, N)
            mask = norm2 < self.kernel_cutoff2
            vals = np.where(mask,
                            amps[np.newaxis, :] * (np.exp(-0.5 * norm2)
                                                   - self.val_at_cutoff),
                            0.0)
            sum_uprob += float(np.sum(vals))

        self.Zed = sum_uprob / self.sum_weights / n
        self.Zn = self.Zed

    def _updateBiasExpression(self, context=None):
        """
        Update the bias energy expression.

        V(s) = (1-1/γ) * kT * log(P(s)/Z_n + ε)
        """

        if self.use_tabulated_bias:
            if self.bias_function is None:
                return
            values = self._build_bias_table()
            if values is None:
                return
            periodic0 = self._periodic_range(0)
            periodic1 = self._periodic_range(1)
            self.bias_function.setFunctionParameters(
                self.bias_grid_points,
                self.bias_grid_points,
                values,
                periodic0[0],
                periodic0[1],
                periodic1[0],
                periodic1[1]
            )
            if context is not None:
                self.force.updateParametersInContext(context)
            return

        if len(self.kernels) == 0:
            coeff = (1.0 - 1.0/self.bias_factor) * self.kT
            bias_expr = f"{coeff:.12g}*{math.log(self.epsilon):.12g}"
            self.force.setEnergyFunction(bias_expr)
            if context is not None:
                self.force.updateParametersInContext(context)
            return

        # Build kernel sum expression: Σ amp_k * G(s,s_k)
        kernel_terms = []

        for kernel in self.kernels:
            # Gaussian terms for each CV
            gaussian_terms = []
            centers = self._kernel_center(kernel)
            sigma_vals = self._kernel_sigma(kernel)
            for j in range(self.num_cvs):
                center = float(centers[j])
                sigma_j = float(sigma_vals[j])

                if self.periodic[j] is None:
                    diff = f"(cv{j}-{center:.12g})"
                else:
                    periodic = self.periodic[j]
                    if periodic is None:
                        continue
                    pmin = periodic[0]
                    pmax = periodic[1]
                    period = pmax - pmin
                    diff = f"((cv{j}-{center:.12g})-{period:.12g}*floor(((cv{j}-{center:.12g})/{period:.12g})+0.5))"

                gaussian_terms.append(f"(({diff})^2/{(sigma_j**2):.12g})")

            gaussian_expr = "exp(-0.5*(" + "+".join(gaussian_terms) + "))"
            amp = self._kernel_amplitude(kernel)

            kernel_terms.append(f"({amp:.12g}*{gaussian_expr})")

        # P(s) = Σ amp_k * G(s,s_k) / sum_weights
        prob_expr = f"({'+'.join(kernel_terms)})/{self.sum_weights:.12g}"

        # P(s)/Z_n + ε
        regularized_expr = f"({prob_expr})/{self.Zn:.12g}+{self.epsilon:.12g}"

        # V(s) = (1-1/γ) * kT * log(P(s)/Z_n + ε)
        coeff = (1.0 - 1.0/self.bias_factor) * self.kT
        bias_expr = f"{coeff:.12g}*log({regularized_expr})"

        self.force.setEnergyFunction(bias_expr)
        if context is not None:
            self.force.updateParametersInContext(context)

    # ------------------------------------------------------------------
    # Bandwidth adaptation  (Silverman's rule, PLUMED's !fixed_sigma_)
    # ------------------------------------------------------------------

    def _adaptBandwidth(self, neff=None):
        """
        Adapt bandwidth using Silverman's rule.

        σ_i^(n) = σ_i^(0) * [N_eff * (d+2)/4]^(-1/(d+4))

        Uses PLUMED's robust neff formula: (1 + Σw)² / (1 + Σw²).
        """
        if not self.adaptive_sigma:
            return

        if neff is None:
            neff = (1.0 + self.sum_weights) ** 2 / (1.0 + self.sum_weights_sq)

        d = self.num_cvs
        factor = (neff * (d + 2) / 4.0) ** (-1.0 / (d + 4))

        for i in range(self.num_cvs):
            self.sigma_vals[i] = self.sigma0_vals[i] * factor

    # ------------------------------------------------------------------
    # PLUMED-style kernel addition with compression
    # ------------------------------------------------------------------

    def _getMergeableKernel(self, center, exclude_idx):
        """Find the closest kernel within the compression threshold.

        Vectorized with numpy for O(N) with fast constant factors.
        Uses the cached numpy arrays when they are still valid (same size
        as kernel list), avoiding redundant list→array conversion.
        """
        n = len(self.kernels)
        if n == 0:
            return None, self.compression_threshold2

        # Use cached arrays if they have the right size (they are fresh
        # for the first call in _addKernel, before any modification).
        # During recursive merge, kernels may have been popped → size mismatch
        # → rebuild from the list.
        if len(self._np_centers) == n:
            centers_arr = self._np_centers
            sigmas_arr = self._np_sigmas
        else:
            centers_arr = np.array([k[:self.num_cvs] for k in self.kernels],
                                   dtype=np.float64)
            sigmas_arr = np.array([k[self.num_cvs:2 * self.num_cvs]
                                   for k in self.kernels], dtype=np.float64)

        center_arr = np.asarray(center, dtype=np.float64)
        diff = center_arr[np.newaxis, :] - centers_arr          # (N, ncv)
        for j in range(self.num_cvs):
            if self.periodic[j] is not None:
                period = self.periodic[j][1] - self.periodic[j][0]
                diff[:, j] -= period * np.round(diff[:, j] / period)
        norm2 = np.sum((diff / sigmas_arr) ** 2, axis=1)       # (N,)

        if 0 <= exclude_idx < n:
            norm2[exclude_idx] = np.inf

        min_k = int(np.argmin(norm2))
        min_norm2 = float(norm2[min_k])

        if min_norm2 < self.compression_threshold2:
            return min_k, min_norm2
        return None, self.compression_threshold2

    def _mergeKernels(self, target, source):
        """Merge *source* kernel into *target* kernel in-place.

        Matches PLUMED's ``mergeKernels``: heights add, centres and sigmas
        are height-weighted averages preserving the second moment.
        """
        h1 = self._kernel_amplitude(target)
        h2 = self._kernel_amplitude(source)
        h_total = h1 + h2

        c1 = self._kernel_center(target)
        c2 = self._kernel_center(source)
        s1 = self._kernel_sigma(target)
        s2 = self._kernel_sigma(source)

        for j in range(self.num_cvs):
            is_periodic = self.periodic[j] is not None
            if is_periodic:
                # Fix PBC: bring c1 close to c2
                c1[j] = c2[j] + self._periodic_difference(c1[j], c2[j], j)

            c_merged = (h1 * c1[j] + h2 * c2[j]) / h_total
            ss = (h1 * (s1[j] ** 2 + c1[j] ** 2) +
                  h2 * (s2[j] ** 2 + c2[j] ** 2)) / h_total - c_merged ** 2

            if is_periodic:
                pmin, pmax = self.periodic[j]
                period = pmax - pmin
                c_merged = pmin + ((c_merged - pmin) % period)

            target[j] = c_merged
            target[self.num_cvs + j] = math.sqrt(max(ss, 1e-16))

        # Store merged amplitude in the weight slot; height stays 1.0.
        target[2 * self.num_cvs] = h_total
        target[2 * self.num_cvs + 1] = 1.0

    def _addKernel(self, kernel):
        """Add a kernel with PLUMED-style merge + recursive merge.

        Also populates self._delta_kernels for the incremental Zed update.
        """
        if self.compression_threshold2 > 0 and len(self.kernels) > 0:
            center_new = self._kernel_center(kernel)
            taker_k, norm2 = self._getMergeableKernel(center_new, exclude_idx=-1)

            if taker_k is not None:
                # Snapshot the old taker (negative = removed)
                old = self.kernels[taker_k]
                self._delta_kernels.append((
                    -self._kernel_amplitude(old),
                    self._kernel_center(old),
                    self._kernel_sigma(old),
                ))

                # Merge new kernel into the closest existing one
                self._mergeKernels(self.kernels[taker_k], kernel)

                # Snapshot the new taker (positive = added)
                new = self.kernels[taker_k]
                self._delta_kernels.append((
                    self._kernel_amplitude(new),
                    self._kernel_center(new),
                    self._kernel_sigma(new),
                ))

                # Recursive merge: the merged kernel might now overlap another
                giver_k = taker_k
                while True:
                    center_g = self._kernel_center(self.kernels[giver_k])
                    taker_k2, norm2_2 = self._getMergeableKernel(center_g, exclude_idx=giver_k)
                    if taker_k2 is None:
                        break

                    # The last delta (positive) is about to be merged again — pop it
                    self._delta_kernels.pop()

                    # Snapshot old taker2 (negative = removed)
                    old2 = self.kernels[taker_k2]
                    self._delta_kernels.append((
                        -self._kernel_amplitude(old2),
                        self._kernel_center(old2),
                        self._kernel_sigma(old2),
                    ))

                    # Keep the lower index to avoid shifting issues on pop
                    if taker_k2 > giver_k:
                        taker_k2, giver_k = giver_k, taker_k2
                    self._mergeKernels(self.kernels[taker_k2], self.kernels[giver_k])

                    # Snapshot new merged result (positive = added)
                    merged = self.kernels[taker_k2]
                    self._delta_kernels.append((
                        self._kernel_amplitude(merged),
                        self._kernel_center(merged),
                        self._kernel_sigma(merged),
                    ))

                    self.kernels.pop(giver_k)
                    giver_k = taker_k2
                return

        # No merge — append as a new kernel
        self.kernels.append(list(kernel))
        self._delta_kernels.append((
            self._kernel_amplitude(kernel),
            self._kernel_center(kernel),
            self._kernel_sigma(kernel),
        ))

    def _evaluateProbability(self, cv_values):
        """
        Evaluate probability estimate at given CV values.
        P(s) = Σ amp_k * G(s, s_k) / sum_weights
        """

        return self._evaluate_probability_kernel_set(cv_values)

    # ------------------------------------------------------------------
    # Main simulation driver
    # ------------------------------------------------------------------

    def step(self, simulation, steps):
        """Advance the simulation while depositing OPES kernels.

        Follows the PLUMED OPESmetad update() flow:
          1. Compute log_weight = V(s)/kT from the current bias.
          2. raw_weight = exp(log_weight) ; accumulate into sum_weights
             **before** any sigma rescaling.
          3. Compute neff with PLUMED's robust formula (1+Σw)²/(1+Σw²).
          4. Adapt bandwidth (Silverman rule) using neff.
          5. Rescale height: kernel_height = raw_weight × Π(σ0/σ).
          6. Add kernel (with compression / recursive merge).
          7. Update Zed normalisation.
          8. Update the bias force expression / table.
        """

        steps_remaining = int(steps)

        while steps_remaining > 0:
            # Calculate steps until next kernel deposition
            steps_until_next = self.stride - (self.step_count % self.stride)
            steps_to_run = min(steps_until_next, steps_remaining)

            # Run simulation
            simulation.step(steps_to_run)
            self.step_count += steps_to_run
            steps_remaining -= steps_to_run

            # Deposit kernel at stride
            if (self.step_count % self.stride) == 0:
                # PLUMED skips the very first update (useful for restarts).
                if self._is_first_step:
                    self._is_first_step = False
                    continue

                # Get current CV values
                cv_values = self.force.getCollectiveVariableValues(simulation.context)
                cv_values = [float(x) for x in cv_values]

                # Build numpy cache for vectorized bias evaluation
                self._build_numpy_cache()

                # ---- 1. log_weight from the current bias ----
                current_bias = self._evaluateBias(cv_values)
                log_weight = self.beta * current_bias
                raw_weight = math.exp(log_weight)

                # ---- 2. Save old state for incremental Zed ----
                self._old_KDEnorm = self.sum_weights   # KDEnorm BEFORE update
                self._old_nker = len(self.kernels)       # N_kernels BEFORE add

                self._counter += 1
                self.sum_weights += raw_weight
                self.sum_weights_sq += raw_weight ** 2

                # ---- 3. neff (PLUMED's robust formula) ----
                neff = (1.0 + self.sum_weights) ** 2 / (1.0 + self.sum_weights_sq)

                self.kernel_counter += 1

                # ---- 4. Adapt bandwidth every step ----
                self._adaptBandwidth(neff)

                # ---- 5. Sigma-rescaled kernel height ----
                kernel_height = raw_weight
                for s0, s in zip(self.sigma0_vals, self.sigma_vals):
                    if s <= 0:
                        raise ValueError("sigma values must stay positive")
                    kernel_height *= s0 / s

                # ---- 6. Add kernel (with PLUMED-style compression) ----
                self._delta_kernels = []
                kernel = self._kernel_row(cv_values, self.sigma_vals,
                                          kernel_height, 1.0)
                old_nker = len(self.kernels)
                self._addKernel(kernel)
                new_nker = len(self.kernels)
                if new_nker < old_nker:
                    print(f"OPES: Compressed {old_nker + 1} → {new_nker} kernels")

                # ---- 7. Update Zed (incremental when possible) ----
                self._build_numpy_cache()
                self._updateZn()

                # ---- 8. Rebuild bias expression / table ----
                if ((not self.use_tabulated_bias)
                        or (self.kernel_counter % self.bias_update_stride == 0)
                        or self.kernel_counter == 1):
                    self._updateBiasExpression(simulation.context)

                # Save periodically
                if self.biasDir and (self.step_count % self.saveFrequency == 0):
                    self.saveKernels()

    def saveKernels(self):
        """Save kernel information to file."""

        if not self.biasDir:
            return

        os.makedirs(self.biasDir, exist_ok=True)

        txt_name = os.path.join(self.biasDir, f"kernels_{self.step_count}.txt")
        npz_name = os.path.join(self.biasDir, f"kernels_{self.step_count}.npz")

        with open(txt_name, 'w') as f:
            f.write(f"# OPES Kernels at step {self.step_count}\n")
            f.write(f"# Temperature: {self.kT} kJ/mol\n")
            f.write(f"# Barrier: {self.barrier} kJ/mol\n")
            f.write(f"# Bias factor: {self.bias_factor}\n")
            f.write(f"# Epsilon: {self.epsilon}\n")
            f.write(f"# Kernel_cutoff: {self.kernel_cutoff}\n")
            f.write(f"# Z_n: {self.Zn}\n")
            f.write(f"# Zed: {self.Zed}\n")
            f.write(f"# Sum_weights: {self.sum_weights}\n")
            f.write(f"# Sigma_vals: {' '.join(str(s) for s in self.sigma_vals)}\n")
            f.write("# Columns: ")
            for i in range(self.num_cvs):
                f.write(f"cv{i} ")
            for i in range(self.num_cvs):
                f.write(f"sigma{i} ")
            f.write("weight height\n")

            for kernel in self.kernels:
                f.write(" ".join(map(str, kernel)) + "\n")

        np.savez_compressed(
            npz_name,
            kernels=np.array(self.kernels),
            sigma_vals=np.array(self.sigma_vals),
            sigma0_vals=np.array(self.sigma0_vals),
            adaptive_sigma=np.array(self.adaptive_sigma),
            sum_weights=np.array(self.sum_weights),
            sum_weights_sq=np.array(self.sum_weights_sq),
            kT=np.array(self.kT),
            Zed=np.array(self.Zed),
            Zn=np.array(self.Zn),
            barrier=np.array(self.barrier),
            bias_factor=np.array(self.bias_factor),
            kernel_cutoff=np.array(self.kernel_cutoff),
            format_version=np.array(3),
        )

    def loadKernels(self, filename):
        """Load kernels from file."""

        self.kernels = []
        if filename.endswith('.npz'):
            data = np.load(filename)
            arr = data['kernels']
            for row in arr:
                self.kernels.append([float(x) for x in row])
            # Restore running sums if available
            if 'sum_weights' in data:
                self.sum_weights = float(data['sum_weights'])
            if 'sum_weights_sq' in data:
                self.sum_weights_sq = float(data['sum_weights_sq'])
        else:
            with open(filename, 'r') as f:
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue
                    values = list(map(float, line.split()))
                    if len(values) in (self.num_cvs + 2, 2 * self.num_cvs + 2):
                        self.kernels.append(values)

        # Rebuild numpy cache and Zed/bias from loaded kernels
        self._build_numpy_cache()
        self._updateZn_full()
        self._updateBiasExpression()

        print(f"Loaded {len(self.kernels)} kernels from {filename}")

    def getFreeEnergy(self, cv_grid):
        """
        Estimate free energy on a grid (vectorized).

        F(s) = -kT * log(P(s))
        """
        self._build_numpy_cache()

        grids = np.meshgrid(*cv_grid, indexing='ij')
        shape = grids[0].shape

        cv_points = np.array([g.flatten() for g in grids]).T   # (M, ncv)
        M = len(cv_points)

        if len(self.kernels) == 0 or self.sum_weights <= 0:
            return np.zeros(shape)

        free_energy = np.full(M, np.inf)

        # Process in chunks to limit memory: chunk × N × ncv
        chunk_size = 512
        for start in range(0, M, chunk_size):
            end = min(start + chunk_size, M)
            chunk = cv_points[start:end]                          # (C, ncv)

            # (C, 1, ncv) - (1, N, ncv) → (C, N, ncv)
            diff = (chunk[:, np.newaxis, :]
                    - self._np_centers[np.newaxis, :, :])
            for j in range(self.num_cvs):
                if self.periodic[j] is not None:
                    p = self.periodic[j][1] - self.periodic[j][0]
                    diff[:, :, j] -= p * np.round(diff[:, :, j] / p)

            norm2 = np.sum(
                (diff / self._np_sigmas[np.newaxis, :, :]) ** 2,
                axis=2,
            )                                                      # (C, N)
            mask = norm2 < self.kernel_cutoff2
            gaussian = np.where(
                mask,
                np.exp(-0.5 * norm2) - self.val_at_cutoff,
                0.0,
            )                                                      # (C, N)
            prob = (np.sum(self._np_amps[np.newaxis, :] * gaussian, axis=1)
                    / self.sum_weights)                            # (C,)
            valid = prob > 0
            free_energy[start:end] = np.where(
                valid,
                -self.kT * np.log(np.where(valid, prob, 1.0)),
                np.inf,
            )

        free_energy = free_energy.reshape(shape)
        free_energy -= np.nanmin(free_energy)

        return free_energy

