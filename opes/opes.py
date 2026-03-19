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

        # Storage for kernels: [cv0, cv1, ..., weight, height]
        # height is for Gaussian normalization: h = 1/[(2π)^(d/2) * Π σ_i]
        self.kernels = []
        self.kernel_counter = 0

        # Z_n / Zed: normalization over explored CV space
        self.Zed = 1.0
        self.Zn = 1.0

        # Create the bias force
        self._createBiasForce()

        # Statistics
        self.step_count = 0
        self.sum_weights = 0.0
        self.sum_weights_sq = 0.0

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

    def _compute_kernel_height(self, log_weight, sigma_vals):
        """Return the kernel height used by the PLUMED OPES reference.

        The reference stores ``exp(log_weight)`` and rescales by ``sigma0/sigma``
        when adaptive sigma is active. The usual Gaussian normalization constant
        is intentionally omitted because it cancels in the OPES normalization.
        """
        height = math.exp(log_weight)
        for sigma0, sigma in zip(self.sigma0_vals, sigma_vals):
            if sigma <= 0:
                raise ValueError("sigma values must stay positive")
            height *= sigma0 / sigma
        return height

    def _periodic_range(self, cv_index):
        periodic = self.periodic[cv_index]
        if periodic is None:
            raise ValueError("Requested periodic range for a non-periodic CV")
        return periodic

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

    def _evaluate_probability_kernel_set(self, cv_values, kernels=None):
        """Evaluate the KDE probability using the stored kernel set."""
        if kernels is None:
            kernels = self.kernels
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

            weighted_sum += self._kernel_weight(kernel) * self._kernel_height(kernel) * gaussian

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
        return self._kernel_weight(kernel) * self._kernel_height(kernel) * (math.exp(-0.5 * norm2) - self.val_at_cutoff)

    def _build_bias_table(self):
        """Build a 2D tabulated bias from the current kernel set."""
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

        x_grid, y_grid = self.bias_grid
        X, Y = np.meshgrid(x_grid, y_grid)
        prob = np.zeros_like(X)

        for kernel in self.kernels:
            centers = self._kernel_center(kernel)
            sigma_vals = self._kernel_sigma(kernel)
            dx = self._periodic_difference(X, centers[0], 0)
            dy = self._periodic_difference(Y, centers[1], 1)
            norm2 = (dx / sigma_vals[0]) ** 2 + (dy / sigma_vals[1]) ** 2
            gaussian = np.where(norm2 < self.kernel_cutoff2, np.exp(-0.5 * norm2) - self.val_at_cutoff, 0.0)
            prob += self._kernel_weight(kernel) * self._kernel_height(kernel) * gaussian

        if self.sum_weights > 0:
            prob /= self.sum_weights

        regularized = prob / self.Zed + self.epsilon
        regularized = np.clip(regularized, self.epsilon, None)
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

    def _updateZn(self):
        """Update the exact PLUMED-style Zed normalization."""
        if len(self.kernels) == 0 or self.sum_weights <= 0:
            self.Zed = 1.0
            self.Zn = 1.0
            return

        sum_uprob = 0.0
        for kernel in self.kernels:
            center = self._kernel_center(kernel)
            for other_kernel in self.kernels:
                sum_uprob += self._evaluateKernel(other_kernel, center)

        self.Zed = sum_uprob / self.sum_weights / len(self.kernels)
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

        # Build kernel sum expression: Σ w_k * h_k * G(s,s_k)
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
            weight = self._kernel_weight(kernel)
            height = self._kernel_height(kernel)

            kernel_terms.append(f"({weight:.12g}*{height:.12g}*{gaussian_expr})")

        # P(s) = Σ w_k * h_k * G(s,s_k) / Σ w_k
        prob_expr = f"({'+'.join(kernel_terms)})/{self.sum_weights:.12g}"

        # P(s)/Z_n + ε
        regularized_expr = f"({prob_expr})/{self.Zn:.12g}+{self.epsilon:.12g}"

        # V(s) = (1-1/γ) * kT * log(P(s)/Z_n + ε)
        coeff = (1.0 - 1.0/self.bias_factor) * self.kT
        bias_expr = f"{coeff:.12g}*log({regularized_expr})"

        self.force.setEnergyFunction(bias_expr)
        if context is not None:
            self.force.updateParametersInContext(context)

    def _adaptBandwidth(self):
        """
        Adapt bandwidth using Silverman's rule.

        σ_i^(n) = σ_i^(0) * [N_eff * (d+2)/4]^(-1/(d+4))
        """

        if not self.adaptive_sigma or len(self.kernels) < 2:
            return

        # Effective sample size
        N_eff = self.sum_weights**2 / self.sum_weights_sq if self.sum_weights_sq > 0 else 1.0

        # Silverman's rule
        d = self.num_cvs
        factor = (N_eff * (d + 2) / 4.0) ** (-1.0 / (d + 4))

        for i in range(self.num_cvs):
            self.sigma_vals[i] = self.sigma0_vals[i] * factor

    def _compressKernels(self):
        """Compress kernels that are too close together."""

        if len(self.kernels) < 2:
            return

        compressed = []

        for kernel in self.kernels:
            placed = False
            for comp_kernel in compressed:
                # Calculate normalized distance
                dist2 = 0.0
                for j in range(self.num_cvs):
                    diff = self._periodic_difference(kernel[j], comp_kernel[j], j)
                    sigma_j = self._kernel_sigma(comp_kernel)[j]
                    dist2 += (diff / sigma_j) ** 2
                dist = math.sqrt(dist2)

                if dist < self.compression_threshold:
                    # Merge full kernel state (center, sigma, weight, height)
                    w1 = self._kernel_weight(comp_kernel)
                    w2 = self._kernel_weight(kernel)
                    h1 = self._kernel_height(comp_kernel)
                    h2 = self._kernel_height(kernel)
                    tot_w = w1 + w2

                    for j in range(self.num_cvs):
                        c1 = float(comp_kernel[j])
                        c2 = float(kernel[j])
                        periodic = self.periodic[j]
                        if periodic is not None:
                            periodic = self._periodic_range(j)
                            c1 = c2 + self._periodic_difference(c1, c2, j)
                        merged_c = (w1 * c1 + w2 * c2) / tot_w
                        if periodic is not None:
                            pmin = periodic[0]
                            pmax = periodic[1]
                            period = pmax - pmin
                            merged_c = pmin + ((merged_c - pmin) % period)
                        comp_kernel[j] = merged_c

                    sigma1 = self._kernel_sigma(comp_kernel)
                    sigma2 = self._kernel_sigma(kernel)
                    merged_sigma = []
                    for j in range(self.num_cvs):
                        s1 = sigma1[j]
                        s2 = sigma2[j]
                        c1 = float(comp_kernel[j])
                        c2 = float(kernel[j])
                        ss = (w1 * (s1 * s1 + c1 * c1) + w2 * (s2 * s2 + c2 * c2)) / tot_w - ((w1 * c1 + w2 * c2) / tot_w) ** 2
                        merged_sigma.append(max(1e-8, math.sqrt(max(ss, 1e-16))))

                    if len(comp_kernel) >= 2 * self.num_cvs + 2:
                        for j in range(self.num_cvs):
                            comp_kernel[self.num_cvs + j] = merged_sigma[j]
                        comp_kernel[2 * self.num_cvs] = tot_w
                        comp_kernel[2 * self.num_cvs + 1] = (w1 * h1 + w2 * h2) / tot_w
                    else:
                        # Backward-compatible fallback: old kernel rows only store weight/height.
                        comp_kernel[self.num_cvs] += kernel[self.num_cvs]
                    placed = True
                    break

            if not placed:
                compressed.append(list(kernel))

        old_count = len(self.kernels)
        self.kernels = compressed
        new_count = len(self.kernels)

        if new_count < old_count:
            print(f"OPES: Compressed {old_count} kernels to {new_count}")

        # Recalculate statistics after compression/merging.
        self.sum_weights = sum(self._kernel_weight(k) for k in self.kernels)
        self.sum_weights_sq = sum(self._kernel_weight(k) ** 2 for k in self.kernels)

    def _evaluateProbability(self, cv_values):
        """
        Evaluate probability estimate at given CV values.
        P(s) = Σ w_k * h_k * G(s, s_k) / Σ w_k
        """

        return self._evaluate_probability_kernel_set(cv_values)

    def step(self, simulation, steps):
        """Advance the simulation while depositing OPES kernels."""

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
                # Get current CV values
                cv_values = self.force.getCollectiveVariableValues(simulation.context)
                cv_values = [float(x) for x in cv_values]

                # Calculate weight: w_k = exp(β * V_{k-1}(s_k))
                current_bias = self._evaluateBias(cv_values)
                # In the PLUMED reference, the deposited kernel amplitude is
                # carried by the kernel height; here we store that amplitude in
                # the weight slot and keep height as a neutral factor so we do
                # not double-count the kernel contribution during evaluation.
                weight = self._compute_kernel_height(self.beta * current_bias, self.sigma_vals)
                height = 1.0

                # Deposit kernel: [cv0, cv1, ..., sigma0, sigma1, ..., weight, height]
                kernel = self._kernel_row(cv_values, self.sigma_vals, weight, height)
                self.kernels.append(kernel)
                self.kernel_counter += 1

                # Update statistics
                self.sum_weights += weight
                self.sum_weights_sq += weight ** 2

                # Compress kernels periodically BEFORE updating bias
                if self.kernel_counter % 100 == 0:
                    self._adaptBandwidth()
                    self._compressKernels()

                # Recompute the normalization on the final kernel list.
                self._updateZn()

                # Rebuild the tabulated bias only periodically; the
                # deposition math still uses the current kernel set every time.
                if (not self.use_tabulated_bias) or (self.kernel_counter % self.bias_update_stride == 0) or self.kernel_counter == 1:
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
            format_version=np.array(2),
        )

    def loadKernels(self, filename):
        """Load kernels from file."""

        self.kernels = []
        if filename.endswith('.npz'):
            data = np.load(filename)
            arr = data['kernels']
            for row in arr:
                self.kernels.append([float(x) for x in row])
        else:
            with open(filename, 'r') as f:
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue
                    values = list(map(float, line.split()))
                    if len(values) in (self.num_cvs + 2, 2 * self.num_cvs + 2):
                        self.kernels.append(values)

        # Recalculate statistics
        self.sum_weights = sum(self._kernel_weight(k) for k in self.kernels)
        self.sum_weights_sq = sum(self._kernel_weight(k)**2 for k in self.kernels)
        self._updateZn()
        self._updateBiasExpression()

        print(f"Loaded {len(self.kernels)} kernels from {filename}")

    def getFreeEnergy(self, cv_grid):
        """
        Estimate free energy on a grid.

        F(s) = -kT * log(P(s))
        """

        grids = np.meshgrid(*cv_grid, indexing='ij')
        shape = grids[0].shape

        cv_points = np.array([g.flatten() for g in grids]).T

        free_energy = np.zeros(len(cv_points))
        for i, point in enumerate(cv_points):
            prob = self._evaluateProbability(point)
            if prob > 0:
                free_energy[i] = -self.kT * math.log(prob)
            else:
                free_energy[i] = np.inf

        free_energy = free_energy.reshape(shape)
        free_energy -= np.nanmin(free_energy)

        return free_energy

