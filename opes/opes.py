"""
OPES (On-the-fly Probability Enhanced Sampling) for OpenMM
Based on PLUMED implementation: https://github.com/plumed/plumed2/blob/master/src/opes/OPESmetad.cpp
Reference: Invernizzi & Parrinello, J. Phys. Chem. Lett. 2020, 11, 2731-2736
"""

import math
import numpy as np
import os
from openmm import CustomCVForce
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
            self.periodic = [None] * self.num_cvs
        else:
            if len(periodic) != self.num_cvs:
                raise ValueError("`periodic` must have the same length as `variables`")
            self.periodic = periodic

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
                if self.periodic[i] is not None:
                    val = s.value_in_unit(unit.radian)
                else:
                    val = s.value_in_unit(unit.nanometer)
            else:
                val = float(s)
            self.sigma0_vals.append(float(np.abs(val)))
            self.sigma_vals.append(float(np.abs(val)))

        # Storage for kernels: [cv0, cv1, ..., weight, height]
        # height is for Gaussian normalization: h = 1/[(2π)^(d/2) * Π σ_i]
        self.kernels = []
        self.kernel_counter = 0

        # Z_n: normalization over explored CV space
        self.Zn = 1.0

        # Create the bias force
        self._createBiasForce()

        # Statistics
        self.step_count = 0
        self.sum_weights = 0.0
        self.sum_weights_sq = 0.0

    def _createBiasForce(self):
        """Create the CustomCVForce for OPES bias."""

        self.force = CustomCVForce("0")

        for i, var in enumerate(self.variables):
            self.force.addCollectiveVariable(f"cv{i}", var)

        self.force.setForceGroup(15)
        self.system.addForce(self.force)

    def _periodic_difference(self, value, center, cv_index):
        """Calculate periodic difference for a CV."""

        if self.periodic[cv_index] is None:
            return float(value - center)

        period_min, period_max = self.periodic[cv_index]
        period = period_max - period_min

        diff = float(value - center)
        diff = diff - period * round(diff / period)
        return float(diff)

    def _gaussian_height(self):
        """Calculate Gaussian normalization height: h = 1/[(2π)^(d/2) * Π σ_i]"""
        h = 1.0
        for i in range(self.num_cvs):
            h /= (self.sigma_vals[i] * math.sqrt(2.0 * math.pi))
        return h

    def _evaluateProbability(self, cv_values):
        """
        Evaluate probability estimate at given CV values.

        P(s) = Σ w_k * h_k * G(s, s_k) / Σ w_k
        """

        if len(self.kernels) == 0:
            return 0.0  # Will be regularized by epsilon

        weighted_sum = 0.0

        for kernel in self.kernels:
            # Gaussian kernel (already normalized by height)
            gaussian = 1.0
            for j in range(self.num_cvs):
                center = kernel[j]
                sigma_j = self.sigma_vals[j]
                diff = self._periodic_difference(cv_values[j], center, j)
                gaussian *= math.exp(-0.5 * (diff / sigma_j)**2)

            weight = kernel[self.num_cvs]      # w_k
            height = kernel[self.num_cvs + 1]  # h_k (normalization)

            weighted_sum += weight * height * gaussian

        # Normalize by sum of weights
        prob_estimate = weighted_sum / self.sum_weights if self.sum_weights > 0 else 0.0
        return prob_estimate

    def _evaluateBias(self, cv_values):
        """
        Evaluate bias at given CV values.

        V(s) = (1 - 1/γ) * kT * log(P(s)/Z_n + ε)
        """

        prob = self._evaluateProbability(cv_values)

        # Regularized probability: P/Z + ε
        regularized_prob = prob / self.Zn + self.epsilon

        # Avoid log of non-positive values
        if regularized_prob <= 0 or not np.isfinite(regularized_prob):
            regularized_prob = self.epsilon

        # Well-tempered bias
        bias = (1.0 - 1.0/self.bias_factor) * self.kT * math.log(regularized_prob)

        return float(bias)

    def _updateZn(self):
        """
        Update Z_n: normalization over explored CV space.

        Z_n estimates the volume of CV space that has been explored.
        Simple estimate: Z_n = effective_sample_size
        """

        # Effective sample size: N_eff = (Σ w_k)^2 / Σ w_k^2
        if self.sum_weights_sq > 0:
            N_eff = self.sum_weights**2 / self.sum_weights_sq
            self.Zn = max(1.0, N_eff)
        else:
            self.Zn = 1.0

    def _updateBiasExpression(self):
        """
        Update the bias energy expression.

        V(s) = (1-1/γ) * kT * log(P(s)/Z_n + ε)
        """

        if len(self.kernels) == 0:
            # No kernels yet: V = (1-1/γ) * kT * log(ε)
            coeff = (1.0 - 1.0/self.bias_factor) * self.kT
            bias_expr = f"{coeff:.12g}*{math.log(self.epsilon):.12g}"
            self.force.setEnergyFunction(bias_expr)
            return

        # Build kernel sum expression: Σ w_k * h_k * G(s,s_k)
        kernel_terms = []

        for kernel in self.kernels:
            # Gaussian terms for each CV
            gaussian_terms = []
            for j in range(self.num_cvs):
                center = float(kernel[j])
                sigma_j = float(self.sigma_vals[j])

                if self.periodic[j] is None:
                    diff = f"(cv{j}-{center:.12g})"
                else:
                    pmin, pmax = self.periodic[j]
                    period = pmax - pmin
                    diff = f"((cv{j}-{center:.12g})-{period:.12g}*floor(((cv{j}-{center:.12g})/{period:.12g})+0.5))"

                gaussian_terms.append(f"(({diff})^2/{(sigma_j**2):.12g})")

            gaussian_expr = "exp(-0.5*(" + "+".join(gaussian_terms) + "))"
            weight = float(kernel[self.num_cvs])
            height = float(kernel[self.num_cvs + 1])

            kernel_terms.append(f"({weight:.12g}*{height:.12g}*{gaussian_expr})")

        # P(s) = Σ w_k * h_k * G(s,s_k) / Σ w_k
        prob_expr = f"({'+'.join(kernel_terms)})/{self.sum_weights:.12g}"

        # P(s)/Z_n + ε
        regularized_expr = f"({prob_expr})/{self.Zn:.12g}+{self.epsilon:.12g}"

        # V(s) = (1-1/γ) * kT * log(P(s)/Z_n + ε)
        coeff = (1.0 - 1.0/self.bias_factor) * self.kT
        bias_expr = f"{coeff:.12g}*log({regularized_expr})"

        self.force.setEnergyFunction(bias_expr)

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
                    sigma_j = self.sigma_vals[j]
                    dist2 += (diff / sigma_j)**2
                dist = math.sqrt(dist2)

                if dist < self.compression_threshold:
                    # Merge: add weights (heights are the same since sigma is global)
                    comp_kernel[self.num_cvs] += kernel[self.num_cvs]
                    placed = True
                    break

            if not placed:
                compressed.append(list(kernel))

        old_count = len(self.kernels)
        self.kernels = compressed
        new_count = len(self.kernels)

        # Recalculate sum_weights after compression
        self.sum_weights = sum(k[self.num_cvs] for k in self.kernels)
        self.sum_weights_sq = sum(k[self.num_cvs]**2 for k in self.kernels)

        if new_count < old_count:
            print(f"OPES: Compressed {old_count} kernels to {new_count}")

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
                weight = math.exp(self.beta * current_bias)

                # Gaussian height (normalization factor)
                height = self._gaussian_height()

                # Deposit kernel: [cv0, cv1, ..., weight, height]
                kernel = cv_values + [weight, height]
                self.kernels.append(kernel)
                self.kernel_counter += 1

                # Update statistics
                self.sum_weights += weight
                self.sum_weights_sq += weight**2

                # Update Z_n
                self._updateZn()

                # Adapt bandwidth periodically
                if self.kernel_counter % 100 == 0:
                    self._adaptBandwidth()

                # Compress kernels periodically
                if self.kernel_counter % 100 == 0:
                    self._compressKernels()

                # Rebuild bias expression (not every step for efficiency)
                if self.kernel_counter % 10 == 0:
                    self._updateBiasExpression()
                    simulation.context.reinitialize(preserveState=True)

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
            f.write(f"# Z_n: {self.Zn}\n")
            f.write("# Columns: ")
            for i in range(self.num_cvs):
                f.write(f"cv{i} ")
            f.write("weight height\n")

            for kernel in self.kernels:
                f.write(" ".join(map(str, kernel)) + "\n")

        np.savez_compressed(npz_name, kernels=np.array(self.kernels))

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
                    if len(values) == self.num_cvs + 2:
                        self.kernels.append(values)

        # Recalculate statistics
        self.sum_weights = sum(k[self.num_cvs] for k in self.kernels)
        self.sum_weights_sq = sum(k[self.num_cvs]**2 for k in self.kernels)
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