"""
OPES (On-the-fly Probability Enhanced Sampling) for OpenMM
Based on: Invernizzi & Parrinello, J. Phys. Chem. Lett. 2020, 11, 2731-2736
"""

import numpy as np
from openmm import CustomCVForce, Context
from openmm.unit import kilojoules_per_mole, nanometer, picosecond
from openmm import unit
import os


class OPES:
    """
    On-the-fly Probability Enhanced Sampling (OPES) method.

    Parameters
    ----------
    system : openmm.System
        The system to which the bias will be applied
    variables : list of openmm.Force or callable
        The collective variables to bias
    temperature : openmm.unit.Quantity
        The temperature at which the simulation is run
    barrier : openmm.unit.Quantity
        The energy barrier estimate (default: 10 kJ/mol)
    sigma : list of openmm.unit.Quantity
        Width of kernels for each CV
    stride : int
        Deposition frequency in time steps (default: 500)
    compression_threshold : float
        Threshold for kernel compression (default: 1.0)
    saveFrequency : int
        Frequency to save kernels and bias (default: 10000 steps)
    biasDir : str
        Directory to save bias information (default: None)
    """

    def __init__(self, system, variables, temperature, barrier=10 * kilojoules_per_mole,
                 sigma=None, stride=500, compression_threshold=1.0,
                 saveFrequency=10000, biasDir=None):

        self.system = system
        self.variables = variables
        self.temperature = temperature
        self.barrier = barrier
        self.stride = stride
        self.compression_threshold = compression_threshold
        self.saveFrequency = saveFrequency
        self.biasDir = biasDir

        # Thermodynamic beta (unitless)
        from openmm import unit
        kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
        kT_quantity = (kB * temperature)
        self.kT = kT_quantity.value_in_unit(kilojoules_per_mole)  # Unitless kT in kJ/mol
        self.beta = 1.0 / self.kT  # Unitless beta in 1/(kJ/mol)

        # Number of CVs
        self.num_cvs = len(variables)

        # Kernel widths
        if sigma is None:
            self.sigma = [0.1 * nanometer] * self.num_cvs
        else:
            self.sigma = sigma

        # Storage for kernels: each kernel is [cv1, cv2, ..., height, Z]
        self.kernels = []
        self.kernel_counter = 0

        # OPES-specific parameters
        self.epsilon = np.exp(-self.barrier.value_in_unit(kilojoules_per_mole) / self.kT)

        # Create the bias force
        self._createBiasForce()

        # Statistics
        self.step_count = 0

    def _createBiasForce(self):
        """Create the CustomCVForce for OPES bias."""

        # Create custom CV force with initial zero bias
        self.force = CustomCVForce("0")  # Changed from "" to "0"

        # Add collective variables
        for i, var in enumerate(self.variables):
            self.force.addCollectiveVariable(f"cv{i}", var)

        # We'll update the energy expression dynamically as kernels are added
        self.force.setForceGroup(15)  # Separate force group for bias
        self.system.addForce(self.force)

    def _updateBiasExpression(self):
        """Update the bias energy expression based on current kernels."""

        if len(self.kernels) == 0:
            self.force.setEnergyFunction("0")
            return

        # Build kernel sum expression
        # V_OPES(s) = -(1/β) * log[1 + Σ_i exp(-β * kernel_i(s))]

        kernel_terms = []
        for i, kernel in enumerate(self.kernels):
            # Gaussian kernel: exp(-0.5 * Σ_j (cv_j - center_j)^2 / sigma_j^2)
            gaussian_terms = []
            for j in range(self.num_cvs):
                center = kernel[j]
                # Extract sigma in its native unit (no conversion)
                sigma = self.sigma[j]._value  # Use ._value to get unitless value
                gaussian_terms.append(f"((cv{j}-{center})^2/{sigma ** 2})")

            gaussian_expr = "exp(-0.5*(" + "+".join(gaussian_terms) + "))"

            # Height and Z value for this kernel
            height = kernel[self.num_cvs]
            Z = kernel[self.num_cvs + 1]

            # Kernel contribution: height * gaussian * exp(β*Z)
            kernel_terms.append(f"({height}*{gaussian_expr}*exp({self.beta}*{Z}))")  # beta is now unitless

        # Full OPES bias expression
        sum_kernels = "+".join(kernel_terms)
        bias_expr = f"-{self.kT}*log(1.0+{sum_kernels})"

        self.force.setEnergyFunction(bias_expr)

    def _evaluateBias(self, cv_values):
        """Evaluate current bias at given CV values."""

        if len(self.kernels) == 0:
            return 0.0

        # Sum of kernel contributions
        kernel_sum = 0.0
        for kernel in self.kernels:
            # Gaussian kernel value
            gaussian = 1.0
            for j in range(self.num_cvs):
                center = kernel[j]
                sigma = self.sigma[j]._value  # Use ._value to get unitless value
                gaussian *= np.exp(-0.5 * ((cv_values[j] - center) / sigma) ** 2)

            height = kernel[self.num_cvs]
            Z = kernel[self.num_cvs + 1]

            kernel_sum += height * gaussian * np.exp(self.beta * Z)  # beta is now unitless

        # OPES bias
        bias = -self.kT * np.log(1.0 + kernel_sum)
        return bias

    def _computeZ(self, cv_values):
        """Compute the Z value (related to the negative log probability)."""

        # Z(s) = -log[p(s)/p_0]
        # In OPES, this is estimated from the current bias
        bias = self._evaluateBias(cv_values)
        Z = -bias / self.kT

        return Z

    def _compressKernels(self):
        """Compress kernels that are too close together."""

        if len(self.kernels) < 2:
            return

        compressed = []
        merged_indices = set()

        for i in range(len(self.kernels)):
            if i in merged_indices:
                continue

            kernel_i = self.kernels[i]
            merged = False

            # Check if this kernel should be merged with any in compressed list
            for j, kernel_j in enumerate(compressed):
                # Distance in CV space
                dist = 0.0
                for k in range(self.num_cvs):
                    sigma_k = self.sigma[k]._value  # Use ._value
                    dist += ((kernel_i[k] - kernel_j[k]) / sigma_k) ** 2
                dist = np.sqrt(dist)

                # If close enough, merge
                if dist < self.compression_threshold:
                    # Merge: weighted average of positions, sum of heights
                    weight_i = kernel_i[self.num_cvs]
                    weight_j = kernel_j[self.num_cvs]
                    total_weight = weight_i + weight_j

                    for k in range(self.num_cvs):
                        kernel_j[k] = (weight_i * kernel_i[k] + weight_j * kernel_j[k]) / total_weight

                    kernel_j[self.num_cvs] += kernel_i[self.num_cvs]  # Sum heights
                    kernel_j[self.num_cvs + 1] = (weight_i * kernel_i[self.num_cvs + 1] +
                                                  weight_j * kernel_j[self.num_cvs + 1]) / total_weight

                    merged = True
                    break

            if not merged:
                compressed.append(list(kernel_i))

        old_count = len(self.kernels)
        self.kernels = compressed
        new_count = len(self.kernels)

        if new_count < old_count:
            print(f"OPES: Compressed {old_count} kernels to {new_count}")

    def step(self, simulation, steps):
        """
        Advance the simulation while depositing OPES kernels.

        Parameters
        ----------
        simulation : openmm.app.Simulation
            The simulation to advance
        steps : int
            Number of steps to run
        """

        for i in range(steps):
            simulation.step(1)
            self.step_count += 1  # Changed from self.step

            # Deposit kernel
            if self.step_count % self.stride == 0:  # Changed from self.step
                # Get current CV values
                state = simulation.context.getState(getEnergy=True, getForces=True)
                cv_values = []
                for j in range(self.num_cvs):
                    cv_val = self.force.getCollectiveVariableValues(simulation.context)[j]
                    cv_values.append(cv_val)

                # Compute Z value
                Z = self._computeZ(cv_values)

                # Compute kernel height
                # In OPES, height is related to the probability: h ∝ exp(-β*F(s))
                # We use: h = ε / (1 + N_eff)
                N_eff = len(self.kernels)  # Simplified effective count
                height = self.epsilon / (1.0 + N_eff)

                # Add kernel: [cv1, cv2, ..., height, Z]
                kernel = cv_values + [height, Z]
                self.kernels.append(kernel)
                self.kernel_counter += 1

                # Compress kernels periodically
                if self.kernel_counter % 100 == 0:
                    self._compressKernels()

                # Update bias expression
                self._updateBiasExpression()

                # Reinitialize context to apply new bias
                state = simulation.context.getState(getPositions=True, getVelocities=True)
                simulation.context.reinitialize(preserveState=True)

            # Save data
            if self.biasDir and self.step_count % self.saveFrequency == 0:  # Changed from self.step
                self.saveKernels()

    def saveKernels(self):
        """Save kernel information to file."""

        if not self.biasDir:
            return

        os.makedirs(self.biasDir, exist_ok=True)

        filename = os.path.join(self.biasDir, f"kernels_{self.step_count}.txt")

        with open(filename, 'w') as f:
            f.write(f"# OPES Kernels at step {self.step_count}\n")
            f.write(f"# Temperature: {self.temperature}\n")
            f.write(f"# Barrier: {self.barrier}\n")
            f.write(f"# Columns: ")
            for i in range(self.num_cvs):
                f.write(f"cv{i} ")
            f.write("height Z\n")

            for kernel in self.kernels:
                f.write(" ".join(map(str, kernel)) + "\n")

    def loadKernels(self, filename):
        """Load kernels from file."""

        self.kernels = []
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                values = list(map(float, line.split()))
                if len(values) == self.num_cvs + 2:  # cv values + height + Z
                    self.kernels.append(values)

        self._updateBiasExpression()
        print(f"Loaded {len(self.kernels)} kernels from {filename}")

    def getFreeEnergy(self, cv_grid):
        """
        Estimate free energy on a grid.

        Parameters
        ----------
        cv_grid : list of arrays
            Grid points for each CV

        Returns
        -------
        free_energy : array
            Estimated free energy on the grid
        """

        # Create mesh grid
        grids = np.meshgrid(*cv_grid, indexing='ij')
        shape = grids[0].shape

        # Flatten for evaluation
        cv_points = np.array([g.flatten() for g in grids]).T

        # Evaluate bias at each point
        free_energy = np.zeros(len(cv_points))
        for i, point in enumerate(cv_points):
            bias = self._evaluateBias(point)
            # In converged OPES, bias ≈ -F(s) + C
            free_energy[i] = -bias

        # Reshape and shift minimum to zero
        free_energy = free_energy.reshape(shape)
        free_energy -= np.min(free_energy)

        return free_energy