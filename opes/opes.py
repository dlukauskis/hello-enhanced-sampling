"""
OPES (On-the-fly Probability Enhanced Sampling) for OpenMM
Based on: Invernizzi & Parrinello, J. Phys. Chem. Lett. 2020, 11, 2731-2736

This file contains a corrected and more robust implementation of OPES for
small-molecule tests (e.g. alanine dipeptide). The key fixes are:
 - correct initialization order (num_cvs before using it)
 - consistent unit handling: numeric kT in kJ/mol, numeric sigmas in the CV units
 - valid OpenMM expression syntax using pow(x,2)
 - avoid extremely frequent Context.reinitialize() by rebuilding bias only
   every `rebuild_every` kernels (default 10)
 - store kernel centers/heights/Z as plain Python floats
 - fix step counting and CV collection

This is intended as a compact, pragmatic implementation for validation and
small-scale testing. A production-grade implementation would expose more
options and optimize kernel storage/merging further.
"""

import math
import numpy as np
import os
from openmm import CustomCVForce
from openmm import unit
# use unit.kilojoules_per_mole, unit.nanometer, unit.radian explicitly to avoid
# static-analysis name resolution issues


class OPES:
    """
    On-the-fly Probability Enhanced Sampling (OPES) method.

    Parameters
    ----------
    system : openmm.System
        The system to which the bias will be applied
    variables : list of openmm.Force or callable
        The collective variables to bias (e.g. CustomTorsionForce objects)
    temperature : openmm.unit.Quantity
        The temperature at which the simulation is run
    barrier : openmm.unit.Quantity
        The energy barrier estimate (default: 10 kJ/mol)
    sigma : list of openmm.unit.Quantity or single Quantity
        Width of kernels for each CV (must be provided in the appropriate unit,
        e.g. `0.35 * radian` for torsions). If a single Quantity is given,
        it is broadcast for all CVs.
    stride : int
        Deposition frequency in time steps (default: 500)
    compression_threshold : float
        Threshold for kernel compression in units of sigma (default: 1.0)
    saveFrequency : int
        Frequency to save kernels and bias (default: 10000 steps)
    biasDir : str
        Directory to save bias information (default: None)
    periodic : list of (min,max) tuples or None
        Periodic ranges for each CV (e.g. [(-pi,pi),(-pi,pi)] for torsions)
    rebuild_every : int
        Rebuild the CustomCVForce energy expression and reinitialize the
        context only every `rebuild_every` kernels to avoid excessive overhead.
    max_kernels : int or None
        Optional cap on number of kernels (old kernels will be dropped when
        exceeded).
    """

    def __init__(
        self,
        system,
        variables,
        temperature,
        barrier=10 * unit.kilojoules_per_mole,
        sigma=None,
        stride=500,
        compression_threshold=1.0,
        saveFrequency=10000,
        biasDir=None,
        periodic=None,
        rebuild_every=10,
        max_kernels=None,
        capacity=5000,
    ):
        self.system = system
        self.variables = variables

        # Number of CVs (must be known early)
        self.num_cvs = len(variables)

        # Periodic boundaries for each CV
        if periodic is None:
            self.periodic = [None] * self.num_cvs
        else:
            if len(periodic) != self.num_cvs:
                raise ValueError("`periodic` must have the same length as `variables`")
            self.periodic = periodic

        # Thermodynamic beta (unitless) and kT numeric in kJ/mol
        kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
        kT_quantity = kB * temperature
        # numeric kT in kJ/mol (float)
        self.kT = kT_quantity.value_in_unit(unit.kilojoules_per_mole)
        self.beta = 1.0 / self.kT

        self.barrier = barrier
        self.stride = int(stride)
        self.compression_threshold = compression_threshold
        self.saveFrequency = int(saveFrequency)
        self.biasDir = biasDir
        self.rebuild_every = max(1, int(rebuild_every))
        self.max_kernels = None if max_kernels is None else int(max_kernels)
        # When capacity is provided, pre-allocate parameterized kernel slots
        # and never rebuild the force expression; this allows fast updates via
        # CustomCVForce global parameters and updateParametersInContext(). If
        # capacity is None or 0, fall back to the previous dynamic-expression
        # behavior.
        self.capacity = None if capacity is None else int(capacity)

        # CV tracking is delegated to an external reporter (OPEsCVReporter).
        # Previously we stored CVs in-memory; that has been removed to mirror
        # the wt-metadetics approach (separate reporter class).

        # Kernel widths: accept single sigma or list
        if sigma is None:
            # Default sigma: assume angular CVs (radian) for safety
            sigma = [0.1 * unit.radian] * self.num_cvs
        if not hasattr(sigma, '__iter__') or isinstance(sigma, unit.Quantity):
            sigma = [sigma] * self.num_cvs
        if len(sigma) != self.num_cvs:
            raise ValueError("`sigma` must have the same length as `variables`")

        # Convert sigma to numeric values in appropriate units: if CV is periodic
        # (angles) prefer radian, otherwise use nanometer. We trust the user to
        # provide reasonable sigma units; fallback choices kept pragmatic.
        self.sigma = list(sigma)
        self.sigma_vals = []
        for i, s in enumerate(self.sigma):
            if hasattr(s, 'unit'):
                # choose radian when periodic range is given (angles)
                if self.periodic[i] is not None:
                    val = s.value_in_unit(unit.radian)
                else:
                    # assume distance-like CV
                    val = s.value_in_unit(unit.nanometer)
            else:
                # bare float: assume already in correct units
                val = float(s)
            # store numeric positive sigma
            self.sigma_vals.append(float(np.abs(val)))

        # Storage for kernels: each kernel is [cv0,..., cvN-1, height, Z]
        self.kernels = []
        self.kernel_counter = 0

        # OPES-specific epsilon (numeric)
        self.epsilon = math.exp(-self.barrier.value_in_unit(unit.kilojoules_per_mole) / self.kT)

        # Create the bias force and add to the system
        self._createBiasForce()
        # If capacity mode is enabled, initialize parameter slots in the force
        if self.capacity and self.capacity > 0:
            self._init_parameter_slots()

        # Statistics
        self.step_count = 0

    def _createBiasForce(self):
        """Create the CustomCVForce for OPES bias and register CVs."""

        # initial zero energy expression
        # We'll set an expression later depending on whether capacity mode is used
        self.force = CustomCVForce("0")

        # Add collective variables (each variable object is passed directly)
        for i, var in enumerate(self.variables):
            # name "cv{i}" will be used in the expression
            self.force.addCollectiveVariable(f"cv{i}", var)

        # put bias into a separate force group
        self.force.setForceGroup(15)
        self.system.addForce(self.force)

    def _init_parameter_slots(self):
        """Pre-allocate global parameters for kernel slots and set a fixed
        energy expression that sums over these slots. This avoids rebuilding
        the expression when adding kernels; only global parameters change.
        """
        N = self.capacity
        # Build expression: sum over k terms of a_k * h_k * G_k * exp(beta*Z_k)
        # where G_k is the multivariate Gaussian built from cv differences.
        kernel_terms = []
        for k in range(N):
            gaussian_terms = []
            for j in range(self.num_cvs):
                # parameter names
                c = f"c_{k}_{j}"
                # periodic handling via cvj and center parameter
                if self.periodic[j] is None:
                    diff = f"(cv{j}-{c})"
                else:
                    pmin, pmax = self.periodic[j]
                    period = pmax - pmin
                    diff = (
                        f"((cv{j}-{c})-{period}*floor(((cv{j}-{c})/{period})+0.5))"
                    )
                gaussian_terms.append(f"(pow({diff},2)/{(self.sigma_vals[j] ** 2):.12g})")

            G = "exp(-0.5*(" + "+".join(gaussian_terms) + "))"
            # parameters: active a_k, height h_k, Z_k
            a = f"a_{k}"
            h = f"h_{k}"
            Z = f"Z_{k}"
            kernel_terms.append(f"({a}*{h}*{G}*exp({self.beta:.12g}*{Z}))")

        sum_kernels = "+".join(kernel_terms) if kernel_terms else "0"
        bias_expr = f"-{float(self.kT):.12g}*log(1.0+({sum_kernels}))"

        # set the energy function once
        self.force.setEnergyFunction(bias_expr)

        # add global parameters with sensible defaults (disabled)
        for k in range(N):
            # centers for each CV
            for j in range(self.num_cvs):
                self.force.addGlobalParameter(f"c_{k}_{j}", 0.0)
            # height, Z, active flag
            self.force.addGlobalParameter(f"h_{k}", 0.0)
            self.force.addGlobalParameter(f"Z_{k}", 0.0)
            self.force.addGlobalParameter(f"a_{k}", 0.0)

        # bookkeeping for slot management
        self.next_slot = 0  # next slot to write into
        self.active_slots = 0
        # mapping from logical kernel index to slot assigned (for saving/dropping)
        self.slot_kernel_map = []

    def _updateBiasExpression(self):
        """Update the bias energy expression based on current kernels.

        Note: this reconstructs a CustomCVForce energy function string. For
        efficiency we recommend calling this only every `rebuild_every` kernel
        additions.
        """

        if self.capacity and self.capacity > 0:
            # In capacity mode the expression is fixed by `_init_parameter_slots`.
            return
        if len(self.kernels) == 0:
            self.force.setEnergyFunction("0")
            return

        kernel_terms = []
        # Precompute numeric constants
        kT_num = float(self.kT)

        for ik, kernel in enumerate(self.kernels):
            # kernel: [cv0,...,cvN-1, height, Z]
            gaussian_terms = []
            for j in range(self.num_cvs):
                center = float(kernel[j])
                sigma_j = float(self.sigma_vals[j])

                if self.periodic[j] is None:
                    # non-periodic difference
                    diff_expr = f"(cv{j}-{center:.12g})"
                else:
                    # periodic wrapping to [-period/2,period/2]
                    pmin, pmax = self.periodic[j]
                    period = pmax - pmin
                    # OpenMM expression to wrap difference
                    diff_expr = (
                        f"((cv{j}-{center:.12g})-{period:.12g}*floor(((cv{j}-{center:.12g})/"
                        f"{period:.12g})+0.5))"
                    )
                # pow instead of ^ and divide by sigma^2
                gaussian_terms.append(f"(pow({diff_expr},2)/{(sigma_j ** 2):.12g})")

            gaussian_expr = "exp(-0.5*(" + "+".join(gaussian_terms) + "))"

            height = float(kernel[self.num_cvs])
            Z = float(kernel[self.num_cvs + 1])

            # Each kernel term: height * exp(-0.5*...) * exp(beta*Z)
            # beta*Z is dimensionless here because Z will be provided as numeric -Z/kT
            kernel_terms.append(f"({height:.12g}*{gaussian_expr}*exp({self.beta:.12g}*{Z:.12g}))")

        sum_kernels = "+".join(kernel_terms)
        bias_expr = f"-{kT_num:.12g}*log(1.0+({sum_kernels}))"

        # Set the energy function
        self.force.setEnergyFunction(bias_expr)

    def _evaluateBias(self, cv_values):
        """Evaluate current bias at given CV values (cv_values: iterable of floats).

        Returns bias in kJ/mol (float).
        """

        if len(self.kernels) == 0:
            return 0.0

        kernel_sum = 0.0
        for kernel in self.kernels:
            gaussian = 1.0
            for j in range(self.num_cvs):
                center = float(kernel[j])
                sigma_j = float(self.sigma_vals[j])
                diff = self._periodic_difference(float(cv_values[j]), center, j)
                gaussian *= math.exp(-0.5 * (diff / sigma_j) ** 2)

            height = float(kernel[self.num_cvs])
            Z = float(kernel[self.num_cvs + 1])

            kernel_sum += height * gaussian * math.exp(self.beta * Z)

        bias = -self.kT * math.log(1.0 + kernel_sum)
        return float(bias)

    def _computeZ(self, cv_values):
        """Compute the Z value (related to the negative log probability).

        Here we use the current bias estimate: Z = -bias / kT
        """
        bias = self._evaluateBias(cv_values)
        Z = -bias / self.kT
        return float(Z)

    def _compressKernels(self):
        """Compress kernels that are too close together.

        This function merges kernels whose distance in units of sigma is below the
        compression_threshold.
        """

        if len(self.kernels) < 2:
            return

        compressed = []

        for kernel in self.kernels:
            placed = False
            for j, k2 in enumerate(compressed):
                # compute normalized distance
                dist2 = 0.0
                for kk in range(self.num_cvs):
                    diff = self._periodic_difference(kernel[kk], k2[kk], kk)
                    sigma_k = self.sigma_vals[kk]
                    dist2 += (diff / sigma_k) ** 2
                dist = math.sqrt(dist2)
                if dist < self.compression_threshold:
                    # merge: weighted by heights
                    h_i = kernel[self.num_cvs]
                    h_j = k2[self.num_cvs]
                    total_h = h_i + h_j if (h_i + h_j) != 0 else 1.0
                    for kk in range(self.num_cvs):
                        k2[kk] = (h_i * kernel[kk] + h_j * k2[kk]) / total_h
                    # heights add
                    k2[self.num_cvs] += kernel[self.num_cvs]
                    # average Z weighted by heights
                    k2[self.num_cvs + 1] = (
                        h_i * kernel[self.num_cvs + 1] + h_j * k2[self.num_cvs + 1]
                    ) / total_h
                    placed = True
                    break
            if not placed:
                compressed.append(list(kernel))

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

        for _ in range(int(steps)):
            simulation.step(1)
            self.step_count += 1

            # Deposit kernel at stride
            if (self.step_count % self.stride) == 0:
                # Get current CV values via the CustomCVForce API
                cv_values = self.force.getCollectiveVariableValues(simulation.context)
                # make sure we have numeric floats
                cv_values = [float(x) for x in cv_values]

                # Compute Z value
                Z = self._computeZ(cv_values)

                # Compute kernel height using the canonical OPES rule (Invernizzi & Parrinello):
                #   h = kT * ln(1 + epsilon / p_est(s))
                # where p_est(s) is the current estimated probability (unnormalized) at the CV point.
                # We have Z = -bias/kT, so p_est_unnorm = exp(Z). Epsilon is exp(-barrier/kT).
                # Add a tiny floor to p_est to avoid division by zero.
                tiny = 1e-300
                p_est = math.exp(Z)
                if not np.isfinite(p_est) or p_est <= 0.0:
                    p_est = tiny
                height = float(self.kT * math.log(1.0 + (self.epsilon / p_est)))

                # Store kernel as numeric list
                kernel = [float(x) for x in cv_values] + [height, float(Z)]

                # If capacity mode is active, write into the next global-parameter slot
                if self.capacity and self.capacity > 0:
                    slot = self.next_slot
                    # set center parameters
                    for j in range(self.num_cvs):
                        pname = f"c_{slot}_{j}"
                        self.force.setGlobalParameterDefaultValue(pname, float(cv_values[j]))
                    # set height and Z, enable slot
                    self.force.setGlobalParameterDefaultValue(f"h_{slot}", float(height))
                    self.force.setGlobalParameterDefaultValue(f"Z_{slot}", float(Z))
                    self.force.setGlobalParameterDefaultValue(f"a_{slot}", 1.0)

                    # update the context with new parameter defaults (fast)
                    try:
                        self.force.updateParametersInContext(simulation.context)
                    except Exception:
                        # if update fails, fallback to reinitialization
                        simulation.context.reinitialize(preserveState=True)

                    # bookkeeping
                    self.slot_kernel_map.append(kernel)
                    self.next_slot = (self.next_slot + 1) % self.capacity
                    self.active_slots = min(self.capacity, self.active_slots + 1)
                    self.kernel_counter += 1

                    # Enforce max_kernels by disabling oldest slot if needed
                    if (self.max_kernels is not None) and (len(self.slot_kernel_map) > self.max_kernels):
                        # compute slot to disable: the oldest is at index 0
                        # find its slot index = (next_slot - active_slots) mod capacity
                        oldest_slot = (self.next_slot - self.active_slots) % self.capacity
                        # disable it
                        self.force.setGlobalParameterDefaultValue(f"a_{oldest_slot}", 0.0)
                        try:
                            self.force.updateParametersInContext(simulation.context)
                        except Exception:
                            simulation.context.reinitialize(preserveState=True)
                        # remove from bookkeeping
                        if self.slot_kernel_map:
                            self.slot_kernel_map.pop(0)

                    # occasional compression of stored python-side kernels
                    if (self.kernel_counter % 100) == 0:
                        self._compressKernels()

                else:
                    # fallback to original behavior: append kernel and rebuild expression periodically
                    if (self.max_kernels is not None) and (len(self.kernels) >= self.max_kernels):
                        self.kernels.pop(0)
                    self.kernels.append(kernel)
                    self.kernel_counter += 1
                    if (self.kernel_counter % self.rebuild_every) == 0:
                        self._updateBiasExpression()
                        try:
                            self.force.updateParametersInContext(simulation.context)
                        except Exception:
                            simulation.context.reinitialize(preserveState=True)

                # Save kernels periodically
                if self.biasDir and (self.step_count % self.saveFrequency == 0):
                    self.saveKernels()

    def saveKernels(self):
        """Save kernel information to file in plain text and numpy compressed format."""

        if not self.biasDir:
            return

        os.makedirs(self.biasDir, exist_ok=True)

        txt_name = os.path.join(self.biasDir, f"kernels_{self.step_count}.txt")
        npz_name = os.path.join(self.biasDir, f"kernels_{self.step_count}.npz")

        with open(txt_name, 'w') as f:
            f.write(f"# OPES Kernels at step {self.step_count}\n")
            f.write(f"# Temperature: {self.kT} kJ/mol\n")
            f.write(f"# Barrier: {self.barrier}\n")
            f.write("# Columns: ")
            for i in range(self.num_cvs):
                f.write(f"cv{i} ")
            f.write("height Z\n")
            for kernel in self.kernels:
                f.write(" ".join(map(str, kernel)) + "\n")

        # Note: CV history is written by the external reporter; we do not
        # duplicate that here.

        # save a binary snapshot
        np.savez_compressed(npz_name, kernels=np.array(self.kernels))

    def loadKernels(self, filename):
        """Load kernels from a plain text or numpy npz file."""

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

        self._updateBiasExpression()
        print(f"Loaded {len(self.kernels)} kernels from {filename}")

    def getFreeEnergy(self, cv_grid):
        """
        Estimate free energy on a grid (kJ/mol).

        Parameters
        ----------
        cv_grid : list of arrays
            Grid points for each CV (numeric, in the same units as the CVs)

        Returns
        -------
        free_energy : array
            Estimated free energy on the grid (kJ/mol), minimum shifted to 0
        """

        grids = np.meshgrid(*cv_grid, indexing='ij')
        shape = grids[0].shape

        cv_points = np.array([g.flatten() for g in grids]).T

        free_energy = np.zeros(len(cv_points))
        for i, point in enumerate(cv_points):
            bias = self._evaluateBias(point)
            free_energy[i] = -bias

        free_energy = free_energy.reshape(shape)
        free_energy -= np.min(free_energy)
        return free_energy

    def _periodic_difference(self, value, center, cv_index):
        """Calculate periodic difference for a CV (returns float).

        Wraps to the interval [-period/2, period/2].
        """

        if self.periodic[cv_index] is None:
            return float(value - center)

        period_min, period_max = self.periodic[cv_index]
        period = period_max - period_min

        diff = float(value - center)
        # wrap using rounding
        diff = diff - period * round(diff / period)
        return float(diff)
