"""Fixed harmonic umbrella windows for OpenMM.

Numeric CV values use the native units returned by the supplied OpenMM Force
(radians for torsions, nanometers for distances); spring constants use
kJ/mol/native-unit**2. Add the bias before constructing a Context.
"""
from dataclasses import dataclass
import math

import numpy as np
from openmm import CustomCVForce, Force


@dataclass(frozen=True)
class UmbrellaVariable:
    force: object
    center: float
    force_constant: float
    periodic: tuple | None = None

    def __post_init__(self):
        if not math.isfinite(self.center):
            raise ValueError("Center must be finite")
        if not math.isfinite(self.force_constant) or self.force_constant <= 0:
            raise ValueError("Spring constant must be finite and positive")
        if self.periodic is not None:
            low, high = self.periodic
            if not all(map(math.isfinite, (low, high))) or high <= low:
                raise ValueError("Periodic bounds must be finite and increasing")


def displacement(values, centers, periodic):
    """Minimum-image CV displacement, broadcasting over leading axes."""
    delta = np.array(np.asarray(values) - np.asarray(centers), dtype=float, copy=True)
    for i, bounds in enumerate(periodic):
        if bounds is not None:
            period = bounds[1] - bounds[0]
            delta[..., i] -= period * np.floor(delta[..., i] / period + 0.5)
    return delta


def bias_energy(values, centers, force_constants, periodic):
    return 0.5 * np.sum(np.asarray(force_constants) * displacement(
        values, centers, periodic)**2, axis=-1)


class UmbrellaSampling:
    """A fixed window; centers and spring constants can be changed in a Context.

    Each variable must own a distinct Force not already attached to a System.
    ``prefix`` must be unique among biases in that System. The harmonic
    potential is continuous at a periodic seam; its derivative has the usual
    minimum-image cusp exactly half a period from the center.
    """

    def __init__(self, system, variables, prefix="umbrella"):
        self.variables = tuple(variables)
        if not self.variables:
            raise ValueError("At least one CV is required")
        if not prefix.isidentifier() or not prefix.isascii():
            raise ValueError("prefix must be an ASCII identifier")
        if len({id(v.force) for v in self.variables}) != len(self.variables):
            raise ValueError("CV Forces must be distinct")
        if any(not isinstance(v.force, Force) or not v.force.thisown for v in self.variables):
            raise ValueError("Each CV must be a fresh, independently owned OpenMM Force")
        names = {name for i in range(len(self.variables))
                 for name in (f"{prefix}_center{i}", f"{prefix}_k{i}")}
        for force in (*system.getForces(), *(v.force for v in self.variables)):
            if hasattr(force, "getNumGlobalParameters"):
                if any(force.getGlobalParameterName(i) in names
                       for i in range(force.getNumGlobalParameters())):
                    raise ValueError("Umbrella prefix collides with existing global parameters")
        occupied = {f.getForceGroup() for f in system.getForces()}
        free_groups = set(range(32)) - occupied
        if not free_groups:
            raise ValueError("No unused force group available for umbrella bias")
        self.force_group = max(free_groups)
        self.system = system  # Keep the owner of the transferred CV Forces alive.
        self.prefix = prefix
        terms, definitions = [], []
        for i, v in enumerate(self.variables):
            center, spring = self._names(i)
            terms.append(f"0.5*{spring}*d{i}^2")
            raw = f"(cv{i}-{center})"
            if v.periodic is not None:
                period = v.periodic[1] - v.periodic[0]
                raw = f"{raw}-{period:.17g}*floor({raw}/{period:.17g}+0.5)"
            definitions.append(f"d{i}={raw}")
        self.force = CustomCVForce("+".join(terms) + ";" + ";".join(definitions))
        self.force.setForceGroup(self.force_group)
        for i, v in enumerate(self.variables):
            self.force.addCollectiveVariable(f"cv{i}", v.force)
            center, spring = self._names(i)
            self.force.addGlobalParameter(center, float(v.center))
            self.force.addGlobalParameter(spring, float(v.force_constant))
        system.addForce(self.force)

    def _names(self, i):
        return f"{self.prefix}_center{i}", f"{self.prefix}_k{i}"

    def set_window(self, context, centers, force_constants=None):
        """Update the running Context through global parameters, without rebuilding."""
        centers = np.asarray(centers, dtype=float)
        if centers.shape != (len(self.variables),) or not np.isfinite(centers).all():
            raise ValueError("Provide one finite center per CV")
        if force_constants is not None:
            force_constants = np.asarray(force_constants, dtype=float)
            if (force_constants.shape != centers.shape or
                    not np.isfinite(force_constants).all() or (force_constants <= 0).any()):
                raise ValueError("Provide one positive finite spring constant per CV")
        for i, center in enumerate(centers):
            c_name, k_name = self._names(i)
            context.setParameter(c_name, float(center))
            if force_constants is not None:
                context.setParameter(k_name, float(force_constants[i]))

    def getCollectiveVariables(self, simulation):
        return self.force.getCollectiveVariableValues(simulation.context)

    def get_window(self, context):
        """Read actual centers/springs from a Context (variables hold defaults)."""
        parameters = [self._names(i) for i in range(len(self.variables))]
        return (np.array([context.getParameter(c) for c, k in parameters]),
                np.array([context.getParameter(k) for c, k in parameters]))

    def step(self, simulation, steps):
        simulation.step(steps)
