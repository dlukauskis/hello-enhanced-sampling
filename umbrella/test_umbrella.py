"""Run with: python -m unittest discover -s umbrella -p 'test_*.py'."""
import io
import unittest
import tempfile
from pathlib import Path
import json
import numpy as np
from openmm import Context, CustomExternalForce, CustomTorsionForce, Platform, System, VerletIntegrator, unit
from umbrella import UmbrellaSampling, UmbrellaVariable, bias_energy
from umbrella_reporter import UmbrellaCVReporter
from analysis_umbrella import WindowData, reconstruct, thermal_energy, independent_indices, load_windows


class ForceTests(unittest.TestCase):
    def context(self, dimension=1, periodic=True):
        system = System()
        system.addParticle(1.)
        variables = []
        for coordinate in "xy"[:dimension]:
            cv = CustomExternalForce(coordinate)
            cv.addParticle(0, [])
            variables.append(UmbrellaVariable(cv, 3., 7., (-np.pi, np.pi) if periodic else None))
        bias = UmbrellaSampling(system, variables)
        integrator = VerletIntegrator(.001)
        context = Context(system, integrator, Platform.getPlatformByName("Reference"))
        context.setPositions([[-3., .5, 0.]])
        return context, integrator, bias

    def test_periodic_force_and_context_update(self):
        context, integrator, bias = self.context(2)
        for centers, springs in [([3., 3.], [7., 7.]), ([1., -2.], [4., 9.])]:
            bias.set_window(context, centers, springs)
            actual_centers, actual_springs = bias.get_window(context)
            np.testing.assert_array_equal(actual_centers, centers)
            np.testing.assert_array_equal(actual_springs, springs)
            state = context.getState(getEnergy=True, getForces=True)
            expected = bias_energy([-3., .5], centers, springs, [(-np.pi, np.pi)] * 2)
            self.assertAlmostEqual(state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole), expected, places=11)
            from umbrella import displacement
            expected_force = -np.array(springs) * displacement([-3., .5], centers, [(-np.pi, np.pi)] * 2)
            np.testing.assert_allclose(state.getForces(asNumpy=True).value_in_unit(
                unit.kilojoules_per_mole / unit.nanometer)[0, :2], expected_force, atol=1e-10)

    def test_nonperiodic(self):
        context, integrator, bias = self.context(periodic=False)
        self.assertAlmostEqual(context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoules_per_mole), .5 * 7 * 6**2)

    def test_torsion_cartesian_gradient(self):
        system = System()
        for _ in range(4):
            system.addParticle(12.)
        cv = CustomTorsionForce("theta")
        cv.addTorsion(0, 1, 2, 3)
        bias = UmbrellaSampling(system, [UmbrellaVariable(cv, 3., 23., (-np.pi, np.pi))])
        integrator = VerletIntegrator(.001)
        context = Context(system, integrator, Platform.getPlatformByName("Reference"))
        positions = np.array([[0., 1., 0.], [0., 0., 0.], [1., 0., 0.], [1., -.9, -.1]])
        context.setPositions(positions)
        forces = context.getState(getForces=True).getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.nanometer)
        for atom in range(4):
            for axis in range(3):
                energies = []
                for sign in [1, -1]:
                    p = positions.copy()
                    p[atom, axis] += sign * 1e-6
                    context.setPositions(p)
                    theta = bias.force.getCollectiveVariableValues(context)[0]
                    energies.append(bias_energy([theta], [3.], [23.], [(-np.pi, np.pi)]))
                self.assertAlmostEqual(forces[atom, axis], -(energies[0] - energies[1]) / 2e-6, places=6)

    def test_validation_and_reporter(self):
        with self.assertRaises(ValueError):
            UmbrellaVariable(None, 0., -1.)
        with self.assertRaises(ValueError):
            UmbrellaVariable(None, 0., 1., (1., 1.))
        context, integrator, bias = self.context()
        with self.assertRaises(ValueError):
            bias.set_window(context, [0.], [0.])
        class Simulation:
            currentStep = 10
        simulation = Simulation()
        simulation.context = context
        out = io.StringIO()
        reporter = UmbrellaCVReporter(out, 5, bias)
        self.assertEqual(reporter.describeNextReport(simulation)["steps"], 5)
        reporter.report(simulation, None)
        self.assertIn("10 -3", out.getvalue())
        reporter.close()
        self.assertFalse(out.closed)

    def test_prefix_collision(self):
        system = System()
        system.addParticle(1.)
        for attempt in range(2):
            cv = CustomExternalForce("x")
            cv.addParticle(0, [])
            variables = [UmbrellaVariable(cv, 0., 1.)]
            if attempt == 0:
                UmbrellaSampling(system, variables)
            else:
                with self.assertRaises(ValueError):
                    UmbrellaSampling(system, variables)
        system = System()
        system.addParticle(1.)
        cv = CustomExternalForce("x+umbrella_center0")
        cv.addGlobalParameter("umbrella_center0", 1.)
        cv.addParticle(0, [])
        with self.assertRaises(ValueError):
            UmbrellaSampling(system, [UmbrellaVariable(cv, 0., 1.)])

    def test_force_ownership_and_groups(self):
        system = System()
        system.addParticle(1.)
        owned = CustomExternalForce("0")
        owned.addParticle(0, [])
        owned.setForceGroup(31)
        system.addForce(owned)
        with self.assertRaises(ValueError):
            UmbrellaSampling(system, [UmbrellaVariable(owned, 0., 1.)])
        cv = CustomExternalForce("x")
        cv.addParticle(0, [])
        bias = UmbrellaSampling(system, [UmbrellaVariable(cv, 0., 1.)])
        self.assertEqual(bias.force_group, 30)

    def test_cli_imports(self):
        # Catch syntax/import failures in the example and plotting entry points.
        import run_aladipep
        import plot_fes_umbrella
        self.assertTrue(callable(run_aladipep.main))
        self.assertTrue(callable(plot_fes_umbrella.main))

    def test_preparation_retry_is_limited(self):
        from unittest.mock import patch
        from contextlib import redirect_stdout
        import run_aladipep
        task = (1, np.array([0.]), {"preparation_steps": 2500})
        with patch.object(run_aladipep, "_run_window", side_effect=[
                run_aladipep.PreparationError("NaN"), (1, "done")]) as worker:
            with redirect_stdout(io.StringIO()):
                self.assertEqual(run_aladipep.run_window(task), (1, "done"))
            self.assertEqual(worker.call_args.args[0][2]["preparation_substeps"], 10)
            self.assertEqual(worker.call_args.args[0][2]["preparation_method"], "steered")
            self.assertNotIn("preparation_substeps", task[2])
        with patch.object(run_aladipep, "_run_window", side_effect=RuntimeError("production")):
            with self.assertRaisesRegex(RuntimeError, "production"):
                run_aladipep.run_window(task)
        with patch.object(run_aladipep, "_run_window", side_effect=run_aladipep.PreparationError("NaN")) as worker:
            with redirect_stdout(io.StringIO()), self.assertRaises(run_aladipep.PreparationError):
                run_aladipep.run_window(task)
            self.assertEqual(worker.call_count, 2)

    def test_reporter_in_openmm_simulation(self):
        from openmm.app import Simulation, Topology, element
        topology = Topology()
        residue = topology.addResidue("X", topology.addChain())
        topology.addAtom("H", element.hydrogen, residue)
        system = System()
        system.addParticle(1.)
        cv = CustomExternalForce("x")
        cv.addParticle(0, [])
        bias = UmbrellaSampling(system, [UmbrellaVariable(cv, 0., 1.)])
        simulation = Simulation(topology, system, VerletIntegrator(.001),
                                Platform.getPlatformByName("Reference"))
        simulation.context.setPositions([[1., 0., 0.]])
        output = io.StringIO()
        reporter = UmbrellaCVReporter(output, 5, bias)
        simulation.reporters.append(reporter)
        bias.step(simulation, 10)
        rows = output.getvalue().splitlines()
        self.assertEqual(len(rows), 3)
        self.assertTrue(rows[1].startswith("5 "))
        self.assertTrue(rows[2].startswith("10 "))


class ReconstructionTests(unittest.TestCase):
    def synthetic(self, dimension):
        rng = np.random.default_rng(1024 + dimension)
        kt = thermal_energy(300.)
        angles = np.linspace(-np.pi, np.pi, 5 if dimension == 1 else 4, endpoint=False)
        centers = np.array(np.meshgrid(*([angles] * dimension), indexing="ij")).reshape(dimension, -1).T
        windows = []
        for center in centers:
            chunks = []
            while sum(len(x) for x in chunks) < 2500:
                x = rng.uniform(-np.pi, np.pi, (12000, dimension))
                target = kt * np.sum(1 - np.cos(x), axis=1)
                bias = bias_energy(x, center, np.full(dimension, 2 * kt), [(-np.pi, np.pi)] * dimension)
                chunks.append(x[rng.random(len(x)) < np.exp(-(target + bias) / kt)])
            windows.append(WindowData(np.concatenate(chunks)[:2500], center, np.full(dimension, 2 * kt)))
        r = reconstruct(windows, [(-np.pi, np.pi)] * dimension, bins=24 if dimension == 1 else 18, decorrelate=False)
        self.assertAlmostEqual(r["probability"].sum(), 1., places=8)
        self.assertGreater(r["overlap_gap"], .01)
        coordinates = np.meshgrid(*r["grid"], indexing="ij")
        target = kt * sum(1 - np.cos(x) for x in coordinates)
        target -= target.min()
        sampled = np.isfinite(r["fes"])
        self.assertLess(np.sqrt(np.mean((r["fes"][sampled] - target[sampled])**2)), .6)
        if dimension == 2:
            probability = r["probability"].sum(axis=1)
            profile = -kt * np.log(probability)
            profile -= profile.min()
            np.testing.assert_allclose(profile, r["profile"], atol=1e-10)
        return windows

    def test_known_1d_fes(self):
        self.synthetic(1)

    def test_known_2d_fes_and_projection(self):
        self.synthetic(2)

    def test_periodic_seam_invariance(self):
        windows = self.synthetic(1)
        a = reconstruct(windows, [(-np.pi, np.pi)], bins=24, decorrelate=False)
        shifted = [WindowData(w.samples + 2 * np.pi, w.center - 2 * np.pi, w.force_constants) for w in windows]
        b = reconstruct(shifted, [(-np.pi, np.pi)], bins=24, decorrelate=False)
        np.testing.assert_allclose(a["fes"], b["fes"], atol=1e-9)

    def test_invalid_data(self):
        with self.assertRaises(ValueError):
            reconstruct([WindowData(np.zeros((5, 1)), [0.], [1.])], [(-np.pi, np.pi)])
        with self.assertRaises(ValueError):
            thermal_energy(-1.)

    def test_nonperiodic_reconstruction(self):
        rng = np.random.default_rng(90)
        kt = thermal_energy(300.)
        # Target U=0.5*x^2, umbrella k=3: exact biased Gaussian distributions.
        windows = [WindowData(rng.normal(3 * c / 4, np.sqrt(kt / 4), (8000, 1)),
                              np.array([c]), np.array([3.])) for c in np.linspace(-3, 3, 9)]
        r = reconstruct(windows, [None], bounds=[(-7., 7.)], bins=56, decorrelate=False)
        target = .5 * r["grid"][0]**2
        target -= target.min()
        mask = (np.abs(r["grid"][0]) < 3.) & np.isfinite(r["fes"])
        self.assertLess(np.sqrt(np.mean((r["fes"][mask] - target[mask])**2)), .25)
        with self.assertRaises(ValueError):
            reconstruct(windows, [None], bounds=[(-1., 1.)])

    def test_disconnected_overlap_rejected(self):
        rng = np.random.default_rng(2)
        windows = [WindowData(rng.normal(center, 1e-5, (100, 1)),
                              np.array([center]), np.array([1e8])) for center in [-1., 1.]]
        with self.assertRaisesRegex(ValueError, "overlap"):
            reconstruct(windows, [(-np.pi, np.pi)], decorrelate=False)

    def test_decorrelation(self):
        rng = np.random.default_rng(4)
        x = np.zeros((5000, 1))
        for i in range(1, len(x)):
            x[i] = .9 * x[i-1] + rng.normal(0, .05)
        window = WindowData(x, np.array([0.]), np.array([10.]))
        indices, g = independent_indices(x, window, [(-np.pi, np.pi)])
        self.assertGreater(g, 5.)
        self.assertLess(len(indices), len(x) / 5)
        self.assertTrue(np.all(np.diff(indices) > 0))
        with self.assertRaises(ValueError):
            independent_indices(np.zeros((100, 1)), window, [(-np.pi, np.pi)])

    def test_load_grid_validation_and_temporary_files(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            common = dict(samples=np.zeros((100, 1)), force_constants=[10.], temperature=300.,
                          periodic=[[-np.pi, np.pi]], production_ps=20., equilibration_ps=10.,
                          stride_ps=.2, dimension=1)
            np.savez(directory / "window_0000.npz", center=[0.], **common)
            np.savez(directory / "window_0001.tmp.npz", **common)
            (directory / "manifest.json").write_text(json.dumps({"centers": [[0.], [1.]]}))
            with self.assertRaises(ValueError):
                load_windows(directory)
            np.savez(directory / "window_0001.npz", center=[1.], **common)
            windows, metadata = load_windows(directory, .5)
            self.assertEqual(len(windows), 2)
            self.assertEqual(len(windows[0].samples), 50)
            common["temperature"] = 310.
            np.savez(directory / "window_0001.npz", center=[1.], **common)
            with self.assertRaises(ValueError):
                load_windows(directory)


if __name__ == "__main__":
    unittest.main()
