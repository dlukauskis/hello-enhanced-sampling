"""
OPES CV Reporter
Writes collective variable time series to disk (plain text), similar to
`MetadynamicsReporter` in the wt-metad code but for OPES.

Usage:
  reporter = OPEsCVReporter(filename, reportInterval, opes_instance)
  simulation.reporters.append(reporter)

The reporter will write a header line and then rows: <time_ps> <cv0> <cv1> ...
"""
import time
import os
import numpy as np
import openmm.unit as unit


class OPEsCVReporter:
    def __init__(self, file, reportInterval, opes_instance, append=False):
        self._reportInterval = int(reportInterval)
        self._openedFile = isinstance(file, str)
        self._opes = opes_instance
        if self._openedFile:
            self._out = open(file, 'a' if append else 'w')
        else:
            self._out = file
        self._append = append
        self._hasInitialized = False

    def describeNextReport(self, simulation):
        steps = self._reportInterval - (simulation.currentStep % self._reportInterval)
        return {'steps': steps, 'periodic': None, 'include': []}

    def report(self, simulation, state):
        # Initialize header on first report
        if not self._hasInitialized:
            headers = ['time_ps']
            for i in range(self._opes.num_cvs):
                headers.append(f'cv{i}')
            if not self._append:
                print(' '.join(headers), file=self._out)
            try:
                self._out.flush()
            except Exception:
                pass
            self._hasInitialized = True

        # get time in ps
        t_ps = round(state.getTime().value_in_unit(unit.picosecond))

        # get CV values from OPES force
        try:
            vals = self._opes.force.getCollectiveVariableValues(simulation.context)
            vals = [float(v) for v in vals]
        except Exception:
            # fallback: empty values
            vals = [0.0] * self._opes.num_cvs

        # write line
        fields = [str(t_ps)] + [f"{v:.12g}" for v in vals]
        print(' '.join(fields), file=self._out)
        try:
            self._out.flush()
        except Exception:
            pass

    def __del__(self):
        if self._openedFile:
            try:
                self._out.close()
            except Exception:
                pass

