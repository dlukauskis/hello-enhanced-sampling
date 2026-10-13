"""CV reporting compatible with OpenMM's Simulation reporter interface."""
class UmbrellaCVReporter:
    def __init__(self, file, reportInterval, umbrella):
        if not isinstance(reportInterval, int) or reportInterval <= 0:
            raise ValueError("reportInterval must be a positive integer")
        self.interval = reportInterval
        self.umbrella = umbrella
        self._owns_file = isinstance(file, (str, bytes)) or hasattr(file, "__fspath__")
        self._out = open(file, "w") if self._owns_file else file
        self._out.write("# step " + " ".join(
            f"{field}{i}" for field in ("cv", "center", "k")
            for i in range(len(umbrella.variables))) + "\n")

    def describeNextReport(self, simulation):
        return {"steps": self.interval - simulation.currentStep % self.interval,
                "periodic": None, "include": []}

    def report(self, simulation, state):
        values = self.umbrella.getCollectiveVariables(simulation)
        centers, springs = self.umbrella.get_window(simulation.context)
        self._out.write(str(simulation.currentStep) + " " +
                        " ".join(f"{v:.12g}" for v in (*values, *centers, *springs)) + "\n")
        self._out.flush()

    def close(self):
        if self._owns_file:
            self._out.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
