# hello-enhanced-sampling
Repository of OpenMM scripts and recipes for testing and analysing various enhanced
sampling approaches in MD on a few toy systems:
- Alanine dipeptide in vacuum
- Beta cyclodextrin in vacuum

Current enhanced sampling approaches tested:
- Well-tempered metadynamics (with a CTMD example script)

## Dependencies

Replicate my environment using the `environment.yml` file:
```commandline
conda env create -f environment.yml
```