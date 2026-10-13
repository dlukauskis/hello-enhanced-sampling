# hello-enhanced-sampling
Repository of OpenMM scripts and recipes for testing and analysing various enhanced
sampling approaches in MD on a few toy systems:
- Alanine dipeptide in vacuum
- Beta cyclodextrin in vacuum

Current enhanced sampling approaches tested:
- Well-tempered metadynamics (with a CTMD example script)
- On-the-fly probability enhanced sampling (OPES)
- [Umbrella sampling](umbrella/README.md), with 1D/2D alanine dipeptide examples
  and MBAR FES reconstruction

## Dependencies

Replicate my environment using the `environment.yml` file:
```commandline
conda env create -f environment.yml
```

Umbrella reconstruction additionally requires PyMBAR:
```commandline
conda activate openmm
python -m pip install -r umbrella/requirements.txt
```
