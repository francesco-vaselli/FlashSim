## FlashSim: A Deep Learning Solution to the HEP simulation problem


## PLEASE NOTE THAT THIS IS AN OLD PROTOTYPE FOR MY MASTER THESIS. CURRENT DEVELOPMENT IS HAPPENING INTERNALLY IN CMS @ CERN

[![Made with MkDocs](https://img.shields.io/badge/docs-online-green)](https://francesco-vaselli.github.io/FlashSim/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

The current repository presents the original code implementation for a *Flash Simulation* approach at the CMS experiment based upon *Normalizing Flows*.
Please consult the [docs](https://francesco-vaselli.github.io/FlashSim/) for a more comprehensive discussion.
The basic idea is expressed below:

![toa](figures/pipeline.png)

The proposed FlashSim would be able of performing realistic
NanoAOD production and effectively bypassing all the intermediate steps. The FullSim chain is showed above, along with the
CMS FastSim and our FlashSim approaches. We show below the
real data processing chain: the RECO and file formats steps are
in common between the two. 

### End-to-end analysis sample generator

![toa](figures/endtoend.png)

We also present the code of the general idea for an end-to-end analysis
sample generator in the NanoAOD format. The key concept can be
easily grasped through the figure above: a FullSim NanoAOD file gets
processed and its Gen-level values extracted for eventual preprocessing. Then, the values, along with random noise, are passed to the two networks, which generate
raw samples which are finally postprocessed to reobtaine physical
distributions and combined into a single, NanoAOD-like file format.
The whole process can be executed by a single call to a Python script,
which leverages the ROOT C interpreter for running the extraction and
the uproot package for structuring and saving the data directly in the
*.root* format, in a corresponding TTree data structure.
