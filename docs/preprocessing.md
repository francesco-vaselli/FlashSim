## What we did and why we did it

We processed the NanoAOD files and extracted all the Jet objects matched to a GenJet object and the Muon objects matched to a GenMuon across all the events in the file.
The output of the *extraction* step is another `.root` file containing just the selected objects. We need this 1-to-1 matching because we want to pass the RECO objects as targets and the corresponding Gen objects as *conditioning* to the models (see [Trainings][1] section), that is we want the output to depend on the physical inputs of the various possible processes.

The resulting file is still organized according to the Events structure (that is, a single event may contain an arbitrary number of jets and muons). Besides, we know that many machine learning algorithms work best when specific distributions are *preprocessed* according to specifc criteria. Normalizing Flows are no exception. Specifically, there are four key features which should be accounted for and modified through preprocessing before training:


1. Because NF are learning actual pdfs, *large gaps* between values of the distribution may disturb training and trick the network to *bridge* the extremes of the distribution by creating spurious samples in the gap. When possible, the gaps should be reduced and the values packed closer together;

2. As NF assume a continuous and differentiable pdf, they are not well suited to deal with *discrete* distributions. Thus, we should apply a process known as *dequantization*, that is applying some sort of smearing to the discrete values to make them similar to those sampled from a continuous distribution;

3. For similar reasons as before, when possible it would be beneficial to widen and normalize steeply falling distributions through invertible transforms such as log(x). If well separated, eventual peaks may be dequantized as well;

4. Finally, we opted for *saturating* long tails of distributions to some limiting values, in order to make it easier for the model to learn the pdf in the more populated region.


Apart from possibly dequantization, we stress that all of this transformations were implemented to make training easier but are not strictly necessary--the models revealed themselves as powerful enough to deal with complex, sharply peaked, long tailed distributions. However, having already implemented the preprocessing pipeline and because it did not introduce a big overhead in the procedure, we decided to keep it for the present work. An example of one of the possible preprocessing operations is shown in the following figure:


![The basic idea](img/preproce.pdf-1.jpg)

Sharply peaked distribution are being converted to more broad ones during the preprocessing step. In this example the `ip3d`, the 3D impact parameter of the $\mu$ w.r.t. the primary vertex, variable gets transformed as log(`ip3d`+0.001).

All of these transforms may be implemented with a clear and natural syntax in the `Python` programming language, specifically thanks to the `pandas` package, which implements a convenient dataframe structure.

## Extraction details

First, we extract the Gen and RECO objects from an existing NanoAOD file. With the use of `C/C++` code (e.g. [muons_extraction.cpp][2] )
for the `ROOT` data analysis framework, this operation can be performed rather quickly thanks to the *compiled* nature of the language being used and the powerful `ROOT::RDataFrame()` class, offering a modern, high-level interface for the manipulation of data stored in a NanoAOD `TTree`, as well as `multi-threading` and other low-level optimizations.
The output of the \emph{extraction} step is another \texttt{.root} file containing just the selected objects.

```python
your_code = do_some_stuff
```


 [1]: <https://francesco-vaselli.github.io/FlashSim/trainings/> "The next section" 

 [2]: <https://github.com/francesco-vaselli/FlashSim/blob/main/preprocessing/muons_extraction.cpp> "muons script" 