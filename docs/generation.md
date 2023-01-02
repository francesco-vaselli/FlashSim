!!! warning "Change of behavior with Uproot update"

    Since the release of Uproot 5.0, we can no longer obtain a multiindex pandas dataframe with events of different length with the sintax `df = tree.arrays(library="pd", *args, **kwargs)` as done in the code used here. 
    
    The new syntax is `df = ak.to_dataframe(tree.arrays(library="ak", *args, **kwargs))`

## Generating synthetic NanoAOD files

The following section explains the code used for generating *synthetic* NanoAOD files starting from the GEN content of a pre-existing NanoAOD. The basics steps are:

1. Extract the GEN info from the `.root` file using a `C++` macro executed directly into the `Python` code thanks to the `ROOT.gInterpreter.Declare` interpreter. This is the same code used for extracting the training data, but there is no matching as we want just the GEN info. The ouptut is saved to an intermediate file.

2. Read the GEN information into various pandas dataframes with `Uproot`, *get the single events structure (i.e. how many objects per events)* to reconstruct the event later on. Additionally save global event information as well as some electron variables needed for the analysis test. Finally, check for events without muons/jets/electrons and adjust the events structure accordingly.

3. Generate the jets and muons data loading the trained models, adjust preprocessed outputs to get the physical quantities.

4. Convert results to jagged `awkward arrays` to save them directly into a `.root` file with the correct `TTree` structure.

Steps 2 and 4 required a careful handling of the events structure and are discussed below in greater detail.