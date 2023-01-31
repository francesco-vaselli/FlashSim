!!! warning "Change of behavior with Uproot update"

    Since the release of Uproot 5.0, we can no longer obtain a multiindex pandas dataframe with events of different length with the sintax `df = tree.arrays(library="pd", *args, **kwargs)` as done in the code used here. 
    
    The new syntax to get the same behavior is `df = ak.to_dataframe(tree.arrays(library="ak", *args, **kwargs))`

## Generating synthetic NanoAOD files

The following section explains the code used for generating *synthetic* NanoAOD files starting from the GEN content of a pre-existing NanoAOD. The basics steps are:

1. Extract the GEN info from the `.root` file using a `C++` macro executed directly into the `Python` code thanks to the `ROOT.gInterpreter.Declare` interpreter. This is the same code used for extracting the training data, but there is no matching as we want just the GEN info. The ouptut is saved to an intermediate file.

2. Read the GEN information into various pandas dataframes with `Uproot`, *get the single events structure (i.e. how many objects per events)* to reconstruct the event later on. Additionally save global event information as well as some electron variables which have not been simulated and are needed for the analysis test. Finally, check for events without muons/jets/electrons and adjust the events structure accordingly.

3. Generate the jets and muons data loading the trained models, adjust preprocessed outputs to get the physical quantities.

4. Convert results to jagged `awkward arrays` to save them directly into a `.root` file with the correct `TTree` structure.

All the steps are repeated for each NanoAOD file in the dataset, by calling the `nbd` (*Nano Builder*) function in a for loop.

Steps 2 and 4 required a careful handling of the events structure and are discussed below in greater detail.

### Getting the events structure

Let's have a look at code of the reading and structuring for muons:

```python
    # read muon data to df
    dfm = tree.arrays(muon_cond, library="pd").astype("float32").dropna()
    dfm = dfm[~dfm.isin([np.nan, np.inf, -np.inf]).any(1)]
    phys_pt = dfm["MGenMuon_pt"].values  # for later rescaling
    print(phys_pt.shape)

    # preprocess conditioning variables
    dfm["MGenMuon_pt"] = dfm["MGenMuon_pt"].apply(
        lambda x: np.log(x)
    )  # for conditioning
    dfm["MClosestJet_pt"] = dfm["MClosestJet_pt"].apply(lambda x: np.log1p(x))
    dfm["MClosestJet_mass"] = dfm["MClosestJet_mass"].apply(lambda x: np.log1p(x))
    dfm["Pileup_sumEOOT"] = dfm["Pileup_sumEOOT"].apply(lambda x: np.log(x))
    dfm["Pileup_sumLOOT"] = dfm["Pileup_sumLOOT"].apply(lambda x: np.log1p(x))
    dfm = dfm[~dfm.isin([np.nan, np.inf, -np.inf]).any(1)]
    print(dfm)
    # crucial step: save original multiindex structure to restructure outputs later
    muons_ev_index = np.unique(dfm.index.get_level_values(0).values)
    print(muons_ev_index)
    events_structure_muons = (
        dfm.reset_index(level=1).index.value_counts().sort_index().values
    )
    print(len(events_structure_muons))
    print(sum(events_structure_muons))

    # reset dataframe index for performing 1to1 generation
    dfm.reset_index(drop=True)
```

The variable `muons_ev_index` stores the event number. This is needed because some events may be missing muons and will not be listed: if event 3 has no muons the list will be `[0, 1, 2, 4, ..., 1287960]`.  

The `events_structure_muons` variable is a list of the number of muons in each event (i.e. `[3, 2, ..., 1, 2]`), ordered in the same way as `muons_ev_index`. Both are being extracted directly from the pandas multiindex structure which is generated directly from Uproot. This has the advantage of avoiding needles looping and counting over the events.

Then, because we need to include 0s where there are no muons in order to restructure the final events, we are manually putting them in the correct positions:

```python
    # adjust structure if some events have no muons
    # dfe = dataframe event containing global info for each event (and thus all events)
    zeros = np.zeros(len(dfe), dtype=int)
    print(len(muons_ev_index), len(events_structure_muons))
    # puts number of muons in corresponding event, otherwise leave 0
    np.put(zeros, muons_ev_index, events_structure_muons, mode="rise")
    events_structure_muons = zeros
    print(events_structure_muons.shape, events_structure_muons)
    print(sum(events_structure_muons))
```

### Structuring the results

Once we have the outputs of our networks, we simply restructure them into a jagged awkward array and save it to file:

```python
    # get a muon collection with the original event structure and branches
    # muon_names = branches names
    # totalm = final output for muons
    to_ttreem = dict(zip(muon_names, totalm.T))
    to_ttreem = ak.Array(to_ttreem)
    to_ttreem = ak.unflatten(to_ttreem, events_structure_muons)

    # once we have all objects save file
    # use uproot recreate to save directly akw arrays to .root file
    new_path = str(os.path.join(new_root, file_path))
    new_path = os.path.splitext(new_path)[0]
    with uproot.recreate(f"{new_path}_synth.root") as file:
        file["Events"] = {
            "Jet": to_ttreej,
            "Muon": to_ttreem,
            "Electron": to_ttreel,
            "event": to_ttreee.event,
            "run": to_ttreee.run,
        }

    return
```

## Upsampling

Because we are generating events so fast, the production of new GEN samples becomes the next speed bottleneck. We investigated the *upsampling* procedure, that is sampling multiple times from the same GEN sample, as a way to mitigate the bottleneck. The results and statistical handling are discussed in the next section; here we discuss the changes to the generation code.

We used the following strategy:

```python
    # main difference from 1to1 case: we need to produce 
    # an upsampled index for saving the right topology later
    numb_ev = dfm.index.get_level_values(0).values
    # use list comprehension to save the new event list
    # (if UPSAMPLING_FACTOR=1 GO from [0, 1, 2] to [0, 1, 2, 3, 4, 5])
    l = [numb_ev + n * (numb_ev[-1] + 1) for n in np.arange(0, UPSAMPLE_FACTOR)]
    numb_ev = np.concatenate(l, axis=0)
    # main upsampling idea: concatenate multiple copies 
    # of the original df
    dfm = pd.concat([dfm] * UPSAMPLE_FACTOR, axis=0)
    numb_sub_ev = dfm.index.get_level_values(1).values
    up_index = pd.MultiIndex.from_arrays(
        [numb_ev, numb_sub_ev], names=["event", "object"]
    )
    dfm = dfm.set_index(up_index)
```

We defined a global `UPSAMPLE_FACTOR = n` defining how many throws should be performed for each event. Then, we created a new event index repeating the original event structure n times. We concatenated the original dataframe n times and applied the new structure. 

In this way, each event is repeated exactly n times, one repetition at a time in the original ordering.