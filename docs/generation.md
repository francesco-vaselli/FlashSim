!!! warning "Change of behavior with Uproot update"

    Since the release of Uproot 5.0, we can no longer obtain a multiindex pandas dataframe with events of different length with the sintax `df = tree.arrays(library="pd", *args, **kwargs)` as done in the code used here. The new syntax should be `df = ak.to_dataframe(tree.arrays(library="ak", *args, **kwargs))`