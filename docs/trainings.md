## A brief intro to Normalizing Flows


## The key training concepts

The code for the training is commented (see [the GitHub repo][1]), thus you should be able to infer most of the steps from comments alone.
This page serves to discuss the most relevant training facts and choices in terms of hyperparameters, and to point at possible improvements and variations.

We trained on 5 millions jet/muons from the $t\overline{t}$ process and validated on 100k jets/muons. The typical training times on a V100 32gb GPU were about 5 days for both models.

The most relevant hyperparameters are defined in the snippet below (for the jets model):

```py
    # define hyperparams
    lr = 1e-5
    total_epochs = 600
    batch_size = 2048

    #stuff 

    # define additional model parameters
    param_dict = {
        "num_transform_blocks": 10,
        "activation": "relu",
        "batch_norm": True,
        "num_bins": 128,
        "hidden_dim": 298,
    }

    # create model
    flow = create_NDE_model(17, 14, 17, param_dict)
```

We will proceed to list them with a brief explanation for the chosen value:

- The *learning rate* has been fixed to a relatively small value for two main reasons. Having 50+ millions of parameters for the full model, we wanted a smooth descent, without overshooting. A small value helped in this regard, however we experimented with values of around 0.001 and found that the model was still capable of convergence in way less epochs. However, letting the training go on longer with a lower lr resulted in better results regarding the *conditioning* (how the Gen-level information influences the outputs), a feature not captured by the loss. We thus left a smaller value than what's actually needed.
- *Epochs* and *batch size* have been guessed as reasonable values after a few initial trainings. With such a large training sample, a large batch size helped with both training speed and by providing a large number of events to average upon (useful for our loss)
- The *param dict* defines key quantities for our neural networks defining the single *splines* of the full NF model:
  - *number of transformation blocks*: specifying how many layers the network should have;
  - *activation type*: we found the ReLU reliable as always;
  - *batch normalization*: we turned it on as it supposedly helps with large models such as ours;
  - *number of bins*: how many bins should the spline defined by the neural network have. This is actually a key parameter, and we had massive improvements once we switched from low numbers to higher ones. **We could experiment with increasing this number while reducing layers/blocks**;
  - The *hidden dimension* specify how many nodes per layer the network should have.
- Finally, while creating the full model we must specify three numbers: the input variables number, the conditioning variables number and the *number of flow steps*, specifying how many splines the full model should have. We went with the exact same number as the input parameters. Because of the linear permutations and the coupling split at each flow step, this meant that eventually all variables will be generated with the others as conditioning--an ideal scenario for ensuring good correlations.

## The dataset classes

Remember that we trained on 5 millions of jets/muons. Because jets are numerous in our training process ($t\overline{t}$), we used a simple dataset for jets as we needed to open just one file to access the whole 5 millions jets.


```py
class MyDataset(Dataset):
    """Very simple Dataset for reading hdf5 data
        This is way simpler than muons as we heve enough jets in a single file
        Still, dataloading is a bottleneck even here
    Args:
        Dataset (Pytorch Dataset): Pytorch Dataset class
    """

    def __init__(self, h5_paths, limit):

        self.h5_paths = h5_paths
        self._archives = [h5py.File(h5_path, "r") for h5_path in self.    h5_paths]
        self._archives = None

        y = self.archives[0]["data"][:limit, 0:14] # conditioning
        x = self.archives[0]["data"][:limit, 14:31] # targets
        self.x_train = torch.tensor(x, dtype=torch.float32)  # .to(device)
        self.y_train = torch.tensor(y, dtype=torch.float32)  # .to(device)

    @property
    def archives(self):
        if self._archives is None:  # lazy loading here!
            self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]
        return self._archives

    def __len__(self):
        return len(self.y_train)
    # trivial get item by index
    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]
```

Because muons are more scarce, we actually had to open up multiple files during training to have at our disposal the 5 millions. The dataset is thus more complex:

```py
class H5Dataset(Dataset):
    """Pytorch Dataset for reading input data from hdf5 files on disk
    Expects hdf5 files containing a "data" Dataset, which in turn contains correctly processed data
    (there is no preprocessing here), and returns two separate tensor for each instance
    Uses np.searchsorted to getitems from different files (thanks @Nadya!)
    However, dataloading is a current bottleneck and should be investigated
    x: target variables (expects a 30:52 ordering on each row)
    y: conditioning variables (expects a 0:30 ordering on each row)
    Args:
        Dataset (Pytorch Dataset): Pytorch Dataset class
    """

    def __init__(self, h5_paths, limit=-1):
        """Initialize the class, set indexes across datasets and define lazy loading
        Args:
            h5_paths (strings): paths to the various hdf5 files to include in the final Dataset
            limit (int, optional): optionally limit dataset length to specified values, if negative
                returns the full length as inferred from files. Defaults to -1.
        """
        max_events = int(5e9)
        self.limit = max_events if limit == -1 else int(limit)
        self.h5_paths = h5_paths
        self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]

        self.strides = []
        for archive in self.archives:
            with archive as f:
                self.strides.append(len(f["data"])) 

        self.len_in_files = self.strides[1:]
        self.strides = np.cumsum(self.strides)
        self._archives = None

    @property
    def archives(self):
        if self._archives is None:  # lazy loading here!
            self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]
        return self._archives

    # smart get item through searchsorted for finding the right file
    # and getting the actual index in the file
    def __getitem__(self, index):
        file_idx = np.searchsorted(self.strides, index, side="right")
        idx_in_file = index - self.strides[max(0, file_idx - 1)]
        y = self.archives[file_idx]["data"][idx_in_file, 0:30]
        x = self.archives[file_idx]["data"][idx_in_file, 30:52]
        y = torch.from_numpy(y)
        x = torch.from_numpy(x)
        # x = x.float()
        # y = y.float()

        return x, y

    def __len__(self):
        # return self.strides[-1] #this will process all files
        if self.limit <= self.strides[-1]:
            return self.limit
        else:
            return self.strides[-1]
```

Despite the different approaches, we observed that **the data loading step may actually be a severe bottleneck in our training** as the GPU utilization heavily fluctuated between 40% and 70% most of the time. When we tried to implement a dataset similar to the muons' one for the jets we observed a significant slowdown, suggesting that the dataset classes may be part of the problem.

## The cosine annealing

It should be noted that the learning rate for both models was constantly updated through the *cosine annealing* procedure.

Cosine Annealing is a type of learning rate schedule that has the effect of starting with a large learning rate that is relatively rapidly decreased to a minimum value before being increased rapidly again. The resetting of the learning rate acts like a simulated restart of the learning process and the re-use of good weights as the starting point of the restart is referred to as a *warm restart* in contrast to a *cold restart* where a new set of small random numbers may be used as a starting point. The proper formula its:

$\eta_{t} = \eta_{min} + \frac{1}{2}\left(\eta_{max}-\eta_{min}\right)\left(1+\cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)$

were $\eta$ is the learning rate and $T$ the training epoch.

## The losses

We show here the losses for both models during training. Please note that the validation loss has been averaged over the last 5 epochs.


![The jets loss](img/lossesjets.png)

![The muons loss](img/lossesmuons.png)

The jets model shows clear signs for improvements, but it was stopped because we had to put results together for my thesis. The muons models, on the other hand, shows signs of stalling after epoch 200, but it was left in training as this vastly improved *conditioning*, a performance not captured by our loss.

 [1]: <https://github.com/francesco-vaselli/FlashSim/tree/main/trainings> "The git repo, training section" 
