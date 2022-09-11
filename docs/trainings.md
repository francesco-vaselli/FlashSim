## A brief intro to Normalizing Flows


## The key training concepts

The code for the training is commented (see [the GitHub repo][1]), thus you should be able to infer most of the steps from comments alone.
This page serves to discuss the most relevant training facts and choices in terms of hyperparameters, and to point at possible improvements and variations.

We trained on 5 millions jet/muons and validated on 100k jets/muons. The typical training times on a V100 32gb GPU were about 5 days for both models.

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

We will proceed to list them with a brief explanation for the chose value:

- The *learning rate* has been fixed to a relatively small value for two main reasons. Having 50+ millions of parameters for the full model, we wanted a smooth descent, without overshooting. A small value helped in this regard, however we experimented with values of around 0.001 and found that the model was still capable of convergence in way less epochs. However, letting the training go on longer with a lower lr resulted in better results regarding the *conditioning* (how the Gen-level information influences the outputs), a feture not captured by the loss. We thus left a smaller value than what's actually needed.
- *Epochs* and *batch size* have been guessed as reasonable values after a few initial trainings. With such a large training sample, a large batch size helped with both training speed and by providing a large number of events to average upon (useful for our loss)
- The *param dict* defines key quantities for our neural networks defining the single *splines* of the full NF model:
  - *number of transformation blocks*: specifying how many layers the network should have;
  - *activation type*: we found the ReLU reliable as always;
  - *batch normalization*: we turned it on as it supposedly helps with large models such as ours;
  - *number of bins*: how many bins should the spline defined by the neural network have. This is actually a key parameter, and we had massive improvements once we switched from low numbers to higher ones. **We could experiment with increasing this number while reducing layers/blocks**;
  - The *hidden dimension* specify how many nodes per layer the network should have.
- Finally, while creating the full model we must specify three numbers: the input variables number, the conditioning variables number and the *number of flow steps*, specifying how many splines the full model should have. We went with the exact same number as the input parameters. Because of the linear permutations and the coupling split at each flow step, this meant that eventually all variables will be generated with the others as conditioning--an ideal scenario for ensuring good correlations.
## The dataset issue

## The cosine annealing

 [1]: <https://github.com/francesco-vaselli/FlashSim/tree/main/trainings> "The git repo, training section" 
