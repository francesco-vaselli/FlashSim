This long section serves to present in a comprehensive way the results obtained.

To perform a sound and reasonable comparison, we extract the Gen-values for conditioning from $10^{5}$ t$\overline{\text{t}}$ samples coming from an unseen test set for jets and from the validation set (which was not used for training but just for evaluating the loss over time) for muons. We then generate new analysis samples using FlashSim and starting from the same Gen-values, to get a one-to-one correspondence between the them and FullSim samples.

We performed two main types of comparison on the obtained samples: we compared the 1-d *distribution* and the 2-d *correlations* for any pair of variables.
We inspected the latter visually thanks to *contour plots*, but we wanted to have a precise measure for the similarity of the empirical distributions between two samples. We thus choose the *Wasserstein distance*, defined as:

$$W_1(u, v) = \inf_{\pi \in \Gamma(u, v)}\int_{\mathbb{R}\times\mathbb{R}}|x-y|d\pi(x,y) = \int_{-\infty}^{+\infty} |U - V|$$

where $\Gamma(u, v)$ is the set of (probability) distributions on $\mathbb{R}\times\mathbb{R}$ whose marginals are $u$ and $v$ on the first and second factors respectively, and $U$, $V$ are the respective CDFs. Intuitively, if each distribution is viewed as a unit amount of earth (soil), the metric is the minimum \emph{cost} of turning one pile into the other, which is assumed to be the amount of earth that needs to be moved times the mean distance it has to be moved, and thus this metric is also know informally as the *earth mover distance*.

## Jets Results

### 1D

We show in the following figure four 1-d distributions out of the total of 17 target variables obtained for jets. We emphasize once more that the model actually learned to generate the 17 values simultaneously, preserving the correct correlations as well as producing convincing distribution.

Regarding the distributions, we observe that the model has correctly learned all the multi-modal, sharply peaked tagging distributions with Wasserstein scores of the order of $0.001$, testifying good convergence. The log scale of btagDeepB actually shows an instance of *bridging*, where a small set of values were generated between two separate peaks. Single-mode distributions such as ptRatio have been larned as well, as were the Ids thanks to dequantization. The jetId outputs were rounded to the closest value between 0, 2 and 6, the only admissible ones.

Finally, we also observe a worse performance on two distributions: bRegCorr, a rather simple, skewed one-mode distribution which is expected to improve with further training (current Wasserstein distance is $\approx$ 0.02), and nConstituents. The latter result is probably in stronger disagreement because the target actually consists of integer values--as we discussed before, the NF approach expects continuous distributions, and so the model performs bridging in an attempt to obtain a reasonable continuous distribution. However, it has been observed in previous trainings for similar architectures that the model is actually capable of partially overcoming this limitation by brute force alone: if left in training for long enough it may eventually learn to output values close to the integer ones.


![deepB](img/eval3.pdf-1.jpg)
![ptratio](img/eval12.pdf-1.jpg)
![jetid](img/eval16.pdf-1.jpg)
![nconst](img/eval10.pdf-1.jpg)

### Jets correlations

The correlations between jets variables, inspected visually, show good agreement with those from FullSim. The following figures shows the highly non-trivial correlations between the tagging distributions, with *quantiles* plotted at 0.5, 0.9, 0.99. The same choice for quantiles has been adopted for all the following figures.

![corrjet1](img/corrjet1.pdf-1.jpg)

We can also observe in the following figures how the models have learned to capture the correlations between the qgl score, which is correctly correlated to the number of constituents as a lower number of constituents is expected for the u, d, s quarks when compared to gluons. Additionally, correlations between the physical p$_T$ and mass distributions, obtained from the original p$_T$Ratio and massRatio outputs of the network, have been learned as well.

![corrjet2](img/corrjet2.pdf-1.jpg)
![corrjet3](img/corrjet3.pdf-1.jpg)

## Muons results

### 1D

For the muons, we obtained similar results--good, convincing general convergence and correlations apart from a subset of the target variables. It should be noted that a larger number of target variables for this case were actually Boolean Ids, and as discussed before were approached through dequantization. The following figures shows 4 distributions out of 22 target variables. Aside from good convergence on the firs two, we can observe that for a series of them, such as the impact parameters errors dxyErr and dzErr the training is complicated by the fact that the NanoAOD format stores the variables in a low-precision format: this is reflected by the jagged structure in the plot for FullSim and it causes the model to perform bridging to reach convergence.

![mpt](img/meval2.pdf-1.jpg)
![mdxy](img/meval4.pdf-1.jpg)
![mdxyerr](img/meval6.pdf-1.jpg)
![softmva](img/meval21.pdf-1.jpg)

### Correlations

Finally, as a last example of correlations, we show in the following that the model has actually learned to capture complex correlations such as the ones between the *impact parameter* ip3d and the quantity $\sqrt{\texttt{dxy}^2 + \texttt{dz}^2}$, which is closely related to the definition of the impact parameter itself.

![mcorr](img/mcorrs.pdf-1.jpg)

## Conditioning

Another extremely important feature of our approach is the desired ability to obtain specific results starting from certain Gen-level inputs, a characteristic we called conditioning. The idea is that we want to learn not just $p^*_x(\mathbf{x})$, but $p^*_x(\mathbf{x}|\text{Gen})$.

We can readily see that this is possible by focusing on specific results obtained for the jets model. The following shows that the final, NanoAOD level reconstructed p$_T$ is correctly correlated to the GenJet p$_T$ for both FullSim and FlashSim: as we would expect the Gen-p$_T$ is crucial in determining the final-state p$_T$. What is more, in the same figure we also show the *profile histogram* and RMS ($\sigma_{p_T}$/p$_T$) for the GenJet p$_T$ versus the p$_T$Ratio. As expected, not only does the p$_T$Ratio decrease as the GenJet p$_T$ increases (highly energetic jets have a reconstructed p$_T$ closer to the Gen-value), but the RMS correctly decreases as well, as constant terms in the p$_T$ resolution due to pile-up are divided by bigger terms as GenJet p$_T$ increases.

![corjet4](img/corrjet4.pdf-1.jpg)
![profhist](img/profhist.pdf-1.jpg)

Additionally, because the partonFlavour conditioning variable allow us to specify the quark content of a jet, we can study how related quantities depend on this input. As a key example, we study the behaviour of the btagDeepB b-tagging distribution as we vary the parton input for the jet generation. The next figures show how the distribution changes according to the ground truth value specified as input: as expected, jets being conditioned with a b content present higher values of b-tagging, with a sharp peak at one, while those coming from u, d, s are clearly peaked around smaller values. Now we could think of defining a threshold and assign a reconstructed b content to all those jets higher than that value. We would naturally mistag some events, leading us to define a *flase-positive* ratio and a *true-positive* one. A standard figure of merit for these cases is the *Receiving operating characteristic* (ROC) curve, which plots the TPR against the FPR for all possible threshold choices. The last figure shows it for our model in log scale, showing minimal deviations from the target FullSim curve.

![btagj](img/btagj.pdf-1.jpg)
![roc1](img/rocj1.png)

Because our results are not as close to FullSim as it was for 2-d correlations, we would like to compare them with other competing approaches to asses the goodness of our own methodology. In order to do so, for a previous training with a lower number of jet target variables, presented at the CMS Machine learning Forum of April 2022, we compared the ROC curves between FullSim, FastSim and FlashSim on a $10^{6}$ t$\overline{\text{t}}$ samples set (not previously seen during training). Results are shown in the next Figure. We can see that while the ROC between our approach and FullSim is actually indistiguishable for TPR higher than 0.8, the FastSim ROC completely *overshoots* the target, due to oversimplifications in the simulation approach. With longer training times and additional loss terms addressing this type of conditioning, which is currently not considered by the model loss function, we are confident that the performance of FlashSim could be improved even more.

![allrocs](img/allrocs.pdf-1.jpg)

## Speed

A crucial result obtained is the *generation speed*: for both jets and muons we managed to generate samples in batches of $10^{4}$ in $\approx$ 0.3 seconds each, corresponding to a generation speed of raw samples of about 33,300 *samples per second* (33 kHz) *meaning a six orders of magnitude speedup when compared to FullSim and four orders of magnitude speedup when compared to FastSim*! Even considering possible reduction due to preprocessing and data loading, this result testify to the potential of the current methodology to completely redefine our approach to event simulation, at least at the NanoAOD level.

What is more, the $10^{4}$ batch size for generation was limited only by the VRAM of the GPU being used, meaning that more powerful GPUs, ideally working in parallel, could achieve even faster generation times.

## Results on unseen processes

We actually extended the use of the models to *unseen, different physical processes*: *Drell-Yan two-jets* (DY), *Electroweak two-muons two-jets* (EWK LLJJ) and two *Signal* (H$\rightarrow\mu^+\mu^-$) datasets were processed as well and stored for the comparison of the next chapter. Some results, are showed below and emphasize how our approach has correctly learned to simulate the interaction and reconstruction of the CMS detector, giving consistent results independently from the input process.

![ewk](img/ewkeval7.pdf-1.jpg)
![hmm](img/hmmeval2.pdf-1.jpg)
![dy](img/dycorrs.pdf-1.jpg)

## Benchmark analysis

Having described in detail our innovative approach to event simulation, and having applied it to generated events for which we have the FullSim sample, we decided to repeat the basic steps of a recent analysis to demonstrate the feasibility of our model in a real-case scenario.

## Upsampling