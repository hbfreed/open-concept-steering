# Update on how we train SAEs

We've made improvements to how we train SAEs since Towards Monosemanticity with the goal of lowering the SAE loss. While the new setup is a significant improvement over what we published in Towards Monosemanticity we believe there are further improvements to be made. We haven't ablated every decision so it's likely some simplifications could be made. This work was explicitly focused on lowering loss and didn't grapple with loss not being the ultimate objective we care about. Here's a summary of our current SAE training setup:

Let n be the input and output dimension and m be the autoencoder hidden layer dimension. Let s be the size of the dataset. Given encoder weights $W_e \in \mathbb{R}^{m \times n}$, decoder weights $W_d \in \mathbb{R}^{n \times m}$, and biases $\mathbf{b}_e \in \mathbb{R}^{m}, \mathbf{b}_d \in \mathbb{R}^{n}$, the operations and loss function over a dataset $X \in \mathbb{R}^{s,n}$ are:

$$
\begin{aligned}
\mathbf{f}(x) &= \text{ReLU}( W_e \mathbf{x}+\mathbf{b}_e ) \\ 
\hat{\mathbf{x}} &= W_d \mathbf{f}(x)+\mathbf{b}_d \\ 
\mathcal{L} &= \frac{1}{|X|} \sum_{\mathbf{x}\in X} ||\mathbf{x}-\hat{\mathbf{x}}||_2^2 + \lambda\sum_i |\mathbf{f}_i(x)| ||W_{d,i}||_2 
\end{aligned}
$$

Note that the columns of $W_d$ have an unconstrained L2 norm (in Towards Monosemanticity they were constrained to norm one) and the sparsity penalty (second term) has been changed to include the L2 norm of the columns of $W_d$. We believe this was the most important change we made from Towards Monosemanticity.

$\mathbf{b}_e$ and $\mathbf{b}_d$ are initialized to all zeros. The elements of $W_d$ are initialized such that the columns point in random directions and have fixed L2 norm of 0.05 to 1 (set in an unprincipled way based on n and m, 0.1 is likely reasonable in most cases). $W_e$ is initialized to $W_d^T$.

The rows of the dataset X are shuffled. The dataset is scaled by a single constant such that $\mathbb{E}_{\mathbb{x} \in X}[||x||_2] = \sqrt{n}$. The goal of this change is for the same value of $\lambda$ to mean the same thing across datasets generated by different size transformers.

During training we use Adam optimizer beta1=0.9, beta2=0.999 and no weight decay. Our learning rate varies based on scaling laws, but 5e-5 is a reasonable default. The learning rate is decayed linearly to zero over the last 20% of training. We vary training steps based on scaling laws, but 200k is a reasonable default. We use batch size 2048 or 4096 which we believe to be under the critical batch size. The gradient norm is clipped to 1 (using clip_grad_norm). We vary $\lambda$ during training, it is initially 0 and linearly increases to its final value over the first 5% of training steps. A reasonable default for $\lambda$ is 5.

We do not use resampling or ghost grads because less than 1% of our features are dead at the end of training (dead means not activating for 10 million samples). We don't do any fine tuning after training.

Conceptually a feature's activation is now $\mathbf{f}_i ||W_{d,i}||_2$ instead of $\mathbf{f}_i$. To simplify our analysis code we construct a model which makes identical predictions but has an L2 norm of 1 on the columns of $W_d$. We do this by $W_e' = W_e ||W_d||_2$, $b_e' = b_e ||W_d||_2$, $W_d' = \frac{W_d}{||W_d||_2}$ and $b_d'=b_d$.

Our initialization likely needs improvement. As we increase m the reconstruction loss at initialization increases. This may cause problems for sufficiently large m. Potentially with improved initialization we could remove gradient clipping.

We haven’t seen improvements in loss from resampling or ghost grads, but it’s possible resampling “low value” features would improve loss.

It’s plausible some sort of post training (for example Addressing Feature Suppression in SAEs) would be helpful.

Improving shrinkage is an area for improvement.

There are likely other areas for improvement we don’t know about.

Given a fixed dataset X as we increase m the loss consistently decreases. We’ve been able to increase m to single digit millions without issues. This holds across a variety of transformer sizes and mlp activations or the residual stream. Our setup from Towards Monosemanticity would frequently have higher loss, many dead features, or many nearly identical features when run with large values of m.

We make changes to our training setup by looking at loss across a variety of values of \lambda, m, transformer sizes, and mlp or residual stream runs. We’re generally excited by a change that consistently decreases loss by at least 1%, or a change with roughly equal loss that simplifies our training setup. With our setup, comparing runs on (L0, MSE) or (L0, % of MLP loss recovered) requires care because L0 can be unstable. For example we’ve had cases where training twice as long with half the number of features leads to a &lt;1% change in MSE and L1 but a 30% decrease in L0.

Here are some results from small models. All runs have 131,072 features, 200k train steps, batch size 4096. Note that L1 of f depends on our specific normalization of activations.

| Type of Run                  |   Lambda |     L0(f) |    L1(f) |   Normalized MSE |   Frac Cross Entropy Loss Recovered |
|------------------------------|----------|-----------|----------|------------------|-------------------------------------|
| 1L MLP                       |        2 |  99.6239  | 17.2256  |          0.03054 |                             0.98305 |
| 1L MLP                       |        5 |  38.6873  | 11.5959  |          0.06609 |                             0.96398 |
| 1L MLP                       |       10 |  20.0687  |  7.12194 |          0.1312  |                             0.91426 |
| 4L MLP (layer 2)             |        2 | 264.029   | 95.0349  |          0.06824 |                             0.96824 |
| 4L MLP (layer 2)             |        5 |  69.9276  | 56.9238  |          0.12546 |                             0.92904 |
| 4L MLP (layer 2)             |       10 |  26.4846  | 39.4266  |          0.18485 |                             0.88438 |
| 4L Residual Stream (layer 2) |        2 |  81.5859  | 30.3732  |          0.09543 |                             0.9572  |
| 4L Residual Stream (layer 2) |        5 |  33.2312  | 19.1226  |          0.16295 |                             0.90443 |
| 4L Residual Stream (layer 2) |       10 |   8.71466 | 12.5389  |          0.25455 |                             0.83883 |


Features uncovered by sparse autoencoders are optimized to reconstruct model activity while remaining sparsely active. Our team and others have observed that these features often appear to encode specific, interpretable concepts. However, a potential concern about using these features for an interpretability agenda is that, despite their semantic significance to humans, these features may not capture the abstractions that the model uses for its computation. We have conducted preliminary experiments that suggest that models do in fact “listen” to feature values significantly more than would be expected by chance.

Our experiment works as follows: We train a sparse autoencoder (SAE) on the residual stream following the third layer of a trained four-layer transformer. For each SAE feature, we take a representative sample of datapoints for which that feature has a nonzero activation, scale the value of that activation by a factor of either 0 (“feature ablation”) and 2 (“feature doubling”), and propagate the updated value through to the model according to the same procedure as in Towards Monosemanticity (the case of scaling factor equal to zero corresponds to a feature ablation). We compute the average increase in the model’s loss following this procedure.

Our goal is to determine whether the loss increase from rescaling feature activation values is especially high compared to other model perturbations with similar statistics. If so, it would provide evidence that feature directions exert “special” influence on downstream computation in the model. To this end, we compare feature rescaling to several controls:

- Apply a random perturbation to the residual stream activity at the same layer, matching the magnitude of the random perturbation to the magnitude of the perturbation induced by the feature rescaling (“random perturbation, model activations” in the figure).

This control is meant to test whether feature ablations are more significant than random perturbations. Arguably, this is a weak baseline, as the variance of model activations is likely not isotropic, and thus some dimensions of residual stream activity may be less consequential to model behavior. SAE feature vectors are trained to reconstruct model activity, and as a result probably concentrate in more important dimensions. Thus, as a stronger baseline, we tried the following:

- Apply a random perturbation of the same magnitude as the feature rescaling on the feature activations vector (“random perturbation, feature activations” in the figure), and propagate that perturbation through to the model according to the same procedure used for the feature rescalings.

These experiments revealed several interesting findings:

- We found that feature ablations have significantly greater impact on model performance than either of the controls.

- Interestingly, ablating the feature activation has a substantially greater effect on model performance than amplifying the feature by the same amount, suggesting that the influence of features on model outputs may (partially) saturate at higher feature activations.

We also compared the effect of feature perturbations to other, more structured forms of perturbations.  In all cases we match the magnitude of the perturbation in model activation space to be equal to that of the corresponding feature ablation.

- Controlling for perturbation magnitude, applying perturbations in a direction antiparallel to the SAE reconstructions (“dampen feature activations” in the figure) – equivalent to “spreading out” a feature ablation across all features, in proportion to their activity – produces similar effects as single-feature ablations.

- Controlling for magnitude, perturbations antiparallel to the model activity (“dampen model activations” in the figure) have less impact than feature dampening. Note that dampening model activations is different than dampening feature activations, as model activations include two components – the bias term of the SAE, and the error vector left unexplained by the SAE – that are not affected by feature dampening. This result indicates that the feature dampening effect is not explainable simply due to its effect on activation norm.  In fact, in later layers, the effect of dampening model activations almost vanishes.

- Consistent with the results of Gurnee (discussed in detail elsewhere in this update), magnitude-controlled perturbations along the reconstruction error direction -(x - SAE(x)) (“perturb along residual” in the figure) are also substantially more impactful than random perturbations, though less impactful than feature ablations.  While not conclusive, these results suggest that Gurnee’s findings are consistent with a model in which SAE reconstruction errors lie primarily along feature directions, as this would explain their greater-than-random impact on model outputs.

- The outsized impact of feature ablations is more pronounced for residual stream features than MLP layer features. This suggests that residual stream and MLP features may play different functional roles in the network, though our understanding of this result is limited.

These results are preliminary, but generally support the idea that feature directions uncovered by SAEs are high-leverage “levers” for influencing model outputs.
