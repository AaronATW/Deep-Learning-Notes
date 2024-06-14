## $\beta$-VAE
[https://openreview.net/pdf?id=Sy2fzU9gl](https://openreview.net/pdf?id=Sy2fzU9gl)
### Introduction
- **Goal**: Learning **interpretable**, **factorized** latent **representations** in a **unsupervised** manner.
- **Disentangled Representation**: Single latent units are sensitive to single generative factors (how we interpret the world).
- **Unsupervised Learning**: Recent work requires prior knowledge about the number or the nature of generative factors, but not really practical in the real world.
- A relevant unsupervised disentanglement approach is **InfoGAN**.
	- Capable of discovering some data generative factors.
	- Instable training, insufficient sample diversity, sensitive to priors.
- What $\beta$-VAE does:
	- Add a hyperparameter $\beta (\beta > 1)$ to add a learning constraint based on VAE.
	- Latent information channel capacity is limited, and emphasize on factor independency.
	- Devise a quantitative protocol of disentanglement.
	- Achieve SOTA.
### Theory base
**Data framework:** 
- $\mathcal{D}=\{X,V,W\}$
- Images: $\mathbf x\in\mathbb{R}^N$
- Conditionally independent factors: $\mathbf v\in\mathbb{R}^K$, $\log p(\mathbf v|\mathbf x)=\sum_k \log p(v_k|\mathbf x)$
- Conditionally dependent factors: $\mathbf w\in\mathbb{R}^H$
- World simulator: $p(\mathbf x|\mathbf v,\mathbf w)=\mathbf{Sim}(\mathbf v, \mathbf w)$
**Learning Goal**:
- To learn generative latent factors $\mathbf z\in\mathbb{R}^M\ (M\geq K)$
- $p(\mathbf x|\mathbf z)\approx p(\mathbf x|\mathbf v,\mathbf w)=\mathbf{Sim}(\mathbf v, \mathbf w)$
- We want to maximize the marginal likelihood of observed data $\mathbf x$.
- We also want to give an isotropic unit Gaussian prior $p(\mathbf z)=\mathcal{N}(\mathbf{0},I)$, to encourage lower capacity of latent information bottleneck and statistical independence.
**Loss Function**:
- Maximizing a Lagrangian: $$\mathcal{F}(\theta,\phi,\beta;\mathbf x,\mathbf z)=\mathbb E_{q_\phi(\mathbf z|\mathbf x)}[\log p_\theta(\mathbf x|\mathbf z)]-\beta(D_{KL}(q_\phi(\mathbf z|\mathbf x)||p(\mathbf z)-\epsilon))$$- Lower bound as the loss function:$$\mathcal{F}(\theta,\phi,\beta;\mathbf x,\mathbf z)=\mathcal{L}(\theta,\phi;\mathbf x,\mathbf z,\beta)+\beta\epsilon\geq\mathcal{L}(\theta,\phi;\mathbf x,\mathbf z,\beta)=E_{q_\phi(\mathbf z|\mathbf x)}[\log p_\theta(\mathbf x|\mathbf z)]-\beta(D_{KL}(q_\phi(\mathbf z|\mathbf x)||p(\mathbf z))$$
- Letting $\beta>1$ will put a stronger constraint on latent variables, and higher $\beta$ values encourage learning a disentangled representation of $\mathbf v$.
- Too large $\beta$ will decrease the reconstruction fidelity.
### Disentanglement Metric
**Defining our expectations on disentangled representations:**
- If a representation of data is disentangled with true generative factors, we can use robust classification methods to representations and get good classification results (**interpretability**). 
- **Independence**: important, but it doesn't guarantee interpretability.
**Illustrations and Pseudocodes**:
![[Pasted image 20240613122733.png]]
![[Pasted image 20240613122751.png]]
### Experiments
**Latent variables traversal**:
![[Pasted image 20240613123055.png]]
**Representation learned by different models:**
![[Pasted image 20240613123142.png]]
**The effect of $\beta$**:
![[Pasted image 20240613123251.png]]
## Challenging Common Assumptions
### Basic Ideas
There are some common assumptions of unsupervised learning of disentanglement representations. 
We want to find a transformation $r(\mathbf x)$ which is interpretable. But they have some problems:
- It can be proved that it's impossible to do so without inductive biases.
- It does not always work well on downstream tasks.
- Some experiments are not reproducible, highly relying on random seeds and hyperparameters. It makes model choosing unreliable.
### Why Impossibility?
**Theorem**:
- If $\mathbf z$ is factorizable: $p(\mathbf z)=\prod_{i=1}^d p(\mathbf z_i)$.
- There exists an infinite family of bijective functions $f: \text{supp}(\mathbf z)\to\text{supp}(\mathbf z)$:
	- Such that $\frac{\partial f_i(\mathbf u)}{\partial u_j}\neq 0$ almost everywhere. ($\mathbf z$ and $f(\mathbf z)$ are completely entangled)
	- And $P(\mathbf z \leq \mathbf u)=P(f(\mathbf z)\leq\mathbf u)$ for all $\mathbf u\in \text{supp}(\mathbf z)$. (They have the same marginal distribution)
**Proof**:
1. We have a generative model $p(\mathbf z)$ and $P(\mathbf x|\mathbf z)$.
2. We find an $r(\mathbf x)$ totally disentangled with $\mathbf z$.
3. Based on the theorem, there should be another generative model with $\mathbf{\hat{z}}=f(\mathbf z)$. It's totally entangled with $\mathbf z$ and $r(\mathbf x)$.
4. As both generative models have the same marginal distribution of $\mathbf x$, $r(\mathbf x)$ should at least be entangled with one of the latent variables, and we don't know how to recognize that because we only have access to $\mathbf x$.
**Result:**
We can only do the unsupervised disentanglement introducing inductive biases. We should make it explicit in future work.
### Experiment Results
1. Individual dimensions are not correlated doesn't mean that mean representations are uncorrelated. ![[Pasted image 20240613130012.png]]
2. Different disentanglement metrics have different levels of correlations. ![[Pasted image 20240613130113.png]]
3. The disentanglement scores seem to rely less on models themselves (objective functions), but more on hyperparameters and random seeds. ![[Pasted image 20240613130243.png]]
4. Unsupervised model selection is not solved yet.
	- No consistent hyperparameter choosing strategy for different models.
	- Rank correlations between unsupervised metrics and disentanglement metrics show no clear pattern.
	- We cannot transfer good hyperparameters to other datasets.
1. It's not clear to us that downstream tasks perform better because of better disentanglements. ![[Pasted image 20240613130512.png]]
## Non-linear ICA
### Basic Ideas:
- We want to find useful representations of high-dimensional data, which needs disentanglement.
- Linear ICA has been successful.
- **Non-linear ICA** is problematic because it lacks **identifiability** (uniqueness of the representation).
- Utilizing temporal structures or auxiliary information can make it possible.
### Linear ICA:
$$\mathbf x= \mathbf A \mathbf s$$
- $\mathbf s$ is the latent variable, $\mathbf A$ is the mixing matrix, and $\mathbf x$ is the observation.
- If $\mathbf s$ is Gaussian, we cannot recover it from the mixture.
- If $s_1,s_2, ... ,s_d$ are statistically independent, and non-Gaussian, the true components can be estimated by learning a demixing matrix $\mathbf B$ such that  $\mathbf z:= \mathbf B\mathbf x$.
### Non-linear ICA's Identifiability Problem
$$\begin{align*}
\mathbf{x} &= \mathbf{f}(\mathbf{s}) \\
\mathbf{z} &:= \mathbf{g}(\mathbf{x})
\end{align*}$$Instead of matrix, we use a non-linear mixing and demixing function.
However, it can be proved that in arbitrary case, we can construct a trivial disentanglement which is absurd. Therefore, there exists the identifiability problem. ![[Pasted image 20240613132929.png]]
### VAE
- $\mathbf x=\mathbf f(\mathbf z)+\mathbf n$
- **Unidentifiability** is more serious.
- Can be seen as a non-linear PCA.
### Identifiable Nonlinear ICA
$$\mathbf x(t)=\mathbf f(\mathbf s(t))$$
At each time $t$, $\mathbf x(t)$ and $\mathbf s(t)$ are both vectors (varying over $t$). 
#### Time-Contrastive Learning (TCL)
- **Assumption**: Sources are independent, and also nonstationary time series.
- When we cut time into segments, they become piece-wise stationary and every segment is independent.
- Latent Distribution Modeling: $$\log p_\tau(s_i)=\log q_{i,0}(s_i)+\sum_{j=1}^k \lambda_{i,j}(\tau)q_{i,j}(s_i)-\log Z_i(\lambda_{i,1}(\tau),...,\lambda_{i,k}(\tau))$$- $\mathbf g=\mathbf f^{-1}$ is learned by **self-supervised learning (SSL)**, which make a combination of a neural network and a softmax layer predict which segment $\tau$ this $\mathbf x(t)$ come from.
- Possible to conduct dimension reduction.
![[Pasted image 20240613134717.png]]
#### Permutation-Contrastive Learning (PCL)
- **Assumption:** Independent components have temporal dependencies (**autocorrelations**):$$s_i(t)=r_i(s_i(t-1))+n_i(t)$$
- **SSL algorithm**:
	- Original data: $\mathbf y(t) = \begin{pmatrix} \mathbf x(t) \\ \mathbf x(t-1) \end{pmatrix}$
	- Randomly permuted data: $\mathbf y^*(t) = \begin{pmatrix} \mathbf x(t) \\ \mathbf x(t^*) \end{pmatrix}$
	- Learn to discriminate them.
- Possible to conduct dimension reduction.
![[Pasted image 20240613135108.png]]
#### Combining Temporal Dependencies and Non-Stationarity
$$\begin{align*}
s_i(t) &=r(s_i(t-1))+\sigma_i(t)n_i(t) \\
\mathbf x(t) &=\mathbf f(\mathbf x(t-1),\mathbf s(t))
\end{align*}$$
#### Auxiliary Variables
$$\begin{align*}
p(\mathbf s|\mathbf u) &=\prod_i p_i(s_i|\mathbf u), \\
\mathbf{\tilde x} &= (\mathbf x,\mathbf u), \\
\mathbf{\tilde x}^* &= (\mathbf x,\mathbf u^*) \\
\end{align*}$$
Non-linear logistic regression is performed to discriminate two samples.
#### Instead of SSL...
- **Noise-free model**: The second term is computational demanding. Not allowing for dimension reductions![[Pasted image 20240613135657.png]]
- **Variational approximations**: Adding a noise term to make computations easier. Allowing for dimension reductions.
- **Energy-based modeling**: Compromising between statistical efficiency and computational efficiency.