This is my notes based on these tutorials:
https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
https://cvpr2022-tutorial-diffusion-models.github.io/
https://www.youtube.com/watch?v=cS6JQpEY9cs
https://arxiv.org/abs/2201.09865
## Forward Diffusion Process
Totally T steps of adding Gaussian noises into the real data.
$$\begin{align}
q(\mathbf{x}_t\mid\mathbf{x}_{t-1})=\mathcal{N}(\mathbf{x}_t;\sqrt{1-\beta_t}\mathbf{x}_{t-1},\beta_t\mathbf{I}) \\
q(\mathbf{x}_{1:T}\mid\mathbf{x}_0)=\prod_{t=1}^Tq(\mathbf{x}_t\mid\mathbf{x}_{t-1})
\end{align}$$
When $T\rightarrow\infty$, $\mathbf{x}_T$ is equivalent to a Gaussian distribution.
**Nice property**: sampling $\mathbf{x}_t$ is easy. Let $\alpha_t=1-\beta_t$ and $\bar{\alpha}_t=\prod_{i=1}^t\alpha_i$, we have that$$\begin{align}
\mathbf{x}_t=\sqrt{\bar\alpha_t}\mathbf{x_0}+\sqrt{1-\bar\alpha_t}{\boldsymbol\epsilon} \\
q(\mathbf{x}_t\mid\mathbf{x}_0)=\mathcal{N}(\mathbf{x}_t;\sqrt{\bar{\alpha}_t}\mathbf{x}_0,(1-\bar{\alpha}_t)\mathbf{I})
\end{align}$$ and usually $\beta_1<\beta_2<\ldots<\beta_T$.

## Reverse Diffusion Process
We want to reverse the forward process and sample from $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$. When $\beta_t$ is small enough, this will be approximately Gaussian distribution.
It's hard to estimate, so we need to **learn a model** $p_\theta$ to approximate following conditional probabilities. $$\begin{align}
p_\theta(\mathbf{x}_{0:T})=p(\mathbf{x}_T)\prod_{t=1}^Tp_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t) \\ 
p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)=\mathcal{N}(\mathbf{x}_{t-1};\boldsymbol\mu_\theta(\mathbf{x}_t,t),\mathbf{\Sigma}_\theta(\mathbf{x}_t,t))
\end{align}$$
Meanwhile, although $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$ is intractable, it's tractable when conditioned on $\mathbf{x}_0$:
$$q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)=\mathcal{N}(\mathbf{x}_{t-1};\boldsymbol{\tilde\mu_t}(\mathbf{x}_t,\mathbf{x}_0),\tilde\beta_t\mathbf{I})$$
where $$\begin{align}
\mathbf{\tilde\mu_t}(\mathbf{x}_t,\mathbf{x}_0)&=\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\mathbf{x}_t+\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}\mathbf{x}_0 \\
&= \frac{1}{\sqrt{\alpha_t}}(\mathbf{x}_t-\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\boldsymbol\epsilon_t)\\
\\
& \tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\cdot\beta_t &
\end{align}$$ (calculated with Bayes' rule, $q(\mathbf{x}_t\mid\mathbf{x}_0)$, and $q(\mathbf{x}_t\mid\mathbf{x}_{t-1})$)

## Variational Lower Bound
This can be proved by the definition of KL divergence that:
$$L_{VLB}=\mathbb{E}_{q(\mathbf{x}_{0:T})}[\log\frac{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})}]\geq\mathbb{E}_{q(\mathbf{x_0})}\log p_\theta(\mathbf{x}_0)=NLL$$
We can rewrite $L_{VLB}$ in this form to analytically calculate the bound.
$$\begin{align*} L_{\text{VLB}} &= L_T + L_{T-1} + \cdots + L_0 \\ \text{where } L_T &= D_{\text{KL}}(q(\mathbf{x}_T \mid \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_T)) \\ L_t &= D_{\text{KL}}(q(\mathbf{x}_t \mid \mathbf{x}_{t+1}, \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_t \mid \mathbf{x}_{t+1})) \quad \text{for } 1 \leq t \leq T - 1 \\ L_0 &= -\log p_\theta (\mathbf{x}_0 \mid \mathbf{x}_1) \end{align*}$$
$L_T$ is constant here, because no learnable parameters are here. $L_0$ is using a discrete decoder and has a special form. $L_t (1 \leq t \leq T - 1)$ is easier to calculate.
Here we use a reparameterization trick to simplify $L_t$. It's the KL-divergence between two Gaussian distributions, so here the authors assumed that they want to minimize the difference of their means. After we simplify the goal, and do the **following reparameterization**, the goal is clearer:
$$\boldsymbol\mu_\theta(\mathbf{x}_t,t)=\frac{1}{\sqrt{\alpha_t}}(\mathbf{x}_t-\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\boldsymbol\epsilon_\theta(\mathbf{x}_t,t))$$
Plugging in, we get that:
$$L_t=\mathbb{E}_{\mathbf{x}_0, \boldsymbol\epsilon} \left[ \frac{(1 - \alpha_t)^2}{2 \alpha_t (1 - \bar\alpha_t)\|\boldsymbol\Sigma_\theta\|_2^2} \left\| \boldsymbol\epsilon_t - \boldsymbol\epsilon_\theta \left(\sqrt{\bar\alpha_t} x_0 + \sqrt{1 - \bar\alpha_t} \boldsymbol\epsilon_t, t\right) \right\|^2 \right]$$
We can notice that $L_t$ is a weighted form of a square difference. To simply it, they removed these weights and aggregate these terms together. Considering those weights should also work, but has some computational issues.
The final objective used by the algorithm is :$$\begin{align*} L_t^{\text{simple}} &= \mathbb{E}_{t \sim [1,T],\mathbf x_0,\boldsymbol\epsilon_t} \left[ \left\| \boldsymbol\epsilon_t - \boldsymbol\epsilon_\theta(\mathbf x_t, t) \right\|^2 \right] \\ &= \mathbb{E}_{t \sim [1,T],\mathbf x_0,\boldsymbol\epsilon_t} \left[ \left\| \boldsymbol\epsilon_t - \boldsymbol\epsilon_\theta\left(\sqrt{\bar\alpha_t} \mathbf x_0 + \sqrt{1 - \bar\alpha_t} \boldsymbol\epsilon_t, t\right) \right\|^2 \right] \end{align*}$$
## Training and Sampling Algorithm (Simplified)
![[Pasted image 20240603134616.png]]
## Progressive Coding
A method of image encoding that uses multiple scans rather than a single scan. (A scan is a single pass through the data of one or more components in an image.) The first scan encodes a rough approximation to the image that is recognizable and can be transmitted quickly compared to the total transmission time of the image. Subsequent scans refine the appearance of the image. Progressive encoding is one of the options in the [JPEG](https://www.encyclopedia.com/science-and-technology/computers-and-electrical-engineering/computers-and-computing/jpeg#1O11JPEG) standard. (pasted from https://www.encyclopedia.com/computing/dictionaries-thesauruses-pictures-and-press-releases/progressive-encoding)
The algorithm pseudocode mentioned in the DDPM paper:
![[Pasted image 20240605122821.png]]
Following plot shows something... I don't quite understand but it should be helpful to understand latent diffusion models. 
At each time t, the **distortion** is calculated as the root mean squared error $\sqrt{\| \mathbf x_0 - \hat{\mathbf x}_0 \|^2 / D}$, and the **rate** is calculated as the cumulative number of bits received so far at time t.
![[Pasted image 20240605123422.png]]

## Interpolation
The idea is to mix two pictures in the latent space with a weight factor $\lambda$, and then use the reverse steps to recover the interpolation image.
$$\begin{align*}
\mathbf x_0, \mathbf x_0' &\sim q(\mathbf x_0) \\
\mathbf x_t, \mathbf x_t' &\sim q(\mathbf x_t\mid\mathbf x_0) \\
\bar{\mathbf x}_t &= (1-\lambda)\mathbf x_0 + \lambda \mathbf x_0' \\
\bar{\mathbf x}_0 &\sim p(\mathbf x_0 \mid \bar{\mathbf x}_t)
\end{align*}$$
![[Pasted image 20240605105848.png]]
From the image above, we can see some patterns:
- If we only use 0 steps, which implies we do not use any diffusion step, it is just a vanilla mixture of two pictures. 
- With more steps (about 500 steps), we can see some natural and smooth transitions semantically. 
- However, when we use all of the 1000 steps, the latent variable seems to be totally Gaussian, and the interpolation images seem to be random. 
- More steps allow for more deviation from the original two images.
##  Inpainting
Pasted from the paper https://arxiv.org/abs/2201.09865:
![[Pasted image 20240605130820.png]]
## Latent Diffusion Model
![[Pasted image 20240604104349.png]]
Adding a **discriminator** will improve the quality of the images.
![[Pasted image 20240604104633.png]]![[Pasted image 20240604104937.png]]