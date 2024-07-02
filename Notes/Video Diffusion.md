## Video-LDM
### Introduction
Generative models of images have been successful, while video modeling has lagged behind.
- Significant **computational** cost.
- Lack large-scale, general and publicly available **datasets**.
- Most work generates **short** and **low-resolution** videos.
**Video-LDM** can generate high-resolution and long videos with following steps:
- Pre-train on images, leveraging image datasets.
- Adding a temporal dimension into the latent space DM, with spatial layers fixed.
- Fine-tune the AE's decoder to achieve temporal consistency.
- Temporally align (L)DM upsamplers for video SR.
## Model Structure
### Overview
![[Pasted image 20240619103750.png]]
(**Top**: Autoencoder **Bottom**: LDM)
The original datapoints are videos with T RGB frames, with same heights and widths.
$$\mathbf x\in\mathbb{R}^{T\times3\times\tilde H\times\tilde W},\ \mathbf x\sim p_{data}$$
The autoencoder maps these datapoints into the latent space, where C is the number of latent channels. $$\mathcal E(\mathbf x)=\mathbf z\in\mathbb R^{T\times C\times H\times W}$$
### Temporal Autoencoder Finetuning
- The encoder is frozen.
- Introduce additional temporal layers for the decoder (patch-wise temporal discriminator).
### LDM Finetuning
![[Pasted image 20240619110430.png]]
- Only the spatial layers do not have **temporal awareness**.
- Additional temporal layers define the **video-aware** temporal backbone.
- How can we align the temporal information? Process videos also in a time dimension.
> [!NOTE] Reshaping
> $$\mathbf z' \leftarrow \text{rearrange}(\mathbf z,\text{(b t) c h w} \rightarrow \text{b c t h w})$$$$\mathbf z' \leftarrow l_\phi^i(\mathbf z',\mathbf c) \text{ \\\\This is the temporal layer.}$$  $$\mathbf z' \leftarrow \text{rearrange}(\mathbf z,\text{b t c h w} \rightarrow \text{(b t) c h w})$$
---
- Two kinds of temporal mixing layers: **temporal attention** and **Conv3D**.
- Positional encoding for time: **sinusoidal embedding**.
- **Merge parameter** $\alpha$: can be learned. By setting $\alpha^i_\phi=1$ for each layer, we can simply skip the temporal block. Easy to switch between pre-training (with images) and fine-tuning (with videos).
![[Pasted image 20240619112600.png]]
### Prediction, Interpolation and SR Models
![[Pasted image 20240619134252.png]]
We feed the mask and masked encoded video frames in the model for conditioning.
$$\mathbf c_S = (\mathbf m_S\circ\mathbf z,\mathbf m_S)$$
The new objective is:$$\mathbb{E}_{\mathbf x\sim p_{\text{data}},\mathbf m_S\sim p_S,\tau\sim p_\tau,\epsilon}[\|\mathbf y-\mathbf f_{\theta,\phi}(\mathbf z_r;\mathbf c_S,\mathbf c,\tau)\|_2^2]$$ where $p_S$ represents the mask sampling distribution, and $\mathbf c$ represents the text prompt information.
- **Prediction**: 
	- Give (first) $S$ frames, to predict the rest $T-S$ frames. 
	- First, it synthesize a single context frame, and generates a sequence. 
	- Then it conditions on two context frames to encode movement.
	- **Classifier-free diffusion guidance**: stabilize this process with a guidance scale $s\geq 1$.
- **Interpolation**:
	- Mask the frames to be interpolated.
	- $T\rightarrow 4T$ interpolation, $4T\rightarrow 16T$ interpolation.
- **SR Models**:
	- Scale up the Video LDM outputs by $4\times$ with upsampler (L)DM.
	- Conditioning on a noisy low-resolution image $\mathbf c_{\tau_\gamma}=\alpha_{\tau_\gamma}\mathbf x+\sigma_{\tau_\gamma}\epsilon$.
	- To make it video-aware, we condition on a low-resolution sequence of length $T$, and concatenate these images frame-by-frame.
### Experiment Environment
- **Datasets**: (i) RDS videos. (ii) WebVid-10M.
- **Evaluation Metrics**: FID, FVD, human evaluation.
- **Model Architecture**: Stable diffusion (U-Net).
- **Sampling**: DDIM.
## Stable Video Diffusion
### Introduction
- Previous work mainly focus on the arrangement of spatial and temporal layers.
- However, the effect of **data** and **training strategies** needs to be studied.
	- Data curation.
	- 3 training stages: image pretraining, video pretraining and video finetuning.
- We use this method to train SOTA text-to-video and image-to-video models.
- It can also provide strong priors for multi-view generators.
### Curating Data for HQ Video Synthesis
#### Data Processing and Annotation
- **Cut** detection pipeline: avoid cuts and fades leaking into synthesized videos.
- **Captioning**:
	- Image captioner: CoCa.
	- Video captioner: V-BLIP.
	- Summarization: LLM-based of first two captions.
- **Removing** degrading data:
	- Less motion: optical flow threshold.
	- Excessive text presence: OCR.
	- Low aesthetic value / text-image similarity: CLIP embedding.
#### Stage I: Image Pretraining
- Grounded on **Stable Diffusion**.
- Experiments show that image-pretrained model is preferred in both **quality** and **prompt-following.**
![[Pasted image 20240620101935.png]]
#### Stage II: Curating a Video Pretraining Dataset
- This paper introduced a **systematic approach** to video data curation.
- The idea is to set a threshold for each annotation type (CLIP scores, aesthetic scores, OCR detection rates, synthetic captions, optical flow scores).
- The best threshold combination is based on experiments with Elo ranking for human preference votes.
- **Curated training data improves performance.** LVD-10M-F is a better dataset in both spatiotemporal quality and prompt alignment compared to other datasets.
- **Data curation helps at scale.** This advantage also works on larger datasets. And when we use bigger dataset, we also get better performance.
![[Pasted image 20240620103033.png]]
#### Stage III: High-Quality Finetuning
- We finetune our model on a smaller finetuning dataset for some certain tasks.
- It is shown from the experiment that video pretraining is beneficial, and should occur on a large scale, curated dataset.
### Experiments: Training at Scale
- Pretrained Base Model
- HR Text-to-Video Model
- HR Image-to-Video Model
- Frame Interpolation
- Multi-View Generation
