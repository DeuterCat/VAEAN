# VAEAN
simple practice of fine-tuning in few-shot image classification task 



## notion

The auxiliary set $\mathcal{A}$ is denoted as $\{X_i^{(\mathcal{A})},y_{i}^{(\mathcal{A})}\}_{i=1}^{n_\mathcal{A}}$, where $n_\mathcal{A}$ is the size of auxiliary set, and $X_i^{(\mathcal{A})}$,$y_{i}^{(\mathcal{A})}$ is the $i^{\textrm{th}}$ image and label from the $i^{\textrm{th}}$ sample, with $X_i^{(\mathcal{A})} \in \mathbb{R}^{H\times W\times 3}$ and the one-hot label $y_{i}^{(\mathcal{A})} \in \{0,1\}^{N_{\mathcal{A}}}$, where $N_{\mathcal{A}}$ is the number of categories of image in auxiliary set. And the support set can be denoted similarly as $S=\{X_i^{(\mathcal{S})},y_{i}^{(\mathcal{S})}\}_{i=1}^{n_\mathcal{S}}$. Let $f_\theta$ denote the embedding model with parameter $\theta$ and $C_\omega$ denote the classification model with parameter $\omega$, and $L$ is the loss function.

### **Baseline method of fine-tuning:**

**Pre-training stage**
$$
  \theta^{\prime},\omega^{\prime}=  \mathop{\arg\min}\limits_{\theta,\omega} 
   \sum_{i=1}^{n_\mathcal{A}}
   L(C_\omega(f_\theta(X_i^{(\mathcal{A})})),y_{i}^{(\mathcal{A})})

$$
**Fine-tuning stage**
$$
\omega^{*} = \mathop{\arg\min}\limits_{\omega} \sum_{i=1}^{n_{\mathcal{S}}} 
    L(C_\omega(f_{\theta^{\prime}}(X_i^{(\mathcal{S})})),y_{i}^{(\mathcal{S})})
$$

### Variational AutoEncoder Augmented Neural Network:

**Pre-training stage**
$$
  L_i =  L_1(C_\omega(f_\theta(X_i^{(\mathcal{A})})),y_{i}^{(\mathcal{A})})+L_2(g_\varphi(f_\theta(X_i^{(\mathcal{A})})),X_i^{(\mathcal{A})})
$$

$$
\theta^{\prime},\omega^{\prime}, \varphi^{\prime}=  \mathop{\arg\min}\limits_{\theta,\omega} 
   \sum_{i=1}^{n_\mathcal{A}}
   L_i
$$



**Fine-tuning stage**
$$
 \mathcal{S}^\prime = \{
    \mathcal{S}
    ,
    g_{\varphi^\prime}(f_{\theta^\prime}(\mathcal{S}))
    \}
$$

$$
\omega^{*} = \mathop{\arg\min}\limits_{\omega} \sum_{i=1}^{n_{\mathcal{S}^{\prime}}} 
    L_1(C_\omega(f_{\theta^{\prime}}(X_i^{(\mathcal{S}^{\prime})})),y_{i}^{(\mathcal{S}^{\prime})})
$$

In VAEAN, We set up a Reparadecoder (a decoder with reparameterization trick, the mean and variation of the hidden variable is calculated and used to form the hidden variable with noise) denoted as $g_\varphi$ with parameter $\varphi$ . $L_i$ stands for the $i^{\textrm{th}}$ total loss of VAEAN, $L_1$ is the loss of classification, and $L_2$ is the adapted loss of Variational AutoEncoder. ${\mathcal{S}}^\prime$ is the augmented support set containing images produced by decoder, and $n_{\mathcal{S}^{\prime}}$ is the size of $\mathcal{S}^{\prime}$. In VAEAN, the embedding model also plays the role as the encoder, the output of the embedding model is passed to both the Reparadecoder and the classifier. The decoder is trained in pre-training stage, and used to create augmented images in fine-tuning stage.

