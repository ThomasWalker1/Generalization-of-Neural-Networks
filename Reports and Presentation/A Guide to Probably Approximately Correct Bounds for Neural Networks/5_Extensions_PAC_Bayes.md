## Disintegrated PAC-Bayes Bounds\label{Section-Disintegration}

The majority of the PAC-Bayes bounds we have discussed so far have been derived to hold for all posterior distributions. The intention of disintegrated PAC-Bayes bounds is to refine these results by only requiring them to hold for a single posterior distribution. We now study the work of (Viallard, 2021) that sets out a general framework for deriving such bounds. The setup is the same as the one we have considered so far, with the added assumption that $C=1$ and the additional consideration of a deterministic learning algorithm $A:\mathcal{Z}^m\to\mathcal{M}(\mathcal{W})$ that is applied to the training sample $S$.

**Definition** (Viallard, 2021) *The two distributions $P$ and $Q$ defined on the some sample space $\mathcal{X}$, then for any $\alpha>1$ their Renyi divergence is defined to be $$D_{\alpha}(Q,P)=\frac{1}{\alpha-1}\log\left(\mathbb{E}_{x\sim P}\left(\frac{Q(x)}{P(x)}\right)^{\alpha}\right).$$*
 
**Theorem** (Viallard, 2021) *For any distribution $\mathcal{D}$ on $\mathcal{Z}$, for any parameter space $\mathcal{W}$, for any prior distribution $\pi$ on $\mathcal{W}$, for any $\phi:\mathcal{W}\times\mathcal{Z}^m\to\mathbb{R}^+$, for any $\alpha>1$, for any $\delta>0$ and for any deterministic learning algorithm $A:\mathcal{Z}^m\to\mathcal{M}(\mathcal{W})$ the following holds $$\mathbb{P}_{\mathcal{S}\sim\mathcal{D}^m,\mathbf{w}\sim\rho_S}\left(\frac{\alpha}{\alpha-1}\log\left(\phi(\mathbf{w},S)\right)\leq\frac{2\alpha-1}{\alpha-1}\log\left(\frac{2}{\delta}\right)+D_{\alpha}(\rho_{S},\pi)+\log\left(\mathbb{E}_{S^\prime\sim\mathcal{D}^m}\mathbb{E}_{\mathbf{w}^\prime\sim\pi}\phi(\mathbf{w}^\prime,S^\prime)^{\frac{\alpha}{\alpha-1}}\right)\right)\geq1-\delta,$$ where $\mathcal{\rho}_S:=A(S)$.*
 
*Proof.*$\square$

### Application to Neural Network Classifiers

We can contextualize this bound to over-parameterized neural networks. Suppose that $\mathbf{w}\in\mathbb{R}^d$ is a weight vector of a neural network, with $d\gg m$. Assume that the network is trained for $T$ epochs and that these epochs are used to generate $T$ priors $\mathbf{P}=\{\pi_t\}_{t=1}^T$. Let the priors be of the form $\pi_t=\mathcal{N}\left(\mathbf{w}_t,\sigma^2\mathbf{I}_d\right)$ where $\mathbf{w}_t$ is the weight vector obtained after the $t^\text{th}$ epoch. We assume that the priors are obtained from the learning algorithm being applied to the sample $S_{\mathrm{prior}}$ where $S_{\mathrm{prior}}\cap S=\emptyset$.

**Corollary**\label{Corollary-Disintegration on Neural Networks 1} *For any distribution $\mathcal{D}$ on $\mathcal{Z}$, for any set $\mathcal{W}$, for any set $\mathbf{P}$ of $T$ priors on $\mathcal{W}$, for any learning algorithm $A:\mathcal{Z}^m\to\mathcal{M}(\mathcal{W})$, for any loss $l:\mathcal{W}\times\mathcal{Z}\to[0,1]$ and for any $\delta>0$ then for any $\pi_t\in\mathbf{P}$ we have that $$\mathbb{P}_{\mathcal{S}\sim\mathcal{D}^m,\mathbf{w}\sim\rho_{S}}\left(\mathrm{kl}\left(\hat{R}(\mathbf{w}),R(\mathbf{w})\right)\leq\frac{1}{m}\left(\frac{\Vert\mathbf{w}-\mathbf{w}_t\Vert_2^2}{\sigma^2}+\log\left(\frac{16T\sqrt{m}}{\delta^3}\right)\right)\right)\geq1-\delta.$$*
 
**Corollary** *Under the assumptions of Corollary \ref{Corollary-Disintegration on Neural Networks 1} with $\delta\in(0,1)$ and for all $\pi_t\in\mathbf{P}$ we have that $$\begin{align*}\mathbb{P}_{S\sim\mathcal{D}^m,\mathbf{w}\sim\rho_S}&\left(\mathrm{kl}\left(\hat{R}(\mathbf{w}),R(\mathbf{w})\right)\leq\frac{1}{m}\left(\frac{\Vert\mathbf{w}+\mathbf{\epsilon}-\mathbf{w}_t\Vert_2^2-\Vert\mathbf{\epsilon}\Vert_2^2}{2\sigma^2}+\log\left(\frac{2T\sqrt{m}}{\delta}\right)\right)\right),\\\mathbb{P}_{S\sim\mathcal{D}^m,\mathbf{w}\sim\rho_S}&\left(\mathrm{kl}\left(\hat{R}(\mathbf{w}),R(\mathbf{w})\right)\leq\frac{1}{m}\left(\frac{m+1}{m}\frac{\Vert\mathbf{w}+\mathbf{\epsilon}-\mathbf{w}_t\Vert_2^2-\Vert\mathbf{\epsilon}\Vert_2^2}{2\sigma^2}+\log\left(\frac{T(m+1)}{\delta}\right)\right)\right),\end{align*}$$ and for all $c\in\mathbf{C}$ $$R(\mathbf{w})\leq\frac{1-\exp\left(-c\hat{R}(\mathbf{w})-\frac{1}{m}\left(\frac{\Vert\mathbf{w}+\mathbf{\epsilon}-\mathbf{w}_t\Vert_2^2-\Vert\mathbf{\epsilon}\Vert_2^2}{2\sigma^2}+\log\left(\frac{T\vert\mathbf{C}\vert}{\delta}\right)\right)\right)}{1-\exp(-c)}.$$ Where $\mathbf{\epsilon}\sim\mathcal{N}\left(\mathbf{0},\sigma^2\mathbf{I}_d\right)$ is Gaussian noise such that $\mathbf{w}+\mathbf{\epsilon}$ acts as the weights sampled from $\mathcal{N}(\mathbf{w},\sigma^2\mathbf{I}_d)$, and $\mathbf{C}$ is a set of hyper-parameters fixed a priori.*
 

## PAC-Bayes Compression Bounds

We will now see how compression ideas can be capitalized to tighten PAC-Bayes bounds. The work of (Zhou, 2019) evaluates generalization bounds by first measuring the effective compressed size of a neural network and then substituting this into the bounds. We have seen that compression techniques can efficiently reduce the effective size of a network, and so accounting for this can lead to tighter bounds. This also captures the intuition that we expect a model to overfit if it is more difficult to compress. Therefore, these updated bounds also incorporate a notion of model complexity. The work of (Zhou, 2019) utilizes a refined version of Theorem \ref{Theorem-Catoni Bound 2}.

**Theorem** (Catoni, 2007)\label{Theorem-Catoni Bound 2 Refined} *Let $L$ be a $0$-$1$ valued loss function. Let $\pi$ be a probability measure on the parameter space, and let $\alpha>1,\delta>0$. Then, $$\mathbb{P}_{S\sim\mathcal{D}^m}\left(R(\rho)\leq\inf_{\lambda>1}\Phi^{-1}_{\lambda/m}\left(\hat{R}(\rho)+\frac{\alpha}{\lambda}\left(\mathrm{KL}(\rho,\pi)-\log(\delta)+2\log\left(\frac{\log\left(\alpha^2\lambda\right)}{\log(\alpha)}\right)\right)\right)\right)\geq1-\delta.$$*
 
*Proof.*$\square$

The intention now is to motivate the choice of $\pi$ using ideas of compressibility such that $\mathrm{KL}(\rho,\pi)$ is kept small. To do this we will choose a prior $\pi$ that assigns greater probability mass to models with a shorter code length.

**Theorem** (Zhou, 2019) \label{Theorem-KL Divergence Bound using Coding Schemes} *Let $\vert h\vert_c$ denote the number of bits required to represent hypothesis $h$ using some pre-specified coding $c$. Let $\rho$ denote the point mass distribution at the compressed model $\hat{h}$. Let $M$ denote any probability measure on the positive integers. Then there exists a prior $\pi_c$ such that $$\mathrm{KL}(\rho,\pi_c)\leq\left\vert\hat{h}\right\vert_c\log(2)-\log\left(M\left(\left\vert\hat{h}\right\vert_c\right)\right).$$*
 
**Remark** *An example of a coding scheme $c$ could be the Huffman encoding. However, such a compression scheme is agnostic to any structure of the hypothesis class $\mathcal{H}$. By exploiting structure in the hypothesis class the bound can be improved substantially.*
 
We now formalise compression schemes to allow us to refine Theorem \ref{Theorem-KL Divergence Bound using Coding Schemes}. Denote a compression procedure by a triple $(S,C,Q)$ where
- $S=\{s_1,\dots,s_k\}\subseteq\{1,\dots,d\}$ is the location of the non-zero weights,
- $C=\{c_1,\dots,c_r\}\subseteq\mathbb{R}$, is a codebook, and 
- $Q=(q_1,\dots,q_k)$ for $q_i\in\{1,\dots,r\}$ are the quantized values.

Define the corresponding weights $\mathbf{w}(S,Q,C)\in\mathbb{R}^d$ as,
$$w_i(S,Q,C)=\begin{cases}c_{q_j}&i=s_j\\0&\text{otherwise}.\end{cases}$$
Training a neural network is a stochastic process due to the randomness of SGD. So to analyse the generalization error we try to capture randomness in the analysis by applying Gaussian noise to weights. For this we use $\rho\sim\mathcal{N}\left(\mathbf{w},\sigma^2J\right)$, with $J$ being a diagonal matrix.

**Theorem** (Zhou, 2019)\label{Theorem-KL Divergence Bound Using Compression Prior} *Let $(S,C,Q)$ be the output of a compression scheme, and let $\rho_{S,C,Q}$ be the stochastic estimator given by the weights decoded from the triplet and variance $\sigma^2$. Let $c$ denote an arbitrary fixed coding scheme and let $M$ denote an arbitrary distribution on the positive integers. Then for any $\tau>0$, there is a prior $\pi$ such that $$\begin{align*}\mathrm{KL}(\rho_{S,C,Q},\pi)\leq&(k\lceil\log(r)\rceil+\vert S\vert_c+\vert C\vert_c)\log(2)-\log(M(k\lceil\log(r)\rceil+\vert S\vert_c+\vert C\vert_c))\\&+\sum_{i=1}^k\mathrm{KL}\left(\mathcal{N}\left(c_{q_i},\sigma^2\right),\sum_{j=1}^r\mathcal{N}\left(c_j,\tau^2\right)\right)\end{align*}$$*
 
Choosing the prior alluded to by Theorem \ref{Theorem-KL Divergence Bound Using Compression Prior} and utilizing Theorem \ref{Theorem-Catoni Bound 2 Refined} one can obtain a PAC-Bayes generalization bound that exploits notions of compressibility.