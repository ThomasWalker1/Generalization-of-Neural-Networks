 ## Introduction to PAC-Bayes Theory

 ###  Bayesian Machine Learning

Here we will outline an introduction to Bayesian learning given by (Guedj, 2019). This will provide some context to the framework under which PAC-Bayes bounds are derived. As before we suppose that our training data $S_m=\{(x_i,y_i)\}_{i=1}^m$ consists of samples from the distribution $\mathcal{D}$ defined on $\mathcal{Z}$. Bayesian machine learning is used to find a parameter $\hat{\mathbf{w}}$ that corresponds to a hypothesis $h_{\hat{\mathbf{w}}}$ with the property that $h_{\hat{\mathbf{w}}}(x)\approx y$. To do this a learning algorithm is employed, which is simply a map from the data space to the parameter space, $\mathcal{W}$. In this framework, our loss function takes the form $l:\mathcal{X}\times\mathcal{Y}\to\mathbb{R}_+$.
The learning algorithm requires some prior distribution, $\pi$, to be defined on $\mathcal{W}$. Then using the training data the posterior distribution, $\rho$, is formed from the prior distribution. From the posterior distribution, there are many methodologies to then determine the parameter $\hat{\mathbf{w}}$. For example, one could take $\hat{\mathbf{w}}$ to be the mean, median or a random realisation of $\rho$.

 ###  Introducing PAC-Bayes Bounds\label{Subsubsection-Introducing PAC-Bayes Bounds}

Bayesian machine learning is a way to manage randomness and uncertainty in the learning task. PAC-Bayes are PAC bounds that operate under this framework.

**Definition** *Let $\mathcal{M}(\mathcal{W})$ be a set of probability distributions defined over $\mathcal{W}$. A data-dependent probability measure is a function $$\hat{\rho}:\bigcup_{n=1}^{\infty}(\mathcal{X}\times\mathcal{Y})^n\to\mathcal{M}(\mathcal{W}).$$*
 
For ease of notation we will simple write $\hat{\rho}$ to mean $\hat{\rho}((X_1,Y_1),\dots,(X_n,Y_n))$. The Kullback-Liebler (KL) divergence is a measure of similarity between probability measures defined on the same measurable space.
**Definition** *Given two probability measures $Q$ and $P$ defined on some sample space $\mathcal{X}$, the KL divergence between $Q$ and $P$ is $$\mathrm{KL}(Q,P)=\int\log\left(\frac{dQ(x)}{dP(x)}\right)Q(dx)$$
when $Q$ is absolutely continuous with respect to $P$. Otherwise, $\mathrm{KL}(Q,P)=\infty$.*
 
**Remark** *When $Q, P$ are probability measures on Euclidean space $\mathbb{R}^d$ with densities $q,p$ respectively. The KL divergence is $$\mathrm{KL}(Q, P):=\int\log\left(\frac{q(x)}{p(x)}\right)q(x)dx.$$
Note that KL divergence can take values in the range $[0,\infty]$. Also, note the asymmetry in the definition.*
 
For the multivariate normal distributions $N_{q}\sim\mathcal{N}(\mu_{q},\Sigma_{q})$ and $N_{p}\sim\mathcal{N}(\mu_{p},\Sigma_{p})$ defined on $\mathbb{R}^d$ we have that,
$$\mathrm{KL}(N_q, N_p)=\frac{1}{2}\left(\mathrm{tr}\left(\Sigma_p^{-1}\Sigma_q\right)-d+(\mu_p-\mu_q)^\top\Sigma_p^{-1}(\mu_p-\mu_q)+\log\left(\frac{\det\Sigma_p}{\det\Sigma_q}\right)\right).$$
Similarly, for Bernoulli distributions $\mathcal{B}(q)\sim\mathrm{Bern}(q)$ and $\mathcal{B}(p)\sim\mathrm{Bern}(p)$ it follows that
$$\mathrm{kl}(q, p):=\mathrm{KL}(\mathcal{B}(q),\mathcal{B}(p))=q\log\left(\frac{q}{p}\right)+(1-q)\log\left(\frac{1-q}{1-p}\right),$$
For $p^*\in[0,1]$ bounds of the form $\mathrm{KL}(q, p^*)\leq c$ for some $q\in[0,1]$ and $c\geq0$ are of interest. Hence, we introduce the notation
$$\mathrm{kl}^{-1}(q, c):=\sup\{p\in[0,1]:\mathrm{kl}(q, p)\leq c\}.$$

 **Proposition** *For any probability measures $Q$ and $P$ it follows that $\mathrm{KL}(Q,P)\geq0$ with equality if and only if $Q$ and $P$ are the same probability distribution.*
 

 ###  PAC-Bayes Bounds

The first PAC-Bayes bounds we will encounter is known as Catoni's bound. Recall, that under the Bayesian framework, we first fix a prior distribution, $\pi\in\mathcal{M}(\mathcal{W})$.

**Theorem** (Alquier, 2023)\label{Theorem-Catoni Bound} *For all $\lambda>0$, for all $\rho\in\mathcal{M}(\mathcal{W})$, and $\delta\in(0,1)$ it follows that $$\mathbb{P}_{S\sim\mathcal{D}^m}\left(\mathbb{E}_{\mathbf{w}\sim\rho}\left(\hat{R}(\mathbf{w})\right)\leq\frac{\lambda C^2}{8m}+\frac{\mathrm{KL}(\rho,\pi)+\log\left(\frac{1}{\delta}\right)}{\lambda}\right)\geq1-\delta.$$*
*Proof.*  $\square$

Theorem \ref{Theorem-Catoni Bound} motivates the study of the data-dependent probability measure
\label{Equation-Minimizer of Catoni Bound}$$\begin{equation}\hat{\rho}_{\lambda}=\mathrm{argmin}_{\rho\in\mathcal{M}(\mathcal{W})}\left(\mathbb{E}_{\mathbf{w}\sim\rho}\left(\hat{R}(\mathbf{w})\right)+\frac{\mathrm{KL}(\rho,\pi)}{\lambda}\right).\end{equation}$$

**Definition**  *The optimization problem defined by Equation \ref{Equation-Minimizer of Catoni Bound} has the solution $\hat{\rho}_{\lambda}=\pi_{-\lambda\hat{R}}$ given by $$\hat{\rho}_{\lambda}(d\mathbf{w})=\frac{\exp\left(-\lambda\hat{R}(\mathbf{w})\right)\pi(d\mathbf{w})}{\mathbb{E}_{\mathbf{w}\sim\pi}\left(\exp\left(-\lambda\hat{R}(\mathbf{w})\right)\right)}.$$
This is distribution is known as the Gibbs posterior.*
 
**Corollary** *For all $\lambda>0$, and $\delta\in(0,1)$ it follows that $$\mathbb{P}_{S\sim\mathcal{D}^m}\left(\mathbb{E}_{\mathbf{w}\sim\hat{\rho}_{\lambda}}\left(R(\mathbf{w})\right)\leq\inf_{\rho\in\mathcal{M}(\mathcal{W})}\left(\mathbb{E}_{\mathbf{w}\sim\rho}\left(\hat{R}(\mathbf{w})\right)+\frac{\lambda C^2}{8m}+\frac{\mathrm{KL}(\rho,\phi)+\log\left(\frac{1}{\delta}\right)}{\lambda}\right)\right)\geq1-\delta.$$*

For a learning algorithm, we noted that there are different methodologies for how the learned classifier is sampled from the posterior. In the case where consider a single random realisation of the posterior distribution, we have the following result.

**Theorem** (Alquier, 2023) *For all $\lambda>0$, $\delta\in(0,1)$, and data-dependent probability measure $\tilde{\rho}$ we have that $$\mathbb{P}_{S\sim\mathcal{D}^m}\mathbb{P}_{\tilde{\mathbf{w}}\sim\tilde{\rho}}\left(R\left(\tilde{\mathbf{w}}\right)\leq\hat{R}\left(\tilde{\mathbf{w}}\right)+\frac{\lambda C^2}{8m}+\frac{\log\left(\frac{d\rho\left(\tilde{\mathbf{w}}\right)}{d\pi\left(\tilde{\mathbf{w}}\right)}\right)+\log\left(\frac{1}{\delta}\right)}{\lambda}\right)\geq1-\delta$$*
*Proof.*  $\square$

Note that Theorem \ref{Theorem-Catoni Bound} is a bound in probability. We now state an equivalent bound that holds in expectation.

**Theorem** (Alquier, 2023) *For all $\lambda>0$, and data-dependent probability measure $\tilde{\rho}$, we have that $$\mathbb{E}_{S\sim\mathcal{D}^m}\mathbb{E}_{\mathbf{w}\sim\tilde{\rho}}(R(\mathbf{w}))\leq\mathbb{E}_{S\sim\mathcal{D}^m}\mathbb{E}_{\mathbf{w}\sim\tilde{\rho}}\left(\hat{R}(\mathbf{w})+\frac{\lambda C^2}{8m}+\frac{\mathrm{KL}(\tilde{\rho},\pi)}{\lambda}\right).$$*
 
**Corollary** *For $\tilde{\rho}=\hat{\rho}_{\lambda}$, the following holds $$\mathbb{E}_{S\sim\mathcal{D}^m}\mathbb{E}_{\mathbf{w}\sim\hat{\rho}_{\lambda}}(R(\mathbf{w}))\leq\mathbb{E}_{S\sim\mathcal{D}^m}\left(\inf_{\rho\in\mathcal{M}(\mathcal{W})}\mathbb{E}_{\mathbf{w}\sim\rho}\left(\hat{R}(\mathbf{w})\right)+\frac{\lambda C^2}{8m}+\frac{\mathrm{KL}(\rho,\pi)}{\lambda}\right).$$*
 

In the results that follow we will consider the $0$-$1$ loss. This is a measurable function $l:\mathcal{Y}\times\mathcal{Y}\to\{0,1\}$ defined by $l(y,y^\prime)=\mathbf{1}(y\neq y^\prime)$.

**Theorem** (McAllester, 1998) *For all $\rho\in\mathcal{M}(\mathcal{W})$ and $\delta>0$ we have that $$\mathbb{P}_{S\sim\mathcal{D}^m}\left(\mathbb{E}_{\mathbf{w}\sim\rho}(R(\mathbf{w}))\leq\mathbb{E}_{\mathbf{w}\sim\rho}\left(\hat{R}(\mathbf{w})\right)+\sqrt{\frac{\mathrm{KL}(\rho,\pi)\log\left(\frac{1}{\delta}\right)+\frac{5}{2}\log(m)+8}{2m-1}}\right)\geq1-\delta.$$*
 
**Theorem** (Catoni, 2007)\label{Theorem-Catoni Bound 2} *For $a>0$ and $p\in(0,1)$ let $$\Phi_{a}(p)=\frac{-\log\left(1-p(1-\exp(-a)\right)}{a}.$$
Then for any $\lambda>0$, $\delta>0$ and $\rho\in\mathcal{M}(\mathcal{W})$ we have that $$\mathbb{P}_{S\sim\mathcal{D}^m}\left(\mathbb{E}_{\mathbf{w}\sim\rho}(R(\mathbf{w}))\leq\Phi^{-1}_{\frac{\lambda}{m}}\left(\mathbb{E}_{\mathbf{w}\sim\rho}\left(\hat{R}(\mathbf{w})\right)+\frac{\mathrm{KL}(\rho,\pi)+\log\left(\frac{1}{\delta}\right)}{\lambda}\right)\right)\geq1-\delta.$$*
 
**Theorem** (Maurer, 2004)\label{Theorem-Maurer Bound} *For any $\delta>0$ and $\rho\in\mathcal{M}(\mathcal{W})$ then we have that $$\mathbb{P}_{S\sim\mathcal{D}^m}\left(\mathbb{E}_{\mathbf{w}\sim\rho}(R(\mathbf{w}))\leq\mathrm{kl}^{-1}\left(\mathbb{E}_{\mathbf{w}\sim\rho}\left(\hat{R}(\mathbf{w})\right),\frac{\mathrm{KL}(\rho,\pi)+\log\left(\frac{2\sqrt{m}}{\delta}\right)}{m}\right)\right)\geq1-\delta.$$*
 
 ## Optimizing PAC-Bayes Bounds via SGD
In practice, it is often the case that these bounds are not useful. Despite providing insight into how generalization relates to each of the components of the learning process they do not have much utility in providing non-vacuous bounds on the performance of neural networks on the underlying distribution. The significance of the KL divergence between the posterior and the prior can be noted in each of the bounds of Section \ref{Subsubsection-Introducing PAC-Bayes Bounds}. This motivated the work of (Dziugaite, 2017) who successfully minimized this term to provide non-vacuous results in practice. They considered a restricted problem that lends itself to efficient optimization. They use stochastic gradient descent to refine the prior, which is effective as SGD is known to find flat minima. This is important as around flat minima such as $\mathbf{w}^*$ we have that $\hat{R}(\mathbf{w})\approx\hat{R}(\mathbf{w}^*)$ (Alquier, 2023). The setup considered by (Dziugaite, 2017) is the same as the one we have considered throughout this report. With $\mathcal{X}\subset\mathbb{R}^k$ and labels being $\pm 1$. That is, we are considering binary classification based on a set of features. We explicitly state our hypothesis set as
$$\mathcal{H}=\left\{h_{\mathbf{w}}:\mathbb{R}^k\to\mathbb{R}:\mathbf{w}\in\mathbb{R}^d\right\}.$$
We are still considering the $0$-$1$, however, because our classifiers output real numbers we modify the loss slightly to account for this. That is, we let $l:\mathbb{R}\to\{\pm1\}$ be defined as $l(y,y^\prime)=\mathbf{1}(\mathrm{sgn}(y^\prime)=y)$.
For optimization purposes we use the convex surrogate loss function $\tilde{l}:\mathbb{R}\times\{\pm1\}\to\mathbb{R}_+$
$$\tilde{l}(y,\hat{y})=\frac{\log\left(1+\exp\left(-\hat{y}y\right)\right)}{\log(2)}.$$ 
For the empirical risk under the convex surrogate loss we write
$$\tilde{R}(\mathbf{w})=\frac{1}{m}\sum_{i=1}^m\tilde{l}(h_{\mathbf{w}}(x_i),y_i).$$
Recall, that this definition implicitly depends on the training sample $S_m$. We now introduce some new notations to simplify the upcoming bounds. For a randomized classifier $\rho$ on $\mathcal{W}$ we define the expected risk and the expected empirical risk as
$$R(\rho)=\mathbb{E}_{\mathbf{w}\sim\rho}(R(\mathbf{w})),\text{ and }\hat{R}(\rho)=\mathbb{E}_{\mathbf{w}\sim\rho}\left(\hat{R}(\mathbf{w})\right)$$
respectively. We will use these definitions throughout the remaining sections of this report as well. As noted previously the work (Dziugaite, 2017) looks to minimize the KL divergence between the prior and the posterior to achieve non-vacuous bounds. To do this they work under a restricted setting and construct a process to find the posterior $\rho$ that minimizes the divergence. To being (Dziugaite, 2017) utilize the following bound.

**Theorem** (Dziugaite, 2017) \label{Theorem-Bound on KL Divergence on Errors} *For every $\delta>0$,$m\in\mathbb{N}$, distribution $\mathcal{D}$ on $\mathbb{R}^k\times\{\pm 1\}$, distribution $\pi$ on $\mathcal{W}$ and distribution $\rho\in\mathcal{M}(\mathcal{W})$, we have that $$\mathbb{P}_{S\sim\mathcal{D}^m}\left(\mathrm{kl}\left(\hat{R}(\rho), R(\rho)\right)\leq\frac{\mathrm{KL}(\rho,\pi)+\log\left(\frac{m}{\delta}\right)}{m-1}\right)\geq1-\delta.$$*
 
**Remark**  *Note how this is a slightly weaker statement than Theorem \ref{Theorem-Maurer Bound}. This is because (Dziugaite, 2017) cited this Theorem from (Seeger, 2001), however, since then (Maurer, 2004) was able to tighten the result by providing Theorem \ref{Theorem-Maurer Bound}. In the following we will update the work of (Dziugaite, 2020) and use the tightened result provided by \ref{Theorem-Maurer Bound}.*
 
This motivates the following PAC-Bayes learning algorithm.
1. Fix a $\delta>0$ and a distribution $\pi$ on $\mathcal{W}$,
2. Collect an $\mathrm{i.i.d}$ sample $S_m$ of size $m$,
3. Compute the optimal distribution $\rho$ on $\mathcal{W}$ that minimizes \label{Equation-Optimization Equation for PAC-Bayes via SGD}
    $$\begin{equation}\mathrm{kl}^{-1}\left(\hat{R}(\rho),\frac{\mathrm{KL}(\rho,\pi)+\log\left(\frac{2\sqrt{m}}{\delta}\right)}{m}\right),\end{equation}$$
4. Then return the randomized classifier given by $\rho$.

Implementing such an algorithm in this general form is intractable in practice. Recall, that we are considering neural networks and so $\mathbf{w}$ represents the weights and biased of our neural network. To make the algorithm more practical we therefore consider
$$\mathcal{M}(\mathcal{W})=\left\{\mathcal{N}_{\mathbf{w},\mathbf{s}}=\mathcal{N}(\mathbf{w},\mathrm{diag}(\mathbf{s})):\mathbf{w}\in\mathbb{R}^d,\mathbf{s}\in\mathbb{R}_+^d\right\}.$$
Utilizing the bound $\mathrm{kl}^{-1}(q,c)\leq q+\sqrt{\frac{c}{2}}$ in Equation (\ref{Equation-Optimization Equation for PAC-Bayes via SGD}) and replacing the loss with the convex surrogate loss we obtain the updated optimization problem
\label{Equation-Updated Optimization Equation for PAC-Bayes via SGD}
$$\begin{equation}\min_{\mathbf{w}\in\mathbb{R}^d,\mathbf{s}\in\mathbb{R}^d_+}\tilde{R}\left(\mathcal{N}_{\mathbf{w},\mathbf{s}}\right)+\sqrt{\frac{\mathrm{KL}(\mathcal{N}_{\mathbf{w},\mathbf{s}},\pi)+\log\left(\frac{2\sqrt{m}}{\delta}\right)}{2m}}.\end{equation}$$
We now suppose our prior $\pi$ is of the form $\mathcal{N}(\mathbf{w}_0,\lambda I)$. As we will see the choice of $\mathbf{w}_0$ is not too impactful, as long as it is not $\mathbf{0}$. However, to efficiently choose a judicious value for $\lambda$ we discretize the problem, with the side-effect of expanding the eventual generalization bound. We let $\lambda$ have the for $c\exp\left(-\frac{j}{b}\right)$ for $j\in\mathbb{N}$, so that $c$ is an upper bound and $b$ controls precision. By ensuring that Theorem \ref{Theorem-Maurer Bound} holds with probability $1-\frac{6\delta}{\pi^2j^2}$ for each $j\in\mathbb{N}$ then we can apply a union bound argument to ensure that we get results that hold for probability $1-\delta$. A union bound argument refers to applying Theorem \ref{Theorem-Union Bound}. 

**Theorem** (Dziugaite, 2017)\label{Theorem-Union Bound} *Let $E_1,E_2,\dots$ be events. Then $\mathbb{P}\left(\bigcup_nE_n\right)\leq\sum_n\mathbb{P}(E_n).$*
 
Treating $\lambda$ as continuous during the optimization process and then discretizing at the point of evaluating the bound yields the updated optimization problem 
\label{Equation-Updated Optimization PAC-Bayes via SGD with Prior}
$$\begin{equation}\min_{\mathbf{w}\in\mathbb{R}^d,\mathbf{s}\in\mathbb{R}^d_+,\lambda\in(0,c)}\tilde{R}(\mathcal{N}_{\mathbf{w},\mathbf{s}})+\sqrt{\frac{1}{2}B_{\mathrm{RE}}(\mathbf{w},\mathbf{s},\lambda;\delta)}
\end{equation}$$

where

$$B_{\mathrm{RE}}(\mathbf{w},\mathbf{s},\lambda;\delta)=\frac{\mathrm{KL}(\mathcal{N}_{w,s},\mathcal{N}(\mathbf{w}_0,\lambda I))+2\log\left(b\log\left(\frac{c}{\lambda}\right)\right)+\log\left(\frac{\pi^2\sqrt{m}}{3\delta}\right)}{m}.$$

To optimize Equation (\ref{Equation-Updated Optimization PAC-Bayes via SGD with Prior}) we would like to compute its gradient and apply SGD. However, this is not feasible in practice for $\tilde{R}(\mathcal{N}_{\mathbf{w},\mathbf{s}})$. Instead we compute the gradient of $\tilde{R}\left(\mathbf{w}+\xi\odot\sqrt{\mathbf{s}}\right)$ where $\xi\sim\mathcal{N}_{0,\mathbf{1}_d}$.  Once good candidates for this optimization problem are found we return to (\ref{Equation-Optimization Equation for PAC-Bayes via SGD}) to calculate the final error bound. With the choice of $\lambda$ it follows that with probability $1-\delta$, uniformly over all $\mathbf{w}\in\mathbb{R}^d,\mathbf{s}\in\mathbb{R}^d_+$ and $\lambda$ (of the discrete form) the expected risk of $\rho=\mathcal{N}_{\mathbf{w},\mathbf{s}}$ is bounded by

$$\mathrm{kl}^{-1}\left(\hat{R}(\rho), B_{\mathrm{RE}}(\mathbf{w},\mathbf{s},\lambda;\delta)\right).$$

However, it is often not possible to compute $\hat{R}(\rho)$ due to the intractability of $\rho$. So instead an unbiased estimate is obtained by estimating $\rho$ using a Monte Carlo approximation. Given $n$ $\mathrm{i.i.d}$ samples $\mathbf{w}_1,\dots,\mathbf{w}_n$ from $\rho$ we use the Monte Carlo approximation $\hat{\rho}_n=\sum_{i=1}^n\delta_{\mathbf{w}_i}$, to get the bound
$$\hat{R}(\rho)\leq\overline{\hat{R}_{n,\delta^\prime}}(\rho):=\mathrm{kl}^{-1}\left(\hat{R}\left(\hat{\rho}_n\right),\frac{1}{n}\log\left(\frac{2}{\delta^\prime}\right)\right),$$
which holds with probability $1-\delta^\prime$. Finally, by Theorem \ref{Theorem-Union Bound}
$$R(\rho)\leq\mathrm{kl}^{-1}\left(\overline{\hat{R}_{n,\delta^\prime}}(\rho), B_{\mathrm{RE}}(\mathbf{w},\mathbf{s},\lambda;\delta)\right),$$
holds with probability $1-\delta-\delta^\prime.$ Now all that is left is to do is to determine optimal values for $\mathbf{w}$ and $\mathbf{s}$. To do this first train a neural network via SGD to get a value of $\mathbf{w}$. Then instantiate a stochastic neural network with the multivariate normal distribution $\rho=\mathcal{N}_{\mathbf{w},\mathbf{s}}$ over the weights, with $\mathbf{s}=\vert\mathbf{w}\vert$. Next apply Algorithm 4 to deduce values of $\mathbf{w},\mathbf{s}$ and $\lambda$ that give a tighter bound.

<font size="3"> **Algorithm 4** Optimizing the PAC Bounds</font>
> **Require:**\
$\mathbf{w}_0\in\mathbb{R}^d$, the network parameters at initialization.\
$\mathbf{w}\in\mathbb{R}^d$, the network parameters after SGD.\
$S_m$, training examples.\
$\delta\in(0,1)$, confidence parameter.\
$b\in\mathbb{N},c\in(0,1)$, precision and bound for $\lambda$.\
$\tau\in(0,1), T$, learning rate.\
**Ensure:** Optimal $\mathbf{w},\mathbf{s},\lambda$.\
$\zeta=\vert\mathbf{w}\vert$\Comment{$\mathbf{s}(\zeta)=e^{2\zeta}$}\
$\rho=-3$\Comment{$\lambda(\rho)=e^{2\rho}$}\
$B(\mathbf{w},\mathbf{s},\lambda,\mathbf{w}^\prime)=\tilde{R}(\mathbf{w})+\sqrt{\frac{1}{2}B_{\mathrm{RE}}(\mathbf{w},\mathbf{s},\lambda)}$\
**for** $t=1\to T$ **do**\
----Sample $\xi\sim\mathcal{N}(0,I_d)$\
----$\mathbf{w}^\prime(\mathbf{w},\zeta)=\mathbf{w}+\xi\odot\sqrt{\mathbf{s}(\zeta)}$\
----$\begin{pmatrix}\mathbf{w}\\\zeta\\\rho\end{pmatrix}=-\tau\begin{pmatrix}\nabla_{\mathbf{w}}B(\mathbf{w},\mathbf{s}(\zeta),\lambda(\rho),\mathbf{w}^\prime(\mathbf{w},\zeta))\\\nabla_\zeta B(\mathbf{w},\mathbf{s}(\zeta),\lambda(\rho),\mathbf{w}^\prime(\mathbf{w},\zeta))\\\nabla_\rho B(\mathbf{w},\mathbf{s}(\zeta),\lambda(\rho),\mathbf{w}^\prime(\mathbf{w},\zeta))\end{pmatrix}$\
**end for**\
**return** $\mathbf{w},\mathbf{s}(\zeta),\lambda(\rho)$

Once the values of $\mathbf{w},\mathbf{s}$ and $\lambda$ are found we then need to compute $\overline{\hat{R}_{n,\delta^\prime}}(\rho):=\mathrm{kl}^{-1}\left(\hat{R}\left(\hat{\rho}_n\right),\frac{1}{n}\log\left(\frac{2}{\delta^\prime}\right)\right)$ to get our bound. We note that
$$\hat{R}(\hat{\rho}_n)=\sum_{i=1}^n\delta_{\mathbf{w}_i}\left(\frac{1}{m}\sum_{j=1}^ml(h_{\mathbf{w}_i}(x_j),y_j)\right).$$
Then to invert the kl divergence we employ Newton's method, in the form of Algorithm 5, to get an approximation for our bound.

<font size="3"> **Algorithm 5** Newton's Method for Inverting kl Divergence</font>
> **Require:** $q,c$, initial estimate $p_0$ and $N\in\mathbb{N}$\
**Ensure:** $p$ such that $p\approx\mathrm{kl}^{-1}(q,c)$\
**for** $n=1\to N$ **do**\
----**if** $p\geq1$ **then**\
--------**return** $1$\
----**else**\
--------$p_0=p_0-\frac{q\log\left(\frac{q}{c}\right)+(1-q)\log\left(\frac{1-q}{1-c}\right)-c}{\frac{1-q}{1-p}-\frac{q}{p}}$\
----**end if**\
**end for**\
**return** $p_0$