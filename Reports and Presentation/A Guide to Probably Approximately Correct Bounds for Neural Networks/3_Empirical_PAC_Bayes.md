# Empirical PAC-Bayes Bounds

## 3.1 Introduction to PAC-Bayes Theory

### 3.1.1 Bayesian Machine Learning

Here we will outline an introduction to Bayesian machine learning given by (Guedj, 2019). This will provide some context to the framework under which PAC-Bayes bounds are derived. As before we suppose that our training data $S=\{(x_i,y_i)\}_{i=1}^m$ consists of samples from the distribution $\mathcal{D}$ defined on $\mathcal{Z}$. Bayesian machine learning is used to find a parameter $\hat{\mathbf{w}}$ that corresponds to a hypothesis $h_{\hat{\mathbf{w}}}$ with the property that $h_{\hat{\mathbf{w}}}(x)\approx y$. To do this a learning algorithm is employed, which is simply a map from the data space to the parameter space, $\mathcal{W}$. The learning algorithm requires some prior distribution, $\pi$, to be defined on $\mathcal{W}$. Then using the training data the posterior distribution, $\rho$, is formed from the prior distribution. From the posterior distribution, there are many methodologies to then determine the parameter $\hat{\mathbf{w}}$. For example, one could take $\hat{\mathbf{w}}$ to be the mean, median or a random realization of $\rho$.

### 3.1.2 Notations and Definitions

Bayesian machine learning is a way to manage randomness and uncertainty in the learning task. PAC-Bayes bounds are derived under this framework.

**Definition 3.1** (Alquier, 2023) Let $\mathcal{M}(\mathcal{W})$ be a set of probability distributions defined over $\mathcal{W}$. A data-dependent probability measure is a function $$\hat{\rho}:\bigcup_{n=1}^{\infty}(\mathcal{X}\times\mathcal{Y})^n\to\mathcal{M}(\mathcal{W}).$$
 
For ease of notation we will simple write $\hat{\rho}$ to mean $\hat{\rho}((X_1,Y_1),\dots,(X_n,Y_n))$. The Kullback-Liebler (KL) divergence is a measure of similarity between probability measures defined on the same measurable space.

**Definition 3.2** (Alquier, 2023) Given two probability measures $Q$ and $P$ defined on some sample space $\mathcal{X}$, the KL divergence between $Q$ and $P$ is $$\mathrm{KL}(Q,P)=\int\log\left(\frac{dQ(x)}{dP(x)}\right)Q(dx)$$ when $Q$ is absolutely continuous with respect to $P$. Otherwise, $\mathrm{KL}(Q,P)=\infty$.
 
**Remark 3.3** (Dziugaite, 2017) When $Q, P$ are probability measures on Euclidean space $\mathbb{R}^d$ with densities $q,p$ respectively. The KL divergence is $$\mathrm{KL}(Q, P):=\int\log\left(\frac{q(x)}{p(x)}\right)q(x)dx.$$ Note that KL divergence can take values in the range $[0,\infty]$. Also, note the asymmetry in the definition.
 
For the multivariate normal distributions (Dziugaite, 2017) $N_{q}\sim\mathcal{N}(\mu_{q},\Sigma_{q})$ and $N_{p}\sim\mathcal{N}(\mu_{p},\Sigma_{p})$ defined on $\mathbb{R}^d$ we have that, $$\mathrm{KL}(N_q, N_p)=\frac{1}{2}\left(\mathrm{tr}\left(\Sigma_p^{-1}\Sigma_q\right)-d+(\mu_p-\mu_q)^\top\Sigma_p^{-1}(\mu_p-\mu_q)+\log\left(\frac{\det\Sigma_p}{\det\Sigma_q}\right)\right).$$ Similarly, for Bernoulli distributions (Dziugaite, 2017) $\mathcal{B}(q)\sim\mathrm{Bern}(q)$ and $\mathcal{B}(p)\sim\mathrm{Bern}(p)$ it follows that $$\mathrm{kl}(q, p):=\mathrm{KL}(\mathcal{B}(q),\mathcal{B}(p))=q\log\left(\frac{q}{p}\right)+(1-q)\log\left(\frac{1-q}{1-p}\right),$$ For $p^*\in[0,1]$ bounds of the form $\mathrm{kl}(q, p^*)\leq c$ for some $q\in[0,1]$ and $c\geq0$ are of interest. Hence, we introduce the notation $$\mathrm{kl}^{-1}(q, c):=\sup\{p\in[0,1]:\mathrm{kl}(q, p)\leq c\}.$$ For a distribution $Q$ defined on $\mathcal{W}$ we will use the notation $$\mathbb{E}_{\mathbf{w}\sim Q}(R(\mathbf{w}))=R(Q)\text{ and }\mathbb{E}_{\mathbf{w}\sim Q}\left(\hat{R}(\mathbf{w})\right)=\hat{R}(Q)$$ for convenience. The first PAC-Bayes bounds we will encounter is known as Catoni's bound. Recall, that under the Bayesian framework, we first fix a prior distribution, $\pi\in\mathcal{M}(\mathcal{W})$.

### 3.1.3 PAC-Bayes Bounds

**Theorem 3.4** (Catoni, 2009) For all $\lambda>0$, for all $\rho\in\mathcal{M}(\mathcal{W})$, and $\delta\in(0,1)$ it follows that $$\mathbb{P}_{S\sim\mathcal{D}^m}\left(R(\rho)\leq\hat{R}(\rho)+\frac{\lambda C^2}{8m}+\frac{\mathrm{KL}(\rho,\pi)+\log\left(\frac{1}{\delta}\right)}{\lambda}\right)\geq1-\delta.$$

<details>
<summary>Proof</summary>
<br>

**Theorem 3.4.1 (Jensen's Inequality)**
For a convex function $f(x)$ (with a Taylor expansion) and a random variable $X$ defined on sample space $\mathcal{X}$, if $\mathbb{E}(f(X))$ and $f(\mathbb{E}(X))$ are finite then $$\mathbb{E}(f(X))\geq f(\mathbb{E}(X)).$$ Equality holds if and only if $f$ is a linear function on some convex set $A$ such that $\mathbb{P}(X\in A)=1$. If $f$ doesn't have this property then equality holds if and only if the random variable is constant.
<details>
<summary>Proof (Mitzenmacher, 2005)</summary>
<br>

Let $\mu=\mathbb{E}(X)$, so by assumption we know there is a $c$ such that $$f(x)=f(\mu)+f^{\prime}(\mu)(x-\mu)+\frac{f^{\prime\prime}(c)(x-\mu)^2}{2}\geq f(\mu)+f^\prime(\mu)(x-\mu).$$ Where we have used the fact that $f^{\prime\prime}(c)>0$ due to the convexity of $f$. Taking expectations of both sides we conclude that $$\begin{align*}\mathbb{E}(f(X))&\geq\mathbb{E}\left(f(\mu)+f^{\prime}(u)(X-\mu)\right)\\&=\mathbb{E}(f(\mu))+f^\prime(\mu)\left(\mathbb{E}(X)-\mu\right)\\&=f(\mu)=f\left(\mathbb{E}(X)\right),\end{align*}$$ which completes the proof of the theorem. $\square$

</details>

**Proposition 3.4.2** For any probability measures $Q$ and $P$ it follows that $\mathrm{KL}(Q,P)\geq0$ with equality if and only if $Q$ and $P$ are the same probability distribution.
<details>
<summary>Proof</summary>
<br>

Note that $\log$ is a concave function so Jensen's inequality is reversed. Therefore, $$\begin{align*}-\mathrm{KL}(Q,P)&=-\int_{\mathcal{X}}\log\left(\frac{q(x)}{p(x)}\right)q(x)dx\\&=\int_{\mathcal{X}}\log\left(\frac{p(x)}{q(x)}\right)q(x)dx\\&=\mathbb{E}_{Q}\left(\log\left(\frac{p(x)}{q(x)}\right)\right)\\&\leq\log\left(\mathbb{E}_Q\left(\frac{p(x)}{q(x)}\right)\right)\\&=\log\left(\int_{\mathcal{X}}p(x)dx\right)\\&=\log(1)=0,\end{align*}$$ where Jensen's inequality has been used to get the inequality. This shows that $\mathrm{KL}(Q,P)\geq0$. Note that if $\mathrm{KL}(Q,P)=0$ then equality must hold for Jensen's inequality which implies that $\frac{q(x)}{p(x)}=1$ which implies that $Q$ and $P$ are the same probability distribution. On the other hand, if $Q$ and $P$ are the same probability distribution on the sample space $\mathcal{X}$ then, $$\mathrm{KL}(Q,P)=\int_{\mathcal{X}}\log\left(\frac{q(x)}{p(x)}\right)q(x)dx=\int_{\mathcal{X}}\log(1)q(x)dx=0.$$
 
</details>

**Lemma 3.4.3** For any measurable, bounded function $f:\mathcal{W}\to\mathbb{R}$ we have, $$\log\left(\mathbb{E}_{\mathbf{w}\sim\pi}\left(e^{f(\mathbf{w})}\right)\right)=\sup_{\rho\in\mathcal{M}(\mathcal{W})}\left(\mathbb{E}_{\mathbf{w}\sim\rho}\left(f(\mathbf{w})\right)-\mathrm{KL}(\rho,\pi)\right).$$ Moreover, the supremum with respect to $\rho$ is achieved for the Gibbs posterior $\pi_f$ defined by its density with respect to $\pi$ as $$\frac{d\pi_f(\mathbf{w})}{d\pi_f(\mathbf{w})}=\frac{e^{f(\mathbf{w})}}{\mathbb{E}_{\mathbf{w}\sim\pi_f}\left(e^{f(\mathbf{w})}\right)}.$$
<details>
<summary>Proof</summary>
<br>

From the definition of $\pi_f(\mathbf{w})$ we have that $$\pi_f(\mathbf{w})=\frac{e^{f(\mathbf{w})}}{\mathbb{E}_{\mathbf{w}\sim\pi_f}\left(e^{f(\mathbf{w})}\right)}\pi_f(\mathbf{w}).$$ Therefore, $$\begin{align*}\mathrm{KL}\left(\rho,\pi_f\right)&=\int_{\mathbf{w}\in\mathcal{W}}\log\left(\frac{\rho(\mathbf{w})}{\pi_f(\mathbf{w})}\right)\rho(\mathbf{w})d\mathbf{w}\\&=\int_{\mathbf{w}\in\mathcal{W}}\log(\rho(\mathbf{w}))\rho(\mathbf{w})d\mathbf{w}-\int_{\mathbf{w}\in\mathcal{W}}\log\left(\frac{e^{h(\mathbf{w})}\pi_f(\mathbf{w})}{\mathbb{E}_{\mathbf{w}\sim\pi_f}\left(e^{f(\mathbf{w})}\right)}\right)\rho(\mathbf{w})d\mathbf{w}\\&=\int_{\mathbf{w}\in\mathcal{W}}\log\left(\frac{\rho(\mathbf{w})}{\pi_f(\mathbf{w})}\right)\rho(\mathbf{w})d\mathbf{w}-\int_{\mathbf{w}\in\mathcal{W}}h(\mathbf{w})\rho(\mathbf{w})d\mathbf{w}+\log\left(\mathbb{E}_{\mathbf{w}\sim\pi_f}\left(e^{f(\mathbf{w})}\right)\right)\\&=\mathrm{KL}(\rho,\pi_f)-\mathbb{E}_{\rho}(f(\mathbf{w}))+\log\left(\mathbb{E}_{\mathbf{w}\sim\pi_f}\left(e^{f(\mathbf{w})}\right)\right).\end{align*}$$ By Proposition 3.4.2 the left hand side is non-negative and equal to $0$ only when $\rho=\pi_f$, which completes the proof. $\square$

</details>

Recall, from the proof of Theorem 2.1 that for any $t>0$ we have that $$\mathbb{E}_{S\sim\mathcal{D}^m}\left(\exp\left(tm\left(R(\mathbf{w})-\hat{R}(\mathbf{w})\right)\right)\right)\leq\exp\left(\frac{mt^2C^2}{8}\right).$$ Letting $t=\frac{\lambda}{m}$ we deduce that $$\mathbb{E}_{S\sim\mathcal{D}^m}\left(\exp\left(\lambda\left(R(\mathbf{w})-\hat{R}(\mathbf{w})\right)\right)\right)\leq\exp\left(\frac{\lambda^2C^2}{8m}\right).$$ Integrating this with respect to $\pi$ gives $$\mathbb{E}_{\mathbf{w}\sim\pi}\mathbb{E}_{S\sim\mathcal{D}^m}\left(\exp\left(\lambda\left(R(\mathbf{w})-\hat{R}(\mathbf{w})\right)\right)\right)\leq\exp\left(\frac{\lambda^2C^2}{8m}\right).$$ To which we can apply Fubini's theorem to interchange the order of integration $$\mathbb{E}_{S\sim\mathcal{D}^m}\exp\left(\lambda\left(R(\pi)-\hat{R}(\pi)\right)\right)\leq\exp\left(\frac{\lambda^2C^2}{8m}\right),$$ and then apply Lemma 3.4.3 to get $$\mathbb{E}_{S\sim\mathcal{D}^m}\left(\exp\left(\sup_{\rho\in\mathcal{M}(\mathcal{W})}\left(\lambda\left(R(\rho)-\hat{R}(\rho)\right)\right)-\mathrm{KL}(\rho,\pi)-\frac{\lambda^2C^2}{8m}\right)\right)\leq 1.$$ Now fix $s>0$ and apply Chernoff bound to get that $$\mathbb{P}_{S\sim\mathcal{D}^m}\left(\sup_{\rho\in\mathcal{M}(\mathcal{W})}\left(\lambda\left(R(\rho)-\hat{R}(\rho)\right)\right)-\mathrm{KL}(\rho,\pi)-\frac{\lambda^2C^2}{8m}>s\right)\leq\mathbb{E}_{S\sim\mathcal{D}^m}\left(\exp\left(\sup_{\rho\in\mathcal{M}(\mathcal{W})}\left(\lambda\left(R(\rho)-\hat{R}(\rho)\right)\right)-\mathrm{KL}(\rho,\pi)\right)\right)e^{-s}\leq e^{-s}.$$ Setting $s=\log\left(\frac{1}{\delta}\right)$ and rearranging completes the proof. $\square$

</details>

Theorem 3.4 motivates the study of the data-dependent probability measure $$\begin{equation}\hat{\rho}_{\lambda}=\mathrm{argmin}_{\rho\in\mathcal{M}(\mathcal{W})}\left(\hat{R}(\rho)+\frac{\mathrm{KL}(\rho,\pi)}{\lambda}\right).\end{equation}$$

**Definition 3.5** (Alquier, 2023) The optimization problem defined by Equation $(1)$ has the solution $\hat{\rho}_{\lambda}=\pi_{-\lambda\hat{R}}$ given by $$\hat{\rho}_{\lambda}(d\mathbf{w})=\frac{\exp\left(-\lambda\hat{R}(\mathbf{w})\right)\pi(d\mathbf{w})}{\mathbb{E}\left(\exp\left(-\lambda\hat{R}(\pi)\right)\right)}.$$ This is distribution is known as the Gibbs posterior.
 
**Corollary 3.6** (Alquier, 2023) For all $\lambda>0$, and $\delta\in(0,1)$ it follows that $$\mathbb{P}_{S\sim\mathcal{D}^m}\left(R(\hat{\rho}_{\lambda})\leq\inf_{\rho\in\mathcal{M}(\mathcal{W})}\left(\hat{R}(\rho)+\frac{\lambda C^2}{8m}+\frac{\mathrm{KL}(\rho,\pi)+\log\left(\frac{1}{\delta}\right)}{\lambda}\right)\right)\geq1-\delta.$$

For a learning algorithm, we noted that there are different methodologies for how the learned classifier is sampled from the posterior. In the case where consider a single random realization of the posterior distribution, we have the following result.

**Theorem 3.7** (Alquier, 2023) For all $\lambda>0$, $\delta\in(0,1)$, and data-dependent probability measure $\tilde{\rho}$ we have that $$\mathbb{P}_{S\sim\mathcal{D}^m}\mathbb{P}_{\tilde{\mathbf{w}}\sim\tilde{\rho}}\left(R\left(\tilde{\mathbf{w}}\right)\leq\hat{R}\left(\tilde{\mathbf{w}}\right)+\frac{\lambda C^2}{8m}+\frac{\log\left(\frac{d\rho\left(\tilde{\mathbf{w}}\right)}{d\pi\left(\tilde{\mathbf{w}}\right)}\right)+\log\left(\frac{1}{\delta}\right)}{\lambda}\right)\geq1-\delta.$$
<details>
<summary>Proof</summary>
<br>

The beginning of this proof proceeds in the same way as that of Theorem 3.4 up to the point where we conclude that $$\mathbb{E}_{\mathbf{w}\sim\pi}\mathbb{E}_{S\sim\mathcal{D}^m}\left(\exp\left(\lambda\left(R(\mathbf{w})-\hat{R}(\mathbf{w})\right)\right)\right)\leq\exp\left(\frac{\lambda^2C^2}{8m}\right).$$ For any non-negative function $h$ we have that $$\begin{align*}\mathbb{E}_{\mathbf{w}\sim\pi}(h(\mathbf{w}))&=\int_{\mathbf{w}\in\mathcal{W}}h(\mathbf{w})\pi(d\mathbf{w})\\&=\int_{\left\{\frac{d\tilde{\rho}}{d\pi}(\mathbf{w})>0\right\}}h(\mathbf{w})\pi(d\mathbf{w})\\&=\int_{\left\{\frac{d\tilde{\rho}}{d\pi}(\mathbf{w})>0\right\}}h(\mathbf{w})\frac{d\pi}{d\tilde{\rho}}(\mathbf{w})\tilde{\rho}(d\mathbf{w})\\&=\mathbb{E}_{\mathbf{w}\sim\tilde{\rho}}\left(h(\mathbf{w})\exp\left(-\log\left(\frac{d\tilde{\rho}}{d\pi}(\mathbf{w})\right)\right)\right)\end{align*}$$ which means that $$\mathbb{E}_{\mathbf{w}\sim\pi}\mathbb{E}_{S\sim\mathcal{D}^m}\left(\exp\left(\lambda\left(R(\mathbf{w})-\hat{R}(\mathbf{w})\right)-\log\left(\frac{d\tilde{\rho}}{d\pi}(\mathbf{w})\right)\right)\right)\leq\exp\left(\frac{\lambda^2C^2}{8m}\right).$$ Now in the same way as the proof of Theorem 3.4 we apply the Chernoff bound, set $\delta$ and then re-arrange the terms to complete the proof. $\square$

</details>

Note that Theorem 3.4 is a bound in probability. We now state an equivalent bound that holds in expectation.

**Theorem 3.8** (Alquier, 2023) For all $\lambda>0$, and data-dependent probability measure $\tilde{\rho}$, we have that $$\mathbb{E}_{S\sim\mathcal{D}^m}(R(\tilde{\rho}))\leq\mathbb{E}_{S\sim\mathcal{D}^m}\left(\hat{R}(\tilde{\rho})+\frac{\lambda C^2}{8m}+\frac{\mathrm{KL}(\tilde{\rho},\pi)}{\lambda}\right).$$

<details>
<summary>Proof</summary>
<br>

Once again we proceed in the same way as Theorem 3.4 to the point where we deduce that $$\mathbb{E}_{S\sim\mathcal{D}^m}\left(\exp\left(\sup_{\rho\in\mathcal{M}(\mathcal{W})}\left(\lambda\left(R(\rho)-\hat{R}(\rho)\right)\right)-\mathrm{KL}(\rho,\pi)-\frac{\lambda^2C^2}{8m}\right)\right)\leq 1.$$ Now we apply Jensen's inequality to get that $$\exp\left(\mathbb{E}_{S\sim\mathcal{D}^m}\left(\sup_{\rho\in\mathcal{M}(\mathcal{W})}\left(\lambda\left(R(\rho)-\hat{R}(\rho)\right)\right)-\mathrm{KL}(\rho,\pi)-\frac{\lambda^2C^2}{8m}\right)\right)\leq 1,$$ which implies that $$\mathbb{E}_{S\sim\mathcal{D}^m}\left(\sup_{\rho\in\mathcal{M}(\mathcal{W})}\left(\lambda\left(R(\rho)-\hat{R}(\rho)\right)\right)-\mathrm{KL}(\rho,\pi)-\frac{\lambda^2C^2}{8m}\right)\leq0.$$ In particular this holds for our data-dependent probability measure $\tilde{\rho}$. Therefore, $$\mathbb{E}_{S\sim\mathcal{D}^m}\left(\lambda\left(R(\tilde{\rho})-\hat{R}(\tilde{\rho})\right)-\mathrm{KL}(\tilde{\rho},\pi)-\frac{\lambda^2C^2}{8m}\right)\leq0,$$ and so using the linearity of expectation and rearranging completes the proof. $\square$

</details>

**Corollary 3.9** (Alquier, 2023) For $\tilde{\rho}=\hat{\rho}_{\lambda}$, the following holds $$\mathbb{E}_{S\sim\mathcal{D}^m}(R(\tilde{\rho}))\leq\mathbb{E}_{S\sim\mathcal{D}^m}\left(\inf_{\rho\in\mathcal{M}(\mathcal{W})}\left(\hat{R}(\rho)\right)+\frac{\lambda C^2}{8m}+\frac{\mathrm{KL}(\rho,\pi)}{\lambda}\right).$$

In the results that follow we will consider the $0$-$1$ loss. This is a measurable function $l:\mathcal{Y}\times\mathcal{Y}\to\{0,1\}$ defined by $l(y,y^\prime)=\mathbf{1}(y\neq y^\prime)$.

**Theorem 3.10** (McAllester, 1999) For all $\rho\in\mathcal{M}(\mathcal{W})$ and $\delta>0$ we have that $$\mathbb{P}_{S\sim\mathcal{D}^m}\left(R(\rho)\leq\hat{R}(\rho)+\sqrt{\frac{\mathrm{KL}(\rho,\pi)+\log\left(\frac{1}{\delta}\right)+\frac{5}{2}\log(m)+8}{2m-1}}\right)\geq1-\delta.$$

<details>
<summary>Proof</summary>
<br>

Refer to (McAllester, 1999) for the proof of this theorem.

</details>

**Theorem 3.11** (Catoni, 2007) For $a>0$ and $p\in(0,1)$ let $$\Phi_{a}(p)=\frac{-\log\left(1-p(1-\exp(-a))\right)}{a}.$$ Then for any $\lambda>0$, $\delta>0$ and $\rho\in\mathcal{M}(\mathcal{W})$ we have that $$\mathbb{P}_{S\sim\mathcal{D}^m}\left(R(\rho)\leq\Phi^{-1}_{\frac{\lambda}{m}}\left(\hat{R}(\rho)+\frac{\mathrm{KL}(\rho,\pi)+\log\left(\frac{1}{\delta}\right)}{\lambda}\right)\right)\geq1-\delta.$$
<details>
<summary>Proof</summary>
<br>

Refer to (Catoni, 2007) for the proof of this theorem.

</details>
 
**Theorem 3.12** (Maurer, 2004) For any $\delta>0$ and $\rho\in\mathcal{M}(\mathcal{W})$ then we have that $$\mathbb{P}_{S\sim\mathcal{D}^m}\left(R(\rho)\leq\mathrm{kl}^{-1}\left(\hat{R}(\rho),\frac{\mathrm{KL}(\rho,\pi)+\log\left(\frac{2\sqrt{m}}{\delta}\right)}{m}\right)\right)\geq1-\delta.$$

<details>
<summary>Proof</summary>
<br>

For $X_1,\dots,X_n$ $\mathrm{i.i.d}$ random variables in $[0,1]$ and with $\mathbb{E}(X_i)=\mu$ let $\mathbf{X}=(X_1,\dots,X_n)$ and $$M(\mathbf{X})=\frac{1}{n}\sum_{i=1}^nX_i.$$ For any random variable $X$ in $[0,1]$ let $X^\prime$ denote the Bernoulli random variables with parameter $\mathbb{E}(X)$ and let $\mathbf{X}^\prime=(X_1^\prime,\dots,X_n^\prime)$. 

**Theorem 3.12.1** For $n\geq2$ with the notation as above we have that $$\mathbb{E}\left(\exp\left(n\mathrm{kl}(M(\mathbf{X}),\mu)\right)\right)\leq\exp\left(\frac{1}{12n}\right)\sqrt{\frac{\pi n}{2}}+2.$$
<details>
<summary>Proof</summary>
<br>

For the proof of this theorem refer to (Maurer, 2004).

</details>

**Corollary 3.12.2** For $n\geq2$ we have that $$\mathbb{E}\left(\exp\left(n\mathrm{kl}(M(\mathbf{X}),\mu)\right)\right)\leq2\sqrt{n}.$$
<details>
<summary>Proof</summary>
<br>

Replace $n$ with the continuous variable $x\in(0,\infty)$. Let $f(x)=\exp\left(\frac{1}{12x}\right)\sqrt{\frac{\pi x}{2}}+2$ and $g(x)=2\sqrt{x}$, then $$f^\prime(x)=g^{\prime}(x)\left(\sqrt{\frac{\pi}{2}}\exp\left(\frac{1}{12x}\right)\left(\frac{1}{2}-\frac{1}{12x}\right)\right).$$ From which it is clear that $f^\prime(x)<g^{\prime}(x)$. Therefore, as one can numerically see that $g(x)>f(x)$ for $x\approx 7.5$ we can conclude that for all $n\geq8$ we have that $\exp\left(\frac{1}{12n}\right)\sqrt{\frac{\pi n}{2}}+2\leq2\sqrt{n}$ which completes the proof of the corollary. $\square$

</details>

Recall, that $$\hat{R}(\mathbf{w})=\frac{1}{m}\sum_{i=1}^ml(h_{\mathbf{w}}(x_i),y_i)$$ and $R({\mathbf{w}})=\mathbb{E}_{(x,y)\sim\mathcal{D}}\left(l(h(x),y)\right)$. As we are considering a loss function bounded to the interval $[0,1]$ we can consider each of the $l(h_{\mathbf{w}}(x_i),y_i)$ as $\mathrm{i.i.d}$ random variables with mean $R(\mathbf{w})$. Therefore, for any $\mathbf{w}\in\mathcal{W}$ we can apply Corollary 3.12.2 to deduce that $$\mathbb{E}\left(m\mathrm{kl}\left(\hat{R}(\mathbf{w}),R(\mathbf{w})\right)\right)\leq2\sqrt{m}.$$ Now applying Jensen's inequality to the convexity of $\mathrm{kl}$ divergence and the exponential function we have that $$\begin{align*}\mathbb{E}-{S\sim\mathcal{D}^m}\left(\exp\left(m\mathrm{kl}\left(\hat{R}(\rho),R(\rho)\right)-\mathrm{kl}\left(\rho,\pi\right)\right)\right)&\leq\mathbb{E}_{S\sim\mathcal{D}^m}\left(\exp\left(\mathbb{E}_{\mathbf{w}\sim\rho}\left(m\mathrm{kl}\left(\hat{R}(\mathbf{w}),R(\mathbf{w})\right)-\log\left(\frac{d\rho(\mathbf{w})}{d\pi(\mathbf{w})}\right)\right)\right)\right)\\&\leq\mathbb{E}_{S\sim\mathcal{D}^m}\left(\mathbb{E}_{\mathbf{w}\sim\rho}\left(\exp\left(m\mathrm{kl}\left(\hat{R}(\mathbf{w}),R(\mathbf{w})\right)-\log\left(\frac{d\rho(\mathbf{w})}{d\pi(\mathbf{w})}\right)\right)\right)\right)\\&=\mathbb{E}_{S\sim\mathcal{D}^m}\left(\mathbb{E}_{\mathbf{w}\sim\pi}\left(\exp\left(m\mathrm{kl}\left(\hat{R}(\mathbf{w}),R(\mathbf{w})\right)\right)\left(\frac{d\rho}{d\pi}\right)^{-1}\frac{d\rho}{d\pi}\right)\right)\\&\leq\mathbb{E}_{\mathbf{w}\sim\rho}\left(\mathbb{E}_{S\sim\mathcal{D}^m}\left(\exp\left(m\mathrm{kl}\left(\hat{R}(\mathbf{w}),R(\mathbf{w})\right)\right)\right)\right)\\&\leq2\sqrt{m}.\end{align*}$$ Applying Markov's inequality we conclude that $$\begin{align*}\mathbb{P}_{S\sim\mathcal{D}^m}\left(\mathrm{kl}\left(\hat{R}(\mathbf{w}),R(\mathbf{w})\right)>\frac{\mathrm{kl}(\rho,\pi)+\log\left(\frac{2\sqrt{m}}{\delta}\right)}{m}\right)&=\mathbb{P}_{S\sim\mathcal{D}^m}\left(\exp\left(m\mathrm{kl}\left(\hat{R}(\mathbf{w}),R(\mathbf{w})\right)-\mathrm{kl}(\rho,\pi)\right)>\frac{2\sqrt{m}}{\delta}\right)\\&\leq\delta.\end{align*}$$ Taking the complement of this completes the proof. $\square$

</details>
 
## 3.2 Optimizing PAC-Bayes Bounds via SGD

In practice, it is often the case that these bounds are not useful. Despite providing insight into how generalization relates to each of the components of the learning process they do not have much utility in providing non-vacuous bounds on the performance of neural networks on the underlying distribution. The significance of the KL divergence between the posterior and the prior can be noted in each of the bounds of Section 3.1.2. This motivated the work of (Dziugaite, 2017) who successfully minimized this term to provide non-vacuous results in practice. They considered a restricted problem that lends itself to efficient optimization. They use stochastic gradient descent to refine the prior, which is effective as SGD is known to find flat minima. This is important as around flat minima such as $\mathbf{w}^*$ we have that $\hat{R}(\mathbf{w})\approx\hat{R}(\mathbf{w}^*)$ (Alquier, 2023). The setup considered by (Dziugaite, 2017) is the same as the one we have considered throughout this report. With $\mathcal{X}\subset\mathbb{R}^k$ and labels being $\pm 1$. That is, we are considering binary classification based on a set of features. We explicitly state our hypothesis set as $$\mathcal{H}=\left\{h_{\mathbf{w}}:\mathbb{R}^k\to\mathbb{R}:\mathbf{w}\in\mathbb{R}^d\right\}.$$ We are still considering the $0$-$1$, however, because our classifiers output real numbers we modify the loss slightly to account for this. That is, we let $l:\mathbb{R}\to\{\pm1\}$ be defined as $l(y,y^\prime)=\mathbf{1}(\mathrm{sgn}(y^\prime)=y)$. For optimization purposes we use the convex surrogate loss function $\tilde{l}:\mathbb{R}\times\{\pm1\}\to\mathbb{R}_+$ $$\tilde{l}(y,\hat{y})=\frac{\log\left(1+\exp\left(-\hat{y}y\right)\right)}{\log(2)}.$$  For the empirical risk under the convex surrogate loss we write $$\tilde{R}(\mathbf{w})=\frac{1}{m}\sum_{i=1}^m\tilde{l}(h_{\mathbf{w}}(x_i),y_i).$$ Recall, that this definition implicitly depends on the training sample $S_m$. As noted previously the work (Dziugaite, 2017) looks to minimize the KL divergence between the prior and the posterior to achieve non-vacuous bounds. To do this they work under a restricted setting and construct a process to minimize the divergence between the prior and the posterior when the learning algorithm is stochastic gradient descent (SGD). To begin (Dziugaite, 2017) utilize the following bound.

**Theorem 3.13** (Dziugaite, 2017) For every $\delta>0$,$m\in\mathbb{N}$, distribution $\mathcal{D}$ on $\mathbb{R}^k\times\{\pm 1\}$, distribution $\pi$ on $\mathcal{W}$ and distribution $\rho\in\mathcal{M}(\mathcal{W})$, we have that $$\mathbb{P}_{S\sim\mathcal{D}^m}\left(\mathrm{kl}\left(\hat{R}(\rho), R(\rho)\right)\leq\frac{\mathrm{KL}(\rho,\pi)+\log\left(\frac{m}{\delta}\right)}{m-1}\right)\geq1-\delta.$$
 
**Remark 3.14**  Note how this is a slightly weaker statement than Theorem 3.12. This is because (Dziugaite, 2017) cited this Theorem from (Seeger, 2001), however, since then (Maurer, 2004) was able to tighten the result by providing Theorem 3.12. In the following we will update the work of (Dziugaite, 2017) and use the tightened result provided by Theorem 3.12.
 
This motivates the following PAC-Bayes learning algorithm.
1. Fix a $\delta>0$ and a distribution $\pi$ on $\mathcal{W}$,
2. Collect an $\mathrm{i.i.d}$ sample $S_m$ of size $m$,
3. Compute the optimal distribution $\rho$ on $\mathcal{W}$ that minimizes
    $$\begin{equation}\mathrm{kl}^{-1}\left(\hat{R}(\rho),\frac{\mathrm{KL}(\rho,\pi)+\log\left(\frac{2\sqrt{m}}{\delta}\right)}{m}\right),\end{equation}$$
4. Then return the randomized classifier given by $\rho$.

Implementing such an algorithm in this general form is intractable in practice. Recall, that we are considering neural networks and so $\mathbf{w}$ represents the weights and biases of our neural network. To make the algorithm more practical we therefore consider $$\mathcal{M}(\mathcal{W})=\left\{\mathcal{N}_{\mathbf{w},\mathbf{s}}=\mathcal{N}(\mathbf{w},\mathrm{diag}(\mathbf{s})):\mathbf{w}\in\mathbb{R}^d,\mathbf{s}\in\mathbb{R}_+^d\right\}.$$ Utilizing the bound $\mathrm{kl}^{-1}(q,c)\leq q+\sqrt{\frac{c}{2}}$ and replacing the loss with the convex surrogate loss in Equation $(2)$ we obtain the updated optimization problem $$\begin{equation}\min_{\mathbf{w}\in\mathbb{R}^d,\mathbf{s}\in\mathbb{R}^d_+}\tilde{R}\left(\mathcal{N}_{\mathbf{w},\mathbf{s}}\right)+\sqrt{\frac{\mathrm{KL}(\mathcal{N}_{\mathbf{w},\mathbf{s}},\pi)+\log\left(\frac{2\sqrt{m}}{\delta}\right)}{2m}}.\end{equation}$$ We now suppose our prior $\pi$ is of the form $\mathcal{N}(\mathbf{w}_0,\lambda I)$. As we will see the choice of $\mathbf{w}_0$ is not too impactful, as long as it is not $\mathbf{0}$. However, to efficiently choose a judicious value for $\lambda$ we discretize the problem, with the side-effect of expanding the eventual generalization bound. We let $\lambda$ have the for $c\exp\left(-\frac{j}{b}\right)$ for $j\in\mathbb{N}$, so that $c$ is an upper bound and $b$ controls precision. By ensuring that Theorem 3.12 holds with probability $1-\frac{6\delta}{\pi^2j^2}$ for each $j\in\mathbb{N}$ we can then apply a union bound argument to ensure that we get results that hold for probability $1-\delta$. Treating $\lambda$ as continuous during the optimization process and then discretized at the point of evaluating the bound yields the updated optimization problem $$\begin{equation}\min_{\mathbf{w}\in\mathbb{R}^d,\mathbf{s}\in\mathbb{R}^d_+,\lambda\in(0,c)}\tilde{R}(\mathcal{N}_{\mathbf{w},\mathbf{s}})+\sqrt{\frac{1}{2}B_{\mathrm{RE}}(\mathbf{w},\mathbf{s},\lambda;\delta)}\end{equation}$$ where $$B_{\mathrm{RE}}(\mathbf{w},\mathbf{s},\lambda;\delta)=\frac{\mathrm{KL}(\mathcal{N}_{\mathbf{w},\mathbf{s}},\mathcal{N}(\mathbf{w}_0,\lambda I))+2\log\left(b\log\left(\frac{c}{\lambda}\right)\right)+\log\left(\frac{\pi^2\sqrt{m}}{3\delta}\right)}{m}.$$ To optimize Equation $(4)$ we would like to compute its gradient and apply SGD. However, this is not feasible in practice for $\tilde{R}(\mathcal{N}_{\mathbf{w},\mathbf{s}})$. Instead we compute the gradient of $\tilde{R}\left(\mathbf{w}+\xi\odot\sqrt{\mathbf{s}}\right)$ where $\xi\sim\mathcal{N}_{0,\mathbf{1}_d}$.  Once good candidates for this optimization problem are found we return to $(2)$ to calculate the final error bound. With the choice of $\lambda$ it follows that with probability $1-\delta$, uniformly over all $\mathbf{w}\in\mathbb{R}^d,\mathbf{s}\in\mathbb{R}^d_+$ and $\lambda$ (of the discrete form) the expected risk of $\rho=\mathcal{N}_{\mathbf{w},\mathbf{s}}$ is bounded by $$\mathrm{kl}^{-1}\left(\hat{R}(\rho), B_{\mathrm{RE}}(\mathbf{w},\mathbf{s},\lambda;\delta)\right).$$ However, it is often not possible to compute $\hat{R}(\rho)$ due to the intractability of $\rho$. So instead an unbiased estimate is obtained by estimating $\rho$ using a Monte Carlo approximation. Given $n$ $\mathrm{i.i.d}$ samples $\mathbf{w}_1,\dots,\mathbf{w}_n$ from $\rho$ we use the Monte Carlo approximation $\hat{\rho}_n=\sum_{i=1}^n\delta_{\mathbf{w}_i}$, to get the bound $$\hat{R}(\rho)\leq\overline{\hat{R}_{n,\delta^\prime}}(\rho):=\mathrm{kl}^{-1}\left(\hat{R}\left(\hat{\rho}_n\right),\frac{1}{n}\log\left(\frac{2}{\delta^\prime}\right)\right),$$ which holds with probability $1-\delta^\prime$. Finally, by the union bound $$R(\rho)\leq\mathrm{kl}^{-1}\left(\overline{\hat{R}_{n,\delta^\prime}}(\rho), B_{\mathrm{RE}}(\mathbf{w},\mathbf{s},\lambda;\delta)\right),$$ holds with probability $1-\delta-\delta^\prime.$ Now all that is left is to do is to determine optimal values for $\mathbf{w}$ and $\mathbf{s}$. To do this first train a neural network via SGD to get a value of $\mathbf{w}$. Then instantiate a stochastic neural network with the multivariate normal distribution $\rho=\mathcal{N}_{\mathbf{w},\mathbf{s}}$ over the weights, with $\mathbf{s}=\vert\mathbf{w}\vert$. Next apply Algorithm 4 to deduce values of $\mathbf{w},\mathbf{s}$ and $\lambda$ that give a tighter bound.

<font size="3"> **Algorithm 4** Optimizing the PAC Bounds</font>
> **Require:**\
> $\mathbf{w}_0\in\mathbb{R}^d$, the network parameters at initialization.\
> $\mathbf{w}\in\mathbb{R}^d$, the network parameters after SGD.\
> $S_m$, training examples.\
> $\delta\in(0,1)$, confidence parameter.\
> $b\in\mathbb{N},c\in(0,1)$, precision and bound for $\lambda$.\
> $\tau\in(0,1), T$, learning rate.\
> **Ensure:** Optimal $\mathbf{w},\mathbf{s},\lambda$.\
> $\zeta=\vert\mathbf{w}\vert$ $\quad\quad$ Comment: $\mathbf{s}(\zeta)=e^{2\zeta}$\
> $\rho=-3$ $\quad\quad$ Comment: $\lambda(\rho)=e^{2\rho}$\
> $B(\mathbf{w},\mathbf{s},\lambda,\mathbf{w}^\prime)=\tilde{R}(\mathbf{w})+\sqrt{\frac{1}{2}B_{\mathrm{RE}}(\mathbf{w},\mathbf{s},\lambda)}$\
> **for** $t=1\to T$ **do**\
> ----Sample $\xi\sim\mathcal{N}(0,I_d)$\
> ----$\mathbf{w}^\prime(\mathbf{w},\zeta)=\mathbf{w}+\xi\odot\sqrt{\mathbf{s}(\zeta)}$\
> ----$\begin{pmatrix}\mathbf{w}\\\zeta\\\rho\end{pmatrix}=-\tau\begin{pmatrix}\nabla_{\mathbf{w}}B(\mathbf{w},\mathbf{s}(\zeta),\lambda(\rho),\mathbf{w}^\prime(\mathbf{w},\zeta))\\\nabla_\zeta B(\mathbf{w},\mathbf{s}(\zeta),\lambda(\rho),\mathbf{w}^\prime(\mathbf{w},\zeta))\\\nabla_\rho B(\mathbf{w},\mathbf{s}(\zeta),\lambda(\rho),\mathbf{w}^\prime(\mathbf{w},\zeta))\end{pmatrix}$\
> **end for**\
> **return** $\mathbf{w},\mathbf{s}(\zeta),\lambda(\rho)$

Once the values of $\mathbf{w},\mathbf{s}$ and $\lambda$ are found we then need to compute $\overline{\hat{R}_{n,\delta^\prime}}(\rho):=\mathrm{kl}^{-1}\left(\hat{R}\left(\hat{\rho}_n\right),\frac{1}{n}\log\left(\frac{2}{\delta^\prime}\right)\right)$ to get our bound. We note that $$\hat{R}(\hat{\rho}_n)=\sum_{i=1}^n\delta_{\mathbf{w}_i}\left(\frac{1}{m}\sum_{j=1}^ml(h_{\mathbf{w}_i}(x_j),y_j)\right).$$ Then to invert the kl divergence we employ Newton's method, in the form of Algorithm 5, to get an approximation for our bound.

<font size="3"> **Algorithm 5** Newton's Method for Inverting kl Divergence</font>
> **Require:** $q,c$, initial estimate $p_0$ and $N\in\mathbb{N}$\
> **Ensure:** $p$ such that $p\approx\mathrm{kl}^{-1}(q,c)$\
> **for** $n=1\to N$ **do**\
> ----**if** $p\geq1$ **then**\
> --------**return** $1$\
> ----**else**\
> --------$p_0=p_0-\frac{q\log\left(\frac{q}{c}\right)+(1-q)\log\left(\frac{1-q}{1-c}\right)-c}{\frac{1-q}{1-p}-\frac{q}{p}}$\
> ----**end if**\
> **end for**\
> **return** $p_0$