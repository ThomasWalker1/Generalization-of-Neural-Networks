# PAC

## 2.1 Introducing PAC Bounds

###  2.1.1 Notation

We will first introduce some basic notation that is for the most part consistent with (Alquier, 2023) and will remain constant throughout the report. Along the way, we will need to introduce some more specialized notation for the different sections. The problems we will concern ourselves most with will be supervised classification tasks. This means, we have a feature space $\mathcal{X}$ and a label space $\mathcal{Y}$ which combine to form the data space $\mathcal{Z}=\mathcal{X}\times\mathcal{Y}$ for which some unknown $\mathcal{D}$ is defined on. The challenge now is to learn a classifier $h:\mathcal{X}\to\mathcal{Y}$ that correctly labels samples from $\mathcal{X}$ according to $\mathcal{D}$. The training data $S_m=\{(x_i,y_i)\}_{i=1}^m$ consists of $m$ $\mathrm{i.i.d}$ samples from $\mathcal{D}$. As we are considering neural networks, a classifier will be parameterised by a weight vector $\mathbf{w}$ which we will denote $h_{\mathbf{w}}$. Let $\mathcal{W}$ denote the set of possible weights for a classifier and the set of all possible classifiers $\mathcal{H}$ will sometimes be referred to as the hypothesis set. We will often denote the set of probability distributions over $\mathcal{W}$ as $\mathcal{M}(\mathcal{W})$. To assess the quality of a classifier we define a measurable function $l:\mathcal{Y}\times\mathcal{Y}\to[0,\infty)$ called the loss function and we will assume that $0\leq l\leq C$. As our training data is just a sample from the underlying (unknown) distribution $\mathcal{D}$ there is the possibility that our classifier performs well on the training data, but performs poorly on the true distribution. Let the risk of our classifier be defined as $$R(h_{\mathbf{w}})=\mathbb{E}_{(x,y)\sim\mathcal{D}}\left(l(h(x),y)\right).$$ As our classifier is parameterised $\mathbf{w}$ we will instead write $R(\mathbf{w})$ for the risk of our classifier. Similarly, we define the empirical risk of our classifier to be $$\hat{R}(\mathbf{w})=\frac{1}{m}\sum_{i=1}^ml(h_{\mathbf{w}}(x_i),y_i).$$ Note that $\mathbb{E}_{S\sim\mathcal{D}^m}\left(\hat{R}(\mathbf{w})\right)=R(\mathbf{w})$.

###  2.1.2 PAC Bounds

PAC bounds refer to a general class of bounds on the performance of a learned classifier. They aim to determine with high probability what the performance of a classifier will be like on the distribution $\mathcal{D}$ when trained on some training data taken from this distribution.

**Theorem 2.1** (Alquier, 2023) Let $\vert\mathcal{W}\vert=M<\infty$, $\delta\in(0,1)$, and $\mathbf{w}\in\mathcal{W}$ then it follows that $$\mathbb{P}_{S\sim\mathcal{D}^m}\left(R(\mathbf{w})\leq\hat{R}(\mathbf{w})+C\sqrt{\frac{\log\left(\frac{M}{\epsilon}\right)}{2n}}\right)\geq 1-\delta.$$
<details>
<summary>Proof</summary>
<br>

**Lemma 1** (Scott, 2014) Let $U_1,\dots,U_n$ be independent random variables taking values in an interval $[a,b]$. Then for any $t>0$ we have that $$\mathbb{E}\left(\exp\left(t\sum_{i=1}^n\left(U_i-\mathbb{E}(U_i)\right)\right)\right)\leq\exp\left(\frac{nt^2(b-a)^2}{8}\right).$$
<details>
<summary>Proof</summary>
<br>

**Proof.** For $s>0$ the function $x\mapsto e^{sx}$ is convex 
so that
$$e^{sx}\leq\frac{x-a}{b-a}e^{sb}+\frac{b-x}{b-a}e^{sa}.$$
Let $V_i=\mathbb{E}(U_i-\mathbb{E}(U_i))$, then as $\mathbb{E}(V_i)=0$ it follows that
$$\mathbb{E}\left(\exp(sV_i)\right)\leq\frac{b}{b-a}e^{sa}-\frac{a}{b-a}e^{sb}.$$
With $p=\frac{b}{b-a}$ and $u=(b-a)s$ consider
$$\psi(u)=\log\left(pe^{sa}+(1-p)e^{sb}\right)=(p-1)u+\log\left(p+(1-p)e^u\right).$$
This is a smooth function so that by Taylor's theorem we have that for any $u\in\mathbb{R}$ there exists $\xi=\xi(u)\in\mathbb{R}$ such that
$$\psi(u)=\psi(0)+\psi^\prime(0)u+\frac{1}{2}\psi^{\prime\prime}(\xi)u^2.$$
As
$$\psi^\prime(u)=(p-1)+1-\frac{p}{p+(1-p)e^u}$$
we have that $\psi(0)=0$ and $\psi^\prime(0)=0$. Furthermore, as
$$\psi^{\prime\prime}(u)=\frac{p(1-p)e^u}{(p+(1-p)e^u)^2},\text{ and }\psi^{(3)}(u)=\frac{p(1-p)e^u(p+(1-p)e^u)(p-(1-p)e^u)}{(p+(1-p)e^u)^2}$$
we see that $\psi^{\prime\prime}(u)$ has a stationary point at $u^*=\log\left(\frac{p}{p-1}\right)$. For $u$ slightly less than $u^*$ we have $\psi^{(3)}(u)>0$ and for $u$ slightly larger than $u^*$ we have $\psi^{(3)}(u)<0$. Therefore, $u^*$ is a maximum point and so
$$\psi^{\prime\prime}(u)\leq\psi^{\prime\prime}(u^*)=\frac{1}{4}.$$
Hence, $\psi(u)\leq\frac{u^2}{8}$ which implies that
$$\log\left(\mathbb{E}\left(\exp(sV_i)\right)\right)\leq\frac{u^2}{8}=\frac{s^2(b-a)^2}{8}.$$
Therefore,
$$\begin{align*}\mathbb{E}\left(\exp\left(t\sum_{i=1}^n\left(U_i-\mathbb{E}(U_i)\right)\right)\right)&=\prod_{i=1}^n\mathbb{E}\left(\exp\left(t(U_i-\mathbb{E}(U_i))\right)\right)\\&\leq\prod_{i=1}^n\exp\left(\frac{t^2(b-a)^2}{8}\right)\\&\leq\exp\left(\frac{nt^2(b-a)^2}{8}\right)\end{align*}$$
which completes the proof. $\square$

</details>

Recall that we have our random sample $S=\{(x_i,y_i)\}_{i=1}^m\sim\mathcal{D}^m$. If we fix $\mathbf{w}\in\mathcal{W}$ we can let $l_i(\mathbf{w})=l(h_{\mathbf{w}}(x_i),y_i)$. This is a random variable due to the randomness of $S$ and so we can apply Lemma 1 to $U_i=\mathbb{E}(l_i(\mathbf{w}))-l_i(\mathbf{w})$ to get that
$$\mathbb{E}_{S\sim\mathcal{D}^m}\left(\exp\left(tm\left(R(\mathbf{w})-\hat{R}(\mathbf{w})\right)\right)\right)\leq\exp\left(\frac{mt^2C^2}{8}\right).$$
Therefore, for any $s>0$ we can apply Markov's Inequality to get that
$$\begin{align*}\mathbb{P}_{S\sim\mathcal{D}^m}\left(R(\mathbf{w})-\hat{R}(\mathbf{w})>s\right)&=\mathbb{P}_{S\sim\mathcal{D}^m}\left(\exp\left(mt\left(R(\mathbf{w})-\hat{R}(\mathbf{w})\right)\right)>\exp(mts)\right)\\&\leq\frac{\mathbb{E}_{S\sim\mathcal{D}^m}\left(\exp\left(mt\left(R(\mathbf{w})-\hat{R}(\mathbf{w})\right)\right)\right)}{\exp(mts)}\\&\leq\exp\left(\frac{mt^2C^2}{8}-mts\right).\end{align*}$$
This bound is minimized for $t=\frac{4s}{C^2}$ so that
$$\mathbb{P}_{S\sim\mathcal{D}^m}\left(R(\mathbf{w})>\hat{R}(\mathbf{w})+s\right)\leq\exp\left(-\frac{2ms^2}{C^2}\right).$$

The above bound holds for fixed $\mathbf{w}\in\mathcal{W}$ so develop a uniform bound we consider the following.
$$\begin{align*}\mathbb{P}_{\mathcal{S}\sim\mathcal{D}^m}\left(\sup_{\mathbf{w}\in\mathcal{W}}\left(R(\mathbf{w})-\hat{R}(\mathbf{w})\right)>s\right)&=\mathbb{P}_{S\sim\mathcal{D}^m}\left(\bigcup_{\mathbf{w}\in\mathcal{W}}\left\{R(\mathbf{w})-\hat{R}(\mathbf{w})>s\right\}\right)\\&\leq\sum_{\mathbf{w}\in\mathcal{W}}\mathbb{P}_{S\sim\mathcal{D}^m}\left(R(\mathbf{w})>\hat{R}(\mathbf{w})+s\right)\\&\leq M\exp\left(-\frac{2ms^2}{C^2}\right).\end{align*}$$
Now taking $\delta=M\exp\left(-\frac{2ms^2}{C^2}\right)$ we get that
$$\mathbb{P}_{S\sim\mathcal{D}^m}\left(\sup_{\mathbf{w}\in\mathcal{W}}\left(R(\mathbf{w})-\hat{R}(\mathbf{w})\right)>C\sqrt{\frac{\log\left(\frac{M}{\delta}\right)}{2m}}\right)\leq\delta$$
which upon taking complements completes the proof of the theorem. $\square$

</details>
 
Theorem 2.1 says that with arbitrarily high probability we can bound the performance of our trained classifier on the unknown distribution $\mathcal{D}$. However, there is nothing to guarantee that the bound is useful in practice. Note that requiring bounds to hold for greater precision involves sending $\epsilon$ to $0$ which increases the bound. If the bound exceeds $C$ then it is no longer useful as we know already that $R(\mathbf{w})\leq C$. It is important to note at this stage that are two ways in which PAC bounds can hold. One set of bounds holds in expectation whilst the other hold in probability. Risk is a concept that will develop bounds in expectation. In Section 2.3 we will introduce definitions that will let us work with bounds that hold in probability. There are two general forms of PAC bounds, we have uniform convergence bounds and algorithmic-dependent bounds (Viallard, 2021). Uniform convergence bounds have the general form $$\mathbb{P}_{\mathcal{S}\sim\mathcal{D}^m}\left(\sup_{\mathbf{w}\in\mathcal{W}}\left\vert R(\mathbf{w})-\hat{R}(\mathbf{w})\right\vert\leq\epsilon\left(\frac{1}{\delta},\frac{1}{m},\mathcal{W}\right)\right)\geq 1-\delta.$$ This can be considered as a worst-case analysis of hypothesis generalization, and so in practice will lead to vacuous bounds. Algorithmic-dependent bounds involve the details of a learning algorithm $A$ and take the form $$\mathbb{P}_{\mathcal{S}\sim\mathcal{D}^m}\left(\left\vert R\left(A(S)\right)-\hat{R}\left(A(S)\right)\right\vert\leq\epsilon\left(\frac{1}{\delta},\frac{1}{m},A\right)\right)\geq1-\delta.$$ These bounds can be seen as a refinement of the uniform convergence bounds as they are only required to hold for the output of the learning algorithm. It will be the subject of Section 5.1 to explore such bounds further.

###  2.1.3 Occam Bounds

Occam bounds are derived under the assumption that $\mathcal{H}$ is countable and that we have some bias $\pi$ defined on the hypothesis space. Note that in our setup this does not necessarily mean that $\mathcal{W}$ is countable, as multiple weights may correspond to the same classifier. However, as the Occam bounds hold true for all $h\in\mathcal{H}$ it must also be the case that they hold for all classifiers corresponding to the weight $\mathbf{w}\in\mathcal{W}$. Using this we will instead assume that $\pi$ is defined over $\mathcal{W}$.

**Theorem 2.2** (McAllester, 2013) Simultaneously for all $\mathbf{w}\in\mathcal{W}$ and $\delta\in(0,1)$ the following holds, $$\mathbb{P}_{S\sim\mathcal{D}^m}\left(R(\mathbf{w})\leq\inf_{\lambda>\frac{1}{2}}\frac{1}{1-\frac{1}{2\lambda}}\left(\hat{R}(\mathbf{w})+\frac{\lambda C}{m}\left(\log\left(\frac{1}{\pi(\mathbf{w})}\right)+\log\left(\frac{1}{\delta}\right)\right)\right)\right)\geq1-\delta.$$
<details>
<summary>Proof</summary>
<br>

For the proof we consider the case when $C=1$, with the more general case following by rescaling the loss function. For $\mathbf{w}\in\mathcal{W}$ let
$$\epsilon(\mathbf{w})=\sqrt{\frac{2R(\mathbf{w})\left(\log\left(\frac{1}{\pi(\mathbf{w})}\right)+\log\left(\frac{1}{\delta}\right)\right)}{m}}.$$
Then the relative Chernoff bound states that
$$\mathbb{P}_{S\sim\mathcal{D}^m}\left(\hat{R}(\mathbf{w})\leq R(\mathbf{w})-\epsilon(\mathbf{w})\right)\leq\exp\left(-\frac{m\epsilon(\mathbf{w})^2}{2R(\mathbf{w})}\right)=\delta\pi(\mathbf{w}).$$
Summing over all $\mathbf{w}$ and applying the union bound we conclude that the probability that a $\mathbf{w}$ exists with the property that $R(\mathbf{w})>\hat{R}(\mathbf{w})+\epsilon(\mathbf{w})$ is $\delta$. Therefore, for all $\mathbf{w}$
$$\mathbb{P}_{S\sim\mathcal{D}^m}\left(R(\mathbf{w})\leq\hat{R}(\mathbf{w})+\sqrt{R(\mathbf{w})\left(\frac{2\left(\log\left(\frac{1}{\pi(\mathbf{w})}\right)+\log\left(\frac{1}{\delta}\right)\right)}{m}\right)}\right)\geq1-\delta.$$
Using $\sqrt{ab}=\inf_{\lambda>0}\left(\frac{a}{2\lambda}+\frac{\lambda b}{2}\right)$ we get that
$$\mathbb{P}_{S\sim\mathcal{D}^m}\left(R(\mathbf{w})\leq\hat{R}(\mathbf{w})+\frac{R(\mathbf{w})}{2\lambda}+\frac{\lambda\left(\log\left(\frac{1}{\pi(\mathbf{w})}\right)+\log\left(\frac{1}{\delta}\right)\right)}{m}\right)\geq1-\delta,$$
which upon rearrangement completes the proof. $\square$

</details>

## 2.2 Expected Risk Minimization

In light of Theorem 2.1 it may seem reasonable to want to identify the parameter value $\hat{\mathbf{w}}_{\mathrm{ERM}}$ that minimizes $\hat{R}(\cdot)$. This optimization process is known as Empirical Risk Minimization (ERM) and is formally defined as $$\hat{\mathbf{w}}_{\mathrm{ERM}}=\inf_{\mathbf{W}\in\mathcal{W}}\hat{R}(\mathbf{w}).$$

## 2.3 Compression

We now show how PAC bounds can be used to bound the performance of a compressed neural network. In classical statistical theory only as many parameters as training samples are required to overfit. So in practice, neural networks would be able to overfit the training data as they have many more parameters than training samples. Although overfitting to the training sample will yield a low empirical risk, in practice neural networks do not overfit to the data. This suggests that there is some capacity of the network that is redundant in expressing the learned function. In (Arora, 2018) compression frameworks are constructed that aim to reduce the effective number of parameters required to express the function of a trained network whilst maintaining its performance. To do this (Arora, 2018) capitalize on how a neural network responds to noise added to its weights. We first introduce the compression techniques for linear classifiers and then proceed to work with fully connected ReLU neural networks.

###  2.3.1 Establishing the Notion of Compression

We are in a scenario where we have a learned classifier $h$ that achieves low empirical loss but is complex. In this case, we are considering $\mathcal{Y}=\mathbb{R}^k$ so that the output of $h$ in the $i^\text{th}$ can be thought of as a relative probability that the input belongs to class $i$. With this, we define the classification margin loss for $\gamma>0$ to be $$L_{\gamma}(h)=\mathbb{P}_{(x,y)\sim\mathcal{D}}\left(h(x)[y]\leq\gamma+\max_{i\neq y}f(x)[i]\right).$$ Similarly, we have the empirical classification margin loss defined as $$\hat{L}_{\gamma}(h)=\frac{1}{m}\left\vert\left\{f(x_i)[y_i]\leq\gamma+\max_{i\neq y_i}f(x_i)[i]\right\}\right\vert.$$ Suppose that our neural network has $d$ fully connected layers and let $x^i$ be the vector before the activation at layer $i=0,\dots,d$ and as $x^0$ is the input denote it $x$. Let $A^i$ be the weight matrix of layer $i$ and let layer $i$ have $n_i$ hidden layers with $n=\max_{i=1}^dn_i$. The classifier calculated by the network will be denoted $h_\mathbf{w}(x)$, where $\mathbf{w}$ can be thought of as a vector containing the weights of the network. For layers $i\leq j$ the operator for composition of the layers will be denoted $M^{i,j}$, the Jacobian of the input $x$ will be denoted $J_x^{i,j}$ and $\phi(\cdot)$ will denote the component-wise ReLU. With these the following hold, $$x^i=A^i\phi\left(x^{i-1}\right),\;x^j=M^{i,j}\left(x^i\right),\text{ and }M^{i,j}\left(x^i\right)=J_{x^i}^{i,j}x^i.$$ For a matrix $B$, $\Vert B\Vert_F$ will be its Frobenius norm, $\Vert B\Vert_2$ its spectral norm and $\frac{\Vert B\Vert_F^2}{\Vert B\Vert_2^2}$ its stable rank.

**Definition 2.3** Let $h$ be a classifier and $G_{\mathcal{W}}=\{g_{\mathbf{w}}:\mathbf{w}\in\mathcal{W}\}$ be a class of classifiers. We say that $h$ is $(\gamma,S)$-compressible via $G_{\mathcal{W}}$ if there exists $\mathbf{w}\in\mathcal{W}$ such that for any $x\in\mathcal{X}$, $$\left\vert h(x)[y]-g_{\mathbf{w}}(x)[y]\right\vert\leq\gamma$$ for all $y\in\{1,\dots,k\}$.
 

**Definition 2.4** Suppose $G_{\mathcal{W},s}=\{g_{\mathbf{w},s}:\mathbf{w}\in\mathcal{W}\}$ is a class of classifiers indexed by trainable parameters $\mathbf{w}$ and fixed string $s$. A classifier $h$ is $(\gamma,S)$-compressible with respect to $G_{\mathcal{W},s}$ using helper string $s$ if there exists $\mathbf{w}\in\mathcal{W}$ such that for any $x\in \mathcal{X}$, $$\vert h(x)[y]-g_{\mathbf{w},s}(x)[y]\vert\leq\gamma$$ for all $y\in\{1,\dots,k\}$.
 
**Theorem 2.5** Suppose $G_{\mathcal{W},s}=\{g_{\mathbf{w},s}:\mathbf{w}\in\mathcal{W}\}$ where $\mathbf{w}$ is a set of $q$ parameters each of which has at most $r$ discrete values and $s$ is a helper string. Let $S$ be a training set with $m$ samples. If the trained classifier $h$ is $(\gamma,S)$-compressible via $G_{\mathcal{W},s}$ with helper string $s$, then there exists $\mathbf{w}\in\mathcal{W}$ with high probably such that $$L_0(g_{\mathbf{w}})\leq\hat{L}_{\gamma}(h)+O\left(\sqrt{\frac{q\log(r)}{m}}\right)$$ over the training set.
<details>
<summary>Proof</summary>
<br>

For $\mathbf{w}\in\mathcal{W}$, the empirical classification margin $\hat{L}_0(g_{\mathbf{w}})$ is the average of $m$ $\mathrm{i.i.d}$ Bernoulli random variables with parameter $L_0(g_{\mathbf{w}})$. Therefore, by Chernoff bound
$$\mathbb{P}_{S\sim\mathcal{D}^m}\left(L_0(g_{\mathbf{w}})-\hat{L}_0(g_{\mathbf{w}})\geq\tau\right)\leq\exp\left(-2\tau^2m\right).$$
Therefore, with $\tau=\sqrt{\frac{q\log(r)}{m}}$ we have that
$$\mathbb{P}_{S\sim\mathcal{D}^m}\left(L_0(g_{\mathbf{w}})\leq\hat{L}_0(g_{\mathbf{w}})+\tau\right)\geq1-\exp(-2q\log(r)).$$
As there are only $r^q$ different $\mathbf{w}$, we can apply a union bound arguements to conclude that for all $\mathbf{w}\in\mathcal{W}$ we have that
$$\mathbb{P}_{S\sim\mathcal{D}^m}\left(L_0(g_{\mathbf{w}})\leq\hat{L}_0(g_{\mathbf{w}})+\sqrt{\frac{q\log(r)}{m}}\right)\geq1-\exp(-q\log(r)).$$
As $h$ is $(\gamma,S)$-compressible via $G_{\mathcal{W},S}$ then there exists a $\mathbf{w}\in\mathcal{W}$ such that for any $x\in\mathcal{X}$ and any $y$ we have
$$\vert h(x)[y]-g_{\mathbf{w}}(x)[y]\vert\leq\gamma.$$
Therefore, as long as $h$ has a margin at least $\gamma$ the classifier $g_{\mathbf{w}}$ classifies the examples correctly so that
$$\hat{L}_0(g_{\mathbf{w}})\leq\hat{L}_{\gamma}(h).$$
Combining this with the previous observations completes the proof of the theorem. $\square$


</details>
 
**Remark 2.6** Theorem 2.5 only gives a bound for $g_{\mathbf{w}}$ which is the compression of $h$. However, there are no requirements on the hypothesis class, assumptions are only made on $h$ and its properties on a finite training set.
  
**Corollary 2.7** If the compression works for $1-\xi$ fraction of the training sample, then with a high probability $$L_0(g_{\mathbf{w}})\leq\hat{L}_{\gamma}(h)+\xi+O\left(\sqrt{\frac{q\log r}{m}}\right).$$
<details>
<summary>Proof</summary>
<br>

The proof this corollary proceeds in exactly the same ways as the proof of Theorem 2.5, however, in the last step we can use the upper-bound
$$\hat{L}_0(g_{\mathbf{w}})\leq\hat{L}_{\gamma}(h)+\xi.$$
Which arises as for the fraction of the training sample where the compression doesn't work we assume that the loss is maximized, which was assumed to be $1$. 

</details>
 
###  2.3.2 Compression of a Linear Classifier

We now develop an algorithm to compress the decision vector of a linear classifier. We will use linear classifiers to conduct binary classification, where the members of one class have label $1$ and the others have label $-1$. The linear classifiers will be parameterized by the weight vector $\mathbf{w}\in\mathbb{R}^d$ such that for $x\in\mathcal{X}$ we have $h_{\mathbf{w}}(x)=\mathrm{sgn}(\mathbf{w}^\top x)$. Define the margin, $\gamma>0$, of $\mathbf{w}$ to be such that $y\left(\mathbf{w}^\top x\right)\geq\gamma$ for all $(x,y)$ in the training set. In compressing $\mathbf{w}$, according to Algorithm 1, we end up with a linear classifier parameterized by the weight vector $\hat{\mathbf{w}}$ with some PAC bounds relating to its performance.

<font size="3"> **Algorithm 1 $(\gamma,\mathbf{w})$**</font>
> **Require** vector $\mathbf{w}$ with $\Vert\mathbf{w}\Vert\leq 1$, $\eta$.\
**Ensure** vector $\hat{\mathbf{w}}$ such that for any fixed vector $\Vert u\Vert\leq 1$, with probability at least $1-\eta$, $\left\vert\mathbf{w}^\top\mathbf{u}-\hat{\mathbf{w}}^\top\mathbf{u}\right\vert\leq\gamma$. Vector $\hat{\mathbf{w}}$ has $O\left(\frac{\log d}{\eta\gamma^2}\right)$ non-zero entries.\
**for** $i=1\to d$ **do**\
----Let $z_i=1$ with probability $p_i=\frac{2w_i^2}{\eta\gamma^2}$ and $0$ otherwise.\
----Let $\hat{w}_i=\frac{z_iw_i}{p_i}$.\
**end for**\
**return** $\hat{\mathbf{w}}$

**Theorem 2.8** For any number of samples $m$, Algorithm 1 generates a compressed vector $\hat{\mathbf{w}}$, such that $$L(\hat{\mathbf{w}})\leq\tilde{O}\left(\left(\frac{1}{\gamma^2m}\right)^{\frac{1}{3}}\right).$$
<details>
<summary>Proof</summary>
<br>

**Lemma 1** Algorithm 1 $(\gamma,\mathbf{w})$ returns a vector $\hat{\mathbf{w}}$ such that for any fixed $u$, with probability $1-\eta$, $\left\vert\hat{\mathbf{w}}^\top u-\mathbf{w}^\top u\right\vert\leq\gamma$. The vector $\hat{\mathbf{w}}$ has at most $O\left(\frac{\log d}{\eta\gamma^2}\right)$ non-zero entries with high probability.
<details>
<summary>Proof</summary>
<br>

By the construction of Algorithm 1 it is clear that for all $i$ we have $\mathbb{E}\left(\hat{w}_i\right)=w_i$. Similarly, we have that
$$\mathrm{Var}\left(\hat{w}_i\right)=2p_i(1-p_i)\frac{w_i^2}{p_i^2}\leq\frac{2c_i^2}{p_i}\leq\eta\gamma^2.$$
Therefore, for $u$ independent of $\hat{\mathbf{w}}$ we have that
$$\mathbb{E}\left(\hat{\mathbf{w}}^\top u\right)=\mathbf{w}^\top c\text{ and }\mathrm{Var}\left(\hat{\mathbf{w}} u^\top\right)\leq\frac{\Vert u\Vert^2}{4}\leq\eta\gamma^2.$$
Therefore, by Chebyshev's inequality we have that
$$\mathbb{P}\left(\left\vert \hat{\mathbf{w}}^\top u-\mathbf{w}^\top u\right\vert\geq\gamma\right)\leq\eta.$$
With the expected number of non-zero entries in $\hat{\mathbf{w}}$ being
$$\sum_{i=1}^dp_i=\frac{2}{\eta\gamma^2}.$$
So by Chernoff bound we conclude that with high probability that the number of non-zero entries is at most
$$O\left(\frac{\log(d)}{\eta\gamma^2}\right),$$
which completes the proof of the lemma.$\square$

</details>


In the discrete case, a similar result holds.

**Lemma 2** Let Algorithm 1 $\left(\frac{\gamma}{2},\mathbf{w}\right)$ return vector $\tilde{\mathbf{w}}$. Let, $$\hat{w}_i=\begin{cases}0&\vert\tilde{w}_i\vert\geq2\eta\gamma\sqrt{h}\\\text{rounding to nearest multiple of }\frac{\gamma}{2\sqrt{h}}&\text{Otherwise.}\end{cases}$$ Then for any fixed $u$ with probability at least $1-\eta$, $\left\vert\hat{\mathbf{w}}^\top u-\mathbf{w}^\top u\right\vert\leq\gamma.$
<details>
<summary>Proof</summary>
<br>

Let $\mathbf{w}^\prime$ be the vector where
$$w^\prime_i=\begin{cases}w_i&\vert w_i\vert\geq\frac{\gamma}{4\sqrt{h}}\\0&\text{otherwise.}\end{cases}$$
Then $\left\Vert \mathbf{w}^\prime-\mathbf{w}\right\Vert\leq\frac{\gamma}{4}$. Note that $\left\vert\tilde{\mathbf{w}}_i\right\vert\geq2\eta\gamma\sqrt{d}$ if and only if $\left\vert\mathbf{w}_i\right\vert\leq\frac{\gamma}{4\sqrt{d}}$ and also note that $\left\Vert\hat{\mathbf{w}}-\tilde{\mathbf{w}}\right\Vert\leq\frac{\gamma}{4}$. Combining these observations gives
$$\begin{align*}\left\vert\hat{\mathbf{w}}^\top u-\mathbf{w}^\top u\right\vert&\leq\left\vert\hat{\mathbf{w}}^\top u-\tilde{\mathbf{w}}^\top u\right\vert+\left\vert\tilde{\mathbf{w}}^\top u-(\mathbf{w}^\prime)^\top u\right\vert+\left\vert(\mathbf{w}^\prime)^\top u-\mathbf{w}^\top u\right\vert\\&\leq\frac{\gamma}{4}+\frac{\gamma}{2}+\frac{\gamma}{4}=\gamma,\end{align*}$$
which completes the proof of the lemma.$\square$

</details>

Now choose $\eta=\left(\frac{1}{\gamma^2m}\right)^{\frac{1}{3}}$. By Lemma 1 and Lemma 2 we know that Algorithm 1 works with probability $1-\eta$ and has at most $\tilde{O}\left(\frac{\log(d)}{\eta\gamma^2}\right)$ parameters. Using Corollary 2.7 we know that
$$L\left(\hat{\mathbf{w}}\right)\leq\tilde{O}\left(\eta+\sqrt{\frac{1}{\eta\gamma^2m}}\right)\leq\tilde{O}\left(\left(\frac{1}{\gamma^2m}\right)^{\frac{1}{3}}\right)$$
which completes the proof of the theorem.

</details>

**Remark 2.9** The rate is not optimal as it depends on $m^{1/3}$ and not $\sqrt{m}$. To resolve this we employ helper strings.
  
<font size="3"> **Algorithm 2 $(\gamma,\mathbf{w})$**</font>
> **Require:** vector $\mathbf{w}$ with $\Vert\mathbf{w}\Vert\leq 1$, $\eta$.\
**Ensure** vector $\hat{\mathbf{w}}$ such that for any fixed vector $\Vert u\Vert\leq 1$, with probability at least $1-\eta$, $\vert \mathbf{w}^\top u-\hat{\mathbf{w}}^\top u\vert\leq\gamma$.\
Let $k=\frac{16\log\left(\frac{1}{\eta}\right)}{\gamma^2}$.\
Sample the random vectors $v_1,\dots,v_k\sim\mathcal{N}(0,I)$.\
Let $z_i=\langle v_i,\mathbf{w}\rangle$.\
(In Discrete Case) Round $z_i$ to closes multiple of $\frac{\gamma}{2\sqrt{dk}}$.\
**return** $\hat{\mathbf{w}}=\frac{1}{k}\sum_{i=1}^kz_iv_i$

**Remark 2.10** The vectors $v_i$ of Algorithm 2 form the helper string.

**Theorem 2.11** For any number of sample $m$, Algorithm 2 with the helper string generates a compressed vector $\hat{\mathbf{w}}$, such that $$L(\hat{\mathbf{w}})\leq\tilde{O}\left(\sqrt{\frac{1}{\gamma^2m}}\right).$$

<details>
<summary>Proof</summary>
<br>

**Lemma 1** For any fixed vector $u$, Algorithm 2 $(\gamma, \mathbf{w})$ returns a vector $\hat{\mathbf{w}}$ such that with probability at least $1-\eta$, we have $\left\vert\hat{\mathbf{w}}^\top u-\mathbf{w}^\top u\right\vert\leq\gamma$.
<details>
<summary>Proof</summary>
<br>

Observe that
$$\hat{\mathbf{w}}^\top u=\frac{1}{k}\sum_{i=1}^k\langle v_i,\mathbf{w}\rangle\langle v_i,u\rangle.$$
Where,
$$\mathbb{E}\left(\langle v_i,\mathbf{w}\rangle\langle v_i,u\rangle\right)=\mathbb{E}\left(\mathbf{w}^\top v_iv_i^\top u\right)=\mathbf{w}^\top\mathbb{E}\left(v_iv_i^\top\right)=\mathbf{w}^\top u$$
and
$$\mathrm{Var}\left(\hat{\mathbf{w}}^\top u\right)\leq O\left(\frac{1}{k}\right)\leq O\left(\frac{\gamma}{\sqrt{\log(m)}}\right).$$
Therefore, by standard concentration inequalities we have that
$$\mathbb{P}\left(\left\vert\hat{\mathbf{w}}^\top u-\mathbf{w}^\top u\right\vert\geq\frac{\gamma}{2}\right)\leq\exp\left(\frac{-\gamma^2k}{16}\right)\leq\eta.$$
Hence, with high probability the vector after discretization can only change by at most $\frac{\gamma}{2}$, which completes the proof. $\square$

</details>

Choosing $\eta=\frac{1}{m}$ and applying Lemma 1 we see that with probability $1-\eta$, the compressed vector has at most
$$O\left(\frac{\log(m)}{\gamma^2}\right)$$
parameters. So by Corollary 2.7 we know that
$$L\left(\mathbf{w}\right)\leq\tilde{O}\left(\eta+\sqrt{\frac{1}{\gamma^2m}}\right)\leq\tilde{O}\left(\sqrt{\frac{1}{\gamma^2m}}\right)$$
which completes the proof of the theorem. $\square$

</details>
 

### 2.3.3 Compression of a Fully Connected Network

In a similar way, the layer matrices of a fully connected network can be compressed in such a way as to maintain a reasonable level of performance. A similar compression algorithm on how to do this is detailed in Algorithm 3. Throughout we will let $\mathbf{w}$ parameterize our classifier. It can just be thought of as a list of layer matrices for our neural network.

<font size="3"> **Algorithm 3 $(A,\epsilon,\eta)$**</font>
> **Require** Layer matrix $A\in\mathbb{R}^{n_1\times n_2}$, error parameters $\epsilon,\eta$.\
**Ensure** Returns $\hat{A}$ such that for all vectors $\mathbf{u},\mathbf{v}$ we have that $$\mathbb{P}\left(\left\vert\mathbf{u}^\top\hat{A}\mathbf{v}-\mathbf{u}^\top A\mathbf{v}\right\vert\geq\epsilon\Vert A\Vert_F\Vert\mathbf{u}\Vert\Vert\mathbf{v}\Vert\right)\leq\eta$$\
Sample $k=\frac{\log(\frac{1}{\eta})}{\epsilon^2}$ random matrices $M_1,\dots,M_k$ with $\mathrm{i.i.d}$ entries $\pm1$.\
**for** $l=1\to k$ **do**\
----Let $Z_{l}=\langle A,M_{l}\rangle M_{l}$\
**end for**\
**return** $\hat{A}=\frac{1}{k}\sum_{l=1}^kZ_{l}$


**Definition 2.12** If $M$ is a mapping from real-valued vectors to real-valued vectors, and $\mathcal{N}$ is a noise distribution. Then the noise sensitivity of $M$ at $\mathbf{x}$ with respect to $\mathcal{N}$ is $$\psi_{\mathcal{N}}(M,\mathbf{x})=\mathbb{E}\left(\frac{\Vert M(\mathbf{x}+\eta\Vert\mathbf{x}\Vert)-M(\mathbf{x})\Vert^2}{\Vert M(\mathbf{x})\Vert^2}\right),$$ and $\psi_{\mathcal{N},S}(M)=\max_{x\in S}\psi_{\mathcal{N}}(M,\mathbf{x})$.

**Remark 2.13** When $\mathbf{x}\neq\mathbf{0}$ and the noise distribution is the Gaussian distribution $\mathcal{N}(0,I)$ then the noise sensitivity of matrix $M$ is exactly $\frac{\Vert M\Vert_F^2}{\Vert Mx\Vert^2}$. Hence, it is at most the stable rank of $M$.
  
**Definition 2.14** The layer cushion of layer $i$ is defined as the largest $\mu_i$ such that for any $x\in\mathcal{X}$ qw have $$\mu_i\left\Vert A^i\right\Vert_F\left\Vert\phi\left(x^{i-1}\right)\right\Vert\leq\left\Vert A^i\phi\left(x^{i-1}\right)\right\Vert.$$
 
**Remark 2.15** Note that $\frac{1}{\mu_i^2}$ is equal to the noise sensitivity of $A^i$ at $\phi\left(x^{i-1}\right)$ with respect to noise $\eta\sim\mathcal{N}(0,I)$.
  
**Definition 2.16** For layers $i\leq j$ the inter-layer cushion $\mu_{i,j}$ is the largest number such that $$\mu_{i,j}\left\Vert J_{x^i}^{i,j}\right\Vert_F\left\Vert x^i\right\Vert\leq\left\Vert J_{x^i}^{i,j}x^i\right\Vert$$ for any $x\in\mathcal{X}$. Furthermore, let $\mu_{i\to}=\min_{i\leq j\leq d}\mu_{i,j}$.
 
**Remark 2.17** Note that $J_{x^i}^{i,i}=I$ so that $$\frac{\left\Vert J_{x^i}^{i,i}x^i\right\Vert}{\left\Vert J_{x^i}^{i,j}\right\Vert_F\left\Vert x^i\right\Vert}=\frac{1}{\sqrt{h^i}}.$$ Furthermore, $\frac{1}{\mu_{i,j}^2}$ is the noise sensitivity of $J_x^{i,j}$ with respect to noise $\eta\sim\mathcal{N}(0,I)$.
  
**Definition 2.18** The activation contraction $c$ is defined as the smallest number such that for any layer $i$ $$\left\Vert\phi\left(x^i\right)\right\Vert\geq\frac{\left\Vert x^i\right\Vert}{c}$$ for any $x\in\mathcal{X}$.
 
**Definition 2.19** Let $\eta$ be the noise generated as a result of applying Algorithm 3 to some of the layers before layer $i$. Define the inter-layer smoothness $\rho_{\delta}$ to be the smallest number such that with probability $1-\delta$ and for layers $i<j$ we have that $$\left\Vert M^{i,j}\left(x^i+\eta\right)-J_{x^i}^{i,j}\left(x^i+\eta\right)\right\Vert\leq\frac{\Vert\eta\Vert\left\Vert x^j\right\Vert}{\rho_{\delta}\left\Vert x^i\right\Vert}$$ for any $x\in\mathcal{X}$.
 
**Remark 2.20** For a neural network let $x$ be the input, $A$ be the layer matrix and $U$ the Jacobian of the network output with respect to the layer input. Then the network output before compression is given by $UAx$ and after compression the output is given by $U\hat{A}x$.

**Theorem 2.21** For any fully connected network $h_{{\mathbf{w}}}$ with $\rho_{\delta}\geq3d$, any probability $0<\delta\leq 1$ and any margin $\gamma$. Algorithm 3 generates weights $\tilde{{\mathbf{w}}}$ such that with probability $1-\delta$ over the training set, $$L_0(h_{\tilde{{\mathbf{w}}}})\leq\hat{L}_{\gamma}(h_{\mathbf{w}})+\tilde{O}\left(\sqrt{\frac{c^2d^2\max_{x\in S}\Vert h_{\mathbf{w}}(x)\Vert_2^2\sum_{i=1}^d\frac{1}{\mu_i^2\mu_{i\to}^2}}{\gamma^2m}}\right).$$
<details>
<summary>Proof</summary>
<br>

**Lemma 1** For any $0<\delta$ and $\epsilon\leq 1$ let $G=\left\{\left(U^i,x^i\right)\right\}_{i=1}^m$ be a set of matrix-vector pairs of size $m$ where $U\in\mathbb{R}^{k\times n_1}$ and $x\in\mathbb{R}^{n_2}$, let $\hat{A}\in\mathbb{R}^{n_1\times n_2}$ be the output of Algorithm 3 $\left(A,\epsilon,\eta=\frac{\delta}{mk}\right)$. With probability at least $1-\delta$ we have for any $(U,x)\in G$ that $\left\Vert U(\hat{A}-A)x\right\Vert\leq\epsilon \Vert A\Vert_F\Vert U\Vert_F\Vert x\Vert$.
<details>
<summary>Proof</summary>
<br>

For fixed vectors $u,v$ we have that
$$u^\top\hat{A}v=\frac{1}{k}\sum_{l=1}^ku^\top Z_lv=\frac{1}{k}\sum_{l=1}^k\langle A,M_l\rangle\left\langle uv^\top,M_l\right\rangle.$$
By standard concentration inequalities we deduce that
$$\mathbb{P}\left(\left\vert\frac{1}{k}\sum_{l=1}^k\langle A,M_l\rangle\left\langle uv^\top,M_l\right\rangle-\left\langle A,uv^\top\right\rangle\right\vert\geq\epsilon\Vert A\Vert_F\left\Vert uv^\top\right\Vert_F\right)\leq\exp\left(-k\epsilon^2\right).$$
Therefore, for the choice of $k$ from Algorithm 3 we know that
$$\mathbb{P}\left(\left\vert u^\top\hat{A}v-u^\top Av\right\vert\geq\epsilon\Vert A\Vert_F\left\Vert u\right\Vert\left\Vert v\right\Vert\right)\leq\eta.$$
Let $(U,x)\in G$ and $u_i$ be the $i^\text{th}$ row of $U$. We can apply the above result with a union bound to get that
$$\mathbb{P}\left(\left\vert u_i^\top\hat{A}v-u_i^\top Av\right\vert\leq\epsilon\Vert A\Vert_F\left\Vert u_i\right\Vert\left\Vert v\right\Vert\right)\geq1-\delta$$
for all $i$ simultaneously. Furthermore,
$$\left\Vert U\left(\hat{A}-A\right)x\right\Vert^2=\sum_{i=1}^n\left(u_i^\top\left(\hat{A}-A\right)x\right)^2,\text{ and }\Vert U\Vert_F^2=\sum_{i=1}^n\Vert u_i\Vert^2$$
we see that with probability at least $1-\delta$ we have
$$\begin{align*}\left\Vert U(\hat{A}-A)x\right\Vert^2&=\sum_{i=1}^n\left(u_i^\top\left(\hat{A}-A\right)x\right)^2\\&\leq\sum_{i=1}^n\epsilon^2\Vert A\Vert_F^2\Vert u_i\Vert^2\Vert x\Vert\\&=\epsilon^2\Vert A\Vert_F^2\Vert U\Vert^2\Vert x\Vert\end{align*}$$
which completes the proof of the lemma. $\square$
</details>

**Lemma 2** For any fully connected network $h_{\mathbf{w}}$ with $\rho_{\delta}\geq 3d$, any probability $0<\delta\leq 1$ and any $0<\epsilon\leq 1$, Algorithm 3 can generate weights $\tilde{\mathbf{w}}$ for a network with $$\frac{72c^2d^2\log\left(\frac{mdn}{\delta}\right)}{\epsilon^2}\cdot\sum_{i=1}^d\frac{1}{\mu_i^2\mu_{i\to}^2}$$ total parameters such that with probability $1-\frac{\delta}{2}$ over the generated weights $\tilde{w}$, for any $x\in\mathcal{X}$ $$\left\Vert h_{\mathbf{w}}(x)-h_{\tilde{w}}(x)\right\Vert\leq\epsilon\Vert h_{\mathbf{w}}(x)\Vert,$$ where $\mu_i,\mu_{i\to},c$ and $\rho_{\delta}$ are the layer cushion, inter-layer cushion, activation contraction and inter-layer smoother for the network.
<details>
<summary>Proof</summary>
<br>

The proof of this lemma proceeds by induction. For $i\geq0$ let $\hat{x}_i^j$ be the output at layer $j$ if weights $A^1,\dots,A^i$ are replaced with $\tilde{A}^1,\dots,\tilde{A}^i$. We want to show for any $i$ if $j\geq i$ then
$$\mathbb{P}\left(\left\Vert\hat{x}_i^j-x^j\right\Vert\leq\frac{i}{d}\epsilon\left\Vert x^j\right\Vert\right)\geq1-\frac{i\delta}{2d}.$$
For $i=0$ the result is clear as the weight matrices are unchanged. Suppose the result holds true for $i-1$. Let $\hat{A}^i$ be the result of applying Algorithm 3 to $A^i$ with $\epsilon_i=\frac{\epsilon\mu_i\mu_{i\to}}{4cd}$ and $\eta=\frac{\delta}{6d^2h^2m}$. Consider the set 
$$G=\left\{\left(J_{x^i}^{i,j}\right):x\in\mathcal{X},j\geq i\right\}$$
and let $\Delta^i=\hat{A}^i-A^i$. Note that
$$\left\Vert\hat{x}_i^j-x^j\right\Vert\leq\left\Vert\hat{x}_i^j-\hat{x}_{i-1}^j\right\Vert+\left\Vert\hat{x}_{i-1}^j-x^j\right\Vert.$$
The second term is bounded by $\frac{(i-1)\epsilon\left\Vert x^j\right\Vert}{d}$ by inductive assumption. Therefore, it suffices to show that the first term is bounded by $\frac{\epsilon}{d}$ to complete the inductive step. First observe that,
$$\left\Vert\hat{x}_i^j-\hat{x}_{i-1}^j\right\Vert\leq\left\Vert J_{x^i}^{i,j}\left(\Delta^i\phi\left(\hat{x}^{i-1}\right)\right)\right\Vert+\left\Vert M^{i,j}\left(\hat{A}^i\phi\left(\hat{x}^{i-1}\right)\right)-M^{i,j}\left(A^i\phi\left(\hat{x}^{i-1}\right)\right)-J_{x^i}^{i,j}\left(\Delta^i\phi\left(\hat{x}^{i-1}\right)\right)\right\Vert.$$
The first term can be bounded as follows
$$\begin{align*}\left\Vert J_{x^i}^{i,j}\left(\Delta^i\phi\left(\hat{x}^{i-1}\right)\right)\right\Vert&\leq\frac{\epsilon\mu_i\mu_{i\to}}{6cd}\left\Vert J_{x^i}^{i,j}\right\Vert\left\Vert A^i\right\Vert_F\left\Vert\phi\left(\hat{x}^{i-1}\right)\right\Vert\quad\text{Lemma 2}\\&\leq\frac{\epsilon\mu_i\mu_{i\to}}{6cd}\left\Vert J_{x^i}^{i,j}\right\Vert\left\Vert A^i\right\Vert_F\left\Vert\hat{x}^{i-1}\right\Vert\quad\text{Lipschitz of $\phi$}\\&\leq\frac{\epsilon\mu_i\mu_{i\to}}{3cd}\left\Vert J_{x^i}^{i,j}\right\Vert\left\Vert A^i\right\Vert_F\left\Vert x^{i-1}\right\Vert\quad\text{ Inductive Assumption}\\&\leq\frac{\epsilon\mu_{i\to}}{3d}\left\Vert J_{x^i}^{i,j}\right\Vert\left\Vert A^i\phi\left(x^{i-1}\right)\right\Vert\\&\leq\frac{\epsilon\mu_{i\to}}{3d}\left\Vert J_{x^i}^{i,j}\right\Vert\left\Vert x^i\right\Vert\\&\leq\frac{\epsilon}{3d}\left\Vert x^j\right\Vert.\end{align*}$$
The second term can be split as
$$\left\Vert\left(M^{i,j}-J_{x^i}^{i,j}\right)\left(\hat{A}^i\phi\left(\hat{x}^{i-1}\right)\right)\right\Vert+\left\Vert\left(M^{i,j}-J_{x^i}^{i,j}\right)\left(A^i\phi\left(\hat{x}^{i-1}\right)\right)\right\Vert,$$
which can be bounded by inter-layer smoothness. By inductive assumption
$$\left\Vert A^i\phi\left(\hat{x}^{i-1}\right)-x^i\right\Vert\leq\frac{(a-1)\epsilon\left\Vert x^i\right\Vert}{d}\leq\epsilon\left\Vert x^i\right\Vert.$$
Then by inter-layer smoothness
$$\left\Vert\left(M^{i,j}-J_{x^i}^{i,j}\right)\left(A^i\phi\left(\hat{x}^{i-1}\right)\right)\right\Vert\leq\frac{\left\Vert x^b\right\Vert\epsilon}{\rho_{\delta}}\leq\frac{\epsilon}{3d}\left\Vert x^j\right\Vert.$$
On the other hand,
$$\left\Vert\hat{A}^i\phi\left(\hat{x}^{i-1}\right)-x^i\right\Vert\leq\left\Vert A^i\phi\left(\hat{x}^{i-1}\right)-x^i\right\Vert+\left\Vert\Delta^i\phi\left(\hat{x}^{i-1}\right)\right\Vert\leq\frac{(i-1)\epsilon}{d}+\frac{\epsilon}{3d}\leq\epsilon$$
so that
$$\left\Vert\left(M^{i,j}-J_{x^i}^{i,j}\right)\left(A^i\phi\left(\hat{x}^{i-1}\right)\right)\right\Vert\leq\frac{\epsilon}{3d}\left\Vert x^j\right\Vert.$$
This completes the inductive step and the proof of the lemma. $\square$

</details>

**Lemma 3** For any fully connected network $h_{\mathbf{w}}$ with $\rho_{\delta}\geq 3d$, any probability $0<\delta\leq 1$ and any margin $\gamma>0$, $h_{\mathbf{w}}$ can be compressed (with respect to a random string) to another fully connected network $h_{\mathbf{w}}$ such that for $x\in\mathcal{X}$, $\hat{L}_0(h_{\tilde{\mathbf{w}}})\leq\hat{L}_{\gamma}(h_\mathbf{w})$ and the number of parameters in $h_{\tilde{\mathbf{w}}}$ is at most $$\tilde{O}\left(\frac{c^2d^2\max_{x\in\mathcal{X}}\Vert h_{\mathbf{w}}(x)\Vert_2^2}{\gamma^2}\sum_{i=1}^d\frac{1}{\mu_i^2\mu_{i\to}^2}\right).$$
<details>
<summary>Proof</summary>
<br>

In the first case suppose that $\gamma^2>2\max_{x\sin\mathcal{X}}\left\Vert h_{\mathbf{w}}(x)\right\Vert_2^2$, then
$$\left\vert h_{\mathbf{w}}(x)[y]-\max_{i\neq y}h_{\mathbf{w}}(x)[i]\right\vert^2\leq 2\max_{x\in\mathcal{X}}\left\Vert h_{\mathbf{w}}(x)\right\Vert_2^2\leq\gamma^2.$$
Therefore, the margin can be at most $\gamma$ which implies that $\hat{L}_{\gamma}(h_{\mathbf{w}})=1$ and so the statement holds in this case. If instead $\gamma^2\leq2\max_{x\in\mathcal{X}}\left\Vert h_{\mathbf{w}}(x)\right\Vert_2^2$, then setting $\epsilon=\frac{\gamma^2}{2\max_{x\in\mathcal{X}}\left\Vert h_{\mathbf{w}}(x)\right\Vert_2^2}$ in Lemma 2 we conclude that
$$\left\Vert h_{\mathbf{w}}(x)-h_{\tilde{\mathbf{w}}}(x)\right\Vert\leq\frac{\gamma}{\sqrt{2}}$$
for any $x\in\mathcal{X}$. If $\hat{L}_{\gamma}(h_{\mathbf{w}})=1$ then clearly $\hat{L}_0(h_{\mathbf{w}})\leq1=\hat{L}_{\gamma}(h_{\mathbf{w}})$. Suppose instead that $\hat{L}_{\gamma}(h_{\mathbf{w}})<1$. Then there exists an $(x,y)\in\mathcal{Z}$ such that $h_{\mathbf{w}}(x)[y]>\max$

</details>
 

**Lemma 4** For any matrix $A$ let $\hat{A}$ be the truncated version of $A$ where singular values that are smaller than $\delta\left\Vert A\right\Vert_2$.Let $h_{\mathbf{w}}$ be a $d$-layer network with weights $A=\left\{A^1,\dots,A^d\right\}$. Then for any input $x$, weights $A$ and $\hat{A}$, if for any layer $i$, $\left\Vert A^i-\hat{A}^i\right\Vert\leq\frac{1}{d}\Vert A^i\Vert$, then $$\Vert h_\mathbf{w}(x)-h_{\hat{\mathbf{w}}}(x)\Vert\leq e\Vert x\Vert\left(\prod_{i=1}^d\left\Vert A^i\right\Vert_2\right)\sum_{i=1}^d\frac{\left\Vert A^i-\hat{A}^i\right\Vert_2}{\Vert A^i\Vert_2}$$

We can assume without loss of generality that for any $i\neq j$ that
$$\Vert A_i\Vert_F=\Vert A_j\Vert_F=\beta.$$
Therefore, for any $x\in\mathcal{X}$ we have
$$\beta^d=\prod_{i=1}^d\left\Vert A^i\right\Vert_F\leq\frac{c\left\Vert x^1\right\Vert}{\Vert x\Vert\mu_1}\prod_{i=2}^d\left\Vert A^i\right\Vert_F\leq\dots\leq\frac{c^d\left\Vert h_{\mathbf{w}}(x)\right\Vert}{\Vert x\Vert\prod_{i=1}^d\mu_i}.$$
By Lemma 2 we know that $\left\Vert\tilde{A}^i\right\Vert_F\leq\beta\left(1+\frac{1}{d}\right)$. As 
$$\tilde{A}^i=\frac{1}{k}\sum_{l=1}^k\left\langle A^i,M_l\right\rangle M_{l},$$
if $\hat{A}^i$ are the approximations of $\tilde{A}^i$ with accuracy $\mu$ then
$$\left\Vert\hat{A}^i-\tilde{A}^i\right\Vert_F\leq\sqrt{k} h\nu\leq\sqrt{q}h\nu$$
where $q$ is the total number of parameters. Therefore, by Lemma 4 we have that
$$\begin{align*}\left\vert l_{\gamma}(h_{\tilde{w}}(x),y)-l_{\gamma}(h_{\hat{\mathbf{w}}}(x),y)\right\vert&\leq\frac{2e}{\gamma}\Vert x\Vert\left(\prod_{i=1}^d\left\Vert\tilde{A}^i\right\Vert\right)\sum_{i=1}^d\frac{\left\Vert\tilde{A}^i-\hat{A}^i\right\Vert_F}{\left\Vert\tilde{A}^i\right\Vert_F}\\&\leq\frac{e^2}{\gamma}\Vert x\Vert\beta^{d-1}\sum_{i=1}^d\left\Vert \tilde{A}^i-\hat{A}^i\right\Vert_F\\&\leq\frac{e^2c^d\left\Vert h_{\mathbf{w}}(x)\right\Vert\sum_{i=1}^d\left\Vert\tilde{A}^i-\hat{A}^i\right\Vert_F}{\gamma\beta\prod_{i=1}^d\mu_i}\\&\leq\frac{qh\nu}{\beta},\quad\text{By Lemma }10:\frac{e^2d\left\Vert h_{\mathbf{w}}(x)\right\Vert}{\gamma\beta\prod_{i=1}^d\mu_i}\leq\sqrt{q}.\end{align*}$$
The absolute value of each parameter in layer $i$ is at most $\beta h$, therefore, to get an $\epsilon$-cover the logarithm of the number of choices for each parameter is $\log\left(\frac{kh^2}{\epsilon}\right)\leq2\log\left(\frac{kh}{\epsilon}\right)$ giving a covering number of
$$2q\log\left(\frac{kh}{\epsilon}\right).$$
Bounding the Rademacher complexity using Dudley entropy integral completes the proof.

</detail>