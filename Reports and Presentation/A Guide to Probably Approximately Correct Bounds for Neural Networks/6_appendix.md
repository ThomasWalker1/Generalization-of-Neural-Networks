# Rademacher Complexity

Recall, that we have the space $\mathcal{Z}$ on which a distribution $\mathcal{D}$ is defined from which we draw an $\mathrm{i.i.d}$ sampled $S=\{(x_i,y_i)\}_{i=1}^m$. Suppose we have a class of functions $\mathcal{F}=\{f:\mathcal{Z}\to\mathbb{R}\}$.

**Definition 6.1** (Balcan, 2011) The empirical Rademacher complexity of $\mathcal{F}$ is $$\hat{\mathfrak{R}}(\mathcal{F})=\mathbb{E}_{\sigma\in\{\pm1\}}\left(\sup_{f\in\mathcal{F}}\left(\frac{1}{m}\sum_{i=1}^m\sigma_if((x_i,y_i))\right)\right),$$ where each $\sigma_i$ is an independent random variable uniformly distribution on $\{\pm1\}$.

**Definition 6.2** (Balcan, 2011) The Rademacher complexity of $\mathcal{F}$ is $$\mathfrak{R}(\mathcal{F})=\mathbb{E}_{S\sim\mathcal{D}^m}\left(\hat{\mathfrak{R}}(\mathcal{F})\right).$$

**Theorem 6.3** (Balcan, 2011) For a parameter $\delta\in(0,1)$ if $\mathcal{F}\subseteq\{f:\mathcal{Z}\to[0,1]\}$ then $$\mathbb{P}_{S\sim\mathcal{D}^m}\left(\mathbb{E}_{z\sim\mathcal{D}}\left(f(z)\right)\leq\frac{1}{m}\sum_{i=1}^mf(x_i,y_i)+2\mathfrak{R}(\mathcal{F})+\sqrt{\frac{\log\left(\frac{1}{\delta}\right)}{m}}\right)\geq1-\delta,$$ and $$\mathbb{P}_{S\sim\mathcal{D}^m}\left(\mathbb{E}_{z\sim\mathcal{D}}\left(f(z)\right)\leq\frac{1}{m}\sum_{i=1}^mf(x_i,y_i)+2\hat{\mathfrak{R}}(\mathcal{F})+3\sqrt{\frac{\log\left(\frac{2}{\delta}\right)}{m}}\right)\geq1-\delta.$$
<details>
<summary>Proof</summary>
<br>

**Theorem 6.3.1 (McDiarmid Inequality)** Let $x_1,\dots, x_n$ be independent random variables taking values in a set $A$ and let $c_1,\dots, c_n$ be positive real constants. If $\phi:A^n\to\mathbb{R}$ satisfies $$\sup_{x_1,\dots,x_n,x_i^\prime}\left\vert\phi(x_1,\dots,x_i,\dots,x_n)-\phi\left(x_1,\dots,x_i^\prime,\dots,x_n\right)\right\vert\leq c_i,$$ for $1\leq i\leq n$, then $$\mathbb{P}\left(\phi(x_1,\dots,x_n)-\mathbb{E}\left(\phi(x_1,\dots,x_n)\right)\geq\epsilon\right)\leq\exp\left(\frac{-2\epsilon}{\sum_{i=1}^nc_i^2}\right).$$
<details>
<summary>Proof</summary>
<br>

For a proof of this theorem refer to (Scott(b), 2014).

</details>

**Lemma 6.3.2** The function $$\phi(S)=\sup_{h\in\mathcal{F}}\left(\mathbb{E}_{\hat{S}\sim\mathcal{D}^m}\left(h(x,y)\right)-\frac{1}{m}\sum_{i=1}^mh(x_i,y_i)\right)$$ satisfies $$\sup_{z_1,\dots,z_n,z_i^\prime\in\mathcal{Z}}\left\vert\phi(z_1,\dots,z_i,\dots,z_m)-\phi(z_1,\dots,z_i^\prime,\dots,z_m)\right\vert\leq\frac{1}{m}.$$
<details>
<summary>Proof</summary>
<br>

Let $S=\{z_1,\dots,z_m\}$ and $S^\prime=\{z_1,\dots,z_i^\prime,\dots,z_m\}$ then $$\left\vert\phi(S)-\phi\left(S^\prime\right)\right\vert=\left\vert\sup_{h\in\mathcal{F}}\left(\mathbb{E}_{\hat{S}\sim\mathcal{D}^m}\left(h(x,y)\right)-\frac{1}{m}\sum_{(x_j,y_j)\in S}h(x_j,y_j)\right)-\sup_{h\in\mathcal{F}}\left(\mathbb{E}_{\hat{S}\sim\mathcal{D}^m}\left(h(x,y)\right)-\frac{1}{m}\sum_{(x_j,y_j)\in S^\prime}h(x_j,y_j)\right)\right\vert.$$ Let $h^*\in\mathcal{F}$ be the function the maximizes the supremum of $\phi(S)$, then $$\left\vert\phi(S)-\phi\left(S^\prime\right)\right\vert=\left\vert\mathbb{E}_{\hat{S}\sim\mathcal{D}^m}\left(h^*(x,y)\right)-\frac{1}{m}\sum_{(x_j,y_j)\in S}h^*(x_j,y_j)-\sup_{h\in\mathcal{F}}\left(\mathbb{E}_{\hat{S}\sim\mathcal{D}^m}\left(h(x,y)\right)-\frac{1}{m}\sum_{(x_j,y_j)\in S^\prime}h(x_j,y_j)\right)\right\vert$$ and because $h^*$ can at best also maximize $\phi\left(S^\prime\right)$ we also have that $$\begin{align*}\left\vert\phi(S)-\phi\left(S^\prime\right)\right\vert&\leq\left\vert\mathbb{E}_{\hat{S}\sim\mathcal{D}^m}\left(h^*(x,y)\right)-\frac{1}{m}\sum_{(x_j,y_j)\in S}h^*(x_j,y_j)-\mathbb{E}_{\hat{S}\sim\mathcal{D}^m}\left(h^*(x,y)\right)-\frac{1}{m}\sum_{(x_j,y_j)\in S^\prime}h^*(x_j,y_j)\right\vert\\&=\left\vert\frac{1}{m}\sum_{(x_j,y_j)\in S^\prime}h^*(x_j,y_j)-\frac{1}{m}\sum_{(x_j,y_j)\in S}h^*(x_j,y_j)\right\vert.\end{align*}$$ By using the definitions of $S$ and $S^\prime$ this simplifies to $$\begin{align*}\left\vert\phi(S)-\phi\left(S^\prime\right)\right\vert&\leq\frac{1}{m}\left\vert h^*(x_i,y_i)-h^*\left(x_i^\prime,y_i^\prime\right)\right\vert\\&\leq\frac{1}{m},\end{align*}$$ which completes the proof of the lemma. $\square$

</details>

Lemma 6.3.2 shows that $\phi(S)=\sup_{h\in\mathcal{F}}\left(\mathbb{E}_{\hat{S}\sim\mathcal{D}^m}\left(h(x,y)\right)-\frac{1}{m}\sum_{i=1}^mh(x_i,y_i)\right)$ satisfies the conditions of Theorem 6.3.1, therefore, $$\mathbb{P}\left(\phi(S)-\mathbb{E}_{S^\prime\sim\mathcal{D}^m}\left(\phi\left(S^\prime\right)\right)\geq t\right)\leq\exp\left(-\frac{t^2}{m}\right).$$ With $t=\sqrt{\frac{\log\left(\frac{1}{\delta}\right)}{m}}$ we deduce that $$\mathbb{P}_{S\sim\mathcal{D}^m}\left(\mathbb{E}_{\hat{S}\sim\mathcal{D}^m}(f(x,y))\leq\frac{1}{m}\sum_{i=1}^mf(x_i,y_i)+\mathbb{E}_{\hat{S}^\prime\sim\mathcal{D}^m}\left(\phi\left(\hat{S}^\prime\right)\right)\right)\geq1-\delta.$$ Now we need to bound the expectation of $\phi(S)$ using Rademacher complexity to complete the proof. Let $\tilde{S}=\left\{\tilde{z}_1,\dots,\tilde{z}_m\right\}$ be a sample independent but identically distributed to $S$. As $$\mathbb{E}_{\tilde{S}}\left(\frac{1}{m}\sum_{(x,y)\in\tilde{S}}h(x,y)\Bigg\vert S\right)=\mathbb{E}_{z\sim\mathcal{D}}\left(h(z)\right),\text{ and }\;\mathbb{E}_{\tilde{S}}\left(\frac{1}{m}\sum_{(x,y)\in S}h(x,y)\Bigg\vert S\right)=\frac{1}{m}\sum_{(x,y)\in S}h(x,y)$$ we deduce that $$\begin{align*}\mathbb{E}_{S\sim\mathcal{D}^m}\left(\phi(S)\right)&=\mathbb{E}_{S\sim\mathcal{D}^m}\left(\sup_{h\in\mathcal{F}}\left(\mathbb{E}_{\tilde{S}\sim\mathcal{D}^m}\left(\frac{1}{m}\sum_{(x,y)\in\tilde{S}}\left(h(x,y)\right)-\frac{1}{m}\sum_{(x,y)\in S}h(x,y)\Bigg\vert S\right)\right)\right).\end{align*}$$ We can apply Jensen's inequality as $\sup$ is convex to deduce that $$\mathbb{E}_{S\sim\mathcal{D}^m}\left(\sup_{h\in\mathcal{F}}\left(\mathbb{E}_{\tilde{S}\sim\mathcal{D}^m}\left(\frac{1}{m}\sum_{(x,y)\in\tilde{S}}h(x,y)-\frac{1}{m}\sum_{(x,y)\in S}h(x,y)\Bigg\vert S\right)\right)\right)\leq\mathbb{E}_{S\sim\mathcal{D}^m}\mathbb{E}_{\tilde{S}\sim\mathcal{D}^m}\left(\sup_{h\in\mathcal{F}}\left(\frac{1}{m}\sum_{(x,y)\in\tilde{S}}h(x,y)-\frac{1}{m}\sum_{(x,y)\in S}h(x,y)\right)\right).$$ As $\mathbb{E}(\sigma_i)=0$ we can multiply each term by $\sigma_i$, and in distribution we have $-\sigma_i=\sigma_i$ so that $$\begin{align*}\mathbb{E}_{S\sim\mathcal{D}^m}\mathbb{E}_{\tilde{S}\sim\mathcal{D}^m}\left(\sup_{h\in\mathcal{F}}\left(\frac{1}{m}\sum_{(x,y)\in\tilde{S}}h(x,y)-\frac{1}{m}\sum_{(x,y)\in S}h(x,y)\right)\right)&=\mathbb{E}_{\sigma\in\{\pm1\}^m}\mathbb{E}_{S\sim\mathcal{D}^m}\mathbb{E}_{\tilde{S}\sim\mathcal{D}^m}\left(\sup_{h\in\mathcal{F}}\left(\frac{1}{m}\sum_{(x,y)\in\tilde{S},\sigma_i\in\sigma}\sigma_ih(x,y)-\frac{1}{m}\sum_{(x,y)\in S,\sigma_i\in\sigma}\sigma_ih(x,y)\right)\right)\\&\leq\mathbb{E}_{\sigma\in\{\pm1\}^m}\mathbb{E}_{S\sim\mathcal{D}^m}\left(\sup_{h\in\mathcal{F}}\left(\frac{1}{m}\sum_{(x,y)\in S,\sigma_i\in\sigma}\sigma_ih(x,y)\right)\right)+\mathbb{E}_{\sigma\in\{\pm1\}^m}\mathbb{E}_{\tilde{S}\sim\mathcal{D}^m}\left(\sup_{h\in\mathcal{F}}\left(\frac{1}{m}\sum_{(x,y)\in\tilde{S},\sigma_i\in\sigma}\sigma_ih(x,y)\right)\right)\\&=2\mathfrak{R}(\mathcal{F}),\end{align*}$$ which when substituted into our previous bounds completes the proof of the first statement. To obtain the second statement we note that $\hat{\mathfrak{R}}(\mathcal{F})$ satisfies Theorem 6.3.1 with constant $\frac{1}{m}$. Therefore, a second application of Theorem 6.3.1 with confidence level (where a confidence level of $\frac{\delta}{2}$ is used for each application) gives the desired result.

</details>

If we let $\mathcal{F}=\left\{(x,y)\mapsto\mathbb{I}\left(h_{\mathbf{w}}(x)[y]\leq\gamma+\max_{j\neq y}h_{\mathbf{w}}(x)[j]\right):\mathbf{w}\in\mathcal{W}\right\}$ then for any $\delta\in(0,1)$ and $\mathbf{w}\in\mathcal{W}$ we have that $$\mathbb{P}_{S\sim\mathcal{D}^m}\left(L_{\gamma}\left(h_{\mathbf{w}}\right)\leq\hat{L}_{\gamma}(h_{\mathbf{w}})+2\hat{\mathfrak{R}}(\mathcal{F})+3\sqrt{\frac{\log\left(\frac{2}{\delta}\right)}{m}}\right)\geq1-\delta.$$

**Definition 6.4** (Rebeschini, 2022) Given a set $\mathcal{S}$ and a function $\rho:\mathcal{S}\times\mathcal{S}\to\mathbb{R}_+$, we call $(\mathcal{S},\rho)$ a pseudo-metric space if for all $x,y,z\in\mathcal{S}$ we have
- $\rho(x,y)=\rho(y,x)$,
- $\rho(x,z)\leq\rho(x,y)+\rho(y,z)$, and
- $\rho(x,x)=0$.

**Definition 6.5** (Rebeschini, 2022) Let $(\mathcal{S},\rho)$ be a pseudo-metric space and let $\epsilon>0$. Then the set $\mathcal{C}\subseteq\mathcal{s}$ is an $\epsilon$-cover of $(\mathcal{S},\rho)$ if for every $x\in\mathcal{S}$ there is a $y\in\mathcal{C}$ such that $\rho(x,y)\leq\epsilon$. The set $\mathcal{C}$ is a minimal $\epsilon$-cover if there is no other $\epsilon$-cover with lower cardinality. The cardinality of any minimal $\epsilon$-cover is the $\epsilon$-covering number denoted $N(\mathcal{S},\rho,\epsilon)$.

For a given training set $S=\{(x_i,y_i)\}_{i=1}^m$ we can consider the set $$\mathcal{G}=\{(f(x_1,y_1),\dots,f(x_m,y_m)):f\in\mathcal{F}\}.$$

**Theorem 6.6** (Lotz, 2020) Let $\mathcal{F}\subseteq\{f:\mathcal{Z}\to[0,1]\}$ and $S\sim\mathcal{D}^m$ then $$\hat{\mathfrak{R}}(\mathcal{F})\leq\inf_{\epsilon>0}\left(\epsilon+\sqrt{\frac{2N(\mathcal{G},\rho,\epsilon)}{m}}\right).$$
<details>
<summary>Proof</summary>
<br>

**Lemma 6.6.1 (Massart's Lemma)** (Rebeschini, 2022) Let $\mathcal{T}\subseteq\mathbb{R}^n$ then we have that $$\mathfrak{R}(\mathcal{T})\leq\max_{t\in\mathcal{T}}\Vert t\Vert_2\frac{\sqrt{2\log\vert\mathcal{T}\vert}}{n}.$$
<details>
<summary>Proof (Scott(c), 2014)</summary>
<br>



</details>

Let $T\subseteq\mathcal{G}$ be an $\epsilon$-net of size $N(\mathcal{G},\rho,\epsilon)$, then by Lemma 6.6.1 we have that $$\mathbb{E}_{\sigma\in\{\pm1\}^m}\left(\max_{g^\prime\in T}\frac{1}{m}\sigma_ig^\prime(x_i,y_i)\right)\leq\max_{g^\prime\in T}\Vert g(x_i,y_i)\Vert_2\frac{\sqrt{2\log\left(N(\mathcal{G},\rho,\epsilon)\right)}}{m}\leq\sqrt{m}\frac{\sqrt{2\log\left(N(\mathcal{G},\rho,\epsilon)\right)}}{m}=\sqrt{\frac{2\log\left(N(\mathcal{G},\rho,\epsilon)\right)}{m}}.$$ Using this we can conclude that, $$\begin{align*}\hat{\mathfrak{R}}(\mathcal{G})&=\mathbb{E}_{\sigma\in\{\pm1\}^m}\left(\sup_{g\in\mathcal{G}}\left(\frac{1}{m}\sum_{i=1}^m\sigma_ig(x_i,y_i)\right)\right)\\&\leq\mathbb{E}_{\sigma\in\{\pm1\}^m}\left(\sup_{g\in\mathcal{G}}\left(\frac{1}{m}\sum_{i=1}^m\sigma_ig(x_i,y_i)-\sigma_ig^\prime(x_i,y_i)\right)\right)+\mathbb{E}_{\sigma\in\{\pm1\}^m}\left(\frac{1}{m}\sum_{i=1}^m\sigma_ig^\prime(x_i,y_i)\right)\\&\leq\mathbb{E}_{\sigma\in\{\pm1\}^m}\left(\sup_{g\in\mathcal{G}}\left(\frac{1}{m}\sum_{i=1}^m\vert g(x_i,y_i)-g^\prime(x_i,y_i)\vert\right)\right)+\mathbb{E}_{\sigma\in\{\pm1\}^m}\left(\max_{g^\prime\in T}\left(\frac{1}{m}\sum_{i=1}^m\sigma_ig^\prime(x_i,y_i)\right)\right)\\&\leq\sup_{g\in\mathcal{G}}\rho((g(x_1,y_1),\dots,g(x_m,y_m)),(g^\prime(x_1,y_1),\dots,g^\prime(x_m,y_m)))+\sqrt{\frac{2\log\left(N(\mathcal{G},\rho,\epsilon)\right)}{m}}\\&\leq\epsilon+\sqrt{\frac{2\log\left(N(\mathcal{G},\rho,\epsilon)\right)}{m}},\end{align*}$$ which holds for all $\epsilon>0$ which completes the proof of the theorem.

</details>