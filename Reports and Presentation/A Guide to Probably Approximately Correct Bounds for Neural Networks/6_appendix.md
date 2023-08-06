# Rademacher Complexity

Here we utilize the introduction to Rademacher complexity given in (Rebeschini, 2022) to help understand the reasoning behind the proof of Theorem 2.21. Suppose we have independent random variable $\Omega_1,\dots,\Omega_n\in\{\pm1\}$, where $\mathbb{P}(\Omega_i=1)=\frac{1}{2}$. These variables are often referred to as Rademacher random variables and we will use the notation $\Omega=(\Omega_1,\dots,\Omega_n)$.

**Definition** (Rebeschini, 2022) The Rademacher complexity of a set $\mathcal{T}\subseteq\mathbb{R}^n$ is defined as $$\mathfrak{R}(\mathcal{T})=\mathbb{E}_{\Omega\in\{\pm1\}^n}\left(\sup_{t\in\mathcal{T}}\frac{1}{n}\sum_{i=1}^n\Omega_i t_i\right).$$

**Remark** (Rebeschini, 2022) Note that $\mathfrak{R}(\mathcal{T})$ is a measure of the ability of $\mathcal{T}$ to replicate the sign pattern of a random signal.

**Proposition** (Rebeschini, 2022) With $\mathcal{T}\subseteq\mathbb{R}^n$, $v\in\mathbb{R}^n$ and $c\in\mathbb{R}$ let $c\mathcal{T}v=\{ct+v:t\in\mathcal{T}\}$. Then, $$\mathfrak{R}(c\mathcal{T}+v)=\vert c\vert\mathfrak{R}(\mathcal{T}).$$

We can apply Rademacher complexity to bounding the generalization gap by a method known as symmetrization (Rebeschini, 2022). For a training sample $S=\{(x_i,y_i)\}_{i=1}^m$ we consider the set of points $$\mathcal{L}\circ S:=\left\{(l(h_{\mathbf{w}}(x_1)),\dots,l(h_{\mathbf{w}}(x_i,y_i))):\mathbf{w}\in\mathcal{W}\right\}.$$

**Theorem** (Balcan, 2011) For a parameter $\delta\in(0,1)$ if $\mathcal{F}\subseteq\{f:\mathcal{Z}\to[0,1]\}$ then $$\mathbb{P}_{S\sim\mathcal{D}^m}\left(\mathbb{E}_{z\sim\mathcal{D}}\left(f(z)\right)\leq\frac{1}{m}\sum_{i=1}^mf(z_i)+2\hat{\mathfrak{R}}(\mathcal{F})+3\sqrt{\frac{\log\left(\frac{2}{\delta}\right)}{m}}\right)\geq1-\delta.$$

If we let $\mathcal{F}=\left\{(x,y)\mapsto\mathbb{I}\left(h_{\mathbf{w}}(x))[y]\leq\gamma+\max_{j\neq y}h_{\mathbf{w}}(x)[j]\right):\mathbf{w}\in\mathcal{W}\right\}$ then for any $\delta\in(0,1)$ and $\mathbf{w}\in\mathcal{W}$ we have that $$\mathbb{P}_{S\sim\mathcal{D}^m}\left(L_{\gamma}\left(h_{\mathbf{w}}\right)\leq\hat{L}_{\gamma}(h_{\mathbf{w}})+2\hat{\mathfrak{R}}(\mathcal{F})+3\sqrt{\frac{\log\left(\frac{2}{\delta}\right)}{m}}\right)\geq1-\delta.$$

**Definition** Given a set $\mathcal{S}$ and a function $\rho:\mathcal{S}\times\mathcal{S}\to\mathbb{R}_+$, we call $(\mathcal{S},\rho)$ a pseudo-metric space if for all $x,y,z\in\mathcal{S}$ we have
- $\rho(x,y)=\rho(y,x)$,
- $\rho(x,z)\leq\rho(x,y)+\rho(y,z)$, and
- $\rho(x,x)=0$.

**Definition** Let $(\mathcal{S},\rho)$ be a pseudo-metric space and let $\epsilon>0$. Then the set $\mathcal{C}\subseteq\mathcal{s}$ is an $\epsilon$-cover of $(\mathcal{S},\rho)$ if for every $x\in\mathcal{S}$ there is a $y\in\mathcal{C}$ such that $\rho(x,y)\leq\epsilon$. The set $\mathcal{C}$ is a minimal $\epsilon$-cover if there is no other $\epsilon$-cover with lower cardinality. The cardinality of any minimal $\epsilon$-cover is the $\epsilon$-covering number denoted $N(\mathcal{S},\rho,\epsilon)$.

For a given training set $S=\{(x_i,y_i)\}_{i=1}^m$ we can consider the set of $m$ points $$\mathcal{G}=\{(f(x_1,y_1),\dots,f(x_m,y_m)):f\in\mathcal{F}\}.$$

**Theorem** Let $\mathcal{F}\subseteq\{f:\mathcal{Z}\to[0,1]\}$ and $S\sim\mathcal{D}^m$ then $$\hat{\mathfrak{R}}(\mathcal{F})\leq\inf_{\epsilon>0}\left(\epsilon+\sqrt{\frac{N(\mathcal{G},\rho,\epsilon)}{m}}\right)$$