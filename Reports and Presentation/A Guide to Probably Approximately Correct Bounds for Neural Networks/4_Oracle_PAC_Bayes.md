\subsection{Theory of Oracle PAC-Bayes Bounds}

Oracle bounds are theoretical objects that are not suitable for practical applications. Their utility lies in their ability to highlight properties about the behaviour of our bounds. For example, they can take the form
$$\mathbb{P}_{S\sim\mathcal{D}^m}\left(R\left(\hat{\mathbf{w}}\right)\leq\inf_{\mathbf{w}\in\mathcal{W}}R(\mathbf{w})+r_m(\delta)\right)\geq1-\delta.$$
Where $r_m(\delta)$ is a remainder term that tends to $0$ as $m$ tends to $\infty$. Although this bound cannot be computed in practice it is illustrative of the behaviour of the bound. Just like empirical bounds, there exist oracle bounds that hold in expectation and in probability.

\subsubsection{Oracle PAC-Bayes Bounds in Expectation}
\begin{theorem}
    For $\lambda>0$ we have that
    $$\mathbb{E}_{S\sim\mathcal{D}^m}\mathbb{E}_{\mathbf{w}\sim\hat{\rho}_{\lambda}}(R(\mathbf{w}))\leq\inf_{\rho\in\mathcal{M}(\mathcal{W})}\left(\mathbb{E}_{\mathbf{w}\sim\rho}(R(\theta))+\frac{\lambda C^2}{8m}+\frac{\mathrm{KL}(\rho,\pi)}{\lambda}\right).$$
\end{theorem} 
\textit{Proof.}$\square$

\subsubsection{Oracle PAC-Bayes Bounds in Probability}

\begin{theorem}
    For any $\lambda>0$, and $\delta\in(0,1)$ we have that
    $$\mathbb{P}_{S\sim\mathcal{D}^m}\left(\mathbb{E}_{\mathbf{w}\sim\hat{\rho}_{\lambda}}(R(\mathbf{w}))\leq\inf_{\rho\in\mathcal{M}(\mathcal{W})}\left(\mathbb{E}_{\mathbf{w}\sim\rho}\left(R(\mathbf{w})\right)+\frac{\lambda C^2}{4m}+\frac{2\mathrm{KL}(\rho,\pi)+\log\left(\frac{2}{\delta}\right)}{\lambda}\right)\right)\geq1-\delta$$
\end{theorem}
\textit{Proof.}$\square$

\subsubsection{Bernstein's Assumption}

\begin{definition}
    Let $\mathbf{w}^*$ denote a minimizer of $R$ when it exists,
    $$R(\mathbf{w}^*)=\min_{\mathbf{w}\in\mathcal{W}}R(\mathbf{w}).$$
    When $\mathbf{w}^*$ exists and there is a constant $K$ such that for any $\mathbf{w}\in\mathcal{W}$ we have that
    $$\mathbb{E}_{S\sim\mathcal{D}^m}\left(\left(l(h_{\mathbf{w}}(x_i),y_i)-l(h_{\mathbf{w}^*}(x_i),y_i)\right)^2\right)\leq K\left(R(\mathbf{w})-R(\mathbf{w}^*)\right)$$
    we say that Bernstein's assumption is satisfied with constant $K$.
\end{definition}

\begin{theorem}
    Assume Bernstein's assumption is satisfied with some constant $K>0$. Take $\lambda=\frac{m}{\max(2K,C)}$ then we have
    $$\mathbb{E}_{S\sim\mathcal{D}^m}\mathbb{E}_{\mathbf{w}\sim\hat{\rho}_{\lambda}}\left(R(\mathbf{w})\right)-R\left(\mathbf{w}^*\right)\leq2\inf_{\rho\in\mathcal{M}(\mathcal{W})}\left(\mathbb{E}_{\mathbf{w}\sim\rho}(R(\mathbf{w}))-R\left(\mathbf{w}^*\right)+\frac{\max(2K,C)\mathrm{KL}(\rho,\pi)}{m}\right).$$
\end{theorem}
\textit{Proof.}$\square$

\subsection{Data Driven PAC-Bayes Bounds}

A lot of work to obtain non-vacuous PAC-Bayes bounds is to develop priors that reduce the size of the KL divergence between the prior and the posterior. The idea behind the work of (Dziugaite, 2020) is to hold out some of the training data to obtain data-inspired priors. For this section, we use a PAC-Bayes bound that can be thought of as the Bayesian equivalent of Theorem \ref{Theorem-Occam Bound}, however, now we are dealing with potentially uncountable hypothesis sets.
\begin{theorem}[(McAllester, 2013)]\label{Theorem-Occam Style PAC Bayes Bound}
    For $\lambda>\frac{1}{2}$ selected before drawing our training sample, then for all $\rho\in\mathcal{M}(\mathcal{W})$ and $\delta\in(0,1)$ we have that
    $$\mathbb{P}_{S\sim\mathcal{D}^m}\left(R(\rho)\leq\hat{R}(\mathbf{\rho})+\frac{\lambda C}{m}\left(\mathrm{KL}(\rho,\pi)+\log\left(\frac{1}{\delta}\right)\right)\right)\geq1-\delta.$$
\end{theorem}
\textit{Proof.}$\square$
\begin{corollary}[(Dziugaite, 2020)]\label{Corollary-Occam Style PAC Bayes Bound}
    Let $\beta,\delta\in(0,1)$, $\mathcal{D}$ a probability distribution over $\mathcal{Z}$, and $\pi\in\mathcal{M}(\mathcal{W})$. Then for all $\rho\in\mathcal{M}(\mathcal{W})$ we have that
    $$\mathbb{P}_{S\sim\mathcal{D}^m}\left(R(\rho)\leq\Psi_{\beta,\delta}(\rho,\pi;S)\right)\geq1-\delta,$$
    where $\Psi_{\beta,\delta}(\rho,\pi;S)=\frac{1}{\beta}\hat{R}(\rho)+\frac{\mathrm{KL}(\rho,\pi)+\log\left(\frac{1}{\delta}\right)}{2\beta(1-\beta)m}.$
\end{corollary}
\textit{Proof.}$\square$
As we have done previously, we can consider the optimization problem of minimizing the bound of Corollary \ref{Corollary-Occam Style PAC Bayes Bound}.
\begin{theorem}[(Dziugaite, 2020)]
    Let $m\in\mathbb{N}$ and fix a probability kernel $\rho:\mathcal{Z}^m\to\mathcal{M}(\mathcal{W})$. Then for all $\beta,\delta\in(0,1)$ and distributions $\mathcal{D}$ defined on $\mathcal{Z}$ we that $\mathbb{E}_{S\sim\mathcal{D}^m}\left(\Psi_{\beta,\delta}(\rho(S),\pi;S\right)$ is minimized, in $\pi$, by the oracle prior $\pi^*=\mathbb{E}_{S\sim\mathcal{D}^m}(\rho(S))$. 
\end{theorem}
For a subset $J$ of $\{1,\dots,m\}$ of size $n$, we can use it to sample the training data and yield the subset $S_J$. We can then define the data-dependent oracle prior as $$\pi^*(S_J)=\inf_{\pi\in\mathcal{Z}^n\to\mathcal{M}(\mathcal{W})}\mathbb{E}(\mathrm{KL}(\rho(s),\pi(S_J))$$
which turns out to be $\pi^*(S_J)=\mathbb{E}(\rho(S)\vert S_J)$. It can be shown that the data-dependent oracle prior minimizes the bound of Corollary \ref{Corollary-Occam Style PAC Bayes Bound} in expectation. Therefore, despite being a theoretical quantity, as it cannot be computed in practice, it motivates the construction of practical data-dependent priors as a method to tighten the bounds.
\subsubsection{Implementing Data-Dependent Priors}
To implement data-dependent priors we restrict the optimization problem to make it tractable. We only consider the set of Gaussian priors $\mathcal{F}$ that generate Gaussian posteriors. Neural networks are trained via SGD, and hence there is some randomness to the learning algorithm. Let $(\Omega,\mathcal{F},\nu)$ define a probability space and let us focus on the kernels
$$\rho:\Omega\times\mathcal{Z}^m\to\mathcal{M}(\mathcal{W}),\quad\rho(U,S)=\mathcal{N}(\mathbf{w}_s,\mathbf{s}),$$
where $\mathbf{w}_S$ are the learned weights via SGD on the full dataset $S$. The random variable $U$ represents the randomness of the learning algorithm. As before we consider a non-negative integer $n\leq m$ and with $\alpha=\frac{n}{m}$ we define a subset $S_{\alpha}$ of size $n$ containing the first $n$ indices of $S$ processed by SGD. Let $\mathbb{E}^{S_{\alpha},U}[\cdot]$ denote the conditional expectation operator given $S_{\alpha}$ and $U$. Our aim now is to tighten the bound of Corollary \ref{Corollary-Occam Style PAC Bayes Bound} by minimizing $\mathbb{E}^{S_{\alpha},U}(\mathrm{KL}(\rho(U,S),\pi))$. To do this we further restrict the priors of consideration to those of the form $\mathcal{N}(\mathbf{w}_{\alpha},\sigma I)$ such that with $\sigma$ fixed we are left with the minimization problem
\begin{equation}\label{Equation-DataDependent Prior SGD Minimization Problem}
\mathrm{argmin}_{\mathbf{w}_{\alpha}}\left(\mathbb{E}^{S_{\alpha},U}\left(\Vert\mathbf{w}_S-\mathbf{w}_{\alpha}\Vert\right)\right),
\end{equation}
which can be solved to yield $\mathbf{w}_{\alpha}=\mathbb{E}^{S_{\alpha},U}(\mathbf{w}_S)$. This minimizer is unknown in practice so we attempt to approximate it. We first define a so-called ghost sample, $S^G$, which is an independent sample equal in distribution to $S$. We combine a $1-\alpha$ fraction of $S^G$ with $S_{\alpha}$ to obtain the sample $S_{\alpha}^G$. Let $\mathbf{w}_{\alpha}^G$ be the mean of $\rho(U,S_{\alpha}^G)$. By construction, SGD will first process $S_{\alpha}$ then the combined portion of $S^G$ and hence $\mathbf{w}_{\alpha}^G$ and $\mathbf{w}_S$ are equal in distribution when conditioned on $S_{\alpha}$ and $U$. Therefore, $\mathbf{w}_{\alpha}^G$ is an unbiased estimator of $\mathbb{E}^{S_{\alpha},U}(\mathbf{w}_S)$. Before formalizing this process algorithmically we clarify some notation.
\begin{itemize}
    \item The SGD run on $S$ is the base run.
    \item The SGD run on $S_{\alpha}$ is the $\alpha$-prefix run.
    \item The SGD run on $S_{\alpha}^G$ is the $\alpha$-prefix$+$ghost run and obtains the parameters $\mathbf{w}_{\alpha}^G$.
\end{itemize}
The resulting parameters of the $\alpha$-prefix and $\alpha$-prefix$+$ghost run can be used as the centres of the Gaussian priors to give the tightened generalization bounds. However, sometimes the ghost sample is not attainable in practice, and hence one simply relies upon $\alpha$-prefix runs to obtain the mean of the prior. It is not clear whether $\alpha$-prefix$+$ghost run will always obtain a parameter that leads to a tighter generalization bound. Recall, that $\sigma$ is assumed to be fixed in the optimization process. Algorithm \ref{Algorithm-Tighter PAC Bounds via SGD} is independent of this parameter and so it can be optimized afterwards without requiring a re-run of the optimization process.

\begin{algorithm}
\caption{Stochastic Gradient Descent}
\label{Algorithm-SGD}
\begin{algorithmic}
\Require Learning rate $\eta$
\Function{SGD}{$\mathbf{w}_0,S,b,t,\mathcal{E}=-\infty$}
\State $\mathbf{w}\leftarrow\mathbf{w}_0$
\For{$i\leftarrow 1$ to $t$}
\State Sample $S^\prime\in S$ with $\vert S^\prime\vert=b$
\State $\mathbf{w}\leftarrow\mathbf{w}-\eta\nabla l_{S^\prime}(\mathbf{w})$
\If{$l_S^{0\text{-}1}(\mathbf{w})\leq\mathcal{E}$}
\State break
\EndIf
\EndFor
\EndFunction
\end{algorithmic}
\end{algorithm}

\begin{algorithm}
\caption{Obtaining Bound Using SGD Informed Prior}
\label{Algorithm-Tighter PAC Bounds via SGD}
\begin{algorithmic}
\Require Stopping criteria $\mathcal{E}$, Prefix fraction $\alpha$, Ghost Data $S^G$ (If available), Batch size $b$.
\Function{GetBound}{$\mathcal{E},\alpha,T,\sigma_P$}
\State $S_{\alpha}\leftarrow\{z_1,\dots,z_{\alpha\vert S\vert}\subset S\}$
\State $\mathbf{w}_{\alpha}^0\leftarrow$SGD$\left(\mathbf{w}_0,S_{\alpha},b,\frac{\vert S_{\alpha}\vert}{b}\right)$
\State $\mathbf{w}_S\leftarrow$SGD$\left(\mathbf{w}_{\alpha}^0,S,b,\infty,\mathcal{E}\right)$\Comment{Base Run}
\State $\mathbf{w}_{\alpha}^G\leftarrow$SGD$\left(\mathbf{w}_{\alpha}^0,S_{\alpha}^G,b,T,\cdot\right)$\Comment{Ghost run if data available, otherwise prefix run}
\State $\pi\leftarrow\mathcal{N}\left(\mathbf{w}_{\alpha}^G,\sigma I\right)$
\State $\rho\leftarrow\mathcal{N}\left(\mathbf{w}_S,\sigma I\right)$
\State Bound$\leftarrow\Psi_{\delta}^*(\rho,\pi;S\setminus S_{\alpha})$
\State \textbf{return} Bound
\EndFunction
\end{algorithmic}
\end{algorithm}