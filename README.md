# Generalization-of-Neural-Networks

This repository contains the code for an undergraduate research project I conducted in the summer of 2023. The project looks into the theory regarding the generalization of deep neural networks. Throughout the repository, the generalization bounds proposed by various papers are tested and investigated. Some of the work is my own work that extends the concepts in the papers and links them together in a concise and coherent manner. The aim of the project is to understand how current literature intertwines to give us an understanding of how deep neural networks generalize.

A Survey of the Methods for Deriving Bounds on the Generalization Error of Deep Neural Networks.
- Part I - Research on the Generalization Capacity of Deep Neural Networks.
- Part II - PAC-Bayes Bounds
- Part III - Information Theoretic Approach
- Part IV - Appealing to Gradients
- Part V - Other Approaches

| Part | Section of Report | Paper Title | Paper PDF | Project Code
| ----------- | ----------- | ----------- |----------- | ----------- |
| I | Bayesian Machine Learning | A Primer on PAC-Bayesian Learning | [PDF](https://arxiv.org/pdf/1901.05353.pdf) |  |
|  | Complexity Measures | Foundations of Machine Learning | [PDF](https://www.dropbox.com/s/38p0j6ds5q9c8oe/10290.pdf?dl=1) |  |
|  | Complexity Measures | Algorithmic Foundations of Learning | [PDF](https://www.stats.ox.ac.uk/~rebeschi/teaching/AFoL/22/) |  |
| II | Optimizing PAC-Bayes Bounds via SGD | Computing Nonvacuous Generalization Bonds for Deep (Stochastic) Neural Networks with Many More Parameters than Training Data | [PDF](https://arxiv.org/pdf/1703.11008.pdf) | [CODE](https://github.com/ThomasWalker1/Generalization-of-Neural-Networks/tree/main/Generalization%20in%20Deep%20Learning/PAC) |
|  | Compression | Stronger Generalization Bounds for Deep Nets via a Compression Approach | [PDF](https://arxiv.org/pdf/1802.05296.pdf) | [CODE](https://github.com/ThomasWalker1/Generalization-of-Neural-Networks/tree/main/Generalization%20in%20Deep%20Learning/Compression) |
|  | PAC Compression Bounds | Non-Vacuous Generalization Bounds at the ImageNet Scale: A PAC-Bayesian Compression Approach | [PDF](https://arxiv.org/pdf/1804.05862.pdf) | ----------- |
|  | Data Drive PAC-Bayes Bounds | On the role of data in PAC-Bayes bounds | [PDF](https://arxiv.org/pdf/2006.10929.pdf) | ----------- |
| III | Controlling Bias in Data Analysis | Controlling Bias in Adaptive Data Analysis Using Information Theory | [PDF](http://proceedings.mlr.press/v51/russo16.pdf) | ----------- |
|  | Generalizing the Framework | Dependence Measures Bounding the Exploration Bias for General Measurements | [PDF](https://arxiv.org/pdf/1612.05845.pdf) | ----------- |
|  | Evolution of Mutual Information | Opening the Black Box of Deep Neural Networks via Information | [PDF](https://arxiv.org/pdf/1703.00810.pdf) | [CODE](https://github.com/ThomasWalker1/Generalization-of-Neural-Networks/tree/main/Generalization%20in%20Deep%20Learning/Information) |
|  | Generalization Bounds for Learning Algorithms | Information-theoretic analysis of generalization capability of learning algorithms | [PDF](https://arxiv.org/pdf/1705.07809.pdf) | ----------- |
|  | Chaining Mutual Information | Chaining Mutual Information and Tightening Generalization Bounds | [PDF](https://arxiv.org/pdf/1806.03803.pdf) | ----------- |
|  | Conditional Mutual Information | Reasoning About Generalization via Conditional Mutual Information | [PDF](https://arxiv.org/pdf/2001.09122.pdf) | ----------- |
| IV | Stiffness | Stiffness: A New Perspective on Generalization in Neural Networks | [PDF](https://arxiv.org/pdf/1901.09491.pdf) | [CODE](https://github.com/ThomasWalker1/Generalization-of-Neural-Networks/tree/main/Generalization%20in%20Deep%20Learning/Stiffness) |
|  | Neural Tangent Kernels | Implicit Regularization via Neural Feature Alignment | [PDF](https://arxiv.org/pdf/2008.00938.pdf) | [CODE](https://github.com/ThomasWalker1/Generalization-of-Neural-Networks/tree/main/Generalization%20in%20Deep%20Learning/Tangent%20Kernel) |
|  | Geometric Complexity Measure | Why Neural Networks Find Simple Solutions: The Many Regularizers of Geometric Complexity | [PDF](https://arxiv.org/pdf/2209.13083.pdf) | [CODE](https://github.com/ThomasWalker1/Generalization-of-Neural-Networks/tree/main/Generalization%20in%20Deep%20Learning/Geometric%20Complexity) |
|  | Algorithmic Stability | On the Generalization Mystery in Deep Learning | [PDF](https://arxiv.org/pdf/2203.10036.pdf) | [CODE](https://github.com/ThomasWalker1/Generalization-of-Neural-Networks/tree/main/Generalization%20in%20Deep%20Learning/Gradients) |
| V | Unit-Wise Capacity Measures | Towards Understanding the Role of Over-Parameterization in Generalization of Neural Networks | [PDF](https://arxiv.org/pdf/1805.12076.pdf) | [CODE](https://github.com/ThomasWalker1/Generalization-of-Neural-Networks/tree/main/Generalization%20in%20Deep%20Learning/Unit-Wise%20Capacity) |
|  | Validation Paradigm | Generalization In Deep Learning | [PDF](https://arxiv.org/pdf/1710.05468.pdf) | [CODE](https://github.com/ThomasWalker1/Generalization-of-Neural-Networks/tree/main/Generalization%20in%20Deep%20Learning/Validation) |