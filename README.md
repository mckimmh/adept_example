# Automatic Differentiation with Adept

In statistics and machine learning, many algorithms make use of gradients of functions. In Bayesian Statistics, the gradient of the log-density of a posterior distribution is often used. This tutorial presents an example of constructing the posterior distribution of a sensor-location model so that its log-density and grad-log-density may be evaluated. We first specify the posterior distribution of the model, show how to use the Adept library to compute the gradient of its log-density, use numerical differentiation to check that automatic differentiation is workign correctly, then provide an example of how the gradient may be used to sample the posterior using Hamiltonian Monte Carlo.

## The Sensor Model

This sensor network localisation problem first appeared in Ihler et al, 2005, and has been analysed in other work (Pompe et al., 2020; Tak et al., 2017). There are 6 sensors on the real plane. The position of 4 sensors is unknown whilst the position of 2 sensors is known. The distance between sensors i and j is observed with probability
$$\exp ( - || x_i - x_j ||^2 / (2 \times 0.3^2) ). $$
Given that it's observed, the distance is noisy and has distribution:
$$l(y_{ij} | x_i, x_j) \equiv N( || x_i - x_j ||, 0.02^2 ). $$
The prior distribuiton is a product of independent Gaussian distributions, each with variance 100.

## Computing the log-density and its gradient using Adept

The Adept library may be used to compute gradients automatically. Suppose you have written a function to compute the log-density of the posterior using variables of type `double`. To use automatic differentiation with Adpet, for all variables whose value depends on the independent input variables (i.e. the data), replace type `double` with `adept::adouble`.

## Checking Automatic Differentiation against Numerical Differentation

It's a good idea to check the correctness of your code by comparing the gradient computed using automatic differentiation to the gradient computed using numeric differentiation.

## Example of using the gradient: Hamiltonian Monte Carlo

Now that the gradient may be easily computed, it may be used within gradient-based algorithms. For example, the gradient may be used to generate samples from the posterior via Hamiltonian Monte Carlo (HMC). This tutorial does not cover HMC, but we present here samples generated from the posterior using HMC.

## References

Ihler, A. T., Fisher, J. W., Moses, R. L., and Willsky, A. S. (2005). Nonparametric belief propagation for self-localization of sensor networks. IEEE Journal on Selected Areas in Communications.

Pompe, E., Holmes, C., Łatuszyński, K. (2020). A framework for adaptive MCMC targeting multimodal distributions. The Annals of Statistics.

Tak, H., Meng, X.-L., and van Dyk, D. A. (2017). A Repelling-Attracting Metropolis Algorithm for Multimodality. Journal of Computational and Graphical Statistics.
