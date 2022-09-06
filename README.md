# Automatic Differentiation with Adept

In statistics and machine learning, many algorithms make use of gradients of functions. In Bayesian Statistics, the gradient of the log-density of a posterior distribution is often used. This small directory contains an example of constructing the posterior distribution of a sensor-location model so that its log-density and grad-log-density may be evaluated. We first specify the posterior distribution of the model, describe how to use the Adept library to compute the gradient of its log-density, recommend using numeric differentiation to check that automatic differentiation is working correctly, then provide an example showing that the gradient may be used to sample the posterior using Hamiltonian Monte Carlo.

## The Sensor Model

This sensor network localisation problem first appeared in Ihler et al, 2005, and has been analysed in other work (Pompe et al., 2020; Tak et al., 2017). There are 6 sensors with positions $x_1, x_2, \dots, x_6$ on the real plane. The position of 4 sensors is unknown whilst the position of 2 sensors is known. The distance between sensors $i$ and $j$ is observed with probability:
$$\exp \left( - \frac{|| x_i - x_j ||^2}{2 \times 0.3^2} \right). $$
Variables $w_{ij}$ encode whether the ditance between sensor $i$ and sensor $j$ is observed ( $ w_{ij} = 1$ ) or not ( $ w_{ij} = 0 $ ). Given that it's observed, the distance is noisy and has distribution:
$$N( || x_i - x_j ||, 0.02^2 ). $$
The likelihood is then the product:
$$\prod_{i=1}^4 \prod_{j=i+1}^6 l(x_i, x_j | w_{ij}, y_{ij})$$
where
$$l(x_i, x_j | w_{ij} = 1, y_{ij}) =  \exp \left( - \frac{ || x_i - x_j ||^2 }{ 2 \times 0.3^2 } \right) \exp \left( - \frac{ y_{ij} - || x_i - x_j ||)^2 }{ 2 \times 0.02^2 } \right) $$
and
$$l(x_i, x_j | w_{ij} = 0, y_{ij}) = 1 - \exp \left( - \frac{ || x_i - x_j ||^2 }{ 2 \times 0.3^2} \right).$$
The prior distribuiton is a product of independent Gaussian distributions, each with variance 100.

## Computing the log-density and its gradient using Adept

The Adept library (http://www.met.reading.ac.uk/clouds/adept/) may be used to compute gradients automatically. Suppose you have written a function to compute the log-density of the posterior using variables of type `double`. To use automatic differentiation with Adpet, for all variables whose value depends on the independent input variables (i.e. the data), replace type `double` with `adept::adouble`.

## Checking Automatic Differentiation against Numerical Differentation

It's a good idea to check the correctness of your code by comparing the gradient computed using automatic differentiation to the gradient computed using numeric differentiation. The approximates the derivative as, for $\epsilon > 0$ a small constant:
$$\frac{d}{dx} \pi(x) \approx \frac{ \pi(x+\epsilon) - \pi(x) }{ \epsilon }.$$
This approximation is based on the definition of the derivative as the limit as $\epsilon \rightarrow 0$ of:
$$\frac{ \pi(x+\epsilon) - \pi(x) }{ \epsilon }.$$

## Example of using the gradient: Hamiltonian Monte Carlo

Now that the gradient may be easily computed, it may be used within gradient-based algorithms. For example, the gradient may be used to generate samples from the posterior via Hamiltonian Monte Carlo (HMC) (Duane et al., 1987). This tutorial does not cover HMC, but we present here 1000 samples generated from the posterior using HMC.

![1000 samples generated using HMC](https://github.com/mckimmh/adept_example/blob/main/hmc_samples.png)

## References

Duane, S., Kennedy, A.D., Pendleton, B.J., Rowether, D. (1987). Hybrid Monte Carlo. Physics Letters B.

Ihler, A. T., Fisher, J. W., Moses, R. L., and Willsky, A. S. (2005). Nonparametric belief propagation for self-localization of sensor networks. IEEE Journal on Selected Areas in Communications.

Pompe, E., Holmes, C., Łatuszyński, K. (2020). A framework for adaptive MCMC targeting multimodal distributions. The Annals of Statistics.

Tak, H., Meng, X.-L., and van Dyk, D. A. (2017). A Repelling-Attracting Metropolis Algorithm for Multimodality. Journal of Computational and Graphical Statistics.
