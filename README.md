# Automatic Differentiation with Adept

In statistics and machine learning, many algorithms make use of gradients of functions. In Bayesian Statistics, the gradient of the log-density of a posterior distribution is often used. This tutorial presents an example of constructing the posterior distribution of a sensor-location model so that its log-density and grad-log-density may be evaluated. We first specify the posterior distribution of the model, show how to use the Adept library to compute the gradient of its log-density, use numerical differentiation to check that automatic differentiation is workign correctly, then provide an example of how the gradient may be used to sample the posterior using Hamiltonian Monte Carlo.

## The Sensor Model

## Computing the log-density and its gradient using Adept

## Checking Automatic Differentiation against Numerical Differentation

## Example of using the gradient: Hamiltonian Monte Carlo
