/* Sensor Network Localization
 *
 * g++ -Wall -std=c++17 sensor.cpp -lm
 * ./a.out
 */

#include <adept.h>
#include <adept_source.h>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

// Component i,j of the log-likelihood
adept::adouble ll_ij(const adept::adouble x_i0,
                     const adept::adouble x_i1,
                     const adept::adouble x_j0,
                     const adept::adouble x_j1,
                     const int w_ij,
                     const double y_ij);

// Log-likelihood
adept::adouble log_lik(const adept::adouble x[6][2],
                       const std::vector<int>& w,
                       const std::vector<double>& y);

// Log-prior: product of independent Gaussians with variance 100
adept::adouble log_prior(const adept::adouble x[6][2]);

// Log-posterior
double log_posterior(const std::vector<double>& state,
                     std::vector<double>& grad,
                     const std::vector<int>& w,
                     const std::vector<double>& y);

// Generate the data
void generate_data(std::mt19937_64& gen,
                   const double x[6][2],
                   std::vector<int>& w,
                   std::vector<double>& y);

// Numeric Differentiation
void num_diff(std::vector<double>& state,
              const double epsilon,
              std::vector<double>& num_grad,
              const std::vector<int>& w,
              const std::vector<double>& y);

// Print vector
void print_vector(const char* expression,
                  const std::vector<double>& v);

int main()
{
    const int seed = 576;
    std::mt19937_64 gen(seed);
    
    const double x[6][2] = {{0.57, 0.91},
                            {0.10, 0.37},
                            {0.26, 0.14},
                            {0.85, 0.04},
                            {0.50, 0.30},
                            {0.30, 0.70}};
    
    std::vector<int> w;      // Vector of which distances are observed
    std::vector<double> y;   // Vector of observations
    
    generate_data(gen, x, w, y);
    
    std::normal_distribution<double> rnorm(0.0, 1.0);
    
    // State at which to evaluate the log-density
    std::vector<double> state;
    for (int i = 0; i < 8; i++){
        double state_i = rnorm(gen);
        state.push_back(state_i);
    }
    
    // Gradient
    std::vector<double> grad(8, 0.0);
    
    double ld = log_posterior(state, grad, w, y);
    
    std::cout << "Log-density: " << ld << '\n';
    
    print_vector("Gradient        : ", grad);
    
    // Numeric Differentiation
    const double epsilon = 0.00001;
    std::vector<double> num_grad(8, 0.0);
    num_diff(state, epsilon, num_grad, w, y);
    
    print_vector("Numeric Gradient: ", num_grad);
     
    return 0;
}

adept::adouble ll_ij(const adept::adouble x_i0,
                     const adept::adouble x_i1,
                     const adept::adouble x_j0,
                     const adept::adouble x_j1,
                     const int w_ij,
                     const double y_ij)
{
    // Squared Euclidean Distance between sensor i and j
    adept::adouble m = pow(x_i0 - x_j0, 2.0) + pow(x_i1 - x_j1, 2.0);
    
    adept::adouble ll = 0.0;   // log-likelihood
    
    if (w_ij)
    {
        ll -= 1250 * pow(y_ij - sqrt(m), 2.0);
        ll -= 50.0 * m / 9.0;
    } else
    {
        ll += log(1 - exp(-50.0 * m / 9.0));
    }
    
    return ll;
}

adept::adouble log_lik(const adept::adouble x[6][2],
                       const std::vector<int>& w,
                       const std::vector<double>& y)
{
    std::vector<int>::const_iterator w_iter = w.begin();
    std::vector<double>::const_iterator y_iter = y.begin();
    
    adept::adouble ll = 0;
    
    for (int i = 0; i < 4; i++){
        for (int j = i+1; j < 6; j++)
        {
            ll += ll_ij(x[i][0], x[i][1], x[j][0], x[j][1], *w_iter, *y_iter);
            w_iter++;
            y_iter++;
        }
    }
    
    return ll;
}

adept::adouble log_prior(const adept::adouble x[6][2])
{
    adept::adouble lp = 0;
    for (int i = 0; i < 4; i++)
    {
        lp -= pow(x[i][0], 2.0) + pow(x[i][1], 2.0);
    }
    lp /= 200;
    
    return lp;
}

double log_posterior(const std::vector<double>& state,
                     std::vector<double>& grad,
                     const std::vector<int>& w,
                     const std::vector<double>& y)
{
    if (state.size() != 8){
        std::cerr << "Error in log_posterior: state.size() != 8\n";
    }
    if (grad.size() != 8){
        std::cerr << "Error in log_posterior: grad.size() != 8\n";
    }
    
    adept::Stack stack;
    
    const adept::adouble x[6][2] = {{state[0], state[1]},
                                    {state[2], state[3]},
                                    {state[4], state[5]},
                                    {state[6], state[7]},
                                    {0.50    , 0.30},
                                    {0.30    , 0.70}};
    
    stack.new_recording();
    
    adept::adouble lp = log_prior(x);
    adept::adouble ll = log_lik(x, w, y);
    adept::adouble ld = lp + ll;
    
    ld.set_gradient(1.0);
    stack.compute_adjoint();
    
    for (int i = 0; i < 4; i++){
        for (int j = 0; j < 2; j++){
            grad[2*i + j] = x[i][j].get_gradient();
        }
    }
    
    return ld.value();
}

void generate_data(std::mt19937_64& gen,
                   const double x[6][2],
                   std::vector<int>& w,
                   std::vector<double>& y)
{
    std::uniform_real_distribution<double> runif(0.0, 1.0);
    std::normal_distribution<double> rnorm(0.0, 1.0);
    
    // Squared euclidean distances between the sensors
    double m_mat[4][6] = {};
    for (int i = 0; i < 4; i++){
        for (int j = i+1; j < 6; j++){
            double m = pow(x[i][0] - x[j][0], 2.0) + pow(x[i][1] - x[j][1],2.0);
            m_mat[i][j] = m;
        }
    }
    
    // Which distances are observed
    int w_mat[4][6] = {};
    int i = 0;
    while (i < 4)
    {
        int row_sum = 0;
        for (int j = i+1; j < 6; j++)
        {
            double p = exp(-50.0 * m_mat[i][j] / 9.0);
            double u = runif(gen);
            if (u < p)
            {
                w_mat[i][j] = 1;
                row_sum++;
            }
        }
        if (row_sum > 0) i++;
    }
    
    for (int i = 0; i < 4; i++){
        for (int j = i+1; j < 6; j++){
            double w_ij = w_mat[i][j];
            w.push_back(w_ij);
            
            if (w_ij == 1)
            {
                double y_ij = rnorm(gen);
                y_ij *= 0.02;
                y_ij += sqrt(m_mat[i][j]);
                y.push_back(y_ij);
            } else {
                y.push_back(0);
            }
        }
    }
}

void num_diff(std::vector<double>& state,
              const double epsilon,
              std::vector<double>& num_grad,
              const std::vector<int>& w,
              const std::vector<double>& y)
{
    if (state.size() != 8){
        std::cerr << "Error in num_diff: state.size() != 8\n";
    }
    if (epsilon <= 0){
        std::cerr << "Error in num_diff: epsilon <= 0\n";
    }
    if (num_grad.size() != 8){
        std::cerr << "Error in num_diff: num_grad.size() != 8\n";
    }
    
    std::vector<double> grad_aux(8, 0.0);
    
    const double ld = log_posterior(state, grad_aux, w, y);
    
    for (int i = 0; i < 8; i++)
    {
        state[i] += epsilon;
        double ld_epsilon = log_posterior(state, grad_aux, w, y);
        state[i] -= epsilon;
        
        num_grad[i] = (ld_epsilon - ld) / epsilon;
    }
}

void print_vector(const char* expression,
                  const std::vector<double>& v)
{
    std::cout << expression;
    for (std::vector<double>::const_iterator it = v.begin();
         it != v.end(); ++it){
        std::cout << *it << ' ';
    }
    std::cout << '\n';
}
