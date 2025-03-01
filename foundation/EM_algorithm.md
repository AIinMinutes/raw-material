# Comprehensive Notes on the Expectation-Maximization (EM) Algorithm

## 1. Introduction
The Expectation-Maximization (EM) algorithm is an iterative method for finding maximum likelihood estimates of parameters in statistical models with latent (hidden) variables. It's particularly useful when direct maximum likelihood estimation is difficult or impossible.

## 2. Core Concept
The EM algorithm alternates between two steps:
- **Expectation (E-step)**: Calculate the expected value of the log-likelihood function using current parameter estimates and observed data.
- **Maximization (M-step)**: Update parameters to maximize the expected log-likelihood found in the E-step.

## 3. Mathematical Formulation

### 3.1 Latent-Variable Model
In a latent-variable model, we have:
- Observed data: $X$
- Unobserved (latent) variables: $Z$
- Parameters to estimate: $\theta$

The joint probability distribution $p(X,Z|\theta)$ represents the complete likelihood.

### 3.2 Model Specification
The joint distribution is typically specified by:
1. Conditional distribution of observed variables: $p(X|Z,\theta)$
2. Marginal distribution of latent variables: $p(Z|\theta)$

This gives us: $p(X,Z|\theta) = p(X|Z,\theta)p(Z|\theta)$

### 3.3 Derived Distributions
From these, we can derive:
- Marginal distribution of observed data: $p(X|\theta) = \sum_Z p(X,Z|\theta)$
- Conditional distribution of latent variables: $p(Z|X,\theta) = \frac{p(X,Z|\theta)}{p(X|\theta)}$

### 3.4 Maximum Likelihood Problem
The maximum likelihood estimator (MLE) of $\theta$ solves:
$\hat{\theta}_{MLE} = \arg\max_\theta \log p(X|\theta)$

### 3.5 The EM Algorithm Process
Starting with initial guess $\theta^{(0)}$, the $t$-th iteration consists of:

1. **E-step**: 
   - Compute conditional probabilities $p(Z|X,\theta^{(t-1)})$
   - Calculate expected complete log-likelihood:
     $Q(\theta|\theta^{(t-1)}) = E_{Z|X,\theta^{(t-1)}}[\log p(X,Z|\theta)]$
     $= \sum_Z p(Z|X,\theta^{(t-1)}) \log p(X,Z|\theta)$

2. **M-step**:
   - Find new parameters:
     $\theta^{(t)} = \arg\max_\theta Q(\theta|\theta^{(t-1)})$

3. **Convergence Check**:
   - If $||\theta^{(t)} - \theta^{(t-1)}|| < \epsilon$, stop
   - Otherwise, return to step 1

## 4. Example: Coin-Flipping Scenario

### 4.1 Problem Setup
- Two coins (A and B) with unknown biases $\theta_A$ and $\theta_B$
- 5 experiments, each with 10 coin flips
- Observed data: 5, 9, 8, 4, 7 heads in each experiment
- Hidden variable: Which coin was used in each experiment

### 4.2 Initial Parameters
- $\theta_A = 0.6$ (probability of heads for coin A)
- $\theta_B = 0.5$ (probability of heads for coin B)

### 4.3 E-step Example
For experiment 1 (5 heads out of 10):
- $P(x_1=5|z_1=A,\theta_A=0.6) = {10 \choose 5}(0.6)^5(0.4)^5$
- $P(z_1=A|x_1=5,\theta) = \frac{P(x_1=5|z_1=A,\theta_A)P(z_1=A)}{P(x_1=5|z_1=A,\theta_A)P(z_1=A) + P(x_1=5|z_1=B,\theta_B)P(z_1=B)}$
- After calculation: $P(z_1=A|x_1=5,\theta) = 0.86$, $P(z_1=B|x_1=5,\theta) = 0.14$

### 4.4 M-step Example
Update parameter $\theta_A$:
- $\theta_A^{new} = \frac{\sum_{i=1}^5 P(z_i=A|x_i,\theta) \cdot x_i}{\sum_{i=1}^5 P(z_i=A|x_i,\theta) \cdot 10}$
- $= \frac{0.86 \times 5 + 0.73 \times 9 + 0.77 \times 8 + 0.77 \times 4 + 0.82 \times 7}{0.86 \times 10 + 0.73 \times 10 + 0.77 \times 10 + 0.77 \times 10 + 0.82 \times 10}$
- $= \frac{25.5}{39.5} = 0.65$

## 5. Gaussian Mixture Models (GMMs)

### 5.1 GMM Setup
- Data points: $X = \{x_1, x_2, ..., x_n\}$
- Hidden variables: $Z = \{z_1, z_2, ..., z_n\}$ (which component generated each point)
- Parameters: $\theta = \{\pi_1,...,\pi_k, \mu_1,...,\mu_k, \Sigma_1,...,\Sigma_k\}$
  - $\pi_j$: mixing coefficients
  - $\mu_j$: mean vectors
  - $\Sigma_j$: covariance matrices

### 5.2 GMM Likelihood
For a single data point: $P(x_i|\theta) = \sum_{j=1}^k \pi_j \mathcal{N}(x_i|\mu_j, \Sigma_j)$

### 5.3 E-step for GMMs
Calculate "responsibilities":
$\gamma_{ij} = P(z_i=j|x_i,\theta) = \frac{\pi_j \mathcal{N}(x_i|\mu_j, \Sigma_j)}{\sum_{l=1}^k \pi_l \mathcal{N}(x_i|\mu_l, \Sigma_l)}$

### 5.4 M-step for GMMs
Update parameters:
1. Mixing coefficients: $\pi_j^{new} = \frac{1}{n}\sum_{i=1}^n \gamma_{ij}$
2. Means: $\mu_j^{new} = \frac{\sum_{i=1}^n \gamma_{ij}x_i}{\sum_{i=1}^n \gamma_{ij}}$
3. Covariances: $\Sigma_j^{new} = \frac{\sum_{i=1}^n \gamma_{ij}(x_i-\mu_j^{new})(x_i-\mu_j^{new})^T}{\sum_{i=1}^n \gamma_{ij}}$

## 6. Key Properties of EM Algorithm

### 6.1 Theoretical Guarantees
- **Monotonic Improvement**: Each iteration guarantees $p(X|\theta^{(t)}) \geq p(X|\theta^{(t-1)})$
- **Convergence**: Algorithm converges to a stationary point of the likelihood function
- **Local Optima**: Typically finds a local maximum, not necessarily global

### 6.2 Advantages
- Guaranteed to increase likelihood at each iteration
- Simple to implement for many problems
- Often converges reliably
- Handles missing or incomplete data naturally

### 6.3 Limitations
- May converge to local maxima, not global
- Convergence can be slow in some cases
- Sensitive to initial values
- No guarantee of finding the global maximum

## 7. Applications of EM

Common applications include:
- Mixture models (e.g., Gaussian Mixture Models)
- Hidden Markov Models
- Factor analysis
- Latent class analysis
- Image reconstruction
- Machine learning with incomplete data
- Clustering algorithms

## 8. Practical Considerations

### 8.1 Initialization Strategies
- Multiple random initializations
- K-means for initializing GMMs
- Domain knowledge-based initializations
- Hierarchical approaches

### 8.2 Convergence Criteria
- Parameter change threshold: $||\theta^{(t)} - \theta^{(t-1)}|| < \epsilon$
- Log-likelihood change threshold
- Maximum number of iterations

### 8.3 Computational Efficiency
- For large datasets, consider batch or online variations
- Exploit sparsity and structure when possible
- Vectorized implementations for efficiency