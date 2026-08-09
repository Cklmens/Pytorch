# Deep BSDE Solver for High-Dimensional PDEs

This project implements a **Deep BSDE method** for the numerical solution of high-dimensional semilinear PDEs. The approach represents the solution of the PDE through a backward stochastic differential equation (BSDE) and uses neural networks to approximate the process $Z_t$.

The notebook `project.ipynb` contains the solver implementation together with numerical experiments on several benchmark problems.

## Contents

The project includes experiments for:

1. **Allen–Cahn equation**
2. **Hamilton–Jacobi–Bellman (HJB) equation**
3. **European financial derivatives with different borrowing and lending rates**
4. **Multidimensional Burgers-type PDE**
5. **PDE with quadratically growing derivatives**
6. **Time-dependent reaction–diffusion PDE with an oscillating solution**

---

## Method

Consider a semilinear PDE of the general form

$$
\partial_t u(t,x)
+ \mu(t,x)\cdot\nabla_x u(t,x)
+ \frac12 \operatorname{Tr}
\left(\sigma\sigma^\top(t,x)D_x^2u(t,x)\right)
+ f(t,x,u,\sigma^\top\nabla_xu)=0,
$$

with terminal condition

$$
u(T,x)=g(x).
$$

The associated forward SDE is simulated using the Euler–Maruyama scheme:

$$

X_{n+1}
=
X_n+\mu(t_n,X_n)\Delta t
+\sigma(t_n,X_n)\Delta W_n.

$$

The BSDE discretization is then used to propagate $Y$ and $Z$ backward. The initial value $Y_0$, which corresponds to the desired PDE value $u(0,\xi)$, is learned jointly with neural networks approximating $Z_n$.

The training objective is based on the terminal mismatch

$$

Y_N-g(X_N).

$$

The default loss is the mean squared error:

$$

\mathcal L =
\mathbb E\left[
|Y_N-g(X_N)|^2
\right].

$$

A Huber loss is also implemented in the configurable solver.

---

## Neural Network Architecture

The network `BSDENet` maps the $d$-dimensional state $X_n$ to an approximation of the $d$-dimensional $Z_n$.

The architecture used in the notebook is:

```text
Input: d
   │
Linear(d, d+10)
   │
ReLU
   │
Linear(d+10, d+10)
   │
ReLU
   │
Linear(d+10, d)
   │
Output: Z_n
```

The notebook contains implementations with and without Batch Normalization, as well as configurable initialization strategies.

For each interior time step, a separate neural network is created:

```python
self.nets = nn.ModuleList(
    [BSDENet(d) for _ in range(N - 1)]
)
```

The trainable parameters also include the initial values:

- `Y0`
- `Z0`

---

## Implemented Problems

### 1. Allen–Cahn Equation

Parameters used in the notebook:

| Parameter | Value |
|---|---:|
| Dimension $d$ | 100 |
| Time horizon $T$ | 0.3 |
| Time steps $N$ | 20 |
| Initial state | $0\in\mathbb R^{100}$ |
| Diffusion | $\sqrt{2}$ |
| Driver | $f(y)=y-y^3$ |
| Reference $u(0,\xi)$ | 0.052802 |

The experiment uses:

- batch size: 256
- learning rate: $5\times10^{-4}$
- 4000 training iterations
- 10,000 evaluation samples

---

### 2. Hamilton–Jacobi–Bellman Equation

Parameters:

| Parameter | Value |
|---|---:|
| Dimension $d$ | 100 |
| Time horizon $T$ | 1 |
| Time steps $N$ | 20 |
| Initial state | $0\in\mathbb R^{100}$ |
| Diffusion | $\sqrt{2}$ |
| Driver | $f(z)=-\sum_i z_i^2$ |
| Reference $u(0,\xi)$ | 4.6 |

The experiment uses a learning rate of `0.005`, batch size 256 and 4000 iterations.

---

### 3. European Financial Derivative

This experiment considers a financial problem with different rates for lending and borrowing.

Parameters include:

```text
mu_bar = 6%
sigma_bar = 20%
R_l = 4%
R_b = 6%
d = 100
T = 0.5
N = 20
X_0 = 100
```

The forward process has proportional drift and volatility:

$$

\mu(t,x)=\bar\mu x,
\qquad
\sigma(t,x)=\bar\sigma x.

$$

The BSDE driver incorporates the difference between the lending rate $R_l$ and borrowing rate $R_b$.

The notebook uses:

- batch size: 256
- learning rate: $5\times10^{-3}$
- 5000 training iterations
- reference value: 21.22

---

### 4. Multidimensional Burgers-Type PDE

Parameters:

| Parameter | Value |
|---|---:|
| Dimension $d$ | 20 |
| Time horizon $T$ | 1 |
| Time steps $N$ | 80 |
| Initial state | $0$ |
| Reference $u(0,\xi)$ | 1 |

The experiment uses 5000 iterations and a batch size of 64.

The learning rate is selected by the helper function:

```python
def gamma(m):
    if m <= 30000:
        return 1e-2
    elif m <= 50000:
        return 1e-3
    else:
        return 1e-4
```

---

### 5. PDE with Quadratically Growing Derivatives

Parameters:

| Parameter | Value |
|---|---:|
| Dimension $d$ | 100 |
| Time horizon $T$ | 1 |
| Time steps $N$ | 80 |
| $\alpha$ | 0.4 |
| Reference $u(0,\xi)$ | 1 |

The notebook constructs the driver from an explicit solution and its derivatives, including the temporal derivative, gradient norm and Laplacian.

Training:

- batch size: 64
- learning rate: $5\times10^{-3}$
- 4000 iterations

---

### 6. Time-Dependent Reaction–Diffusion PDE

This example is based on a reaction–diffusion-type PDE with an oscillating explicit solution.

Parameters:

```text
d = 100
T = 1
N = 30
kappa = 0.6
lambda = 1 / sqrt(d)
```

The terminal condition is

$$

g(x)
=
1+\kappa+
\sin\left(\lambda\sum_i x_i\right).

$$

Training:

- batch size: 64
- learning rate: `0.01`
- 3000 iterations
- reference value: 2.09

---

## Hyperparameter Optimization

The notebook also contains an **Optuna** configuration for automatic hyperparameter search on the Allen–Cahn problem.

The search includes:

- loss type: MSE / Huber
- Huber parameter
- optimizer: Adam / Momentum / RMSprop
- learning rate
- momentum
- batch size
- initialization of `Y0`
- initialization of `Z0`
- gradient clipping
- learning-rate scheduler
- scheduler decay factor

The optimization is configured for up to:

```text
50 trials
or
1 hour
```

Each trial performs a shorter training run before evaluating the resulting loss.

---

## Optimization

The configurable solver supports:

- **SGD**
- **SGD with momentum**
- **RMSprop**
- **Adam**

The default Adam parameters are:

```python
betas = (0.9, 0.999)
eps = 1e-8
```

Gradient clipping is available to improve training stability.

A `MultiStepLR` learning-rate scheduler is also implemented in the configurable version.

---

## Evaluation

After training, the notebook evaluates the learned value $Y_0$ over multiple simulated batches.

The reported quantities are:

```text
Estimated u(0,xi)
Standard deviation
Relative L1 error
Final loss
Loss standard deviation
```

The relative error is computed as

$$

\text{Relative error}
=
\frac{|\,\widehat u(0,\xi)-u_{\mathrm{ref}}(0,\xi)\,|}
{|u_{\mathrm{ref}}(0,\xi)|}.

$$

Training diagnostics include loss/error plots and moving statistics.

---

## Installation

The project uses Python with the following main dependencies:

```bash
pip install torch numpy matplotlib tqdm optuna
```

A CUDA-enabled PyTorch installation is recommended for high-dimensional experiments when a compatible NVIDIA GPU is available.

The notebook automatically selects CUDA when available:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

Otherwise, it falls back to the CPU.

---

## Usage

Open the notebook:

```bash
jupyter notebook project.ipynb
```

or:

```bash
jupyter lab project.ipynb
```

Then execute the cells in order.

A typical experiment follows this structure:

```python
funct = Functions()

T, N, d, xi, mu, sigma, f, g = funct.allen_cahn_params()

solver = DeepBSDESolver(
    T, N, d, xi, mu, sigma, f, g,
    batch_size=256,
    lr=5e-4,
    true_u0=0.052802,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

losses = solver.train(
    num_iterations=4000,
    log_interval=500
)

solver.plot_error()
solver.plot_moving_stats()

mean_Y0, std_Y0, mean_loss, std_loss = solver.evaluate(
    num_samples=10000
)
```

---

## Project Structure

```text
.
├── project.ipynb
└── README.md
```

At present, the main implementation is contained in the Jupyter notebook.

---

## Notes on the Current Notebook

The notebook contains more than one version of the solver and of the benchmark parameter definitions. In particular, there are implementations using both `Fonction` and `Functions`, and different versions of `BSDENet` / `DeepBSDESolver`.

Before extracting the code into standalone Python modules, these duplicated implementations should be consolidated into a single canonical version.

Some benchmark functions also contain comments indicating assumptions or adaptations from the original problems. These should be checked against the corresponding reference paper before using the implementation for quantitative benchmarking.

---

## References

The implementation follows the **Deep BSDE methodology** for solving high-dimensional nonlinear PDEs through neural-network approximations of the associated BSDE.

The notebook includes benchmark problems commonly used to evaluate Deep BSDE solvers, including Allen–Cahn, HJB, financial, Burgers-type and reaction–diffusion examples.

---

## Author

Project developed as a numerical study of **deep learning methods for high-dimensional PDEs and BSDEs**, with applications to quantitative finance and stochastic analysis.
