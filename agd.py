# Accelerated Gradient Descent
import numpy as np
import matplotlib.pyplot as plt

def nesterov(f, df, x0, lr_scheduler, ms, max_n_evals, restarts = False):
    """
        lr is learning rate
        ms is momentum sustain factor
    """
    x = np.array(x0)
    x_prev = x
    lr_data = None
    lr = 1
    n_evals_elapsed = []
    evals = []
    n_evals = 0
    restarted = True
    n_steps = 0

    while n_evals < max_n_evals:

        x_mom = x + (x - x_prev) * ms

        descent_dir = -df(x_mom)

        restarted = False
        if restarts:
            if np.dot(descent_dir, (x- x_prev)) < 0:
                # restart
                x = x_mom
        if (x == x_mom).all():
            restarted = True

        lr, n_lr_evals = lr_scheduler(lr, f, df, x_mom, descent_dir, restarted)


        grad_step = descent_dir * lr

        x_prev = x
        x = x_mom + grad_step


        n_evals += 1 + n_lr_evals
        n_steps += 1
        n_evals_elapsed.append(n_evals)
        evals.append(f(x))

    print("Average n_evals per step:", n_evals/n_steps)
    print(x)
    return evals, n_evals_elapsed

"""
progressive momentum rate
https://angms.science/doc/CVX/CVX_NAGD.pdf

restart
lr = 1/L, ms = (sqrt_k - 1)/(sqrt_k + 1)
L is the largest eigen value, mu is the smallest
https://arxiv.org/pdf/1204.3982

optimal = lf = 2/(m+L)
https://math.stackexchange.com/questions/3797972/how-to-choose-the-constant-of-strong-convexity

--- For future research ---
https://arxiv.org/pdf/2002.10583
scheduled restart for stochastic gradient descent
https://dl.acm.org/doi/10.1007/s00245-020-09718-8
Backtracking Gradient Descent Method and Some Applications in Large Scale Optimisation. Part 2: Algorithms and Experiments
half of backtrack on restart

accelerated parameters:
---theoretical---
[x] optimal non-accelerated (no momentum)
[x] optimal accelerated
---practical---
[x] non-optimal
[x] backtrack non-accelerated [c = 0, c = 0.5]
[x] 2nd derivative non-accelerated
[x] always backtrack with restart [c = 0, c = 0.5][ms = 1],
[x] 2nd derivative with restart [ms = 1]
[x] backtrack on restart [cc = 0.5, cc = 1.][c = 0, c = 0.5][ms = 1]

c for backtracking must be 0.5 as a good default value
backtrack on restart WORKS
because of momentum, we actually end up taking 2 of the backtracking steps, so we multiply by d = 0.5.

with ms = 1, the momentum can scale infinitely as necessary to solve the problem.
with ms = 0.5, the momentum can only scale to a factor of 2

line search methods:
2nd, derivative
https://en.wikipedia.org/wiki/Backtracking_line_search
Armijo–Goldstein condition

Kalman-optimizer
    does it match Kalman filter? no
    Is it a good estimator? not really

discounting techniques:
kalman n = 1/(1/n + (1-s)/s)
traditional n = sn
moving: drop 1 value every step

when there is a lot of restarts,it is time to increase the horizon.


Stochastic nesterov
    does the look-ahead technique work well with stochastic gradients? yes
    does backtrack-on-restart work well? yes

Flat Network:
    Compare RBF, Inverse Quadratic, Fuzzy, and Leaky Relu on Sine wave, Point and linear
    distribute nodes:
        over spread (all data between two nodes)
        underspread (all node between two through)
        left spread (all nodes left of all data)
        good spread (all nodes spread evenly)0
"""

n_evals = 1000

def meta_ellipse(y):
    def ellipse(x):
        return 0.5 * np.dot(x * y, x * y)
    def d_ellipse(x):
        return x* y*y
    return ellipse, d_ellipse

def min_and_max_eig_vals(y):
    A = np.diag(y)
    A_2 = A @ A
    eigvals = np.linalg.eigvals(A_2)
    return min(eigvals), max(eigvals)

def optimal_params_no_acc(min_eig_val, max_eig_val):
    lr = 2 / (min_eig_val + max_eig_val)
    return (lr, 0)

def optimal_params_acc(min_eig_val, max_eig_val):
    # k is the condition number
    k = max_eig_val / min_eig_val
    sqrt_k = np.sqrt(k)
    lr = 1/max_eig_val
    ms = (sqrt_k - 1)/(sqrt_k + 1)
    return (lr, ms)

def constant_lr_scheduler(lr):
    def inner_constant_lr_scheduler(old_lr, f, df, x_mom, descent_dir, restarted):
        return lr, 0
    return inner_constant_lr_scheduler

def backtracking_lr_scheduler(c, on_restart, d):
    # https://en.wikipedia.org/wiki/Backtracking_line_search
    def inner_backtracking_lr_scheduler(old_lr, f, df, x_mom, descent_dir, restarted):
        if on_restart and not restarted:
            return old_lr, 0
        lr = 3 * old_lr
        m = -np.dot(descent_dir, descent_dir)

        n_lr_evals = 2
        t = -c * m
        fx = f(x_mom)
        fx_new = f(x_mom + lr * descent_dir)

        if fx - fx_new >= lr * t: # initial step is too small
            while fx - fx_new >= lr * t:
                lr *= 2
                n_lr_evals += 1
                fx_new = f(x_mom + lr * descent_dir)
            lr *= 0.5
        else: # initial step is too big
            while fx - fx_new < lr * t:
                lr *= 0.5
                n_lr_evals += 1
                fx_new = f(x_mom + lr * descent_dir)

        return (d * lr, n_lr_evals)


    return inner_backtracking_lr_scheduler

def approx_2nd_deriv(df, neg_grad, lr, x, epsilon = 0.00001):
    step = neg_grad * lr * epsilon
    new_neg_grad = -df(x + step)
    step_norm = np.linalg.norm(step)
    grad_norm = np.linalg.norm(neg_grad)
    normalized_diff = 1 - np.dot(new_neg_grad, neg_grad) / (grad_norm * grad_norm)
    val = grad_norm * normalized_diff / (step_norm + epsilon * epsilon)
    return val

def approx_2nd_deriv_lr_scheduler(c):
    def inner_approx_2nd_deriv_lr_scheduler(old_lr, f, df, x_mom, descent_dir, restarted):
        return c / approx_2nd_deriv(df, descent_dir, old_lr, x_mom), 1
    return inner_approx_2nd_deriv_lr_scheduler

def rosenbrock(x):
    x = np.array(x)
    val = 0
    for i in range(len(x) - 1):
        val += 100 * (x[i+1] - x[i]**2)**2 + (x[i] - 1)**2
    return val

def d_rosenbrock(x):
    x = np.array(x)
    df = 0 * x

    for i in range(len(x) - 1):
        df[i+1] += 200*(x[i+1] - x[i]**2)
        df[i] += -400 * x[i] * (x[i+1] - x[i]**2) + 2 * (x[i] - 1)

    return df

#
# lr, ms = optimal_params_acc(*min_and_max_eig_vals([1,100,34]))
# evals, n_evals_elapsed = nesterov(*meta_ellipse([1,100,34]), [2,20,5], constant_lr_scheduler(lr), ms, 10000, False)
# plt.plot(n_evals_elapsed, np.log(np.array(evals)+0.0000001), label = "B")

evals, n_evals_elapsed = nesterov(rosenbrock, d_rosenbrock, [-1.2, 1.2,1.2, 1], backtracking_lr_scheduler(0.5, False, 1.0), 0, 2000000, False)
plt.plot(n_evals_elapsed, np.log(np.array(evals)+0.0000001), label = "B")

evals, n_evals_elapsed = nesterov(rosenbrock, d_rosenbrock, [-1.2, 1.2,1.2, 1 ], backtracking_lr_scheduler(0.5, False, 0.98), 1, 2000000, True)
plt.plot(n_evals_elapsed, np.log(np.array(evals)+0.0000001), label = "A")

plt.legend(loc="upper right")
plt.show()





