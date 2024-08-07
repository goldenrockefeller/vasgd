import numpy as np
import matplotlib.pyplot as plt
import scipy
import jax
jax.config.update("jax_enable_x64", True)

rng = np.random.default_rng()
target_model = rng.normal(size = (10, 10))
starting_model = rng.normal(size = (10, 10))
starting_model2 = (rng.normal(size = (10, 10)), rng.normal(size = (10, 10)))
n_input_vectors = 1000
condition_multiplier = 10 * 10
noise = 0.01

input_vectors = []
for i in range(n_input_vectors):
    vector = np.hstack((rng.normal(size = 9), [1.]))
    vector[0] *= np.sqrt(condition_multiplier)
    input_vectors.append(vector)

output_vectors = []
for i in input_vectors:
    output_vectors.append(
        (target_model @ i)
        + rng.normal(scale = noise, size = 10)
    )

input_vectors = np.array(input_vectors)
output_vectors = np.array(output_vectors)

A = np.vstack(input_vectors)

ATA = A.T @ A

eigvals = np.linalg.eigvals(ATA)
min_eig, max_eig = min(eigvals), max(eigvals)
print("Condition Number: ", max_eig/min_eig)

@jax.jit
def sample_loss(flattened_model, input, target):
    model = np.reshape(flattened_model, (10, 10))
    error = (model @ input) - target
    return 0.5 * np.sum(error * error)

d_sample_loss = jax.jit(jax.grad(sample_loss))

v_error = jax.vmap(sample_loss, (None, 0, 0))

@jax.jit
def jf(vector, x_s, y_s):
    errors = v_error(vector, x_s, y_s)
    return np.sum(errors)

jdf = jax.jit(jax.grad(jf))

def loss(input_vectors, output_vectors):
    def get_loss(flattened_model):
        return jf(flattened_model, input_vectors, output_vectors)
    return get_loss

def d_loss(input_vectors, output_vectors):
    def get_d_loss(flattened_model):
        return jdf(flattened_model, input_vectors, output_vectors)
    return get_d_loss

@jax.jit
def sample_loss2(flattened_model, input, target):
    M = np.reshape(flattened_model[:100], (10, 10))
    N = np.reshape(flattened_model[100:200], (10, 10))
    MN = M @ N
    error = (MN @ input) - target
    return 0.5 * np.sum(error * error)

d_sample_loss2 = jax.jit(jax.grad(sample_loss2))

v_error2 = jax.vmap(sample_loss2, (None, 0, 0))

@jax.jit
def jf2(vector, x_s, y_s):
    errors = v_error2(vector, x_s, y_s)
    return np.sum(errors)

jdf2 = jax.jit(jax.grad(jf2))

def loss2(input_vectors, output_vectors):
    def get_loss2(flattened_model):
        return jf2(flattened_model, input_vectors, output_vectors)
    return get_loss2

def d_loss2(input_vectors, output_vectors):
    def get_d_loss2(flattened_model):
        return jdf2(flattened_model, input_vectors, output_vectors)
    return get_d_loss2


def backtracking_rate(f, x, descent_dir, old_lr):
    lr = 3 * old_lr
    m = -np.dot(descent_dir, descent_dir)

    c = 0.5 # Armijo condition parameter

    n_lr_evals = 2
    t = -c * m
    fx = f(x)
    fx_new = f(x + lr * descent_dir)

    if fx - fx_new >= lr * t: # initial step is too small
        while fx - fx_new >= lr * t:
            lr *= 2
            n_lr_evals += 1
            fx_new = f(x + lr * descent_dir)
        lr *= 0.5
    else: # initial step is too big
        while fx - fx_new < lr * t:
            lr *= 0.5
            n_lr_evals += 1
            fx_new = f(x + lr * descent_dir)


    return (lr, n_lr_evals)

def lbfgs_grad(x, x_prev, grad, grad_prev, gamma):
    try:
        q = grad
        y = (grad - grad_prev)
        s = (x - x_prev)
        p = 1 / np.dot(y, s)
        if np.dot(s, y) <= 0:
            return None
        alpha = p * np.dot(s, q)
        q = q - alpha * y
        if gamma is None:
            gamma = np.dot(s, y) / np.dot(y, y)
        z = gamma * q
        beta = p * np.dot(y, z)
        z = z + s * (alpha - beta)
        if np.isfinite(z).all():
            return z
        else:
            return None
    except:
        return None

def single_armijo_step(f, x, grad, old_lr, d):
    descent_dir = -grad
    lr = 3 * old_lr
    m = -np.dot(descent_dir, descent_dir)

    c = 0.5 # Armijo condition parameter

    n_lr_evals = 2
    t = -c * m
    fx = f(x)
    fx_new = f(x + lr * descent_dir)

    if fx - fx_new >= lr * t: # initial step is too small
        while fx - fx_new >= lr * t:
            lr *= 2
            n_lr_evals += 1
            fx_new = f(x + lr * descent_dir)
        lr *= 0.5
    else: # initial step is too big
        while fx - fx_new < lr * t:
            lr *= 0.5
            n_lr_evals += 1
            fx_new = f(x + lr * descent_dir)

    x = x + d * lr * descent_dir
    return x, lr, n_lr_evals

def single_wolfe_step(f, df, x, grad, old_lr):
    wolfe_sr, sr, x_grad, success, n_evals = (
        wolfe_search(f, df, x - old_lr * grad, x)
    )
    if success:
        return x - old_lr *sr * grad,old_lr * sr, success, n_evals
    else:
        return x, old_lr, success, n_evals

def bisector_mom(descent_dir, mom):
    bisector = (
        descent_dir/ np.linalg.norm(descent_dir)
        + mom /  np.linalg.norm(mom)
    )
    projected_mom = bisector * np.dot(mom, bisector) / np.dot(bisector, bisector)
    return projected_mom


def partial_restart(df, x, x_prev):
    n_evals = 0
    mom = x - x_prev
    x_mom = x + mom
    descent_dir = -df(x_mom)

    while np.dot(descent_dir, mom) < 0:
        if n_evals == 50:
            return x_prev, x_prev, n_evals

        mom = bisector_mom(descent_dir, mom)
        x_mom = x_prev + 2 * mom
        grad = df(x_mom)
        n_evals += 1
        descent_dir = -grad

    x = x_prev + mom
    return x, x_prev, n_evals



def nesterov_step(f, df, x, x_prev, lr, is_restarting, is_always_backtracking, using_partial_restart):
    n_evals = 0

    if is_restarting:
        if using_partial_restart:
            x, x_prev, n_evals = partial_restart(df, x, x_prev)
        else:
            if (f(x) < f(x_prev)):
                x, x_prev = x, x
            else:
                x, x_prev = x_prev, x_prev
            n_evals += 2

    if is_always_backtracking:
        if (f(x) < f(x_prev)):
            x, x_prev = x, x_prev
        else:
            x, x_prev = x_prev, x_prev
        n_evals += 1 # 2 without caching

    mom = x - x_prev
    x_mom = x + mom
    grad = df(x_mom)
    n_evals += 1
    descent_dir = -grad
    if np.dot(descent_dir, mom) < 0:
        return x, x_prev, lr, True, n_evals, None

    rn = rng.uniform() < 0.1

    if is_always_backtracking or is_restarting or rn:

        x_prev = x
        x, lr, n_lr_evals = single_armijo_step(f, x_mom, grad, lr, 0.5)
        # x, lr, success, n_lr_evals =  single_wolfe_step(f, df, x, grad, lr, 0.25)
        n_evals += n_lr_evals
        return x, x_prev, lr, False, n_evals, None

    grad_step = descent_dir * lr * 0.5
    x_prev = x
    x = x_mom + grad_step

    return x, x_prev, lr, False, n_evals, None


def solver(f, df, x, method, max_n_evals):
    x = np.array(x)
    x_prev = x
    lr = 1
    is_restarting = True
    n_evals_elapsed = []
    evals = []
    n_evals = 0
    n_steps = 0
    data = None

    while n_evals < max_n_evals:
        x, x_prev, lr, is_restarting, n_method_evals, data = method(f, df, x, x_prev, lr, is_restarting, data)

        n_evals += n_method_evals
        n_steps += 1
        n_evals_elapsed.append(n_evals)
        evals.append(min(f(x), f(x_prev)))

    print("Average n_evals per step:", method.__name__, n_evals/n_steps)
    print(x)
    return evals, n_evals_elapsed

def backtracking_nesterov(f, df, x, x_prev, lr, is_restarting,data):
    return nesterov_step(f, df, x, x_prev, lr, is_restarting, True, False)

def backtrack_on_restart_nesterov(f, df, x, x_prev, lr, is_restarting,data):
    return nesterov_step(f, df, x, x_prev, lr, is_restarting, False, False)

def backtracking_nesterov_pr(f, df, x, x_prev, lr, is_restarting,data):
    return nesterov_step(f, df, x, x_prev, lr, is_restarting, True, True)

def backtrack_on_restart_nesterov_pr(f, df, x, x_prev, lr, is_restarting,data):
    return nesterov_step(f, df, x, x_prev, lr, is_restarting, False, True)


def lbfgs2(f, df, x, x_prev, lr, is_restarting, original_lr):
    n_evals = 0
    grad = df(x)
    n_evals += 1

    if is_restarting:
        not_x, a_lr, n_alr_evals = single_armijo_step(f, x, grad, lr, 1.0)
        new_x, lr, success, n_w_evals = single_wolfe_step(f, df, x, grad, 2*a_lr)
        x_prev = x
        x = new_x
        n_evals += n_w_evals + n_alr_evals
        original_lr = lr
        return x, x_prev, lr, False, n_evals, original_lr

    grad_prev = df(x_prev)
    n_evals += 1
    scaled_grad = lbfgs_grad(x, x_prev, grad, grad_prev, None) # None, 1., original_lr
    if scaled_grad is None:
        return x, x_prev, lr, True, n_evals, original_lr
    elif np.dot(scaled_grad,grad) <= 0:
        return x, x_prev, lr, True, n_evals, original_lr
    new_x, lr, success, n_w_evals = single_wolfe_step(f, df, x, scaled_grad, 2 * lr)
    n_evals += n_w_evals
    if not success:
        print("Armijo step taken, Use Nesterov?")
        new_x, lr, n_lr_evals = single_armijo_step(f, x, grad, lr, 1.0)
        n_evals += n_lr_evals
    x_prev = x
    x = new_x
    return x, x_prev, lr, False, n_evals, original_lr


def hybrid_solver(f, df, x, x_prev, lr, is_restarting,data):
    if data is None:
        hybrid_counter = 0
        hybrid_method_on_nag = 0
        hybrid_data = (hybrid_counter, hybrid_method_on_nag)
        method_data = None
    else:
        method_data = data[1]
        hybrid_counter, hybrid_method_on_nag = data[0]

    if hybrid_method_on_nag >= 0:
        x, x_prev, lr, is_restarting, n_method_evals, method_data = (
            backtrack_on_restart_nesterov(f, df, x, x_prev, lr, is_restarting,method_data)
        )
        hybrid_counter += n_method_evals
        if is_restarting:
            hybrid_method_on_nag  += 1
            if hybrid_method_on_nag == 1:
                hybrid_method_on_nag = -1
        method_data = None
    else:
        x, x_prev, lr, is_restarting, n_method_evals, method_data = (
            lbfgs2(f, df, x, x_prev, lr, is_restarting,method_data)
        )
        hybrid_counter -= n_method_evals
        if hybrid_counter <= 0:
            hybrid_counter = 0
            hybrid_method_on_nag = 0
            is_restarting = True
            x_prev = x

    hybrid_data = (hybrid_counter, hybrid_method_on_nag)
    return x, x_prev, lr, is_restarting, n_method_evals, (hybrid_data, method_data)

def wolfe_check(f, df, x_prev, scaled_s, f_x, f_x_prev, grad_prev):
    n_evals = 0
    x = x_prev + scaled_s
    grad = df(x)
    n_evals += 1


    # Positive curvature check
    if np.dot(scaled_s, grad) <= np.dot(scaled_s, grad_prev):
        return 1., None, False, n_evals

    # Finite wolfe point check
    try:
        wolfe_sr =1 -np.dot(scaled_s, grad) / (np.dot(scaled_s, grad) - np.dot(scaled_s, grad_prev))
    except:
        return wolfe_sr, None, False, n_evals
    if not np.isfinite(scaled_s * wolfe_sr).all():
        return wolfe_sr, None, False, n_evals

    f_wp = f(x_prev + scaled_s * wolfe_sr)
    n_evals += 1
    if not np.isfinite(f_wp):
        return wolfe_sr, None, False, n_evals

    if not (f_wp < f_x and f_wp < f_x_prev and f_x < f_x_prev):
        return wolfe_sr, None, False, n_evals

    return wolfe_sr, grad, True, n_evals

"""
restart : armijo, wolfe
find s: wolfe, none (if z bad, restart)
gamma: lr, wiki
z: wolfe-armijo, wolfe-restart, armijo
lr: redo for NAG, old_lr
check z: if bad (restart)
"""

def wolfe_search(f, df, x, x_prev):
    n_evals = 0

    grad_prev = df(x_prev)
    f_prev = f(x_prev)
    n_evals += 2

    # Get search direction
    s = x - x_prev
    sr = 1.
    if np.dot(grad_prev, s) > 0:
        sr = -1.
        x = x_prev + s * sr
        f_x = f(x)
        n_evals += 1
    else:
        f_x = f(x)
        n_evals += 1
        wolfe_sr, grad, success, n_wp_evals = wolfe_check(f, df, x_prev, s * sr, f_x, f_prev, grad_prev)
        n_evals += n_wp_evals
        if success:
            return wolfe_sr * sr, sr, grad, True, n_evals

    # find bounds
    if f_x < f_prev:
        last_fx = f_x
        while f_x <= last_fx:
            sr *= 2
            x = x_prev + s * sr
            last_fx = f_x
            f_x = f(x)
            n_evals += 1
            wolfe_sr, grad, success, n_wp_evals = wolfe_check(f, df, x_prev, s * sr, f_x, f_prev, grad_prev)
            n_evals += n_wp_evals
            if success:
                return wolfe_sr * sr, sr, grad, True, n_evals

    invphi = (np.sqrt(5) - 1) / 2

    lo = 0.
    hi = sr
    f_lo = f_prev
    f_hi = f_x

    lo_plus = hi - (hi - lo) * invphi
    hi_minus = lo + (hi - lo) * invphi
    f_lo_plus = f(x_prev + s * lo_plus)
    f_hi_minus = f(x_prev + s * hi_minus)
    n_evals += 2

    sr = lo_plus
    x = x_prev + s * sr
    f_x = f_lo_plus
    wolfe_sr, grad, success, n_wp_evals = wolfe_check(f, df, x_prev, s * sr, f_x, f_prev, grad_prev)
    n_evals += n_wp_evals
    if success:
        return wolfe_sr * sr, sr, grad, True, n_evals

    sr = hi_minus
    x = x_prev + s * sr
    f_x = f_hi_minus
    wolfe_sr, grad, success, n_wp_evals = wolfe_check(f, df, x_prev, s * sr, f_x, f_prev, grad_prev)
    n_evals += n_wp_evals
    if success:
        return wolfe_sr * sr, sr, grad, True, n_evals

    for i in range(100):
        # print(lo, lo_plus, hi_minus, hi)
        # print(f_lo, f_lo_plus, f_hi_minus, f_hi)
        # print()

        if f_lo < f_lo_plus and f_lo < f_hi_minus:
            hi = hi_minus
            f_hi = f_hi_minus
            hi_minus = lo_plus
            f_hi_minus = f_lo_plus
            lo_plus = hi - (hi - lo) * invphi
            f_lo_plus = f(x_prev + s * lo_plus)
            n_evals += 1

            sr = lo_plus
            x = x_prev + s * sr
            f_x = f_lo_plus
            wolfe_sr, grad, success, n_wp_evals = wolfe_check(f, df, x_prev, s * sr, f_x, f_prev, grad_prev)
            n_evals += n_wp_evals
            if success:
                return wolfe_sr * sr, sr, grad, True, n_evals

        elif f_lo_plus <= f_hi_minus:
            hi = hi_minus
            f_hi = f_hi_minus
            hi_minus = lo_plus
            f_hi_minus = f_lo_plus
            lo_plus = hi - (hi - lo) * invphi
            f_lo_plus = f(x_prev + s * lo_plus)
            n_evals += 1

            sr = lo_plus
            x = x_prev + s * sr
            f_x = f_lo_plus
            wolfe_sr, grad, success, n_wp_evals = wolfe_check(f, df, x_prev, s * sr, f_x, f_prev, grad_prev)
            n_evals += n_wp_evals
            if success:
                return wolfe_sr * sr, sr, grad, True, n_evals
        else:
            lo = lo_plus
            f_lo = f_lo_plus
            lo_plus = hi_minus
            f_lo_plus = f_hi_minus
            hi_minus = lo + (hi - lo) * invphi
            f_hi_minus = f(x_prev + s * hi_minus)
            n_evals += 1

            sr = hi_minus
            x = x_prev + s * sr
            f_x = f_hi_minus
            wolfe_sr, grad, success, n_wp_evals = wolfe_check(f, df, x_prev, s * sr, f_x, f_prev, grad_prev)
            n_evals += n_wp_evals
            if success:
                return wolfe_sr * sr, sr, grad, True, n_evals

    return None, None, None, False, n_evals


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

#[-1.07397447,  1.12972293,  1.2428755 ]
#[-1.2, 1.2,1.2]
#[-0.77565955,  0.61309386,  0.38206344,  0.14597248]
# evals, n_evals_elapsed = solver(rosenbrock, d_rosenbrock, [5, -1.2, 5], lbfgs2, 2500)
# plt.plot(n_evals_elapsed, np.log(np.array(evals)+0.0000001), label = "backtracking_nesterov")
#
#
#
# evals, n_evals_elapsed = solver(rosenbrock, d_rosenbrock, [5, -1.2, 5], backtrack_on_restart_nesterov, 2000)
# plt.plot(n_evals_elapsed, np.log(np.array(evals)+0.0000001), label = "backtrack_on_restart_nesterov")
# plt.show()


x =  np.reshape(starting_model, 100)
evals, n_evals_elapsed = solver(loss(input_vectors, output_vectors), d_loss(input_vectors, output_vectors), x, lbfgs2, 15000)
plt.plot(n_evals_elapsed, np.log(np.array(evals)+0.0000001), "r+", label = "lbfgs2")


evals, n_evals_elapsed = solver(loss(input_vectors, output_vectors), d_loss(input_vectors, output_vectors), x, backtrack_on_restart_nesterov, 15000)
plt.plot(n_evals_elapsed, np.log(np.array(evals)+0.0000001), "b+", label = "brnag")


evals, n_evals_elapsed = solver(loss(input_vectors, output_vectors), d_loss(input_vectors, output_vectors), x, backtracking_nesterov, 15000)
plt.plot(n_evals_elapsed, np.log(np.array(evals)+0.0000001), "y+", label = "full_bnag")

plt.legend()
plt.show()
#
#

x2 =  np.hstack((np.reshape(starting_model2[0], 100), np.reshape(starting_model2[1], 100)))
evals, n_evals_elapsed = solver(loss2(input_vectors, output_vectors), d_loss2(input_vectors, output_vectors), x2, lbfgs2, 15000)
plt.plot(n_evals_elapsed, np.log(np.array(evals)+0.0000001), "r+", label = "lbfgs2")


evals, n_evals_elapsed = solver(loss2(input_vectors, output_vectors), d_loss2(input_vectors, output_vectors), x2, backtrack_on_restart_nesterov, 15000)
plt.plot(n_evals_elapsed, np.log(np.array(evals)+0.0000001), "b+", label = "brnag")


evals, n_evals_elapsed = solver(loss2(input_vectors, output_vectors), d_loss2(input_vectors, output_vectors), x2, backtracking_nesterov, 15000)
plt.plot(n_evals_elapsed, np.log(np.array(evals)+0.0000001), "y+", label = "full_bnag")

plt.legend()
plt.show()