import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

rng = np.random.default_rng()
target_model = rng.normal(size = (10, 10))
starting_model = rng.normal(size = (10, 10))
n_input_vectors = 1000
condition_multiplier = 1000 * 1000
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

A = np.vstack(input_vectors)

ATA = A.T @ A

eigvals = np.linalg.eigvals(ATA)
min_eig, max_eig = min(eigvals), max(eigvals)
print("Condition Number: ", max_eig/min_eig)

def sample_loss(flattened_model, input, target):
    model = np.reshape(flattened_model, (10, 10))
    error = (model @ input) - target
    return 0.5 * np.sum(error * error)

def loss(flattened_model, input_vectors, output_vectors):
    val = 0.
    for input, output in zip(input_vectors, output_vectors):
        val += sample_loss(flattened_model, input, output)
    return val

def backtracking_rate(old_lr, f, df, x_mom, descent_dir, input, output):
    lr = 3 * old_lr
    m = -np.dot(descent_dir, descent_dir)

    c = 0.5
    d = 0.5

    n_lr_evals = 2
    t = -c * m
    fx = f(x_mom, input, output)
    fx_new = f(x_mom + lr * descent_dir, input, output)

    if fx - fx_new >= lr * t: # initial step is too small
        while fx - fx_new >= lr * t:
            lr *= 2
            n_lr_evals += 1
            fx_new = f(x_mom + lr * descent_dir, input, output)
        lr *= 0.5
    else: # initial step is too big
        while fx - fx_new < lr * t:
            lr *= 0.5
            n_lr_evals += 1
            fx_new = f(x_mom + lr * descent_dir, input, output)

    return (d * lr, n_lr_evals)

sample_loss_grad = jax.grad(sample_loss)
print(
    sample_loss_grad(
        np.reshape(starting_model, 100),
        input_vectors[0],
        output_vectors[0]
    )
)

x =  np.reshape(starting_model, 100)
x_prev = x
epsilon = 1e-16
lr = epsilon
n_evals_elapsed = []
evals = []
n_evals = 0
n_steps = 0
restarts = True
cum_mom = 0*x
cum_grad = 0*x
cum_inv_rate = epsilon
horizon = 400 # all the same horizon value!!!
beta = 1 - (1/horizon)

f = jax.jit(sample_loss)
df = jax.jit(sample_loss_grad)

print(loss(x, input_vectors, output_vectors))
print(f"target_model loss = {loss(target_model, input_vectors, output_vectors)}")
restarted = False

avg_log_lr = 1.
avg_log_lr_count = 0.
backtrack_counter = horizon
backtrack_on_restart = False
restart_counter = 0
restart_counter_2 = 0

for i in range(400):
    for j in range(1000):
        k = rng.integers(1000)
        input = input_vectors[k]
        output = output_vectors[k]

        small_mom = (x - x_prev)
        cum_mom = small_mom + beta * cum_mom
        x_mom = x + horizon * small_mom
        descent_dir = -df(x_mom, input, output)
        cum_grad = descent_dir + beta * cum_grad
        #cum_grad = (descent_dir/cum_inv_rate/horizon + beta * cum_grad) / (1 + beta)

        if restarts:
            restart_condition = np.dot(cum_grad, small_mom + 0* cum_grad/cum_inv_rate/horizon) < 0
            #restart_condition = np.dot(cum_grad, small_mom + cum_grad) < 0

            if restart_condition:

                if restart_counter == horizon and restart_counter_2 == horizon:
                    restart_counter = 0
                    print("Restart")
                    # restart
                    x_mom = x
                    descent_dir = -df(x_mom, input, output)
                    cum_mom = 0*x
                    cum_grad = descent_dir
                    small_mom = 0*x
                    n_evals += 1
                    avg_log_lr_count = 1.
                    backtrack_counter = horizon
                    cum_inv_rate = epsilon
                    restart_counter_2 = 0
                    # cum_inv_rate is not zeroed out!
                elif restart_counter < horizon:
                    restart_counter += 1
                elif restart_counter == horizon:
                    restart_counter_2 += 1
            elif restart_counter < horizon:
                restart_counter += 1

        # skippable? YES!
        if backtrack_counter > 0 or not backtrack_on_restart:
            lr, n_lr_evals = backtracking_rate(2 ** avg_log_lr, f, df, x_mom, descent_dir, input, output)
            avg_log_lr = (
                (np.log2(lr) + avg_log_lr_count * avg_log_lr)
                / (1 + avg_log_lr_count)
            )
            avg_log_lr_count += 1.
            backtrack_counter -= 1
            cum_inv_rate = max(1 / (lr+ epsilon) , beta * cum_inv_rate)
        else:
            n_lr_evals = 0

        grad_step = descent_dir / cum_inv_rate / horizon
        x_prev = x
        x = x + small_mom + grad_step

        n_evals += 1 + n_lr_evals
        n_steps += 1

    print(f"loss = {loss(x, input_vectors, output_vectors)}")
    n_evals_elapsed.append(n_evals)
    evals.append(loss(x, input_vectors, output_vectors))

# for i in range(5000):
#     for j in range(1000):
#         k = rng.integers(1000)
#         input = input_vectors[k]
#         output = output_vectors[k]
#
#         small_mom = (x - x_prev)
#         x_mom = x + 0.998* small_mom
#         descent_dir = -df(x_mom, input, output)
#
#         lr, n_lr_evals = 0.5/condition_multiplier, 0
#         grad_step = lr * descent_dir / 10
#         x_prev = x
#         x = x + small_mom + grad_step
#
#         n_evals += 1 + n_lr_evals
#         n_steps += 1
#
#     print(f"loss = {loss(x, input_vectors, output_vectors)}, {horizon=}")
#     n_evals_elapsed.append(n_evals)
#     evals.append(loss(x, input_vectors, output_vectors))


print("Average n_evals per step:", n_evals/n_steps)



