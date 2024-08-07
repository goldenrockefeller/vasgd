# vasgd
Versatile Accelerated Stochastic Gradient Descent Experiments

Mini-batch is the best when available, when not available horizon method is good alternative. Horizon method experiements assumed one sample at a time, but batches are better when available. For full gradient descent, Nesterov Accelerated Gradient with (Armijo) Backtracking on Restart is the most efficient, but backtracking is not too much more expensive. LBFGS (memory-less) is often slower than backtacking NAG on Restart (but faster than full) because it takes a lot more function evaluations to find acceptable Wolfe Conditions. LBFGS can be the fastest on some highly conditioned problems, adding more memory can improve performance. Nesterov Accelerated Gradient does not do well with Wolfe Conditions search, gradient steps should be small so that future steps that are taken under the influence of momentum are not too large. Large steps will put the NAG step into a restart condition too frequently. Partial restart does not have a noticeble effect on convergence. Hybrid technique that tries to combine lbfgs with NAG by alternating when NAG is set to restart does not do noticeably better.

Takeaway, use Nesterov Accelerated Gradient with (Armijo) Backtracking on Restart with appropriately size mini-batch.
Note: Backtracking on Restart will result in less than 2 function evaluations on average!

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

The results of these experiments show that a nearly parameter-free method for stochastically accelerated gradient descent that uses "Adaptive Restart for Accelerated Gradient Schemes" is very effective!

- No more wasting hours on parameter tuning to get fast performance
- Now, a fairer comparison between machine learning algorithms can be performed (as differences in performance will not be blamed on parameter tuning)
- The algorithm is essentially plug-and-play.
  
The main takeaways from these experiments are:

1) Nesterov acceleration, which was initially created for non-stochastic optimization, is modified for application to a stochastic setting by looking farther ahead than 1 momentum step to generate a gradient for the next descent step.  This modificaiton is absolutely necessary for stable learning.
2) The "Adaptive Restart" scheme, which was initially created for non-stochastic optimization, is modified for application to a stochastic setting by taking moving averages of the gradient and momentum. This modificaiton is absolutely necessary for stable learning.
3) The method's sole parameter is a "horizon" parameter that roughly reflects the number of data points needed to get a good approximation (and prevent the gradient descent method from exploding). A longer horizon represents a tradeoff preference for final model accuracy and learning stability over training speed. Potential variations of this method can start with a small (but stable) horizon, and increase the horizon over time. A horizon of 1 is the non-stochastic Adaptive Restart.
4) Backtracking is worth doing and does not slow down the optimal learning rate that much.
5) The number of function evaluations due to backtracking can be minimized by using previous backtracking results to estimate an optimal learning rate for future gradient descent directions. Periodic reestimation of this optimal learning rate is performed throughout the algorithm (e.g. after each Adaptive restart).
6) Kalman filter for parameter estimation is not good, other preconditioners should be used.

Restart Conditions:
-Restart on *horizon* failures, after *horizon* steps
   
