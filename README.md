# vasgd
Versatile Accelerated Stochastic Gradient Descent Experiments

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
   
