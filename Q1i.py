import numpy as np

# running  statistics to save memory and running time.
# For exact solution, Divide an Conquer technique was used.
# global variables were used to store the results outside the recursive function.

def Solve_Q1_exactly(n, m):
    """
    This function will solve the challenge question recursively (Divide and Conquer technique).
    accepts n and m
    updates global variables: x is combination array, M_PROD is product of face values, S2 is variance
    n_combinations is the number of possible combinations.
    """
    global x
    global M_PROD
    global S2
    global n_combinations
    if n == 1:
        n_combinations += 1
        x[0] = m
        prod = x.prod(dtype=np.longfloat)
        if n_combinations == 1:
            M_PROD = prod
            S2  = 0
        else:
            M_old = M_PROD
            M_PROD = M_old + (prod - M_old)/n_combinations
            S2 = S2 + (prod - M_old)*(prod - M_PROD)
    elif (n <= m <= 6 * n):
    # because the face values are restricted to 1~6,
    # the smallest possible value of m is n when all faces are 1,
    # the largest possible value of m is 6*n when all faces are 6.
        for i in range(max(1, m - 6 * (n - 1)), min (6, m - n + 1) + 1):
        # the same rule for the face values applies for this part.
        # Using the Divde and Concur technique, try assigning the face value of the last dice
        # x[n - 1] = i, and call Solve_Q1(n-1,m-i) for recursively solving for the remaining dices.
        # i (face value of the last dice) has to satisfy the following conditions
        # (1 <= i <= 6) and ((n - 1) <= (m - i) <= 6 * (n - 1))
        # or equivalently,
        # (1 <= i <= 6) and (m - 6 * (n - 1) <= i <= m - n + 1)
        # or equivalently,
        # max(1, m - 6 * (n - 1)) <= i <= min (6, m - n + 1)
        # these range was used in the for loop.
            x[n - 1] = i
            Solve_Q1_exactly(n - 1, m - i)

def Solve_Q1_MonteCarlo(n, m, numTimes, tol):
    global x
    global M_PROD
    M_old = M_PROD
    global S2
    global n_combinations
    stdev = 0
    stdev_old = 0
    delta_stdev = 0
    np.random.seed(42)
    for i in range(numTimes):
        x = np.random.randint(1,7,n)
        if x.sum() == m:
            n_combinations += 1
            prod = x.prod(dtype=np.longfloat)
            if n_combinations == 1:
                M_PROD = prod
                S2 = 0
                print('M_prod = %f, S2 = %f with first found combination' % (M_PROD, S2))
                print('x = ' + str(x) + ', x.prod() = %i: first found combination of face values' % prod)
            else:
                M_old = M_PROD
                M_PROD = M_old + (prod - M_old) / n_combinations
                delta_M_PROD = abs(M_PROD - M_old)
                S2 = S2 + (prod - M_old) * (prod - M_PROD)
                stdev_old = stdev
                stdev = (S2 / n_combinations) ** 0.5
                delta_stdev = abs(stdev - stdev_old)
                if delta_M_PROD/M_PROD < tol and delta_stdev/stdev < tol:
                    print('reached tolerance level tol = %f after %i iterations' % (tol,i))
                    break
    return i

################    beginning of script for checking the algorithm Solve_Q1_exactly(N,M) with a few simple cases  ################
# initialize variables (including global variables)
N, M = 4, 10
x = np.ones(N, dtype=np.int)
M_PROD = 1.0  # running average of products
S2 = 0.0  # running variance of products
n_combinations = 0

print('======================================================')
print('N=%i, M=%i' % (N, M))
Solve_Q1_exactly(N, M)
running_stdev = (S2 / n_combinations) ** 0.5

print('running average of products is %.10f' % M_PROD)
print('running standard deviation of products is %.10f' % running_stdev)
print('n_combinations = %i (this should be 80 for N, M = 4, 10)' % n_combinations)
# n_combinations should be equal to P(M,N,6) * (6**N). Reference: "http://mathworld.wolfram.com/Dice.html"
# P(10,4,6) * (6**4) = 80
print('x = ' + str(x) + ': combination of face values at the last iteration step')
################    end of script for checking the algorithm Solve_Q1_exactly(N,M) with a few simple cases  ################

################  beginning of script for exactly solving the 1st part of Q1  ################
# initialize variables (including global variables)
N, M = 8, 24  # parameters for the first half of the question
x = np.ones(N, dtype=np.int)
M_PROD = 1.0  # running average of products
S2 = 0.0    # running variance of products
n_combinations = 0

print('======================================================')
print('N=%i, M=%i, exact method' % (N,M))
Solve_Q1_exactly(N,M)    # 324 ms per loop for N=8 and M=24.
running_stdev = (S2/n_combinations)**0.5

print('running average of products is %.6f' % M_PROD)
print('running standard deviation of products is %.7f' % running_stdev)
# print('difference is %.10f' % (products.std()-running_stdev))
# # For N=8, M =24,
# # benchmark standard deviation of products is 855.0698853474
# # running standard deviation of products is 855.0698853474
# # difference is -0.0000000000
# answer to be submitted: 855.0698853 (10 digits of precision)

# print('benchmark number of combinations = %i' % sums.size)
print('n_combinations = %i (from running statistics)' % n_combinations)
# print('difference is %.10f' % (sums.size-n_combinations))
# # number of combinations = 98813
# # difference is 0
print('x = ' + str(x) + ': combination of face values at the last iteration step')
################  end of script for exactly solving the 1st part of Q1  ################

################  beginning of script for solving the 1st part of Q1 with Monte_Carlo method  ################
# solving this exactly is not possible due to too many possible combinations.
# initialize variables (including global variables)
N, M = 8, 24  # parameters for the first half of the question
x = np.ones(N, dtype=np.int)
M_PROD = 1.0  # running average of products
S2 = 0.0    # running variance of products
n_combinations = 0
n_samples_simulated = 0
n_times = 100000000
tol = 1E-6
print('======================================================')
print('N=%i, M=%i, Monte Carlo Method' % (N,M))
n_samples_simulated = Solve_Q1_MonteCarlo(N,M,n_times, tol)
running_stdev = (S2/n_combinations)**0.5

print('running average of products is %.10f (target 1859.932954)' % M_PROD)
print('S2 = %.10f' % S2)
print('running standard deviation of products is %.10f (target 855.0698853)' % running_stdev)
print('%i out of %i simulated combinations had sum of face values = %i)' % (n_combinations,n_samples_simulated,M))
# print('x = ' + str(x) + ', x.sum() = %i: combination of face values at the last iteration step (may not meet the sum condition)' % x.sum())
################    end of script for solving the 1st part of Q1 with Monte_Carlo method  ################

################  beginning of script for solving the 2nd part of Q1 with Monte_Carlo method  ################
# solving this exactly is not possible due to too many possible combinations.
# initialize variables (including global variables)
N, M = 50, 150  # parameters for the second part of the question
x = np.ones(N, dtype=np.int)
M_PROD = 1.0  # running average of products
S2 = 0.0    # running variance of products
n_combinations = 0
n_samples_simulated = 0
n_times = 10000000
tol = 1E-6
print('======================================================')
print('N=%i, M=%i, Monte Carlo Method' % (N,M))
n_samples_simulated = Solve_Q1_MonteCarlo(N,M,n_times, tol)
running_stdev = (S2/n_combinations)**0.5

print('running average of products is %.10f' % M_PROD)
print('S2 = %.10f' % S2)
print('running standard deviation of products is %.10f)' % running_stdev)
print('%i out of %i simulated combinations had sum of face values = %i)' % (n_combinations,n_samples_simulated,M))
# print('x = ' + str(x) + ', x.sum() = %i: combination of face values at the last iteration step (may not meet the sum condition' % x.sum())
################    end of script for solving the 2nd part of Q1 with Monte_Carlo method  ################

# for submission
# If N=8 and M=24, what is the expected value of the product? If N=8 and M=24, what is the standard deviation of the product?
# 1859.932954 (10 digits of precision)                         855.0698853 (10 digits of precision)
# If N=50 and M=150, what is the expected value of the product? If N=50 and M=150, what is the standard deviation of the product?
# 185316824894790598656.0000000000                             295666668177792761856.0000000000