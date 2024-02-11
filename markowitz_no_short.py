import numpy as np

# w an array of portfolio weights
# mean is the array of expected returns
# cov is the covariance matrix for the returns on the assets in the portfolio
# q is a "risk tolerance" factor, where 0 results in the portfolio with minimal
# risk and infinity results in the portfolio infinitely far out on the frontier
# with both expected return and risk unbounded

# minimising w @ cov @w - q * mean @ w is equivalent to maximising
# the negative of that formula
def portfolio_value(w, mean, cov, q):
    return - w @ cov @ w + q * mean @ w


# If you differentiate w.r.t. w you get
# - 2 w @ cov + q * mean
# Setting this to 0 and solving for w gives
# 0.5 * q * mean @ np.linalg.pinv(cov)
# We can't short anything so set the minimum value to be 0
# and then divide by the sum so that the sum of the positions is 1.
def diff_starting_point(mean, cov, q):
    raw_maximiser = 0.5 * q * mean @ np.linalg.pinv(cov)
    print("Derivative = 0")
    print(raw_maximiser)
    non_negative = np.maximum(raw_maximiser, 0)
    standardised = non_negative / np.sum(non_negative)
    return np.array(standardised)


# Propose a tweak to the current portfolio w,
# sample nudge ~ |N(mean = 0, sd = 0.01)|
# sample indices i, j
# tweak the nudge so that the following two actions below won't push any
# element of w to be outside the interval [0, 1]
# w[i] -> w[i] + nudge
# w[j] -> w[j] - nudge
def propose_new_move(w):
    nudge = np.abs(np.random.normal(0, 0.01, 1)[0])
    i, j = np.random.choice(a=np.arange(len(w)), size=2, replace=False)
    if 1 < w[i] + nudge:
        nudge = 1 - w[i]
    if w[j] - nudge < 0:
        nudge = w[j]
    w[i] += nudge
    w[j] -= nudge
    return w


# calculate the acceptance probability for the new proposal, using the
# formula for simulated annealing. Don't bother capping it above at 1.
def new_move_acceptance_probability(w, prop_w, T, mean, cov, q):
    e = portfolio_value(w, mean, cov, q)
    prop_e = portfolio_value(prop_w, mean, cov, q)
    return np.exp((prop_e - e)/T)


# try multiple starting points and print out the solution that
# they generate in order to check convergence. Increase temperature or
# number of iterations in order to obtain convergence.
def simulated_annealing(mean, cov, q, temperature, its):
    starting_points = [diff_starting_point(mean, cov, q)]
    starting_points.append(np.array([1] * len(mean)) / len(mean))
    for i in range(len(mean)):
        w_temp = np.zeros(len(mean))
        w_temp[i] = 1.0
        starting_points.append(w_temp)
    print("Starting points: ")
    print(starting_points)
    print("\n")

    best_ws = []
    best_portfolio_values = []
    for w in starting_points:
        print("Starting point")
        print(w)
        best_w = w
        best_portfolio_value = portfolio_value(w, mean, cov, q)
        for it in range(its):
            T = temperature * ((its - it) / its)
            prop_w = propose_new_move(w)
            nmap = new_move_acceptance_probability(w, prop_w, T, mean, cov, q)
            if np.random.uniform(size=1)[0] <= nmap:
                w = prop_w
                if best_portfolio_value < portfolio_value(w, mean, cov, q):
                    best_w = w.copy()
                    best_portfolio_value = portfolio_value(w, mean, cov, q)
        print("best_w, best_portfolio_value")
        print(best_w, best_portfolio_value)
        print("\n")
        best_ws.append(best_w.copy())
        best_portfolio_values.append(best_portfolio_value)
    bpv_index = np.argmax(best_portfolio_values)
    return best_ws[bpv_index], best_portfolio_values[bpv_index]
