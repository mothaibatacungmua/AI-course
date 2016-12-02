import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

np.random.seed(1234)

N = 300
data = np.random.randn(N)

fig,ax = plt.subplots()
ax.hist(data, 40, normed=True, color='red')


# mu ~ N(0, r^2)
# x|mu ~ N(mu, sig^2)
# mean(x)|mu ~ N(mu, 1/n*sig^2)
# mu|mean(x) ~ N([(sig^2/n)*mu + r^2*mean(x)]/[(sig^2/n) + r^2], [(sig^2/n)*r^2]/[(sig^2/n) + r^2]

# Compute analytically the posterior distribution
mu0 = 3.
r = 1.
sig = 1.
mu_post = (sig**2*mu0 + r**2*np.sum(data))/(sig**2 + r**2*N)
sig_post = sig**2 * r**2 / (sig**2 + N*r**2)

print "Posterior mean:%05f, Posterior variance:%05f" % (mu_post, sig_post)

posterior = scipy.stats.norm(mu_post, sig_post).pdf
x = np.arange(-6, 6, 0.01)

fig, ax= plt.subplots()
ax.plot(x, posterior(x), linewidth=2, color='g')
ax.set_xlim(-0.2, 0.2)

# Metropolis-Hastings method
proposal_sd = 2.
mu_prior_mu = 3.
mu_prior_sd = 2.
mu_init = 1.5
Ns = 10000
mu_current = mu_init
posterior = [mu_current]

for i in range(0, Ns):
    # suggest new position
    mu_proposal = scipy.stats.norm(mu_current, proposal_sd**2).rvs()

    # Compute likehood, P(X|mu_current) and P(X|mu_proposal)
    likehood_current = scipy.stats.norm(mu_current, 1).pdf(data).prod()
    likehood_proposal = scipy.stats.norm(mu_proposal, 1).pdf(data).prod()

    # Compute prior probability of current and proposed mu
    prior_current = scipy.stats.norm(mu_prior_mu, mu_prior_sd**2).pdf(mu_current)
    prior_proposal = scipy.stats.norm(mu_prior_mu, mu_prior_sd**2).pdf(mu_proposal)

    p_current = likehood_current * prior_current
    p_proposal = likehood_proposal * prior_proposal

    # Accept proposal
    p_accept = p_proposal / p_current

    accept = np.random.rand() < p_accept

    if accept:
        mu_current = mu_proposal

    posterior.append(mu_current)


print 'MCMC posterior mean:%05f, MCMC posterior variance:%05f' % (np.mean(posterior), np.var(posterior))

fig, ax = plt.subplots()
ax.hist(posterior, 30, normed=True)
theorical_mean = 0.0
theorical_var = np.var(data)/len(data)
ax.set_xlim([-1, 1])

i = np.arange(0, len(posterior))
fig, ax = plt.subplots()
ax.plot(i, posterior)
ax.set_ylim([-0.3, 0.3])

plt.show()