from statsmodels.stats.proportion import proportion_confint
# an example of NCBI
alpha = 0.02
total, rate = 964, 0.8959
positives = int(total*rate)
low, _ = proportion_confint(positives, total, method='wilson', alpha=alpha)
lower_bound = rate - low
print('lower bound = {a}'.format(a=lower_bound))