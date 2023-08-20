from evaluate import get_env, get_state_action_size, evaluate
from policy import NeuroevoPolicy
from argparse import ArgumentParser
import numpy as np
import logging
import cma
import time


def oneplus_lambda(x, fitness, gens, lam, std=0.01, rng=np.random.default_rng()):
    x_best = x
    f_best = -np.Inf
    n_evals = 0
    for g in range(gens):
        N = rng.normal(size=(lam, len(x))) * std
        for i in range(lam):
            ind = x + N[i, :]
            f = fitness(ind)
            if f > f_best:
                f_best = f
                x_best = ind
        x = x_best
        n_evals += lam
        # logging.info('\t%d\t%d', n_evals, f_best)
        print(n_evals, f_best)
    return x_best


def approx_gradient(x, fitness, gens, lam, alpha=0.2, verbose=False):
    x_best = x
    f_best = fitness(x)
    fits = np.zeros(gens)
    n_evals = 0
    for g in range(gens):
        N = np.random.normal(size=(lam, len(x)))
        F = np.zeros(lam)
        for i in range(lam):
            ind = x + N[i, :]
            F[i] = fitness(ind)
            if F[i] > f_best:
                f_best = F[i]
                x_best = ind
                if verbose:
                    print(g, " ", f_best)
        fits[g] = f_best
        mu_f = np.mean(F)
        std_f = np.std(F)
        A = F
        if std_f != 0:
            A = (F - mu_f) / std_f
        x = x - alpha * np.dot(A, N) / lam
        n_evals += lam
        print(n_evals, f_best)
    return x_best

def cma_optim(x, fitness, gens):
    es = cma.CMAEvolutionStrategy(2 * [0], 0.1, {'popsize': 20, 'verb_disp': 1})

    es.optimize(fitness)
    # res = es.result()

    # while not es.stop():
    #  solutions = es.ask()
    #  es.tell(solutions, [himmelblau(s) for s in solutions])
    #  es.disp()
    return es.result_pretty()



def fitness(x, s, a, env, params):
    policy = NeuroevoPolicy(s, a)
    policy.set_params(x)
    return evaluate(env, params, policy)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-e', '--env', help='environment', default='small', type=str)
    parser.add_argument('-g', '--gens', help='number of generations', default=100, type=int)
    parser.add_argument('-p', '--pop', help='population size (lambda for the 1+lambda ES)', default=10, type=int)
    parser.add_argument('-s', '--seed', help='seed for evolution', default=0, type=int)
    parser.add_argument('--log', help='log file', default='evolution.log', type=str)
    parser.add_argument('--weights', help='filename to save policy weights', default='weights', type=str)
    args = parser.parse_args()
    # logging.basicConfig(filename=args.log, encoding='utf-8', level=logging.DEBUG,
                        # format='%(asctime)s %(message)s')

    # starting point
    env, params = get_env(args.env)
    s, a = get_state_action_size(env)
    policy = NeuroevoPolicy(s, a)

    # evolution
    rng = np.random.default_rng(args.seed)
    start = rng.normal(size=(len(policy.get_params(),)))
    starttime = time.time()
    def fit(x):
        return fitness(x, s, a, env, params)
    # x_best = oneplus_lambda(start, fit, args.gens, args.pop, rng=rng)
    x_best = approx_gradient(start, fit, args.gens, args.pop)      
    # x_best = cma_optim(start, fit, args.gens)

    endtime = time.time()
    print(endtime - starttime)

    # Evaluation
    policy.set_params(x_best)
    policy.save(args.weights)
    best_eval = evaluate(env, params, policy)
    print('Best individual: ', x_best[:5])
    print('Fitness: ', best_eval)
