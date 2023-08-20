import random
from cma import fitness_functions
from cma.evolution_strategy import CMAEvolutionStrategy, CMAEvolutionStrategyResult
from cma.optimization_tools import BestSolution
from evaluate import get_env, get_state_action_size, evaluate
from policy import NeuroevoPolicy
from argparse import ArgumentParser
import numpy as np
import logging
import cma
import time
import datetime
import cmaes
from cmaes import SepCMA
        

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


def cma_optim(x, fitness, gens, std=0.2):
    x_best = x
    f_best = -np.Inf
    n_evals = 0
    optimizer = SepCMA(mean=np.zeros(len(x)), sigma=std, population_size=int(1000/gens))
    print(f"number of policy parameters = {optimizer.dim}")
    print(f"population size = {optimizer.population_size}")
    for generation in range(gens):
        solutions = []
        for i in range(optimizer.population_size):
            # Check number of evaluations (ADDED eval LOOP WITHOUT TESTING)
            if (n_evals + i) == 1000:
                print("\n\t\tWARNING: ARRIVED AT MAXIMUM NUMBER OF EVALUATIONS 1000, DID NOT UPDATE\n")
                print(f"#{generation} {f_best} {x_best}")
                n_evals += optimizer.population_size
                logging.info('\t%d\t%d', n_evals, f_best)
                return x_best
            # Ask a parameter
            x = optimizer.ask()
            value = fitness(x)
            solutions.append((x, value))
            if value > f_best:
                f_best = value
                x_best = x
        print(f"#{generation} {f_best} {x_best}")
        # Tell evaluation values.
        optimizer.tell(solutions)
        n_evals += optimizer.population_size
        logging.info('\t%d\t%d', n_evals, f_best)
    return x_best


def geneic_mutation(x, fitness, gens,  std=0.01, rng = np.random.default_rng()):
    # Choose the two elites the most fit 
    def select_elites(populations,fitness):
        fit = [fitness(populations[i]) for i in range(len(populations))]
        indice_elite = np.argsort(fit)[len(fit)-2:len(fit)]
        print ("WE choose the best two elites:"+ str(fit[indice_elite[0]]) +" and "+ str(fit[indice_elite[1]]))
        return indice_elite,fit 

    # Crossover to generate 1 offstring (percentage en fonction du two fitness of elites)
    def cross(elites, percentage):
        elites_copy = np.copy(elites)
        if (len(elites_copy)!=2):
            print ("Parents are not 2 !")
            return None
        else:
            for i in range(int(percentage* len(elites_copy[0]))):
                index = random.randint(0, len(elites_copy[0])-1)
                elites_copy[0][index] = elites_copy[1][index]
            return elites_copy[0]
            
    #  Mutation Percentage probability
    def mutate (elistes, probability):
        elistes_copy = np.copy(elistes)
        for i in range(len(elistes_copy)):
            for j in range(len(elistes_copy[0])):
                prob = random.randint(0,9)
                if (prob<=probability*10):
                    # error = rng.uniform(-4,4)  # juste pour small
                    error = rng.uniform(-4,4)*2  # juste pour large
                    elistes_copy[i][j] = elistes_copy[i][j] + error    
        return  elistes_copy
        

    # Initialize variable
    mutation_probality = 0.8
    pop_n = 10
    gen_n = len(x)
    iteration_number = 1000/pop_n
    x_best = x
    X = []
    F = []
    N = rng.normal(size=(pop_n, gen_n)) * std *100
    for people in range(pop_n):
        ind = x + N[people, :]
        X.append(ind)
        F.append(fitness(ind))
    for gen in range(int(iteration_number)):
        # Initialisation of population
        indice_elite =[]
        fit = []
        elites_selected = []
        indice_elite,fit = select_elites(X,fitness)
        cross_offs = []
        mutated_offs = []

        # choose two elites 
        elite1 = X[indice_elite[0]]
        x_best = elite1
        elite2 = X[indice_elite[1]]
        # record their fitness
        fit1 = -fit[indice_elite[0]]
        fit2 = -fit[indice_elite[1]]
        elites_selected.append(elite1)
        elites_selected.append(elite2)
        # calculate the probability of crossover
        prob = (1/fit1)/(1/fit1+1/fit2)
        cross_offs  = [cross(elites_selected, prob) for i in range(pop_n-2)]
        # define the probablity of mutation for each gene
        mutated_offs = mutate(cross_offs,mutation_probality)
        X = []
        X.append(elite1)
        X.append(elite2)
        for j in range(len(mutated_offs)):
           X.append(mutated_offs[j])
        print ("First gnenration: " +  str(np.sort(fit)[len(fit)-1]))
    return x_best


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
    # x_best = approx_gradient(start, fit, args.gens, args.pop)      
    # x_best = cma_optim(start, fit, 300, std = 0.2)
    x_best = geneic_mutation(start, fit, args.gens, rng = rng)  

    endtime = time.time()
    print(endtime - starttime)

    # Evaluation
    policy.set_params(x_best)
    policy.save(args.weights)
    best_eval = evaluate(env, params, policy)
    print('Best individual: ', x_best[:5])
    print('Fitness: ', best_eval)
