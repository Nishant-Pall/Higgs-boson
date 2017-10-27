import numpy as np
import random

class GeneticAlgorithm(object):
    """
        Implement a simple generationl genetic algorithm as described in the instructions
    """

    def __init__(	self, chromosomeShape,
                    errorFunction,
                    elitism = 1,
                    populationSize = 25,
                    mutationProbability  = .1,
                    mutationScale = .5,
                    numIterations = 10000,
                    errorTreshold = 1e-6
                    ):

        self.populationSize = populationSize # size of the population of units
        self.p = mutationProbability # probability of mutation
        self.numIter = numIterations # maximum number of iterations
        self.e = errorTreshold # threshold of error while iterating
        self.f = errorFunction # the error function (reversely proportionl to fitness)
        self.keep = elitism  # number of units to keep for elitism
        self.k = mutationScale # scale of the gaussian noise

        self.i = 0 # iteration counter

        # initialize the population randomly from a gaussian distribution
        # with noise 0.1 and then sort the values and store them internally

        flag = True

        self.population = []
        for _ in range(populationSize):
            chromosome = 0.1 * np.random.randn(chromosomeShape)            
            fitness = self.calculateFitness(chromosome)

            self.population.append((chromosome, fitness))

        # sort descending according to fitness (larger is better)
        self.population = sorted(self.population, key=lambda t: -t[1])

        # computing average and worst fintess to save time during parents selection
        self.averageFit, self.worstFit = self.averageAndWorstFintess()
        # computing selection ordering to save time during parents selection
        self.selectionOrdering = self.computeSelectionOrdering()

    def averageAndWorstFintess(self):
        fins = [x[1] for x in self.population]
        return (sum(fins)/len(fins), fins[-1])
    
    def computeSelectionOrdering(self):
        divisor = 1.0 * self.populationSize * (self.averageFit - self.worstFit)
        result = []
        
        for chromosome, fit in self.population:
            relativeFit = (fit - self.worstFit) / divisor
            result.append((chromosome, fit, relativeFit))
        
        return sorted(result, key=lambda t: -t[2])

    def step(self):
        """
            Run one iteration of the genetic algorithm. In a single iteration,
            you should create a whole new population by first keeping the best
            units as defined by elitism, then iteratively select parents from
            the current population, apply crossover and then mutation.

            The step function should return, as a tuple:
            * boolean value indicating should the iteration stop (True if
                the learning process is finished, False othwerise)
            * an integer representing the current iteration of the
                algorithm
            * the weights of the best unit in the current iteration
        """
        self.i += 1
        #############################
        #       YOUR CODE HERE      #
        #############################

        newPopulation = self.bestN(self.keep)
        iterations = self.populationSize - self.keep

        while iterations:
            p1, p2 = self.selectParents()
            childChromosome = self.crossover(p1[0], p2[0])
            childChromosome = self.mutate(childChromosome)

            newPopulation.append((childChromosome, self.calculateFitness(childChromosome)))
            iterations -= 1

        # sort descending according to fitness (larger is better)
        self.population = sorted(newPopulation, key=lambda t: -t[1])

        # computing average and worst fintess to save time during parents selection
        self.averageFit, self.worstFit = self.averageAndWorstFintess()
        # computing selection ordering to save time during parents selection
        self.selectionOrdering = self.computeSelectionOrdering()

        return (self.i == self.numIter, self.i, self.best()[0][0])

    def calculateFitness(self, chromosome):
        """
            Implement a fitness metric as a function of the error of
            a unit. Remember - fitness is larger as the unit is better!
        """
        chromosomeError = self.f(chromosome)
        #############################
        #       YOUR CODE HERE      #
        #############################
        return -chromosomeError

    def bestN(self, n):
        """
            Return the best n units from the population
        """
        #############################
        #       YOUR CODE HERE      #
        #############################
        return self.population[:n]

    def best(self):
        """
            Return the best unit from the population
        """
        #############################
        #       YOUR CODE HERE      #
        #############################
        return self.bestN(1) 

    def selectParents(self):
        """
            Select two parents from the population with probability of 
            selection proportional to the fitness of the units in the
            population		
        """
        #############################
        #       YOUR CODE HERE      #
        #############################
        firstProb = random.random()
        secondProb = random.random()

        p1 = self.selectBestForProb(firstProb)
        p2 = self.selectBestForProb(secondProb)

        return (p1,p2)

    def selectBestForProb(self, prob):
        lower = 0
        upper = prob
        for specimen in self.selectionOrdering:
            if lower + specimen[2] >= lower and lower + specimen[2] <= upper:
                return (specimen[0], specimen[1])
            lower = upper
            upper = upper + specimen[2]

    def crossover(self, p1, p2): 
        """
            Given two parent units p1 and p2, do a simple crossover by 
            averaging their values in order to create a new child unit
        """
        #############################
        #       YOUR CODE HERE      #
        #############################
        return np.mean(np.array([p1, p2]), axis=0)

    def mutate(self, chromosome):
        """
            Given a unit, mutate its values by applying gaussian noise
            according to the parameter k
        """
        #############################
        #       YOUR CODE HERE      #
        #############################
        for i, weight in enumerate(chromosome):
            if random.random() < self.p:
                chromosome[i] += np.random.normal(0,self.k,1)
        return chromosome
