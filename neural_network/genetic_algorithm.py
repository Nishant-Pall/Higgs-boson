import numpy as np
import random

class GeneticAlgorithm(object):
    """Simple generational genetic algorithm."""

    def __init__(	self, chromosomeShape,
                    errorFunction,
                    elitism = 1,
                    populationSize = 25,
                    mutationProbability  = .1,
                    mutationScale = .5,
                    numIterations = 10000,
                    errorTreshold = 1e-6
                    ):
        
        self.f = errorFunction # The error function (reversely proportionl to fitness)
        self.keep = elitism  # Number of units to keep for elitism
        self.populationSize = populationSize # Size of the population of units
        self.p = mutationProbability # Probability of mutation
        self.k = mutationScale # Scale of the gaussian noise during mutation
        self.numIter = numIterations # Maximum number of iterations
        self.e = errorTreshold # Threshold of error while iterating

        self.i = 0 # Iteration counter

        # Initializes the population randomly from a gaussian distribution
        # with noise 0.1, sorts the values and stores them internally.
        self.population = []
        for _ in range(populationSize):
            chromosome = 0.1 * np.random.randn(chromosomeShape)            
            fitness = self.calculateFitness(chromosome)
            self.population.append((chromosome, fitness))

        # Sort descending according to fitness (larger is better).
        self.population = sorted(self.population, key=lambda t: -t[1])
        # Computing population's selection ordering to save time during parents selection.
        self.selectionOrdering = self.computeSelectionOrdering()

    def calculateFitness(self, chromosome):
        """
        Fitness metric used is negative value of the error function.
        We leverage the fact that error function is reversely proportional
        to fitness. Fitness is larger as the unit is better.
        """
        return -self.f(chromosome)

    def computeSelectionOrdering(self):
        """
        Forms population's selection ordering with regard to specimen's 
        relative fitness. Specimens are sorted in a descending order.
        """
        averageFit, worstFit = self.averageAndWorstFitness()
        divisor = 1.0 * self.populationSize * (averageFit - worstFit)
        result = []
        
        for chromosome, fit in self.population:
            relativeFit = (fit - worstFit) / divisor
            result.append((chromosome, fit, relativeFit))
        
        return sorted(result, key=lambda t: -t[2])

    def averageAndWorstFitness(self):
        """
        Returnes a tuple with average amd worst fitness from the 
        current population.
        """
        fins = [x[1] for x in self.population]
        return (sum(fins)/len(fins), fins[-1])

    def step(self):
        """
        Runs one iteration of the genetic algorithm. In a single iteration,
        we create a whole new population by first keeping the best units as 
        defined by elitism, then we iteratively select parents from the 
        current population, apply crossover and then mutation.

        Function returns a tuple:
            * boolean value indicating should the iteration stop (True if
                the learning process is finished, False othwerise)
            * an integer representing the current iteration of the
                algorithm
            * the weights of the best unit in the current iteration
        """
        self.i += 1
        
        newPopulation = self.bestN(self.keep)
        iterations = self.populationSize - self.keep

        while iterations:
            p1, p2 = self.selectParents()
            childChromosome = self.crossover(p1[0], p2[0])
            childChromosome = self.mutate(childChromosome)

            newPopulation.append((childChromosome, self.calculateFitness(childChromosome)))
            iterations -= 1

        # Sort descending according to fitness (larger is better).
        self.population = sorted(newPopulation, key=lambda t: -t[1])
        # Computing population's selection ordering to save time during parents selection.
        self.selectionOrdering = self.computeSelectionOrdering()

        return (self.i == self.numIter, self.i, self.best()[0][0])

    def bestN(self, n):
        """
        Returns the best 'n' units from the population.
        """
        return self.population[:n]

    def best(self):
        """
        Returns the best unit from the population.
        """
        return self.bestN(1) 

    def selectParents(self):
        """
        Selects two parents from the population.
        """
        p1 = self.selectBestForProb(random.random())
        p2 = self.selectBestForProb(random.random())
        return (p1,p2)

    def selectBestForProb(self, prob):
        """
        Peforms relative proportional selection of two parents from
        the current population. Selection is proportional to the
        fitness of the units in the population.
        """
        lower = 0
        upper = prob
        for specimen in self.selectionOrdering:
            if lower + specimen[2] >= lower and lower + specimen[2] <= upper:
                return (specimen[0], specimen[1])
            lower = upper
            upper = upper + specimen[2]

    def crossover(self, p1, p2): 
        """
        Given two parent units 'p1' and 'p2', function performs a simple 
        crossover by averaging their values in order to create a new 
        child unit.
        """
        return np.mean(np.array([p1, p2]), axis=0)

    def mutate(self, chromosome):
        """
        Given a unit, function mutates its values by applying gaussian 
        noise according to the parameter 'k'.
        """
        for i, weight in enumerate(chromosome):
            if random.random() < self.p:
                chromosome[i] += np.random.normal(0,self.k,1)
        return chromosome
