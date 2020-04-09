# -*- coding: utf-8 -*-
from influxdb import InfluxDBClient
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from intersection import intersection
from matplotlib import rcParams 


class Genetic():
    """[summary]
    """
    def __init__(self, target):
        self.target = target
        self.limit = .0
        self.client = InfluxDBClient(
            'localhost', 
            8086, 
            'root', 
            'root', 
            'PublicBids'
        )
        # Drop the previous optimization measurements referred to the target 
        # operator from InfluxDB
        self.client.query(
            f"DELETE FROM optimization where op = '{self.target}'"
        )

        self.best_out = []

    def getData(self, market):
        """[summary]
        
        Parameters
        ----------
        market : [type]
            [description]
        
        Returns
        -------
        [type]
            [description]
        """
        # Get the demand data from InfluxDB
        res = (
            self.client
            .query(
                f"SELECT * FROM demand{market} WHERE time = '{targetDay}'"
            )
            .raw
        )

        dem =(
            pd
            .DataFrame(
                res['series'][0]['values'], 
                columns = ['time', 'P', 'Q', 'OPS']
            )
            .drop(columns=['time'])
            .set_index('OPS')
        )
        
        # Get the supply data from InfluxDB
        res = (
            self.client
            .query(
                f"SELECT * FROM supply{market} WHERE time = '{targetDay}'"
            )
            .raw
        )
        sup =(
            pd
            .DataFrame(
                res['series'][0]['values'], 
                columns = ['time', 'P', 'Q', 'OPS']
            )
            .drop(columns=['time'])
            .set_index('OPS')
        )
        
        return dem, sup

    def computeClearing(self):
        """[summary]
        
        Returns
        -------
        [type]
            [description]
        """
        sup = self.sup.sort_values('P', ascending=True)
        dem = self.dem.sort_values('P', ascending=False)
        # Cumulative sums of quantity
        sup_cum = np.cumsum(sup['Q'])
        dem_cum = np.cumsum(dem['Q'])
        # Find the curves intersection
        clearing = intersection(
            sup_cum.values, 
            sup.P.values, 
            dem_cum.values, 
            dem.P.values
        )[1][0]
        
        return clearing

    def getFitness(self, pop):
        """[summary]
        
        Parameters
        ----------
        pop : [type]
            [description]
        
        Returns
        -------
        [type]
            [description]
        """
        fitness = np.zeros((CHROMOSOMES))
        cnt = 0
        for individual in pop:
            # Determine the new curves
            self.sup.loc[target] = [individual[0], individual[1]]
            self.dem.loc[target] = [individual[2], individual[3]]
            # Set the 0 demanded price as the default one
            self.dem.P = self.dem.P.replace(0, 3000)
            # Determine the clearing price
            pun = self.computeClearing()
            
            # Compute the profits
            if self.sup.loc[target].P > pun:
                # Rejected bid for the supply
                Qsup = 0.0
            else:
                # Accepted bid for the supply
                Qsup = self.sup.loc[target].Q
            if self.dem.loc[target].P < pun:
                # Rejected bid for the demand
                Qdem = 0.0
            else:
                # Accepted bid for the demand
                Qdem = self.dem.loc[target].Q

            # Compute the profit
            profit = (Qsup - Qdem)*pun
            
            # Determine the fitness
            fitness[cnt] = profit
            cnt+=1
            
        return fitness
    
    def crossover(self, parents, off_size):
        """[summary]
        
        Parameters
        ----------
        parents : [type]
            [description]
        off_size : [type]
            [description]
        
        Returns
        -------
        [type]
            [description]
        """
        off = np.empty(off_size)
        # The point at which crossover takes place between two parents. 
        # Usually, it is at the center.
        xover_point = np.uint8(off_size[1]/2)

        for k in range(off_size[0]):
            # Index of the first parent to mate.
            parent1_idx = k%parents.shape[0]
            # Index of the second parent to mate.
            parent2_idx = (k+1)%parents.shape[0]
            # The new offspring will have its first half of its genes taken 
            # from the first parent.
            off[k, 0:xover_point] = parents[parent1_idx, 0:xover_point]
            # The new offspring will have its second half of its genes taken 
            # from the second parent.
            off[k, xover_point:] = parents[parent2_idx, xover_point:]
        
        return off

    def select_mating_pool(self, pop, fitness, num_parents):
        """[summary]
        
        Parameters
        ----------
        pop : [type]
            [description]
        fitness : [type]
            [description]
        num_parents : [type]
            [description]
        
        Returns
        -------
        [type]
            [description]
        """
        # Selecting the best individuals in the current generation as parents 
        # to produce the offspring of the next generation.
        parents = np.empty((num_parents, pop.shape[1]))
        for parent_num in range(num_parents):
            max_fitness_idx = np.where(fitness == np.max(fitness))
            max_fitness_idx = max_fitness_idx[0][0]
            parents[parent_num, :] = pop[max_fitness_idx, :]
            fitness[max_fitness_idx] = -99999999999
        
        return parents
    
    def mutation(self, off_xover):
        """[summary]
        
        Parameters
        ----------
        off_xover : [type]
            [description]
        
        Returns
        -------
        [type]
            [description]
        """
        mutations_counter = np.uint8(off_xover.shape[1] / MUTATIONS)
        # Mutation changes a number of genes as defined by the num_mutations 
        # argument. The changes are random.
        for idx in range(off_xover.shape[0]):
            locus = mutations_counter - 1
            for i in range(MUTATIONS):
                # The random value to be added to the gene.
                rnd = np.random.uniform(-20.0, 20.0, 1)
                off_xover[idx, locus] = (
                    off_xover[idx, locus] + rnd
                )
                # Check if all values are non-negatives or if the production 
                # limit is not exceeded. Otherwise mutate again until the 
                # constraints are not respected.
                while (not np.all(off_xover[idx, locus])>=0) or \
                    (off_xover[idx, 1] < off_xover[idx, 3] + self.limit):
                        rnd = np.random.uniform(-20.0, 20.0, 1)
                        off_xover[idx, locus] = off_xover[idx, locus] + rnd
                locus = locus + mutations_counter
                
        return off_xover

    def generateObs(self, fitness, pop):
        """[summary]
        
        Parameters
        ----------
        fitness : [type]
            [description]
        pop : [type]
            [description]
        """
        best_match = np.where(fitness == np.max(fitness))
        sol = pop[best_match,:]
        
        body = [{
            'tags':{
                'op':target
            },
            'measurement':f'optimization',
            'fields':{
                'Psup':sol[0][0][0],
                'Qsup':sol[0][0][1],
                'Pdem':sol[0][0][2],
                'Qdem':sol[0][0][3],
                'Profit':fitness[best_match][0]
            }
        }]
        self.client.write_points(body)

    def init_pop(self):
        """[summary]
        
        Returns
        -------
        [type]
            [description]
        """
        self.dem, self.sup = self.getData('MGP')
        # Define the production limit
        self.limit = self.sup.loc[self.target].Q - self.dem.loc[self.target].Q

        # Determine the forecasted/original profit to check
        # the optimization in the end.
        zero_pop = np.asarray(
            [
                self.sup.loc[self.target].P,
                self.sup.loc[self.target].Q,
                self.dem.loc[self.target].P,
                self.dem.loc[self.target].Q
            ]
        )
        zero_profit = self.getFitness(np.asarray([zero_pop]))[0]
        # Start from the forcasted/original solution and
        # create the first population by perturbating it
        # in the [-4, 4] range
        new_pop = np.copy(zero_pop)
        for i in range(CHROMOSOMES-1):
            temp_pop = zero_pop+np.random.uniform(low = -4.0, high=4.0)
            new_pop = np.vstack((new_pop, temp_pop))
        
        return new_pop

    def run(self):
        """[summary]
        """
        new_pop = self.init_pop()

        for generation in range(N_GENERATIONS):
            if generation%100 == 0:
                print(f'Generation: {generation}')
            fitness = self.getFitness(new_pop)
            # Generate Observations and save the best output
            self.generateObs(fitness, new_pop)
            self.best_out.append(np.max(fitness))
            # Select the parents
            parents = self.select_mating_pool(new_pop, fitness, N_PARENTS)
            # Perform Crossover and mutation
            off_size = (POP_SIZE[0]-parents.shape[0], GENES)
            offspring_mutation = self.mutation(
                self.crossover(
                    parents, 
                    off_size
                )
            )
            # Replace the old generation with the new one
            new_pop[0:parents.shape[0], :] = parents
            new_pop[parents.shape[0]:, :] = offspring_mutation

        # Compute the final fitness value    
        fitness = self.getFitness(new_pop)
        best_match = np.where(fitness == np.max(fitness))

        print(f'Best Solution: {new_pop[best_match,:]}')
        print(f'Best Solution Fitness: {fitness[best_match]}')






global GENES
global CHROMOSOMES
global POP_SIZE
global N_PARENTS
global N_GENERATIONS
global MUTATIONS 
# Number of genes (4 variables * 3 markets)
GENES = 4
# Number of chromosome
CHROMOSOMES = 8
# Population size
POP_SIZE = (CHROMOSOMES, GENES)
# Number of parents mating
N_PARENTS = 4
# Number of generations
N_GENERATIONS = 6000
# Number of mutations
MUTATIONS = 2

target = 'IREN ENERGIA SPA'
ga = Genetic(target)
targetDay = datetime.strptime('20170210','%Y%m%d')
ga.run()