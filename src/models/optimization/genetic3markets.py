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
        self.market = []
        self.stop = False
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

    def computeClearing(self, off, bid):
        """[summary]
        
        Returns
        -------
        [type]
            [description]
        """
        sup = off.sort_values('P', ascending=True)
        dem = bid.sort_values('P', ascending=False)
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
        profit = 0
        # For each market
        for individual in pop:
            for i in range(3):
                # Determine the new curves
                self.market[i][1].loc[self.target] = [individual[4*i+0], individual[4*i+1]]
                self.market[i][0].loc[self.target] = [individual[4*i+2], individual[4*i+3]]
                # Set the 0 demanded price as the default one
                self.market[i][0].P = self.market[i][0].P.replace(0, 3000)
                # Determine the clearing price
                pun = self.computeClearing(self.market[i][1], self.market[i][0])
                
                # Compute the profits
                if self.market[i][1].loc[self.target].P > pun:
                    # Rejected bid for the supply
                    Qsup = 0.0
                else:
                    # Accepted bid for the supply
                    Qsup = self.market[i][1].loc[self.target].Q
                if self.market[i][0].loc[self.target].P < pun:
                    # Rejected bid for the demand
                    Qdem = 0.0
                else:
                    # Accepted bid for the demand
                    Qdem = self.market[i][0].loc[self.target].Q

                # Compute the profit
                profit += (Qsup - Qdem)*pun

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
                n_it = 0
                while ((not np.all(off_xover[idx, locus])>=0) or \
                    (off_xover[idx, 1]+off_xover[idx, 5]+off_xover[idx, 9] > off_xover[idx, 3]+off_xover[idx, 7]+off_xover[idx, 11] + self.limit)) and \
                    not self.stop:
                        #rnd = np.random.uniform(-50.0, 50.0, 1)
                        rnd = np.random.uniform(-20.0, 20.0, 1)
                        off_xover[idx, locus] += rnd
                        n_it +=1
                        if n_it == STOP:
                            self.stop = True
                            break
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
                'PsupMGP':sol[0][0][0],
                'QsupMGP':sol[0][0][1],
                'PdemMGP':sol[0][0][2],
                'QdemMGP':sol[0][0][3],
                'PsupMI':sol[0][0][4],
                'QsupMI':sol[0][0][5],
                'PdemMI':sol[0][0][6],
                'QdemMI':sol[0][0][7],
                'PsupMSD':sol[0][0][8],
                'QsupMSD':sol[0][0][9],
                'PdemMSD':sol[0][0][10],
                'QdemMSD':sol[0][0][11],
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
        for m in ['MGP', 'MI', 'MSD']:
            self.market.append(self.getData(m))
        """self.dem, self.sup = self.getData('MGP')"""
        # Define the production limit, where market[0][1] is MGP sup, 
        # market[0][0] is MGP dem.
        """
        for i in range(3):
            self.limit += self.market[i][1].loc[self.target].Q - self.market[i][0].loc[self.target].Q
        """
        self.limit=15000.0
        # Determine the forecasted/original profit to check
        # the optimization in the end.
        zero_pop = np.asarray(
            [
                self.market[0][1].loc[self.target].P,
                self.market[0][1].loc[self.target].Q,
                self.market[0][0].loc[self.target].P,
                self.market[0][0].loc[self.target].Q,
                self.market[1][1].loc[self.target].P,
                self.market[1][1].loc[self.target].Q,
                self.market[1][0].loc[self.target].P,
                self.market[1][0].loc[self.target].Q,
                self.market[2][1].loc[self.target].P,
                self.market[2][1].loc[self.target].Q,
                self.market[2][0].loc[self.target].P,
                self.market[2][0].loc[self.target].Q
            ]
        )
        zero_profit = self.getFitness(np.asarray([zero_pop]))[0]
        self.best_out.append(zero_profit)
        # Start from the forcasted/original solution and
        # create the first population by perturbating it
        # in the [-4, 4] range
        new_pop = np.copy(zero_pop)
        for i in range(CHROMOSOMES-1):
            temp_pop = zero_pop+np.random.uniform(low = 3.0, high=4.0)
            new_pop = np.vstack((new_pop, temp_pop))
        
        return new_pop

    def run(self):
        """[summary]
        """
        new_pop = self.init_pop()

        fitness = self.getFitness(new_pop)

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
        generation = 0
        while not self.stop:
        #for generation in range(N_GENERATIONS):
            print(f'Generation: {generation}')
            fitness = self.getFitness(new_pop)
            # Generate Observations and save the best output
            if max(self.best_out)<=np.max(fitness):
                self.generateObs(fitness, new_pop)
                self.best_out.append(np.max(fitness))
                
            while max(self.best_out) >= np.max(fitness) and not self.stop:
                temp_pop = np.copy(new_pop)
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
                
                temp_pop[0:parents.shape[0], :] = parents
                temp_pop[parents.shape[0]:, :] = offspring_mutation
                fitness = self.getFitness(temp_pop)

                if max(self.best_out)==np.max(fitness):
                    print('Equal')
                    break
            if not self.stop:
                # Replace the old generation with the new one
                new_pop[0:parents.shape[0], :] = parents
                new_pop[parents.shape[0]:, :] = offspring_mutation
                generation+=1

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
global STOP
STOP = 1000
# Number of genes (4 variables * 3 markets)
GENES = 12
# Number of chromosome
CHROMOSOMES = 6
# Population size
POP_SIZE = (CHROMOSOMES, GENES)
# Number of parents mating
N_PARENTS = 4
# Number of generations
N_GENERATIONS = 6000
# Number of mutations
MUTATIONS = 6

target = 'IREN ENERGIA SPA'
ga = Genetic(target)
targetDay = datetime.strptime('20170210','%Y%m%d')
ga.run()