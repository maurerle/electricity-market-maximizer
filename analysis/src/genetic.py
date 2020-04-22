# -*- coding: utf-8 -*-
from influxdb import InfluxDBClient
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .intersection import intersection
from matplotlib import rcParams 
import time

STOP = 500
# Number of genes (4 variables * 3 markets)
GENES = 12
# Number of chromosome
CHROMOSOMES = 10
# Population size
POP_SIZE = (CHROMOSOMES, GENES)
# Number of parents mating
N_PARENTS = 2
# Number of mutations
MUTATIONS = 12

np.random.seed(17)

class Genetic():
	"""Genetic algorithm class for the optimization problem.
    
    Parameters
    ----------
    target : str
        Operator whose profit has to be optimized
    data : pandas.DataFrame
        Prediction data
    day : datetime.datetime
        Day of interest
    offset : float
        Mutation offset
    
    Attributes
    -------
    target : str
        Operator whose profit has to be optimized
    limit : float
        [description]
    stop : bool
        Stopping flag
    client : influxdb.InfluxDBClient
        InfluxDB Client
    best_out : list
        Optimized profits
    data : pandas.DataFrame
        Prediction data
    targetDay : datetime.datetime
        Day of interest
    offset : float
        Mutation offset
    dem1 : pandas.DataFrame
        Demand curve of MGP
    dem2 : pandas.DataFrame
        Demand curve of MI
    dem3 : pandas.DataFrame
        Demand curve of MSD
    sup1 : pandas.DataFrame
        Suuply curve of MGP
    sup2 : pandas.DataFrame
        Suuply curve of MI
    sup3 : pandas.DataFrame
        Suuply curve of MSD
    original_profit : list
        Original profit
    
    Methods
    -------
    secRes()
        Get the secondary reserve data
    computeClearing1(off, bid)
        Simplified computation of the clearing price for MGP/MI
    computeClearing2(off, bid)
        Simplified computation of the clearing price for MSD
    getProfit1(sup, dem, pun)
        Evaluate the profit in MGP/MI market
    getProfit2(sup, dem, punS, punD)
        Evaluate the profit in MSD market
    getFitness(pop)
        Evaluate the fitness values
    crossover(parents, off_size)
        Crossover process
    select_mating_pool(pop, fitness, num_parents)
        Select the best parents
    mutation(off_xover)
        Mutation process
    init_pop()
        Population initialization
    run()
        Run the genetic algorithm
	"""
	def __init__(self, target, data, day, offset):
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
		self.data = data
		self.targetDay = day
		self.offset = offset

	def secRes(self):
		"""Query for the reserve thresholds.
		
		Returns
		-------
		list
			Secondary reserve thresholds
		"""
		res = (
			self.client
			.query(f"SELECT * FROM STRes WHERE time = '{self.targetDay}'")
			.raw
		)
		return res['series'][0]['values'][0][1]

	def computeClearing1(self, off, bid):
		"""Simplified computation of the clearing price for MGP/MI
		
		Parameters
		----------
		off : pandas.DataFrame
			Offers
		bid : pandas.DataFrame
			Bids
		
		Returns
		-------
		float
			Clearing price
		"""
		sup = off[(off >= 0).any(1)]
		dem = bid[(bid >= 0).any(1)]
		# Sort the prices
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

	def computeClearing2(self, off, bid):
		"""Simplified computation of the 'dummy' clearing price for MSD
		
		Parameters
		----------
		off : pandas.DataFrame
			Offers
		bid : pandas.DataFrame
			Bids
		
		Returns
		-------
		float, float
			Clearing price of demand and supply
		"""
		sup = off[(off >= 0).any(1)]
		dem = bid[(bid >= 0).any(1)]
		# Sort the prices
		sup = off.sort_values('P', ascending=True)
		dem = bid.sort_values('P', ascending=False)
		# Cumulative sums of quantity
		sup_cum = np.cumsum(sup['Q'])
		dem_cum = np.cumsum(dem['Q'])
		# Get the MSD quantity threshold
		th = self.secRes()
		# Create the th curve
		x_th = np.array([th, th])
		y_th = np.array([0, np.max(sup.P.values)])

		clearingD = intersection( 
			dem_cum.values, 
			dem.P.values, 
			x_th,
			y_th
		)[1][0]
		clearingS = intersection( 
			sup_cum.values, 
			sup.P.values,
			x_th,
			y_th
		)[1][0]
		
		return clearingD, clearingS

	def getProfit1(self, sup, dem, pun):
		"""Evaluate the profit in MGP/MI market.
		
		Parameters
		----------
		sup : pandas.DataFrame
			Supply curve
		dem : pandas.DataFrame
			Demand curve
		pun : float
			National Single Price
		
		Returns
		-------
		float
			Profit
		"""
		if sup.loc[self.target].P > pun:
			# Rejected bid for the supply
			Qsup = 0.0
		else:
			# Accepted bid for the supply
			Qsup = sup.loc[self.target].Q
		if dem.loc[self.target].P < pun:
			# Rejected bid for the demand
			Qdem = 0.0
		else:
			# Accepted bid for the demand
			Qdem = dem.loc[self.target].Q
		
		return (Qsup - Qdem)*pun
	
	def getProfit2(self, sup, dem, punS, punD):
		"""Evaluate the profit in MSD market.
		
		Parameters
		----------
		sup : pandas.DataFrame
			Supply curve
		dem : pandas.DataFrame
			Demand curve
		punS : float
			National Single Price of supply 
		punD : float
			National Single Price of demand 
		
		Returns
		-------
		float
			Profit
		"""
		if sup.loc[self.target].P > punS:
			# Rejected bid for the supply
			Qsup = 0.0
		else:
			# Accepted bid for the supply
			Qsup = sup.loc[self.target].Q
		Psup = sup.loc[self.target].P
		
		if dem.loc[self.target].P < punD:
			# Rejected bid for the demand
			Qdem = 0.0
		else:
			# Accepted bid for the demand
			Qdem = dem.loc[self.target].Q
		Pdem = dem.loc[self.target].P
		return Qsup*Psup - Qdem*Pdem

	def getFitness(self, pop):
		"""Evaluation of the fitness values.
		
		Parameters
		----------
		pop : numpy.ndarray
			Population
		
		Returns
		-------
		numpy.ndarray
			Fitness values
		"""
		fitness = np.zeros((CHROMOSOMES))
		cnt = 0
		# For each market
		for individual in pop:
			profit = 0
			# Determine the new curves
			self.sup1.loc[self.target] = [individual[0], individual[1]]
			self.dem1.loc[self.target] = [individual[2], individual[3]]
			self.sup2.loc[self.target] = [individual[4], individual[5]]
			self.dem2.loc[self.target] = [individual[6], individual[7]]
			self.sup3.loc[self.target] = [individual[8], individual[9]]
			self.dem3.loc[self.target] = [individual[10], individual[11]]
			# Set the 0 demanded price as the default one
			self.dem1.P = self.dem1.P.replace(0, 3000)
			self.dem2.P = self.dem2.P.replace(0, 3000)
			self.dem3.P = self.dem3.P.replace(0, 3000)
			# Determine the clearing price for MGP and MI
			try:
				pun1 = self.computeClearing1(self.sup1, self.dem1)
				profit += self.getProfit1(self.sup1, self.dem1, pun1)
			except IndexError:
				profit -= 1e8
			try:
				pun2 = self.computeClearing1(self.sup2, self.dem2)
				profit += self.getProfit1(self.sup2, self.dem2, pun2)
			except IndexError:
				profit -= 1e8
			try:
				pun3d, pun3s = self.computeClearing2(self.sup3, self.dem3)
				profit += self.getProfit2(self.sup3, self.dem3, pun3s, pun3d)
			except IndexError:
				profit -= 1e8

			domain = np.all(individual>=0)
			qDem = individual[3]+individual[7]+individual[11]
			qSup = individual[1]+individual[5]+individual[9]
			u_b = qSup - qDem <= self.limit
			l_b = qSup - qDem >= 0

			if not domain:
				profit -= 1e6
			if not u_b:
				profit -= 1e6
			if not l_b:
				profit -= 1e6

			# Define the fitness as the profit
			fitness[cnt] = profit
			cnt+=1
		
		return fitness
	
	def crossover(self, parents, off_size):
		"""Crossover process of the genetic algorithm.
		
		Parameters
		----------
		parents : numpy.ndarray
			Parents
		off_size : int
			Offset size
		
		Returns
		-------
		numpy.ndarray
			Generated children
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
		"""Select the best individuals in the current generation as parents.
		
		Parameters
		----------
		pop : numpy.ndarray
			Population
		fitness : numpy.ndarray
			Fitness values
		num_parents : int
			Number of parents
		
		Returns
		-------
		numpy.ndarray
			Best parents
		"""
		# Selecting the best individuals in the current generation as parents 
		# to produce the offspring of the next generation.
		parents = np.empty((num_parents, pop.shape[1]))
		for parent_num in range(num_parents):
			max_fitness_idx = np.where(fitness == np.max(fitness))
			max_fitness_idx = max_fitness_idx[0][0]
			parents[parent_num, :] = pop[max_fitness_idx, :]
			fitness[max_fitness_idx] = -99999999999*1e10
		
		return parents
	
	def mutation(self, off_xover):
		"""Mutation process of the genetic algorithm.
		
		Parameters
		----------
		off_xover : numpy.ndarray
			Original children
		
		Returns
		-------
		numpy.ndarray
			Modified children
		"""
		mutations_counter = np.uint8(off_xover.shape[1] / MUTATIONS)
		# Mutation changes a number of genes as defined by the num_mutations 
		# argument. The changes are random.
		for idx in range(off_xover.shape[0]):
			locus = mutations_counter - 1
			for i in range(MUTATIONS):
				rnd = np.random.uniform(-self.offset, self.offset, 1)
				off_xover[idx, locus] += rnd

				locus = locus + mutations_counter
				
		return off_xover

	def init_pop(self):
		"""Initialization of the population.
		
		Returns
		-------
		numpy.ndarray
			Population
		"""
		self.limit=1500000000.0
		self.dem1 = (
			self.data[['MGPqD','MGPpD']]
			.rename(columns={'MGPqD':'Q', 'MGPpD':'P'})
		)
		self.sup1 = (
			self.data[['MGPqO','MGPpO']]
			.rename(columns={'MGPqO':'Q', 'MGPpO':'P'})
		)
		self.dem2 = (
			self.data[['MIqD','MIpD']]
			.rename(columns={'MIqD':'Q', 'MIpD':'P'})
		)
		self.sup2 = (
			self.data[['MIqO','MIpO']]
			.rename(columns={'MIqO':'Q', 'MIpO':'P'})
		)
		self.dem3 = (
			self.data[['MSDqD','MSDpD']]
			.rename(columns={'MSDqD':'Q', 'MSDpD':'P'})
		)
		self.sup3 = (
			self.data[['MSDqO','MSDpO']]
			.rename(columns={'MSDqO':'Q', 'MSDpO':'P'})
		)
		original_pop = np.asarray(
			[ 
				self.sup1.loc[self.target].P, self.sup1.loc[self.target].Q,
				self.dem1.loc[self.target].P, self.dem1.loc[self.target].Q,
				self.sup2.loc[self.target].P, self.sup2.loc[self.target].Q,
				self.dem2.loc[self.target].P, self.dem2.loc[self.target].Q,
				self.sup3.loc[self.target].P, self.sup3.loc[self.target].Q,
				self.dem3.loc[self.target].P, self.dem3.loc[self.target].Q
			]
		) 
		# If the target operator has some -1 due to the prediction
		# fail, replace it with random number
		repl = np.where(original_pop==-1.0)
		for i in repl[0]:
			rand = np.random.uniform(low = 0.0, high=10.0)
			original_pop[i] = rand
		
		zero_pop = np.zeros(12)
		
		
		self.original_profit = self.getFitness([original_pop])[0]
		zero_profit = self.getFitness([zero_pop])
		self.best_out.append(zero_profit[0])
		
		# Start from the forcasted/original solution and
		# create the first population by perturbating it
		# in the [3, 4] range
		new_pop = np.copy(zero_pop)
		for i in range(CHROMOSOMES-1):
			temp_pop = zero_pop.copy()
			for j in range(GENES):
				temp_pop[j] = zero_pop[j]+np.random.uniform(low = 0.0, high=10.0)
			new_pop = np.vstack((new_pop, temp_pop))
		
		return new_pop

	def run(self):
		"""Method to run the genetic algorithm.
		
		Returns
		-------
		list, numpy.ndarray, int
			Optimized profits, best solution, number of generations
		"""
		# Create the first population
		new_pop = self.init_pop()
		# Determine the fitness value for the population and store the first
		# fitness value as the original one
		fitness = self.getFitness(new_pop)
		self.best_out.append(np.max(fitness))
		
		# Select the parents
		parents = self.select_mating_pool(new_pop, fitness, N_PARENTS)
		# Perform Crossover and Mutation
		off_size = (POP_SIZE[0]-parents.shape[0], GENES)
		off_xover = self.crossover( parents, off_size)
		off_mutation = self.mutation(off_xover)

		# Start the evolution until the stopping condition is not found
		generation = 1
		#for generation in range(1, 100):
		n_it = 0
		while True:
			#print(f'Generation: {generation}')
			# Get the fitness value of the new population
			fitness = self.getFitness(new_pop)
			if np.max(fitness) == max(self.best_out):
				n_it += 1
				if n_it == 20:
					break
			else:
				n_it = 0
			self.best_out.append(np.max(fitness))
			# Select the parents
			parents = self.select_mating_pool(new_pop, fitness, N_PARENTS)
			# Perform Crossover and Mutation
			off_size = (POP_SIZE[0]-parents.shape[0], GENES)
			off_xover = self.crossover( parents, off_size)
			off_mutation = self.mutation(off_xover)
			# If the stoping condition is not found, replace the old generation 
			# with the new one and proceed with the next generation
			new_pop[0:parents.shape[0], :] = parents
			new_pop[parents.shape[0]:, :] = off_mutation
			generation+=1
			
			# Generate observation by founding the new maximum fitness
			# value and the relative solution
			best_match = np.where(fitness == np.max(fitness))
			best_sol = new_pop[best_match,:][0][0]
		print(generation)
		# Compute the final fitness value	
		fitness = self.getFitness(new_pop)
		best_match = np.where(fitness == np.max(fitness))
		self.best_out.append(fitness[best_match][0])

		print(f'Best Solution: {new_pop[best_match,:]}')
		print(f'Best Solution Fitness: {fitness[best_match]}')

		return self.best_out, new_pop[best_match,:], generation