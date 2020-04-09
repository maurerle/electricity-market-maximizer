# -*- coding: utf-8 -*-
from influxdb import InfluxDBClient
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from intersection import intersection
from matplotlib import rcParams 


class Genetic():
	def __init__(self, target, data):
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


	def secRes(self):
		res = (
			self.client
			.query(f"SELECT * FROM STRes WHERE time = '{targetDay}'")
			.raw
		)
		return res['series'][0]['values'][0][1]


	def computeClearing1(self, off, bid):
		sup = off[(off > 0).any(1)]
		dem = bid[(bid > 0).any(1)]
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
		sup = off[(off > 0).any(1)]
		dem = bid[(bid > 0).any(1)]
		# Sort the prices
		sup = off.sort_values('P', ascending=True)
		dem = bid.sort_values('P', ascending=False)
		# Cumulative sums of quantity
		sup_cum = np.cumsum(sup['Q'])
		dem_cum = np.cumsum(dem['Q'])
		# Get the MSD quantity threshold
		#th = self.secRes()
		th = 6400
		# Create the th curve
		x_th = [th, th]
		y_th = [0, np.max(sup.P.values)]
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
		"""
		plt.figure()
		plt.plot(sup_cum.values, sup.P.values)
		plt.plot(dem_cum.values, dem.P.values)
		plt.plot(x_th, y_th)
		plt.show()
		exit()
		"""

		return clearingD, clearingS

	def getProfit1(self, sup, dem, pun):
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
		fitness = np.zeros((CHROMOSOMES))
		cnt = 0
		profit = 0
		# For each market
		for individual in pop:
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
			pun1 = self.computeClearing1(self.sup1, self.dem1)
			pun2 = self.computeClearing1(self.sup2, self.dem2)
			#pun3d, pun3s = self.computeClearing2(self.sup3, self.dem3)
			pun3 = self.computeClearing1(self.sup3, self.dem3)
						
			# Determine the profit
			profit += self.getProfit1(self.sup1, self.dem1, pun1)
			profit += self.getProfit1(self.sup2, self.dem2, pun2)
			#profit += self.getProfit2(self.sup3, self.dem3, pun3s, pun3d)
			profit += self.getProfit1(self.sup3, self.dem3, pun3)

			# Define the fitness as the profit
			fitness[cnt] = profit
			cnt+=1
		
		return fitness
	
	def crossover(self, parents, off_size):
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
		mutations_counter = np.uint8(off_xover.shape[1] / MUTATIONS)
		# Mutation changes a number of genes as defined by the num_mutations 
		# argument. The changes are random.
		for idx in range(off_xover.shape[0]):
			locus = mutations_counter - 1
			for i in range(MUTATIONS):
				n_it = 0
				while not self.stop:
					rnd = np.random.uniform(-100.0, 100.0, 1)
					to_check = off_xover.copy()
					to_check[idx, locus] += rnd
					# Constraints
					domain = np.all(to_check>=0)
					qDem = to_check[idx, 3]+to_check[idx, 7]+to_check[idx, 11]
					qSup = to_check[idx, 1]+to_check[idx, 5]+to_check[idx, 9]
					#u_b_max = qSup - qDem + to_check[idx, 11] <= self.limit
					#u_b_min = qSup - qDem + to_check[idx, 11] >= 0
					#l_b = qSup - qDem - to_check[idx, 9] >= 0 
					u_b = qSup - qDem <= self.limit
					l_b = qSup - qDem >= 0
					# Until the constraints are not satisfied, increase the 
					# number of iterations and try again
					#if not domain or not u_b_max or not u_b_min or not l_b:
					if not domain or not u_b or not l_b:
						n_it+=1
						# If the number of iterations reaches the stopping
						# condition, stop the algorithm
						if n_it == STOP:
							self.stop = True
					else:
						off_xover = to_check
						break
				locus = locus + mutations_counter
				
		return off_xover

	def init_pop(self):
		self.limit=150000.0
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
		"""
		self.dem1, self.sup1 = self.getData('MGP')
		self.dem2, self.sup2 = self.getData('MI')
		self.dem3, self.sup3 = self.getData('MSD')
		"""
		
		# Determine the forecasted/original profit to check
		# the optimization in the end.
		zero_pop = np.asarray(
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
		"""
		repl = np.where(zero_pop==-1.0)
		for i in repl[0]:
			rand = np.random.uniform(low = 0.0, high=10.0)
			zero_pop[i] = rand
		"""
		#zero_pop = np.random.rand(12)
		zero_pop = np.zeros(12)
		
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
		generation = 0
		while not self.stop:
			print(f'Generation: {generation}')
			# Get the fitness value of the new population
			fitness = self.getFitness(new_pop)
			# If the maximum fitness value is better than the previous one,
			# store it
			if max(self.best_out)<=np.max(fitness):
				self.best_out.append(np.max(fitness))
			# Otherwise start trying the best solution of this generation
			n_it = 0	
			while max(self.best_out) >= np.max(fitness) and not self.stop:
				temp_pop = np.copy(new_pop)
				# Select the parents
				parents = self.select_mating_pool(new_pop, fitness, N_PARENTS)
				# Perform Crossover and mutation
				off_size = (POP_SIZE[0]-parents.shape[0], GENES)
				off_xover = self.crossover( parents, off_size)
				off_mutation = self.mutation(off_xover)
				# Replace the old temporary population with the new one
				temp_pop[0:parents.shape[0], :] = parents
				temp_pop[parents.shape[0]:, :] = off_mutation
				# Compute the new fitness value
				fitness = self.getFitness(temp_pop)
				
				#print(fitness)
				
				n_it+=1
				# Check if the stopping condition is found
				if n_it == STOP:
					self.stop = True
					break
				
			# If the stoping condition is not found, replace the old generation 
			# with the new one and proceed with the next generation
			if not self.stop:
				new_pop[0:parents.shape[0], :] = parents
				new_pop[parents.shape[0]:, :] = off_mutation
				generation+=1
				
				# Generate observation by founding the new maximum fitness
				# value and the relative solution
				best_match = np.where(fitness == np.max(fitness))
				best_sol = new_pop[best_match,:][0][0]
				"""
				if generation >= 5:
					plt.plot(self.best_out)
					plt.ylim(self.best_out[5], self.best_out[-1])
					plt.draw()
					plt.pause(0.001)
				"""
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
CHROMOSOMES = 10
# Population size
POP_SIZE = (CHROMOSOMES, GENES)
# Number of parents mating
N_PARENTS = 2
# Number of generations
N_GENERATIONS = 6000
# Number of mutations
MUTATIONS = 12


#plt.figure()
"""
target = 'IREN ENERGIA SPA'
#target = '2V ENERGY SRL'
ga = Genetic(target, data)
targetDay = datetime.strptime('01/01/2020', '%d/%m/%Y')
ga.run()
"""