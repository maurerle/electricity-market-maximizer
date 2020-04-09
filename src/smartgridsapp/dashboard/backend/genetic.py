# -*- coding: utf-8 -*-
from influxdb import InfluxDBClient
from datetime import datetime
import pandas as pd
import numpy as np
import json
from .intersection import intersection

STOP = 20
OFFSET = 12.0
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

"""
def _rect_inter_inner(x1,x2):
    n1=x1.shape[0]-1
    n2=x2.shape[0]-1
    X1=np.c_[x1[:-1],x1[1:]]
    X2=np.c_[x2[:-1],x2[1:]]
    S1=np.tile(X1.min(axis=1),(n2,1)).T
    S2=np.tile(X2.max(axis=1),(n1,1))
    S3=np.tile(X1.max(axis=1),(n2,1)).T
    S4=np.tile(X2.min(axis=1),(n1,1))
    return S1,S2,S3,S4

def _rectangle_intersection_(x1,y1,x2,y2):
    S1,S2,S3,S4=_rect_inter_inner(x1,x2)
    S5,S6,S7,S8=_rect_inter_inner(y1,y2)

    C1=np.less_equal(S1,S2)
    C2=np.greater_equal(S3,S4)
    C3=np.less_equal(S5,S6)
    C4=np.greater_equal(S7,S8)

    ii,jj=np.nonzero(C1 & C2 & C3 & C4)
    return ii,jj

def intersection(x1,y1,x2,y2):
    ii,jj=_rectangle_intersection_(x1,y1,x2,y2)
    n=len(ii)

    dxy1=np.diff(np.c_[x1,y1],axis=0)
    dxy2=np.diff(np.c_[x2,y2],axis=0)

    T=np.zeros((4,n))
    AA=np.zeros((4,4,n))
    AA[0:2,2,:]=-1
    AA[2:4,3,:]=-1
    AA[0::2,0,:]=dxy1[ii,:].T
    AA[1::2,1,:]=dxy2[jj,:].T

    BB=np.zeros((4,n))
    BB[0,:]=-x1[ii].ravel()
    BB[1,:]=-x2[jj].ravel()
    BB[2,:]=-y1[ii].ravel()
    BB[3,:]=-y2[jj].ravel()

    for i in range(n):
        try:
            T[:,i]=np.linalg.solve(AA[:,:,i],BB[:,i])
        except:
            T[:,i]=np.NaN


    in_range= (T[0,:] >=0) & (T[1,:] >=0) & (T[0,:] <=1) & (T[1,:] <=1)

    xy0=T[2:,in_range]
    xy0=xy0.T
    return xy0[:,0],xy0[:,1]

"""


class Genetic():
	def __init__(self, target, data, day, limit):
		self.target = target
		self.limit = limit*1e6
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

	def secRes(self):
		res = (
			self.client
			.query(f"SELECT * FROM STRes WHERE time = '{self.targetDay}'")
			.raw
		)
		return res['series'][0]['values'][0][1]

	def computeClearing1(self, off, bid):
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
			pun1 = self.computeClearing1(self.sup1, self.dem1)
			profit += self.getProfit1(self.sup1, self.dem1, pun1)

			pun2 = self.computeClearing1(self.sup2, self.dem2)
			profit += self.getProfit1(self.sup2, self.dem2, pun2)
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
			fitness[max_fitness_idx] = -99999999999*1e10
		
		return parents
	
	def mutation(self, off_xover):
		mutations_counter = np.uint8(off_xover.shape[1] / MUTATIONS)
		# Mutation changes a number of genes as defined by the num_mutations 
		# argument. The changes are random.
		for idx in range(off_xover.shape[0]):
			locus = mutations_counter - 1
			for i in range(MUTATIONS):
				rnd = np.random.uniform(-OFFSET, OFFSET, 1)
				off_xover[idx, locus] += rnd

				locus = locus + mutations_counter
				
		return off_xover

	def init_pop(self):
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
			print(f'Generation: {generation}')
			# Get the fitness value of the new population
			fitness = self.getFitness(new_pop)
			if np.max(fitness) == max(self.best_out):
				n_it += 1
				if n_it == STOP:
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

		# Compute the final fitness value	
		fitness = self.getFitness(new_pop)
		best_match = np.where(fitness == np.max(fitness))
		self.best_out.append(fitness[best_match][0])
		
		profits = []
		for i in range(len(self.best_out)):
			obj = {'x':i, 'y':round(self.best_out[i],2)}
			profits.append(obj)

		print(f'Best Solution: {new_pop[best_match,:]}')
		print(f'Best Solution Fitness: {fitness[best_match]}')
		
		return json.dumps(profits), new_pop[best_match,:]