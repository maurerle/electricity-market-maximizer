from influxdb import InfluxDBClient
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from intersection import intersection
import pulp as pl

class Temp():
	def __init__(self, target):
		self.target = target
		self.limit = .0
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

	def getData(self):
		# Get the demand data from InfluxDB
		
		res = (
		    self.client
		    .query(
		        f"SELECT * FROM predictions"
		    )
		    .raw
		)
		cols = res['series'][0]['columns']
		res = res['series'][0]['values']

		data = pd.DataFrame(res, columns=cols)
		data = data.set_index('op').drop(columns='time').replace(-1.0, np.nan)

		#print(pd.DataFrame(data['MGPpO']).dropna())

		return data

	def getFitness(self, market):
		for col in market.columns:
			if 'pD' in col:
				market[:][col] = market[:][col].replace(0.0, 3000.0)
		for m in ['MGP']:
			off = pd.merge(
				market[:][f'{m}pO'].dropna(),
				market[:][f'{m}qO'].dropna(),
				left_index=True,
				right_index=True
			)
			dem= pd.merge(
				market[:][f'{m}pD'].dropna(),
				market[:][f'{m}qD'].dropna(),
				left_index=True,
				right_index=True
			)
			pun = self.computeClearing(off, dem, m)
	 	
		return pun

	def computeClearing(self, off, bid, m):
		sup = off.sort_values(f'{m}pO', ascending=True)
		dem = bid.sort_values(f'{m}pD', ascending=False)
		# Cumulative sums of quantity
		sup_cum = np.cumsum(sup[f'{m}qO'])
		dem_cum = np.cumsum(dem[f'{m}qD'])
		# Find the curves intersection
		clearing = intersection(
		    sup_cum.values, 
		    sup[f'{m}pO'].values, 
		    dem_cum.values, 
		    dem[f'{m}pD'].values
		)[1][0]
		
		return clearing

target = 'IREN ENERGIA SPA'
ga = Temp(target)
market = ga.getData()
pun = ga.getFitness(market)
print(pun)

for col in market.columns:
	if 'pD' in col:
		market[:][col] = market[:][col].replace(0.0, 3000.0)
for m in ['MGP']:
	off = pd.merge(
		market[:][f'{m}pO'].dropna(),
		market[:][f'{m}qO'].dropna(),
		left_index=True,
		right_index=True
	)
	dem= pd.merge(
		market[:][f'{m}pD'].dropna(),
		market[:][f'{m}qD'].dropna(),
		left_index=True,
		right_index=True
	)


def computeClearing(dem, sup, p1O, q1O, p1D, q1D):
	# Replace the target variable
	dem.loc[target][['MGPqD', 'MGPpD']] = [q1D, p1D]
	sup.loc[target][['MGPqO', 'MGPpO']] = [q1O, p1O]
	dem['MGPpD'] = dem['MGPpD'].replace(0.0, 3000.0)
	# Sort the prices
	sup = off.sort_values(f'MGPpO', ascending=True)
	dem = dem.sort_values(f'MGPpD', ascending=False)
	# Cumulative sums of quantity
	sup_cum = np.cumsum(sup[f'MGPqO'])
	dem_cum = np.cumsum(dem[f'MGPqD'])
	# Find the curves intersection
	clearing = intersection(
		sup_cum.values, 
		sup[f'{m}pO'].values, 
		dem_cum.values, 
		dem[f'{m}pD'].values
	)[1][0]
	
	return clearing



# Define the problem
problem = pl.LpProblem("Example", pl.LpMaximize)

# Define the decision variables
qO = pl.LpVariable('qO', cat='Continuous')
qD = pl.LpVariable('qD', cat='Continuous')
pO = pl.LpVariable('pO', cat='Continuous')
pD = pl.LpVariable('pD', cat='Continuous')

# Create the curves
dem = ga.getData()[['MGPqD', 'MGPpD']].dropna()
sup = ga.getData()[['MGPqO', 'MGPpO']].dropna()

# Shapes
O = sup.shape[0]
D = dem.shape[0]

# Define the objective function
func = (qO - qD)*computeClearing(dem, sup, pO, qO, pD, qD)
problem += func

# Constraint
problem += pl.LpConstraint(qO - qD == 50.0)

# Solve the problem
status = problem.solve()
print(pl.LpStatus[status])

"""

qOacc = pl.LpVariable.dicts('qOacc', ((i) for i in range(pO.shape[0])), cat='Continuous')
qOrej = pl.LpVariable.dicts('qOrej', ((i) for i in range(pO.shape[0])), cat='Continuous')
qDacc = pl.LpVariable.dicts('qDacc', ((i) for i in range(pD.shape[0])), cat='Continuous')
qDrej = pl.LpVariable.dicts('qDrej', ((i) for i in range(pD.shape[0])), cat='Continuous')

pOacc = pl.LpVariable.dicts('pOacc', ((i) for i in range(pO.shape[0])), cat='Continuous')
pOrej = pl.LpVariable.dicts('pOrej', ((i) for i in range(pO.shape[0])), cat='Continuous')
pDacc = pl.LpVariable.dicts('pDacc', ((i) for i in range(pD.shape[0])), cat='Continuous')
pDrej = pl.LpVariable.dicts('pDrej', ((i) for i in range(pD.shape[0])), cat='Continuous')

qOlamb = pl.LpVariable(name='qOlamb', lowBound = 0, cat='Continuous')
qDlamb = pl.LpVariable(name='qDlamb', lowBound = 0, cat='Continuous')
pOlamb = pl.LpVariable(name='pOlamb', lowBound = 0, cat='Continuous')
pDlamb = pl.LpVariable(name='pDlamb', lowBound = 0, cat='Continuous')

eps = pl.LpVariable(name='eps', lowBound=0, cat='Continuous')

#problem += ga.computeClearing(dem, off, 'MGP')

for i in range(pO.shape[0]):
	problem += pl.LpConstraint(qOacc[i] + qOrej[i] == qO[i])
for i in range(pD.shape[0]):
	problem += pl.LpConstraint(qDacc[i] + qDrej[i] == qD[i])

problem += pl.LpConstraint(pl.lpSum(qOacc[i] for i in range(pO.shape[0])) <= qOlamb)
problem += pl.LpConstraint(pl.lpSum(qDacc[i] for i in range(pD.shape[0])) >= qDlamb)

problem += pl.LpConstraint(pl.lpSum(qOrej[i] for i in range(pO.shape[0])) >= qOlamb+1e-10)
problem += pl.LpConstraint(pl.lpSum(qDrej[i] for i in range(pD.shape[0])) <= qDlamb+1e-10)

for i in range(pO.shape[0]):
	problem += pl.LpConstraint((pOacc[i] + pOrej[i])/2 == pO[i])
	problem += pl.LpConstraint(pOacc[i] <= pOlamb)
	problem += pl.LpConstraint(pOrej[i] >= pOlamb+1e-10)
for i in range(pD.shape[0]):
	problem += pl.LpConstraint((pDacc[i] + pDrej[i])/2 == pD[i])
	problem += pl.LpConstraint(pDacc[i] >= pDlamb)
	problem += pl.LpConstraint(pDrej[i] <= pDlamb+1e-10)

#problem += pl.LpConstraint(pOlamb*qOlamb - pDlamb*qDlamb <= eps)
problem += pl.LpConstraint(pOlamb - pDlamb <= eps)
problem += pl.LpConstraint(qOlamb - qDlamb <= eps)
problem += pl.LpConstraint(eps >= 1e-10)



"""