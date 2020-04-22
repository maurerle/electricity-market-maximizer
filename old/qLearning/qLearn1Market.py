# -*- coding: utf-8 -*-
from influxdb import InfluxDBClient
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from intersection import intersection
from matplotlib import rcParams 
import itertools


class Market():
    def __init__(self):
        self.client = InfluxDBClient(
            '172.28.5.1', 
            8086, 
            'root', 
            'root', 
            'PublicBids'
        )
        # Demand and Supply curves
        self.dem, self. sup = self.getData('MGP')
        # Action Space
        self.act_space = np.asarray([i for i in itertools.product([0,1, -1], repeat=4)])
        self.act_space[:,1] = 10.0*self.act_space[:,1]
        self.act_space[:,3] = 10.0*self.act_space[:,3]
        self.act_space[:,0] = .5*self.act_space[:,0]
        self.act_space[:,2] = .5*self.act_space[:,2]
        
        # Q-Table initialization
        # Number of Q-Table columns
        self.q_cols = self.act_space.shape[0]+self.act_space.shape[1]
        self.n_var = 4
        self.alpha = 0.6
        self.gamma = 1.0


    def initQ(self, state):
        Q = np.zeros((1, self.q_cols), dtype=float)
        Q[0,:self.n_var] = state
        Q = pd.DataFrame(Q)

        return Q
    
    
    def getData(self, market):
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


    def step(self, state):
        # Determine the new curves
        self.sup.loc[target] = [state[0], state[1]]
        self.dem.loc[target] = [state[2], state[3]]
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

        return profit

    
    def getStateinQ(self, state):
        try:
            # If the actual state is already in the Q-Table
            states = self.Q.loc[:,:self.n_var-1]
            temp = states.where(states == state).dropna()
            # Get its index
            idx = temp.index.values[0]

            return(idx)
        
        except IndexError:
            new_Q = np.zeros((1, self.q_cols), dtype=float)
            new_Q[0,:self.n_var] = state
            new_Q = pd.DataFrame(new_Q)
            self.Q = self.Q.append(new_Q, ignore_index = True)

            return(self.getStateinQ(state))


    def assign_reward(self, delta_p):
        if np.any(self.nxt_state<0):
            return -10
        if self.nxt_state[1]>self.nxt_state[3]:
            return -10
        if delta_p > 0:
            return 20
        elif delta_p < 0:
            return -20
        else:
            return -1


    def exploitExplore(self, state_idx, epsilon=0.2):
        if np.random.uniform(0, 1) < epsilon:
            # Explore: Randomly choose an action
            action_idx = np.random.choice(np.arange(self.act_space.shape[0]))
        else:
            # Exploit: Select the action with max value
            action_idx = self.Q.iloc[state_idx][self.n_var:].argmax()
            
        return action_idx

    def run(self, init_state):
        # The first four columns are the state variables
        self.Q = self.initQ(init_state)
        # Previous Profit
        #self.prev_profit = self.step(state)
        profit_list = []

        for i in range(500):
            print(i)
            state = init_state
            self.prev_profit = self.step(state)
            for j in range(200):
                #Choose an action and determine (S, A)
                state_idx = self.getStateinQ(state)
                action_idx = self.exploitExplore(state_idx)
                action = self.act_space[action_idx]

                # Compute the next_state
                self.nxt_state = state+action
                
                # Compute the next profit
                nxt_profit = self.step(self.nxt_state)
                
                # Compute the profit difference
                delta_profit = nxt_profit - self.prev_profit
                # Compute rewards for the next state
                self.prev_profit = nxt_profit
                reward = self.assign_reward(delta_profit) 
                #reward = delta_profit
                # TD-Update
                # Get the S' in the Q-Table
                nxt_idx = self.getStateinQ(self.nxt_state) 
                # Find the best A' in the Q-Table
                best_nxt = self.Q.iloc[nxt_idx][self.n_var:].argmax()
                # x1 R + g[Q(S', A')]
                td_target = reward + self.gamma * self.Q.iloc[nxt_idx][best_nxt+self.n_var] 
                # x1 - Q(S, A)
                td_delta = td_target - self.Q.iloc[state_idx][action_idx+self.n_var]
                # Q(S, A) <- Q(S, A) + a*x1
                self.Q.iloc[state_idx][action_idx+self.n_var] += self.alpha * td_delta  
                
                # Update state
                state = self.nxt_state

            profit_list.append(self.prev_profit)

        plt.figure()
        plt.plot(profit_list)
        plt.show()





target = 'IREN ENERGIA SPA'
targetDay = datetime.strptime('20170210','%Y%m%d')    


m = Market()

dem, sup = m.getData('MGP')
state = np.asarray(
    [
        sup.loc[target].P,
        sup.loc[target].Q,
        dem.loc[target].P,
        dem.loc[target].Q
    ]
)
#state = np.zeros((4), dtype=float)
m.run(state)