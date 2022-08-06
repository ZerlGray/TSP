import gym
from gym import spaces
import pygame
import numpy as np
import networkx as nx
from scipy.spatial.distance import euclidean


class TSP(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, size=6):
        self.size = size
        
        self.observation_space = spaces.Box(low = np.ones((self.size,self.size)), 
                                            high = np.ones((self.size,self.size))*100,
                                            )
        
        self.mult = np.ones((self.size, self.size)) - np.eye(self.size)
        self.adjacency_matrix = self.observation_space.sample()*self.mult
        
        
        self.path_graph = nx.turan_graph(self.size, 1)
        self.poss = nx.spring_layout(self.path_graph.nodes)
        
        self.action_space = spaces.Discrete(self.size)

        self.coeff = -10**8
        self.possible_action = [1 for i in range(self.size)]
        self.possible_action[0] = self.coeff
        
        self.path = [0]
        
    def action_map(self, action):
#        if action in self.possible_action:
#            idx = self.possible_action.index(action)
#            del self.possible_action[idx]
#            return action
#        else:
            
        return action
        
    def graph_gen(self):
        G = nx.complete_graph(self.size)

        pos = nx.spring_layout(G)

        for (a, b, _) in G.edges(data=True):
            distance = euclidean(pos[a], pos[b])
            G[a][b]['weight'] = distance
        return G

    def update_graph(self):
        self.path_graph.add_edge()
        return self.a_matrix

    def _get_obs(self):
        return {"adjacency_matrix": self.adjacency_matrix.flatten(), "mask": torch.tensor(self.possible_action)}

    def _get_info(self, action):
        return {"distance": self.adjacency_matrix[self.path[-1]][action]}

    def reset(self, seed=None, return_info=False, options=None):
        super().reset(seed=seed)

        self.path = [0]
        self.possible_action = [1 for i in range(self.size)]
        self.possible_action[0] = self.coeff
        
        self.path_graph = nx.turan_graph(self.size, 1)
        self.adjacency_matrix = self.observation_space.sample()*self.mult
        
        observation = self._get_obs()
        info = self._get_info(0)
        return (observation, info) if return_info else observation

    def step(self, action):

        reward = 0
        action = self.action_map(action)

        self.path_graph.add_edge(self.path[-1], action)

        observation = self._get_obs()
        info = self._get_info(action)
        
        self.possible_action[action] = self.coeff

        reward = reward - self.adjacency_matrix[self.path[-1]][action]
        
        done = True if np.sum(self.possible_action) <= self.coeff*self.size else False
        self.path.append(action)
        
        return observation, reward, done, info

    def render(self, mode="human"):

        if mode == "human":
            nx.draw(self.path_graph, pos=self.poss)
        else:
            return nx.to_numpy_array(self.path_graph)

    def close(self):
        return
