import gym
from gym import spaces
import pygame
import numpy as np
import networkx as nx
from scipy.spatial.distance import euclidean


class TSP(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, size=15):
        self.size = size
        self.window_size = 512  # The size of the PyGame window

        self.graph = self.graph_gen()
        self.a_matrix = nx.to_numpy_array(self.graph) #Матрица смежности, веса отвечают расстоянию
        self.dynamical_a_matrix = np.zeros((self.size,self.size)) #Тоже самое, но веса динамически меняются
        self.observation_space = spaces.Dict(
            {
                "adjacency": self.a_matrix + self.dynamical_a_matrix,
                "path": spaces.Discrete(self.size),
            }
        )

        self.path_graph = nx.turan_graph(self.size, 1)
        self._agent_location = 0;
        self._target_location = 0;

        self.action_space = spaces.Discrete(self.size)

        self.window = None
        self.clock = None

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
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {"distance": self.observation_space['adjacency'][self._agent_location][self._target_location]}

    def reset(self, seed=None, return_info=False, options=None):
        super().reset(seed=seed)

        self._agent_location = 0

        self._target_location = self._agent_location

        observation = self._get_obs()
        info = self._get_info()
        return (observation, info) if return_info else observation

    def step(self, action):

        done = True if self.observation_space['path'].size() == (self.size*(self.size + 1)/2) else False

        self._target_location = action

        self.path_graph.add_edge(self._agent_location, self._target_location)
        self.observation_space['path'][action] = np.max(self.observation_space['path']) + 1

        reward = -self.observation_space['adjacency'][self._agent_location][self._target_location]

        self._agent_location = action
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, done, info

    def render(self, mode="human"):

        if mode == "human":
            nx.draw(self.path_graph, pos=nx.spring_layout(self.path_graph.nodes))
        else:  # rgb_array
            return nx.to_numpy_array(self.path_graph)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

