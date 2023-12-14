class SelfOrganizingMap:
    def __init__(self, input_size, map_size, learning_rate=0.1, sigma=1.0):
        self.input_size = input_size
        self.map_size = map_size
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.weights = np.random.rand(map_size[0], map_size[1], input_size)

    def train(self, data, epochs):
        for epoch in tqdm(range(epochs)):
            for input_vector in data:
                bmu_coords = self.find_bmu(input_vector)
                self.update_weights(input_vector, bmu_coords, epoch)

    def find_bmu(self, input_vector):
        distances = np.zeros((self.map_size[0], self.map_size[1]))
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                distances[i, j] = np.linalg.norm(input_vector, self.weights[i, j])
        bmu_coords = np.unravel_index(np.argmin(distances), distances.shape)
        return bmu_coords
    
    def update_weights(self, input_vector, bmu_coords, iteration):
        def _alpha(self, iteration):   
            decay_factor = np.exp(-iteration / self.sigma)
            return self.learning_rate * decay_factor 

        def _theta(self, bmu_coords, neigh_coords):
                return np.exp(-((neigh_coords[0] - bmu_coords[0])**2 + (neigh_coords[1] - bmu_coords[1])**2) / (2 * (self.sigma**2)))

        alpha = self._alpha(iteration)
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                updated = alpha * self._theta(bmu_coords, (i, j)) * (input_vector - self.weights[i, j])
                self.weights[i, j] += update