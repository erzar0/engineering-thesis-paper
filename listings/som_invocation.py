som = SelfOrganizingMap(input_size=e_xy.shape[-1], map_size=(3, 3), learning_rate=0.1, sigma_neigh=0.5, sigma_decay=0.5)
som.train(e_xy_flat_normalized, epochs)
bmus = [[som.find_bmu(e_xy_flat_normalized[i * e_xy.shape[1] + j]) for j in range(e_xy.shape[1])] for i in range(e_xy.shape[0])]
labels = [[bmu[0] * map_size[1] + bmu[1] for bmu in bmus_row] for bmus_row in bmus]