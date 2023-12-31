BATCH_SIZE = 128
EPOCHS = 10

artificial_data = [generate_training_data(element_lines, mu_max_err = 0.002, sigma_max_err = 0.05, mu_max_err_global=0.05, samples = 10000,  elements_per_sample = i, element_numbers = None) for i in range(1, 10)]
X, y = reduce(lambda acc, val: (acc[0] + val[0], acc[1] + val[1]) , artificial_data, ([], []))
X = np.asarray(X).reshape(-1, 1, CHANNELS_COUNT)
y = np.asarray(y)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.9, shuffle=True)
train_loader = create_dataloader(X_train, y_train, DEVICE, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = create_dataloader(X_valid, y_valid, DEVICE, batch_size=BATCH_SIZE, shuffle=True)

optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.BCELoss()

train_history, valid_history, best_loss, best_weights = train(model, train_loader, valid_loader, optimizer, criterion, epochs = EPOCHS)
model.load_state_dict(best_weights)