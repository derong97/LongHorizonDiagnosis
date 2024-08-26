import torch.optim as optim

from embedding_model.model import init_model, run_epoch

LEARNING_RATE = [1e-4, 1e-3, 1e-2]
WEIGHT_DECAY = [1e-7, 1e-6, 1e-5]

def objective(config, model, device, train_loader, val_loader, hyperparams, criterion, model_name):
    optimizer = optim.Adam(model.parameters(), lr=hyperparams["learning_rate"], weight_decay=hyperparams["weight_decay"])

    min_val_loss = float("inf")
    early_stopping_counter = 0
    for epoch in range(config.training.max_epochs):
        train_loss = run_epoch(model, device, train_loader, optimizer, criterion, model_name, training=True)
        val_loss = run_epoch(model, device, val_loader, optimizer, criterion, model_name, training=False)
        
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            early_stopping_counter = 0

        else:
            early_stopping_counter += 1
            if early_stopping_counter >= config.training.patience:
                print(f"Early stopping at epoch {epoch}")
                break

        print('Completed epoch %i; train loss = %.3f; val loss = %.3f' % (
            epoch, train_loss, val_loss), end='\r')
        
    return min_val_loss

def run_study(config, model_name, label, device, train_loader, val_loader, criterion):
    trial_num = 0
    min_loss = float("inf")
    best_hyperparams = {}
    best_trial_num = None
    best_model = None
    
    for lr in LEARNING_RATE:
        for weight_decay in WEIGHT_DECAY:
            model = init_model(model_name, config, label).to(device)
            hyperparams = {"learning_rate": lr, "weight_decay": weight_decay}
            loss = objective(config, model, device, train_loader, val_loader, hyperparams, criterion, model_name)

            if loss < min_loss:
                min_loss = loss
                best_hyperparams = hyperparams
                best_trial_num = trial_num
                best_model = model
                
            print(f"Trial num {trial_num}: loss = {loss}, best params = {hyperparams}")
            print(f"Best so far: trial num {best_trial_num}, loss = {min_loss}, params = {best_hyperparams}")
            trial_num += 1

    return best_model, best_hyperparams
