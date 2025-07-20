"""
Checkpointing and loading training states
=========================================

Demonstrating the ``Trainer``'s saving and loading functionality, 
which makes it easy to checkpoint and resume training states.

"""

# %%
# 
import torch
import matplotlib.pyplot as plt
import sys
from neuralop.models import FNO
from neuralop import Trainer
from neuralop.training import AdamW
from neuralop.data.datasets import load_darcy_flow_small
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss

device = 'cpu'


# %%
# Loading the Navier-Stokes dataset in 128x128 resolution
train_loader, test_loaders, data_processor = load_darcy_flow_small(
        n_train=1000, batch_size=32, 
        test_resolutions=[16, 32], n_tests=[100, 50],
        test_batch_sizes=[32, 32],
)


# %%
# We create an FNO model

model = FNO(n_modes=(16, 16),
             in_channels=1, 
             out_channels=1, 
             hidden_channels=32, 
             projection_channel_ratio=2, 
             factorization='tucker', 
             rank=0.42)

model = model.to(device)

n_params = count_model_params(model)
print(f'\nOur model has {n_params} parameters.')
sys.stdout.flush()


# %%
#Create the optimizer
optimizer = AdamW(model.parameters(), 
                                lr=8e-3, 
                                weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)


# %%
# Creating the losses
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)

train_loss = h1loss
eval_losses={'h1': h1loss, 'l2': l2loss}


# %%


print('\n### MODEL ###\n', model)
print('\n### OPTIMIZER ###\n', optimizer)
print('\n### SCHEDULER ###\n', scheduler)
print('\n### LOSSES ###')
print(f'\n * Train: {train_loss}')
print(f'\n * Test: {eval_losses}')
sys.stdout.flush()


# %% 
# Create the trainer
trainer = Trainer(model=model, n_epochs=20,
                  device=device,
                  data_processor=data_processor,
                  wandb_log=False,
                  eval_interval=3,
                  use_distributed=False,
                  verbose=True)


# %%
# Train the model and collect losses
train_loss_history = []
test_loss_history = []
n_epochs = trainer.n_epochs

for epoch in range(n_epochs):
    train_err, avg_loss, _, epoch_train_time = trainer.train_one_epoch(epoch, train_loader, train_loss)
    train_loss_history.append(avg_loss)

    if epoch % trainer.eval_interval == 0:
        eval_metrics = trainer.evaluate_all(epoch=epoch,
                                            eval_losses=eval_losses,
                                            test_loaders=test_loaders,
                                            eval_modes={}) # Assuming single_step mode
        
        test_loss_history.append(eval_metrics)

# %%
# Plot the losses
plt.figure(figsize=(10, 5))
plt.plot(train_loss_history, label='Training Loss (H1)')

# Prepare data for plotting test losses
test_epochs = range(0, n_epochs, trainer.eval_interval)
for res in test_loaders.keys():
    for loss_name in eval_losses.keys():
        metric_name = f'{res}_{loss_name}'
        loss_values = [m[metric_name] for m in test_loss_history if metric_name in m]
        plt.plot(test_epochs[:len(loss_values)], loss_values, label=f'Test Loss {metric_name}', marker='o')

plt.title('Training and Test Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss_plot.png')
print("Loss plot saved to loss_plot.png")
plt.show()