import pytorch_lightning as pl
import seaborn as sns
import matplotlib.pyplot as plt


def train_and_plot_state_trajectory(
    learner,
    min_epochs=1,
    max_epochs=2,
    plot_dim=0,
    plot_control=True,
    func=lambda x: x,
    x_label="t",
    y_label="x",
    output_file="./trajectory.png"
):
    trainer = pl.Trainer(min_epochs=min_epochs, max_epochs=max_epochs)
    trainer.fit(learner)
    x0 = learner.train_dataloader().dataset[0].unsqueeze(0)
    times, trajectory, controls = learner.forward(x0)

    t = times.detach().numpy()
    y = trajectory[:, 0, plot_dim].squeeze().detach().numpy()
    y = func(y)
    fig, ax1 = plt.subplots()

    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label, color='b')
    sns.lineplot(x=t, y=y, ax=ax1, color='b')

    if plot_control:
        u = controls[:, 0, 0].squeeze().detach().numpy()
        ax2 = ax1.twinx()
        ax2.set_ylabel('u', color='r')
        sns.lineplot(x=t, y=u, ax=ax2, color='r')
    
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
