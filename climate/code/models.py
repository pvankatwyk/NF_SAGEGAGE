from utils import linear_fit
import torch
from torch import optim, nn
from nflows import distributions, flows, transforms
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch


class FairFlow:
    def __init__(
            self,
            X,
            y=None,
            num_flow_transforms=3,
            flow_hidden_features=4,
    ):

        self.num_flow_transforms = num_flow_transforms
        self.flow_hidden_features = flow_hidden_features
        self.conditional = False
        self.batch_size = None
        self.num_epochs = None
        self.loss_history = []
        self.X = X
        self.y = y

        try:
            self.num_features = self.X.shape[1]
        except IndexError:
            self.num_features = 1
            self.X = self.X.reshape(self.X.shape[0], 1)

        # TODO: Fix this -- Conditional flow calling else clause and throws error
        if self.num_features == 1:
            self.sle = X
            self.gsat = y
        else:
            self.sle = X[:, 0]
            self.gsat = X[:, 1]

        self.bounds = {'sle_min': self.sle.min(), 'sle_max': self.sle.max(),
                       'gsat_min': self.gsat.min(), 'gsat_max': self.gsat.max()}

        # Set up flow
        self.base_dist = distributions.normal.StandardNormal(shape=[self.num_features], )

        t = []
        for _ in range(self.num_flow_transforms):
            t.append(transforms.permutations.RandomPermutation(features=self.num_features, ))
            t.append(transforms.autoregressive.MaskedAffineAutoregressiveTransform(
                features=self.num_features,
                hidden_features=self.flow_hidden_features,
                context_features=1,
            ))

        self.t = transforms.base.CompositeTransform(t)

        self.flow = flows.base.Flow(
            transform=self.t,
            distribution=self.base_dist
        )

        self.optimizer = optim.Adam(self.flow.parameters())

    def train(self, num_epochs, batch_size, verbose=True, plot_logprob=False):
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        for epoch in range(num_epochs):
            if verbose and epoch % 10 == 0:
                print(f'---------- EPOCH: {epoch} ----------')

            for i in range(0, len(self.X), batch_size):
                if self.conditional:
                    x = torch.tensor(self.X[i:i + batch_size, :], dtype=torch.float32).reshape(-1, 1)
                    y = torch.tensor(self.y[i:i + batch_size], dtype=torch.float32).reshape(-1, 1)
                    loss = -self.flow.log_prob(inputs=y, context=x).mean()  # P(Y|X)
                else:
                    x = torch.tensor(self.X[i:i + batch_size, :], dtype=torch.float32)
                    loss = -self.flow.log_prob(inputs=x).mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.loss_history.append(loss.detach().numpy())
                self.optimizer.step()

                if verbose and i % 2000 == 0:
                    print(f"-log_prob Loss: {loss}")

                if plot_logprob and i % 5000:
                    xline = torch.linspace(self.bounds['gsat_min'], self.bounds['gsat_max'], steps=100)
                    yline = torch.linspace(self.bounds['sle_min'], self.bounds['sle_max'], steps=100)
                    xgrid, ygrid = torch.meshgrid(xline, yline)
                    xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

                    with torch.no_grad():
                        zgrid = self.flow.log_prob(xyinput).exp().reshape(100, 100)

                    plt.contourf(xgrid.numpy(), ygrid.numpy(), zgrid.numpy())
                    plt.title('iteration {}'.format(i + 1))
                    plt.show()

    def sample(self, num_samples):
        samples_logprobs = self.flow.sample_and_log_prob(num_samples=num_samples)
        new_samples = samples_logprobs[0].detach().numpy()
        log_probs = samples_logprobs[1].detach().numpy()
        return new_samples, log_probs

    def visualize(self, plot='comparison', samples=None, ):
        plot = plot.lower()
        options = ['comparison', 'original', 'sampled', 'comparison_density', 'log_prob']
        assert plot in options, f"plot must be in {options}, received {plot} "
        if plot == "original":
            x_line, y_line, fit = linear_fit(self.X)
            plt.plot(self.gsat, self.sle, 'o')
            plt.plot(x_line, y_line, 'r-')
            plt.title('Sea Level Rise vs Global Air Temperature')
            plt.xlabel('GSAT')
            plt.ylabel('SLE')
            plt.show()

        if plot == "sampled":
            assert samples is not None, f"samples argument cannot be None"
            x_line, y_line, fit = linear_fit(samples)
            plt.plot(samples[:, 0], samples[:, 1], 'o')
            plt.plot(x_line, y_line, 'r-')
            plt.title('Sea Level Rise vs Global Air Temperature')
            plt.xlabel('GSAT')
            plt.ylabel('SLE')
            plt.show()

        if plot == "comparison":
            assert samples is not None, f"samples argument cannot be None"
            x_line, y_line, fit = linear_fit(self.X, self.y)
            x_line_s, y_line_s, fit_s = linear_fit(samples)

            fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True, )
            fig.suptitle('Normalizing Flow vs Climate Model Projections')
            ax1.scatter(self.X[:,0], self.X[:,1], s=5)
            ax1.plot(x_line, y_line, 'r-')
            ax1.set_title("GCM Simulations")
            ax2.scatter(samples[:, 0], samples[:, 1], s=5)
            ax2.plot(x_line_s, y_line_s, 'r-')
            ax2.set_title("Normalizing Flow Samples")
            plt.tight_layout()
            fig.text(0.5, 0.005, 'Global mean temperature change 2015-2100 (C)', ha='center')
            fig.text(0.01, 0.5, 'Sea Level Contribution at 2100 (cm SLE)', va='center', rotation='vertical')
            fig.text(0.035, 0.06, f"Trend: $y={round(fit[0], 4)}x + {round(fit[1], 4)}$")
            fig.text(0.521, 0.06, f"Trend: $y={round(fit_s[0], 4)}x + {round(fit_s[1], 4)}$")
            plt.show()

        if plot == "comparison_density":
            assert samples is not None, f"samples argument cannot be None"
            x_line, y_line, fit = linear_fit(self.X, self.y)
            x_line_s, y_line_s, fit_s = linear_fit(samples)
            fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(15, 8))
            fig.suptitle('Normalizing Flow vs Climate Model Projections')
            ax1.hist2d(self.gsat, self.sle, bins=(20, 20))
            ax1.plot(x_line, y_line, 'r-')
            ax1.set_title("GCM Simulations")
            ax2.hist2d(samples[:, 0], samples[:, 1], bins=(20, 20))
            ax2.plot(x_line_s, y_line_s, 'r-')
            ax2.set_title("Normalizing Flow Samples")
            plt.tight_layout()
            fig.text(0.5, 0.005, 'Global mean temperature change 2015-2100 (C)', ha='center')
            fig.text(0.01, 0.5, 'Sea Level Contribution at 2100 (cm SLE)', va='center', rotation='vertical')
            fig.text(0.035, 0.06, f"Trend: $y={round(fit[0], 4)}x + {round(fit[1], 4)}$")
            fig.text(0.521, 0.06, f"Trend: $y={round(fit_s[0], 4)}x + {round(fit_s[1], 4)}$")
            plt.show()

        if plot == "log_prob":
            xline = torch.linspace(self.bounds['gsat_min'], self.bounds['gsat_max'], steps=100)
            yline = torch.linspace(self.bounds['sle_min'], self.bounds['sle_max'], steps=100)
            xgrid, ygrid = torch.meshgrid(xline, yline)
            xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

            with torch.no_grad():
                zgrid = self.flow.log_prob(xyinput).exp().reshape(100, 100)

            plt.contourf(xgrid.numpy(), ygrid.numpy(), zgrid.numpy())
            plt.title("Log Probability Map")
            plt.xlabel('Global mean temperature change 2015-2100 (C)')
            plt.ylabel('Sea Level Contribution at 2100 (cm SLE)')
            plt.colorbar()
            plt.show()


def format_conditional_df(samples, context_vals):
    # Format dataframe
    samples_df = pd.DataFrame()
    for i, val in enumerate(context_vals):
        sle = samples[i, :].squeeze()
        gsat = np.tile(val, len(sle))
        samples_df = pd.concat([samples_df, pd.DataFrame({'sle': sle, 'gsat': gsat})])
    return samples_df


def find_nearest_neighbor(value, context_vals, means):
    assert all(np.sort(context_vals) == context_vals), "array is not sorted, make sure the correct array is inputted"
    array = np.asarray(context_vals)
    idx = (np.abs(array - value)).argmin()
    return means[idx]


class ConditionalFairFlow(FairFlow):
    def __init__(self, *args, **kwargs, ):
        super().__init__(*args, **kwargs, )

        self.mean_function_df = None
        self.conditional = True

        self.context_vals = list(np.arange(self.bounds['gsat_min'], self.bounds['gsat_max'], step=0.01))

        # Set up flow
        self.base_dist = distributions.normal.ConditionalDiagonalNormal(
            shape=[self.num_features],
            context_encoder=nn.Linear(1, 2 * self.num_features),
        )

        t = []
        for _ in range(self.num_flow_transforms):
            t.append(transforms.permutations.RandomPermutation(features=self.num_features, ))
            t.append(transforms.autoregressive.MaskedAffineAutoregressiveTransform(
                features=self.num_features,
                hidden_features=self.flow_hidden_features,
                context_features=1,
            ))

        self.t = transforms.base.CompositeTransform(t)

        self.flow = flows.base.Flow(
            transform=self.t,
            distribution=self.base_dist
        )

        self.optimizer = optim.Adam(self.flow.parameters())

    def sample(self, num_samples):
        # context = torch.arange(self.bounds['gsat_min'], self.bounds['gsat_max'], step=0.01).reshape(-1, 1)
        context = torch.tensor(self.context_vals, dtype=torch.float32).reshape(-1,1)
        samples_logprobs = self.flow.sample_and_log_prob(num_samples=num_samples, context=context)

        new_samples = samples_logprobs[0].detach().numpy()
        log_probs = samples_logprobs[1].detach().numpy()

        return new_samples, log_probs

    def calculate_mean_function(self, samples, fit_type='polynomial'):
        fit_type = fit_type.lower()
        options = ['polynomial', 'nearest_neighbor', 'linear']
        assert fit_type in options, f"plot must be in {options}, received {fit_type}"

        means = np.mean(samples.squeeze(), axis=1)
        self.mean_function_df = pd.DataFrame({'context_vals': self.context_vals, 'mean_sle': means})

        if fit_type == 'polynomial':
            fxn = np.polyfit(self.context_vals, means, 3)
        elif fit_type == 'linear':
            fxn = np.polyfit(self.context_vals, means, 1)
        else:
            fxn = find_nearest_neighbor

        return fxn

    def visualize(self, plot='hist', samples=None):
        plot = plot.lower()
        options = ['scatter', 'hist']
        assert plot in options, f"plot must be in {options}, received {plot} "

        means = np.mean(samples.squeeze(), axis=1)
        samples_df = format_conditional_df(samples, self.context_vals)

        if plot == 'hist':
            plt.figure(figsize=(15,8))
            plt.hist2d(samples_df['gsat'], samples_df['sle'], bins=(50, 50))
            plt.title("Mean Temperature Change vs Sea Level Contribution")
        elif plot == 'scatter':
            plt.plot(samples_df['gsat'], samples_df['sle'], 'o')
            plt.title('Conditional NF Scatterplot')
        elif plot == 'log_prob':
            xline = torch.linspace(self.bounds['gsat_min'], self.bounds['gsat_max'], steps=100)
            yline = torch.linspace(self.bounds['sle_min'], self.bounds['sle_max'], steps=100)
            xgrid, ygrid = torch.meshgrid(xline, yline)
            xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

            with torch.no_grad():
                zgrid = self.flow.log_prob(xyinput).exp().reshape(100, 100)

            plt.contourf(xgrid.numpy(), ygrid.numpy(), zgrid.numpy())
            plt.title("Log Probability Map")
            plt.colorbar()

        plt.plot(self.context_vals, means, 'r-')
        f = np.polyfit(self.context_vals, means, 3)
        plt.plot(self.context_vals, np.polyval(f, self.context_vals), '-')
        plt.xlabel('Global mean temperature change 2015-2100 (C)')
        plt.ylabel('Sea Level Contribution at 2100 (cm SLE)')
        plt.show()

    def predict(self, context, fxn):
        assert self.mean_function_df is not None, "Must run calculate_mean_function method before continuing."
        means = self.mean_function_df['mean_sle']

        predictions = np.zeros(len(context))
        for i, val in enumerate(context):
            if 'nearest_neighbor' in str(fxn):
                predictions[i] = fxn(val, self.context_vals, means)
            else:
                predictions[i] = np.polyval(fxn, val)
        return predictions
