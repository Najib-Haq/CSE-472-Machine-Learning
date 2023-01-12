import numpy as np
import random
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import imageio

from sklearn.manifold import TSNE

class GMM:
    def __init__(self, data, k, max_iter=100, init_pi_random=True):
        self.data = data
        self.k = k
        self.max_iter = max_iter
        self.dimension = data.shape[1]
        self.init_pi_random = init_pi_random
        
    def init_params(self, k):
        minx = np.min(self.data)
        maxx = np.max(self.data)
        # Initialize parameters
        # self.mu = np.random.randint(minx, maxx, size=(k, self.dimension)) # k models each with mean vector of dimension
        # self.mu = np.random.randn(k, self.dimension)
        cluster_assignments = np.random.randint(0, k, size=self.data.shape[0])
        self.mu = np.array([self.data[cluster_assignments == i].mean(axis=0) for i in range(k)])
        self.sigma = np.array([np.cov(self.data[cluster_assignments == i].T) for i in range(k)])

        # This needs to be a positive semi-definite matrix
        # https://stackoverflow.com/questions/619335/a-simple-algorithm-for-generating-positive-semidefinite-matrices
        # self.sigma = np.random.randn(k, self.dimension, self.dimension) # k models each with covariance matrix of dimension
        # self.sigma = np.ones((k, self.dimension, self.dimension)) # k models each with covariance matrix of dimension
        # self.noise = np.ones((self.dimension, self.dimension))*1e-6 # need to avoid singularity issues
        # for idx in range(k):
        #     # self.sigma[idx] *= np.array([random.uniform(minx, maxx) for i in range(self.dimension*self.dimension)]).reshape(self.dimension, self.dimension)
        #     # self.sigma[idx] *= np.random.randint(minx, maxx, size=(self.dimension, self.dimension)) + self.noise
        #     self.sigma[idx] = np.dot(self.sigma[idx], self.sigma[idx].transpose())

        # self.sigma = np.zeros((k, self.dimension, self.dimension))
        # for idx in range(k):
        #     np.fill_diagonal(self.sigma[idx], 5)

        self.pi = np.random.rand(k) if self.init_pi_random else np.ones(k) # k models each with a weight
        self.pi = self.pi / np.sum(self.pi) # normalize weights

        # Initialize responsibilities
        self.responsibilities = np.zeros((data.shape[0], k))

    def E_step(self):
        # Expectation-step
        # for each model
        for i in range(self.responsibilities.shape[1]):
            # calculate the probability of all data point belonging to the model
            normal_dist = stats.multivariate_normal(self.mu[i], self.sigma[i]) #+self.noise)
            self.responsibilities[:, i] = self.pi[i] * normal_dist.pdf(self.data)
        # normalize the probabilities
        # TODO: check if this is correct, also "nvalid value encountered in true_divide can occur -> add tolerance?"
        # for i in self.responsibilities:
        #     print(i)
        self.responsibilities = self.responsibilities / np.sum(self.responsibilities, axis=1, keepdims=True)
        

    def M_step(self):
        # Maximization-step
        # for each model
        for i in range(self.responsibilities.shape[1]):
            # update the mean vector
            # print("SUM : ", np.isnan(np.sum(self.responsibilities[:, i])))
            if  np.isnan(np.sum(self.responsibilities[:, i])):
                print("NAN")
                for i in self.responsibilities[:, i]:
                    print(i)

            self.responsibilities += 1e-6 # add pseudocount
            self.mu[i] = np.sum(self.responsibilities[:, i].reshape(self.data.shape[0], 1) * self.data, axis=0) / np.sum(self.responsibilities[:, i])
            # update the covariance matrix, TODO: need to check this
            self.sigma[i] = (np.dot((self.responsibilities[:, i].reshape(self.data.shape[0], 1) * (self.data - self.mu[i])).transpose(), self.data - self.mu[i])) / np.sum(self.responsibilities[:, i]) # self.noise added here too
            # update the weight
            self.pi[i] = np.sum(self.responsibilities[:, i]) / self.data.shape[0]

    def log_likelihood(self):
        # calculate the log likelihood of the data
        # for each model
        for i in range(self.responsibilities.shape[1]):
            # calculate the probability of all data point belonging to the model
            try:
                normal_dist = stats.multivariate_normal(self.mu[i], self.sigma[i])
                # normal_dist = stats.multivariate_normal(self.mu[i], self.sigma[i])
            except:
                print("ERROR in stats: ", self.mu, " ; ", self.sigma)
            self.responsibilities[:, i] = self.pi[i] * normal_dist.pdf(self.data)
        # normalize the probabilities
        return np.sum(np.log(np.sum(self.responsibilities, axis=1, keepdims=True)))


    def fit_k(self, k, plot=False):
        self.init_params(k)

        print("sigma: ", self.sigma, " mu: ", self.mu)
        print(f"===============> FOR SOURCES = {k}")

        if plot:
            frames = []

        best_score = 0
        for i in range(self.max_iter):
            self.E_step()
            # print("After E: ", self.sigma, " ; ", self.mu)
            self.M_step()
            
            # print("After M: ", self.sigma, " ; ", self.mu)
            log_likelihood = self.log_likelihood()
            print(f"Sources: {k}, Iteration: {i}, Log-likelihood: {log_likelihood}")

            # plot if log likelihood better 
            if log_likelihood > best_score:
                if plot: frames.append(self.get_graph(k))
                best_score = log_likelihood

        if plot: imageio.mimsave(f'gmm_{k}.gif', frames, fps=2)
        return log_likelihood

    def fit(self):
        if isinstance(self.k, list):
            log_likelihoods = []
            for k in self.k:
                log_likelihoods.append((k, self.fit_k(k, plot=False))) #k in [2, 3])))
            return log_likelihoods
        else:
            return self.fit_k(self.k)

    def get_graph(self, k):
        if self.dimension == 2:
            plot_data = self.data 
            plot_mu = self.mu
            plot_sigma = self.sigma
        else: 
            tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3)
            plot_data = tsne.fit_transform(self.data)
            # TODO: how to only infer and how to handle sigma?
            plot_mu = tsne.fit_transform(self.mu)
            plot_sigma = np.zeros((k, 2, 2))
            plot_sigma[:, 0] = tsne.fit_transform(self.sigma[:, 0])
            plot_sigma[:, 1] = tsne.fit_transform(self.sigma[:, 1])
            # for i in range(self.sigma.shape[0]):
            #     plot_sigma[i] = tsne.fit_transform(self.sigma[i].reshape(1, -1))

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(plot_data[:, 0], plot_data[:, 1], c=self.responsibilities.argmax(axis=1), cmap='viridis')
        for i in range(self.mu.shape[0]):
            normal_dist = stats.multivariate_normal(plot_mu[i], plot_sigma[i])
            ax.contour(np.sort(plot_data[:, 0]), np.sort(plot_data[:, 1]), normal_dist.pdf(np.dstack(np.meshgrid(np.sort(plot_data[:, 0]), np.sort(plot_data[:, 1])))), colors='black', alpha=0.3)
            ax.scatter(self.mu[i, 0], self.mu[i, 1], c='grey', s=100)
        plt.savefig(f"graph_{k}.png", bbox_inches='tight') 
        fig.canvas.draw()
        # Now we can save it to a numpy array.
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data

# def compare_sklearn(data, k=[2, 3, 4, 5, 6, 7, 8, 9, 10]):
#     from sklearn.mixture import GaussianMixture

#     score = []
#     for n_clusters in k:
#         gmm = GaussianMixture(n_components=n_clusters, max_iter=100, init_params='random')
#         gmm.fit(data)
#         score.append(gmm.score(data))
#         print(f"{n_clusters} => {score[-1]}")

#     plt.plot(score)
#     plt.savefig(f"scores_sklearn.png", bbox_inches='tight') 

if __name__ == "__main__":
    # Load data
    data = pd.read_csv("data6D.txt", sep=" ", header=None)
    data = data.values

    # compare_sklearn(data, k=[2, 3, 4, 5, 6, 7, 8, 9, 10])
    gmm = GMM(data, k=[2, 3, 4, 5, 6, 7, 8, 9, 10], max_iter=100, init_pi_random=True)
    print(gmm.fit())


