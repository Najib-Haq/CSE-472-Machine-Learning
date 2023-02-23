import numpy as np
import random
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import imageio
import seaborn as sns
import time

from sklearn.decomposition import PCA

class GMM:
    def __init__(self, data, k, max_iter=100, init_pi_random=True):
        self.data = data
        self.k = k
        self.max_iter = max_iter
        self.dimension = data.shape[1]
        self.init_pi_random = init_pi_random

        # for plot update
        self.ax, self.fig = None, None
        self.plot_data = {
                'contourX': [],
                'contourY': [],
                'contourZ': [],
                'pcaData': None,
                'pcaV': None,
            }
        
    def init_params(self, k):
        # minx = np.min(self.data)
        # maxx = np.max(self.data)
        # Initialize parameters
        # self.mu = np.random.randint(minx, maxx, size=(k, self.dimension)) # k models each with mean vector of dimension
        # self.mu = np.random.randn(k, self.dimension)
        random_cluster_assignments = np.random.randint(0, k, size=self.data.shape[0])
        self.mu = np.array([self.data[random_cluster_assignments == i].mean(axis=0) for i in range(k)])
        # self.sigma = np.array([np.cov(self.data[cluster_assignments == i].T) for i in range(k)])
        self.sigma = np.array([np.eye(self.dimension) for i in range(k)])
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
        self.responsibilities = np.zeros((self.data.shape[0], k))

    def E_step(self):
        # Expectation-step
        # for each model
        for i in range(self.responsibilities.shape[1]):
            # calculate the probability of all data point belonging to the model
            normal_dist = stats.multivariate_normal(self.mu[i], self.sigma[i]) #+self.noise)
            self.responsibilities[:, i] = self.pi[i] * normal_dist.pdf(self.data)
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
            # update the covariance matrix, 
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


    def fit_k(self, k, plot=False, verbose=True):
        self.init_params(k)

        print("INITIAL: sigma: ", self.sigma, " mu: ", self.mu)
        print(f"===============> FOR SOURCES = {k}")

        for i in range(self.max_iter):
            self.E_step()
            # print("After E: ", self.sigma, " ; ", self.mu)
            self.M_step()
            
            # print("After M: ", self.sigma, " ; ", self.mu)
            log_likelihood = self.log_likelihood()
            if verbose: print(f"Sources: {k}, Iteration: {i}, Log-likelihood: {log_likelihood}")
            if plot: self.get_graph(k, step=i)

        return log_likelihood

    def fit(self):
        if isinstance(self.k, list):
            log_likelihoods = {}
            for k in self.k:
                log_likelihoods[k] = self.fit_k(k, plot=False)

            print(log_likelihoods)            
            self.plot_likelihoods(log_likelihoods)

            # get input
            print(">>> Enter the number of clusturs to visualize: ")
            k = int(input())
            # k = 3
            self.fit_k(k, plot=True, verbose=True)
            
            # return log_likelihoods
        else:
            return self.fit_k(self.k)

    def plot_likelihoods(self, data):
        # plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(list(data.keys()), list(data.values()), 'o--', color="#2AAA8A")
        ax.set(
                xlabel="Number of Clusters",
                ylabel="Log Likelihood",
                title=f"GMM Score vs K",
            )
        plt.grid()
        # plt.savefig(f"Compare_K.png", bbox_inches='tight') 
        plt.show()

    def get_graph(self, k, step):
        if self.dimension == 1: 
            print("No plot for 1D dimension")
            return 
        
        # colors
        clrs = iter(cm.rainbow(np.linspace(0, 1, self.mu.shape[0])))
        
        # get plot data 
        if self.dimension == 2:
            plot_data = self.data 
            plot_mu = self.mu
            plot_sigma = self.sigma
        elif self.dimension > 2: 
            if step == 0:
                pca = PCA(n_components=2)
                self.plot_data['pcaData'] = pca.fit_transform(self.data)
                self.plot_data['pcaV'] = [pca.components_, pca.mean_] # transformation matrix to turn to 2D

            # get mu and sigma in 2D
            V = self.plot_data['pcaV'][0]
            plot_data = self.plot_data['pcaData']
            plot_mu = np.zeros((self.mu.shape[0], 2))
            for i in range(self.mu.shape[0]):
                plot_mu[i] = np.dot(V, (self.mu[i] - self.plot_data['pcaV'][1]))
            # https://stats.stackexchange.com/questions/484094/projecting-a-covariance-matrix-to-a-lower-dimensional-space?utm_source=pocket_mylist&fbclid=IwAR0Px6ySM2DhepQKbV5SxsankWyY1vdZyLW1FBO09_iTW9ZHCji4KsKy_-E
            plot_sigma = np.zeros((self.sigma.shape[0], 2, 2))
            for i in range(self.sigma.shape[0]):
                # print(V.shape, self.sigma[i].shape, V.T.shape)
                plot_sigma[i] = np.dot(np.dot(V, self.sigma[i]), V.T)


        if step == 0:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(10, 10))
            
            # don't need to repeat this calculation for every iter
            for i in range(self.mu.shape[0]):
                # this takes too much time
                # self.plot_data['contourZ'].append(np.dstack(np.meshgrid(np.sort(plot_data[:, 0]), np.sort(plot_data[:, 1]))))
                # self.plot_data['contourX'].append(np.sort(plot_data[:, 0]))
                # self.plot_data['contourY'].append(np.sort(plot_data[:, 1]))
                
                # so use linspace
                X = np.linspace(np.min(plot_data[:, 0]), np.max(plot_data[:, 0]))
                Y = np.linspace(np.min(plot_data[:, 1]), np.max(plot_data[:, 1]))
                X, Y = np.meshgrid(X, Y)
                self.plot_data['contourZ'].append(np.dstack((X, Y)))
                self.plot_data['contourX'].append(X)    
                self.plot_data['contourY'].append(Y)


        if step != 0:
            # remove old contours
            plt.cla()
        
        # plot data
        self.ax.scatter(plot_data[:, 0], plot_data[:, 1], c=self.responsibilities.argmax(axis=1), cmap='viridis')
        
        # this takes the most time
        for i in range(self.mu.shape[0]):
            color = next(clrs)
            normal_dist = stats.multivariate_normal(plot_mu[i], plot_sigma[i])
            self.ax.contour(self.plot_data['contourX'][i], self.plot_data['contourY'][i], normal_dist.pdf(self.plot_data['contourZ'][i]), colors=[color], alpha=0.5, extend='min')
            self.ax.scatter(plot_mu[i, 0], plot_mu[i, 1], color=color, s=100)

        # plt.savefig(f"graph_{k}.png", bbox_inches='tight') 
        plt.title(f"GMM {k} clustering => Iteration: {step}")
        if step == 0: 
            plt.show()
        else:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        # turn off interactive mode for last iter
        if step == (self.max_iter-1):
            plt.ioff()
            plt.show()
            
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
    data = pd.read_csv("data2D.txt", sep=" ", header=None)
    data = data.values

    # compare_sklearn(data, k=[2, 3, 4, 5, 6, 7, 8, 9, 10])
    gmm = GMM(data, k=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], max_iter=100, init_pi_random=True)
    gmm.fit()


