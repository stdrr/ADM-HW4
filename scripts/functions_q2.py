from sklearn.decomposition import TruncatedSVD # for features reduction
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import random
from random import randint
from mpl_toolkits.axes_grid1 import ImageGrid
from wordcloud import WordCloud


def estimate_components(docterm_mat, start_n, end_n, step=100):
    """
    This function is used to find the number of components that reatin more than the 60% of the variance
    """
    components_variance = {}
    fig = plt.figure(figsize=(16, 10))
    for n_components in tqdm(range(start_n, end_n, step)): # try the SVD method with different numbers of components
        svd = TruncatedSVD(n_components=n_components)
        svd.fit_transform(docterm_mat)
        components_variance[n_components] = round(np.cumsum(svd.explained_variance_ratio_)[-1], 5) * 100
        del svd
    plt.plot(list(components_variance.keys()), list(components_variance.values()))
    plt.xlabel('Number of components')
    plt.ylabel('Variance')
    plt.grid()
    plt.show()


## ********* K-means implementation *****************************

    # k : total number of the clusters
def random_centers(k, matrix):
    copy_matrix = np.copy(matrix)
    # matrix for cluster centroids
    center_matrix = np.zeros((k, len(matrix[0])))
    
    # randomly choosing k cluster centers without replacement
    for i in range(k):
        new_center_index = random.randint(0, copy_matrix.shape[0]-1)
        center_matrix[i,:] = copy_matrix[new_center_index, :]
        copy_matrix = np.delete(copy_matrix, new_center_index, 0 )
    return center_matrix


def compute_centroids(cluster_dictionary, matrix, k):
    
    cluster_dict_copy = {key:matrix[cluster_dictionary[key], :] for key in cluster_dictionary}
    
    # computes the new centroid values
    for i in range(len(cluster_dictionary.keys())):
        cluster_dict_copy[i] = np.array([cluster_dict_copy[i]])
        cluster_dict_copy[i] = np.mean(cluster_dict_copy[i][0], axis=0)
    
    new_center_matrix = np.zeros((k, len(matrix[0])))
    for i in range(len(cluster_dict_copy.keys())):
        new_center_matrix[i] = cluster_dict_copy[i]
        
    return new_center_matrix


def assignment_to_centroids(matrix, center_matrix, k):
    
    # cluster_dict key corresponds to index of clusters, values correspond to elements that belong to the cluster 
    cluster_dict = {}
    for i in range(k):
        cluster_dict[i] = []
        
    for element_idx, element in zip(range(matrix.shape[0]), matrix): # get the row (product) and its index
        min_dist_value  = np.linalg.norm(element - center_matrix[0])
        cluster_index = 0
        for i in range(1, k):
            new_dist = np.linalg.norm(element - center_matrix[i])
            if min_dist_value > new_dist:
                min_dist_value = new_dist
                cluster_index = i
        cluster_dict[cluster_index].append(element_idx)
    
    return cluster_dict
        
        
def k_means(k, matrix):
    
    center_matrix = random_centers(k, matrix)
    cluster_dict = assignment_to_centroids(matrix, center_matrix, k)
    
    while np.array_equal(center_matrix, compute_centroids(cluster_dict, matrix, k))==False:
        center_matrix = compute_centroids(cluster_dict, matrix, k)
        cluster_dict  = assignment_to_centroids(matrix, center_matrix, k)
    return cluster_dict, center_matrix


def compute_inertia(cluster_dict, center_matrix, matrix):
    inertia = 0
    for cluster in cluster_dict:
        for point_idx in cluster_dict[cluster]:
            inertia += np.sum(np.square(matrix[point_idx] - center_matrix[cluster]))
    return inertia


def elbow_method(matrix, start_n=1, end_n=50, step=10):
    end_n = end_n if end_n < matrix.shape[0] else matrix.shape[0]
    elbow = {}
    for k in tqdm(range(start_n, end_n, step)):
        cluster_dict, center_matrix = k_means(k, matrix)
        elbow[k] = compute_inertia(cluster_dict, center_matrix, matrix)
    return elbow


def plot_elbow(elbow):
    plt.figure(figsize=(16,10))
    plt.plot(list(elbow.keys()), list(elbow.values()))
    plt.grid()
    plt.xlabel('Number of clusters')
    plt.show()

## **************************************************************************************

def get_word_cloud(center_matrix, svd, vectorizer, n_words=50):
    """
    This function takes the matrix of the centroids computed by K-means and
    plots the word cloud representation of the clusters
    """
    center_matrix_inv = svd.inverse_transform(center_matrix) # get the original feature space, i.e. the space of the words in the vocabulary
    terms = vectorizer.get_feature_names() # get the vocabulary
    word_cloud = WordCloud()
    fig = plt.figure(figsize=(16,30))
    grid = ImageGrid(fig, 111, 
                 nrows_ncols=(round(center_matrix.shape[0]/4 + 0.5), 4), 
                 axes_pad=0.5, 
                 ) # plot the images in a grid view
    centroids = np.argsort(center_matrix_inv)[:, ::-1] # sort the features in descending order, based on the TF-IDF score
    for i, ax in zip(range(center_matrix.shape[0]), grid):
        top_term_freq = {}
        for w_idx in centroids[i, :n_words]:
            top_term_freq[terms[w_idx]] = center_matrix_inv[i][w_idx] # dictionary having the words as keys and the TF-IDF as values
        word_cloud.generate_from_frequencies(top_term_freq) # generate the word cloud representation
        ax.set_title(f'Cluster {i+1}')
        ax.axis("off")
        ax.imshow(word_cloud, interpolation="bilinear")
    plt.show()