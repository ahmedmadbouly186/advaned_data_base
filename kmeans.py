from typing import Dict, List, Annotated
from sklearn.cluster import KMeans
import numpy as np
import struct
import os
import gc
import time

taken_cluster = 150 # number of clusters to search for the query in retrieval
clusters = 1000 # number of kmeans clusters
vector_dim = 70 # vector dimensionality
total_dim = 71  # vector dimensionality + id


class VecDBKmeans:
    def __init__(self, index=0, folder_path="saved_db", new_db=True) -> None:
        self.index = index
        self.folder_path = './'+folder_path+'/'+str(self.index)+"kmeans_files"

        folder_path = os.path.join(os.getcwd(), self.folder_path)

        # Check if the folder already exists
        if not os.path.exists(folder_path):
            # Create the folder
            os.makedirs(folder_path)

        if new_db:
            open(f"{self.folder_path}/centroids.csv", 'w').close()
            open(f"{self.folder_path}/data.csv", 'w').close()

    def insert_records(self, rows: List[Annotated[List[float], 70]], ids: List[int]):
        '''
        given the vectors cluster the data and cluster the records using the kmeans algorithm
        '''
        # if it's empty return
        if (len(ids) == 0):
            return
        
        # this is the main function that creates the clusters and indexes the records
        self.create_clusters(ids, rows, k=min(len(ids), clusters))

    def retrive_centers(self, path):
        '''
        given the path of the file of the saved centroids,
        this function retrieves the centroids saved in the file and returns them as a list
        it uses numpy's memmap in order to not load the whole data into the RAM at the same time
        '''
        # get the size of the records to create the memmap with the specified size
        total_bytes = os.path.getsize(path)
        records = (total_bytes//4)//(vector_dim)

        # create an empty numpy array for the centroids with size the number of records and the victor dimension
        centroids_list = np.zeros((records, vector_dim), dtype=np.float32)

        # read the file into the memmap array and add the values to the centroids list
        memmap_array_read = np.memmap(path, dtype=np.float32, mode='r', shape=(records, vector_dim))
        centroids_list[:, :] = memmap_array_read[:, :]

        # free and delete any allocated memory
        del memmap_array_read
        return centroids_list

    def insert_centers(self, path, centers):
        '''
        given the file path and the centers returned from kmeans.fit
        save the centroids of each cluster to 
        '''
        # create a memmap array to write the returned centers from the kmeans to the file
        memmap_array = np.memmap(path, dtype=np.float32, mode='w+', shape=(len(centers), vector_dim))
        memmap_array[:, :] = centers

        # free and delete the memory used
        memmap_array.flush()
        del memmap_array

    def insert_boundries(self, path, boundries):
        '''
        this function saves the given boundaries (start and end) of each cluster
        '''
        memmap_array = np.memmap(path, dtype=np.int32,mode='w+', shape=(len(boundries), 2))
        memmap_array[:, :] = boundries

        # free and delete any allocated memory
        memmap_array.flush()
        del memmap_array

    def retrive_boundries(self, path):
        '''
        given a path read and return the boundaries (start and end) of each cluster
        '''
        # calculate the size of the records being read in bytes
        total_bytes = os.path.getsize(path)
        records = (total_bytes//4)//(2)

        # create an empty numpy array to store the centroids boundaries
        centers_boundries = np.zeros((records, 2), dtype=np.int32)

        # read the boundaries from the file and add them to the boundaries list
        memmap_array_read = np.memmap(path, dtype=np.int32, mode='r', shape=(records, 2))
        centers_boundries[:, :] = memmap_array_read[:, :]

        # free and delete any memory allocated
        del memmap_array_read
        return centers_boundries

    def create_clusters(self, ids, embeddings, k):
        '''
        creates an object of Kmeans and clusters the given data accordingly
        then it creates files with the number of clusters and adds the fitted data
        to the correct file based on the labels produced by the model
        '''

        # create a kmeans object and fits the given embeddings
        kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
        kmeans.fit(embeddings)

        # save centroids with their file name
        self.insert_centers(path=f"{self.folder_path}/centroids.csv", centers=kmeans.cluster_centers_)

        # for each cluster save the vectors with the same label to the part in the file
        # also save the boundaries (start and end) index of each cluster in the file
        memmap_array = np.memmap(f"{self.folder_path}/data.csv",dtype=np.float32, mode='w+', shape=(len(embeddings), total_dim))
        start_index = 0
        centers_boundries = np.zeros((k, 2), dtype=np.int32)

        # loop on the number of clusters to insert the records of each cluster in the file and save its start and end indices (boundaries)
        for i in range(k):
            centers_boundries[i][0] = start_index
            labels_indices = np.where(kmeans.labels_ == i)[0]
            memmap_array[start_index:start_index +len(labels_indices), 0] = ids[labels_indices]
            memmap_array[start_index:start_index +len(labels_indices), 1:] = embeddings[labels_indices]
            start_index += len(labels_indices)
            centers_boundries[i][1] = start_index

        # free and delete the memory allocated
        memmap_array.flush()
        del memmap_array

        # save the boundaries
        self.insert_boundries(path=f"{self.folder_path}/centers_coundries.csv", boundries=centers_boundries)
        gc.collect()

    def retrive(self, query: Annotated[List[float], 70], top_k=5):
        '''
        given a query search for and return the top_k similar vectors in the database
        '''
        # read all the centroids from the file
        # calculates the cosine similarity between the query and all the centroids and takes top 3 centroids to search in
        centroids = self.retrive_centers(path=f"{self.folder_path}/centroids.csv")

        # 
        centers_boundries = self.retrive_boundries(path=f"{self.folder_path}/centers_coundries.csv")

        # calculate the similarity between the query given and all the list of centroids
        c_scores = self._cal_scores(centroids, query.reshape(-1))
        
        # sort the similarity scores to get the target clusters to search for the query in 
        target_clusters = np.argsort(c_scores)[::-1]
        kmeans_scores = []

        # create memmap array to read the data from the file
        data_path = f"{self.folder_path}/data.csv"
        total_bytes = os.path.getsize(data_path)
        records = (total_bytes//4)//(total_dim)
        memmap_array_read = np.memmap(data_path, dtype=np.float32, mode='r', shape=(records, total_dim))

        # loop on all the target clusters and apply linear search using cosine similarity 
        for i in range(len(target_clusters)):
            if (i >= taken_cluster and len(kmeans_scores) >= top_k):
                break
            start, end = centers_boundries[target_clusters[i]]

            # create a list and add the score of each vector in the target clusters to it
            rows = np.empty((end-start, 70), dtype=np.float32)
            rows[:, :] = memmap_array_read[start:end, 1:]
            ids = np.empty((end-start), dtype=np.int32)
            ids[:] = memmap_array_read[start:end, 0]
            scores = self._cal_scores(rows, query.reshape(-1))

            # sort the scores to retrieve the top k in the list
            chosen_ids = np.argsort(scores)[::-1][:min(len(scores), top_k)]

            # add the calculated sorted scores to the list
            kmeans_scores.extend([(scores[id], ids[id]) for id in chosen_ids])
            
            # free and delete the memory allocated each time
            del scores
            del ids
            del rows

        # free and delete the memory allocated each time
        del memmap_array_read
        del centroids
        del centers_boundries
        del c_scores

        # sort the scores descendingly to get the top k similar vectors
        return sorted(kmeans_scores, reverse=True)[:min(len(kmeans_scores), top_k)]

    def _cal_score(vec1, vec2):
        '''
        given two vectors calculate the similarity between the two vectors using cosine similarity
        '''
        # calculates the dot product between the two vectors
        dot_product = np.dot(vec1, vec2)

        # calc the euclidean norm of each vector
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)

        # calculate the cosine similarity
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def _cal_scores(self, vec_list, vec2):
        '''
        given a vector and a list of vectors calculates the cosine similarities between the vector 
        and each vector in the given list and returns a list of the scores
        '''
        # calculates the dot product between the vector and each vector in the given list
        dot_products = np.dot(vec_list, vec2)

        # calc the euclidean norm of each vector in the list
        norm_vec1 = np.linalg.norm(vec_list, axis=1)

        # calculates the euclidean norm of the given vector
        norm_vec2 = np.linalg.norm(vec2)

        # calculate the cosine similarity
        cosine_similarities = dot_products / (norm_vec1 * norm_vec2)
        return cosine_similarities
