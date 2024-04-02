from typing import Dict, List, Annotated
from sklearn.cluster import KMeans
import numpy as np
import struct
import os
import gc
import time
from typing import Dict, List, Annotated
import numpy as np
# from npy_append_array import NpyAppendArray
from math import ceil, floor 
from sklearn.cluster import MiniBatchKMeans
import os
from joblib import dump, load
from shutil import rmtree
import tempfile
import uuid
taken_cluster =  48 # number of clusters to search for the query in retrieval
clusters = 50 # number of kmeans clusters
vector_dim = 70 # vector dimensionality
total_dim = 71  # vector dimensionality + id


class VecDBKmeans:
    def __init__(self, index=0, folder_path="saved_db", new_db=True) -> None:
        self.index = index
        self.folder_path = './'+folder_path+'/'+str(self.index)+"kmeans_files"
        self.reindex_ratio = 2 # reindex the data when the number of records is doubled
        self.centroids_path='centroids.csv'
        self.data_path='data.csv'
        self.monolothic_data_path='monolothic_data.csv'
        self.metadata_path='metadata.csv'
        folder_path = os.path.join(os.getcwd(), self.folder_path)
        
        # Check if the folder already exists
        if not os.path.exists(folder_path):
            # Create the folder
            os.makedirs(folder_path)

        if new_db:
            open(f"{self.folder_path}/{self.centroids_path}", 'w').close()
            open(f"{self.folder_path}/{self.data_path}", 'w').close()
            open(f"{self.folder_path}/{self.monolothic_data_path}", 'w').close()
            open(f"{self.folder_path}/{self.metadata_path}", 'w').close()
            with open(f"{self.folder_path}/{self.metadata_path}", "w") as file:
                file.write("0\n0\n")
            # loop and create the files for the clusters , each cluster will be in separate file
            for i in range(clusters):
                open(f"{self.folder_path}/{str(i)}.csv", 'w').close()

            

    def insert_records(self, rows: List[Annotated[List[float], 70]], ids: List[int]):
        '''
        given the vectors cluster the data and cluster the records using the kmeans algorithm
        '''
        # if it's empty return
        if (len(ids) == 0):
            return
        
        # insert into the monolithic data file
        monolothic_data_path = f"{self.folder_path}/{self.monolothic_data_path}"
        meta_path= f"{self.folder_path}/{self.metadata_path}"        
                # Open the file in read mode
        records_num=0
        indexing_num=0
        with open(meta_path, 'r') as file:
            # Read the first line
            line1 = file.readline().strip()
            line2 = file.readline().strip()

            # Convert to integer
            records_num = int(line1)
            indexing_num= int (line2)
        start_index = records_num
        memmap_array = np.memmap(monolothic_data_path, dtype=np.float32, mode='w+', shape=(len(rows)+records_num, total_dim))
        memmap_array[start_index:start_index + len(rows), 1:] = rows
        memmap_array[start_index:start_index + len(rows), :1] = np.array(ids).reshape((-1, 1))
        # free and delete the memory used
        memmap_array.flush()
        del memmap_array
        records_num+=len(rows)
        # store the new records number in the metadata file at the first line only
        with open(meta_path, "r+") as file:
            present_vec_num = int(file.readline()) + len(rows)
            file.seek(0)
            file.write(f"{present_vec_num}\n")
            file.write(f"{indexing_num}\n")
            
       
        if(records_num>=self.reindex_ratio*indexing_num):
            print("reindexing")
            self.reindex()
        else :
            print("no need to reindexing")
            self.insert_index(ids,rows)
        # this is the main function that creates the clusters and indexes the records
    
    def insert_index(self, ids,rows):
        # insert the new records to the existing clusters
        # each record will be inserted in the correct cluster
        # the correct cluster is where the centroid is nearst to this record
        # read the centroids from the file
        centroids = self.retrive_centers(path=f"{self.folder_path}/centroids.csv")
        
        # find te nearst center for each record 'rows'
                # Find the nearest center for each record
        nearest_centers = []
        for row in rows:
            scores = self._cal_scores(centroids, row)
            nearest_center_index = np.argmax(scores)
            nearest_centers.append(nearest_center_index)
        
        # for each centroid , insert the records in the correct file
        for i in range(clusters):
            # calc the number of records in the cluster
            # read ow many records are in the cluster by divition the size of the file over (total_dim*4)
            current = os.path.getsize(f"{self.folder_path}/{str(i)}.csv")//(total_dim*4)
            indices = np.where(np.array(nearest_centers) == i)[0]
            if len(indices) == 0:
                continue
            memmap_array = np.memmap(f"{self.folder_path}/{str(i)}.csv", dtype=np.float32, mode='r+', shape=(len(indices) +current , total_dim))
            memmap_array[current:current + len(indices), 1:] = rows[indices]
            memmap_array[current:current + len(indices), :1] = np.array(ids)[indices].reshape((-1, 1))
            memmap_array.flush()
            del memmap_array
        
        pass
    
    def reindex(self):
        open(f"{self.folder_path}/{self.centroids_path}", 'w').close()
        open(f"{self.folder_path}/{self.data_path}", 'w').close()
        # loop and create the files for the clusters , each cluster will be in separate file
        for i in range(clusters):
            open(f"{self.folder_path}/{str(i)}.csv", 'w').close()

        # load the whole data from the monolithic file
        monolothic_data_path = f"{self.folder_path}/{self.monolothic_data_path}"
        meta_path= f"{self.folder_path}/{self.metadata_path}"  
        with open(meta_path, "r+") as file:
            present_vec_num = int(file.readline())
        memmap_array_read=np.memmap(monolothic_data_path, dtype=np.float32, mode='r', shape=(present_vec_num, total_dim))
        ids=memmap_array_read[:,0]
        rows=memmap_array_read[:,1:]
        self.create_clusters(ids, rows, k=min(len(ids), clusters))
        with open(meta_path, "r+") as file:
            present_vec_num = int(file.readline())
            file.seek(0)
            file.write(f"{present_vec_num}\n")
            file.write(f"{present_vec_num}\n")
        del memmap_array_read     
        

        
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
        open(path, 'w').close()
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
        # memmap_array = np.memmap(f"{self.folder_path}/data.csv",dtype=np.float32, mode='w+', shape=(len(embeddings), total_dim))
        start_index = 0
        centers_boundries = np.zeros((k, 2), dtype=np.int32)

        # loop on the number of clusters to insert the records of each cluster in the file and save its start and end indices (boundaries)
        for i in range(k):
            centers_boundries[i][0] = start_index
            labels_indices = np.where(kmeans.labels_ == i)[0]
            # print(np.array(labels_indices))
            # store the ids and the embeddings in the file for this cluster
            open(f"{self.folder_path}/{str(i)}.csv", 'w').close()
            memmap_array = np.memmap(f"{self.folder_path}/{str(i)}.csv", dtype=np.float32, mode='w+', shape=(len(labels_indices) , total_dim))

            memmap_array[:, 0] = ids[np.array(labels_indices)]

            memmap_array[: , 1:] = embeddings[labels_indices]
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
        # data_path = f"{self.folder_path}/data.csv"
        # total_bytes = os.path.getsize(data_path)
        # records = (total_bytes//4)//(total_dim)

        # loop on all the target clusters and apply linear search using cosine similarity 
        for i in range(len(target_clusters)):
            if (i >= taken_cluster and len(kmeans_scores) >= top_k):
                break
            cluster = target_clusters[i]
            records= os.path.getsize(f"{self.folder_path}/{str(cluster)}.csv")//(total_dim*4)
            memmap_array = np.memmap(f"{self.folder_path}/{str(cluster)}.csv", dtype=np.float32, mode='r+', shape=(records , total_dim))

            start, end = centers_boundries[target_clusters[i]]

            # create a list and add the score of each vector in the target clusters to it
            rows = np.empty((records, 70), dtype=np.float32)
            rows[:, :] = memmap_array[:, 1:]
            ids = np.empty((records), dtype=np.int32)
            ids[:] = memmap_array[:, 0]
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
        del memmap_array
        del centroids
        del centers_boundries
        del c_scores

        # sort the scores descendingly to get the top k similar vectors
        return [x[1] for x in sorted(kmeans_scores, reverse=True)[:min(len(kmeans_scores), top_k)]]

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
