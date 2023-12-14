from typing import Dict, List, Annotated
from sklearn.cluster import KMeans
import numpy as np
import struct
import os
import gc
import time
# taken_cluster = 150
# clusters = 1000
taken_cluster=150
clusters=1000
vector_dim = 70
total_dim = 71

class VecDBKmeans:
    def __init__(self,index=0, folder_path="saved_db", new_db=True) -> None:
        self.index=index
        self.folder_path = './'+folder_path+'/'+str(self.index)+"kmeans_files"

        folder_path = os.path.join(os.getcwd(), self.folder_path)

        # Check if the folder already exists
        if not os.path.exists(folder_path):
            # Create the folder
            os.makedirs(folder_path)

        if new_db:
            open(f"{self.folder_path}/centroids.csv", 'w').close()
            open(f"{self.folder_path}/data.csv", 'w').close()
           
    def insert_records(self, rows: List[ Annotated[List[float], 70]],ids: List[int]):
        if(len(ids)==0):
            return
        self.create_clusters(ids, rows, k=min(len(ids),clusters))
        self._build_index()
        
    def retrive_centers(self,path):
        total_bytes=os.path.getsize(path)
        records=(total_bytes//4)//(vector_dim) 
        centroids_list=np.zeros((records,vector_dim), dtype=np.float32)
        memmap_array_read = np.memmap(path, dtype=np.float32, mode='r', shape=(records, vector_dim))
        centroids_list[:,:]=memmap_array_read[:,:]
        del memmap_array_read
        return centroids_list
    
    def insert_centers(self,path,centers):
        memmap_array = np.memmap(path, dtype=np.float32, mode='w+', shape=(len(centers),vector_dim) )
        memmap_array[:,:]=centers
        memmap_array.flush()
        del memmap_array
    
    def insert_boundries(self,path,boundries):
        memmap_array = np.memmap(path, dtype=np.int32, mode='w+', shape=(len(boundries),2) )
        memmap_array[:,:]=boundries
        memmap_array.flush()
        del memmap_array
    
    def retrive_boundries(self,path):
        total_bytes=os.path.getsize(path)
        records=(total_bytes//4)//(2) 
        centers_boundries=np.zeros((records,2), dtype=np.int32)
        memmap_array_read = np.memmap(path, dtype=np.int32, mode='r', shape=(records, 2))
        centers_boundries[:,:]=memmap_array_read[:,:]
        del memmap_array_read
        return centers_boundries
  
    def create_clusters(self, ids, embeddings, k):
        '''
        creates an object of Kmeans and clusters the given data accordingly
        then it creates files with the number of clusters and adds the fitted data
        to the correct file based on the labels produced by the model
        '''

        # create a kmeans object and call fit 
        kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
        kmeans.fit(embeddings)
        # save centroids with their file name
        self.insert_centers(path=f"{self.folder_path}/centroids.csv",centers=kmeans.cluster_centers_)
        # for each cluster save the vectors with the same label to the part in the file
        memmap_array = np.memmap(f"{self.folder_path}/data.csv", dtype=np.float32, mode='w+', shape=(len(embeddings),total_dim) )
        start_index=0
        centers_boundries=np.zeros((k,2), dtype=np.int32)
        for i in range(k):
            centers_boundries[i][0]=start_index
            labels_indices = np.where(kmeans.labels_ == i)[0]
            memmap_array[start_index:start_index+len(labels_indices),0]=ids[labels_indices]
            memmap_array[start_index:start_index+len(labels_indices),1:]=embeddings[labels_indices]
            start_index+=len(labels_indices)
            centers_boundries[i][1]=start_index
        memmap_array.flush()
        del memmap_array   
        self.insert_boundries(path=f"{self.folder_path}/centers_coundries.csv",boundries=centers_boundries)
        gc.collect()
    def retrive(self,centroids, query: Annotated[List[float], 70], top_k=5):
        # read all the centroids from the file
        # calculates the cosine similarity between the query and all the centroids and takes top 3 centroids to search in
        centroids = self.retrive_centers(path=f"{self.folder_path}/centroids.csv")
        centers_boundries=self.retrive_boundries(path=f"{self.folder_path}/centers_coundries.csv")
        c_scores = self._cal_scores(centroids, query.reshape(-1))
        target_clusters = np.argsort(c_scores)[::-1]
        kmeans_scores = []
        
        
        # create memmap array to read the data from the file
        data_path=f"{self.folder_path}/data.csv"
        total_bytes=os.path.getsize(data_path)
        records=(total_bytes//4)//(total_dim) 
        memmap_array_read = np.memmap(data_path, dtype=np.float32, mode='r', shape=(records, total_dim))
        # calculate the cosine similarty in the files of similar centroids
        for i in range(len(target_clusters)):
            if(i>=taken_cluster and len(kmeans_scores)>=top_k):
                break
            start,end=centers_boundries[target_clusters[i]]
            # create a list of scores to sort them later
            rows=np.empty((end-start,70), dtype=np.float32)
            rows[:,:]=memmap_array_read[start:end,1:]
            ids=np.empty((end-start), dtype=np.int32)
            ids[:]=memmap_array_read[start:end,0]
            scores=self._cal_scores(rows, query.reshape(-1))
            choosen_ids=np.argsort(scores)[::-1][:min(len(scores),top_k)]
            kmeans_scores.extend([(scores[id],ids[id]) for id in choosen_ids])
        del memmap_array_read
        # sort the scores descendingly to get the top k similar vectors
        return sorted(kmeans_scores, reverse=True)[:min(len(kmeans_scores),top_k)]

    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def _cal_scores(self, vec_list, vec2):
        dot_products = np.dot(vec_list, vec2)
        # calc the euclidean norm of vec1 and vec2
        # norm_vec1 = np.sqrt(np.sum(np.square(vec1)))
        norm_vec1 = np.linalg.norm(vec_list, axis=1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarities = dot_products / (norm_vec1 * norm_vec2)
        return cosine_similarities

    def _build_index(self):
        pass

