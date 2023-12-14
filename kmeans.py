from typing import Dict, List, Annotated
from sklearn.cluster import KMeans
import numpy as np
import struct
import os
import gc
taken_cluster = 100
clusters = 1000
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
            open(f"{self.folder_path}/centroids.csv", 'w').close()

        if new_db:
            open(f"{self.folder_path}/centroids.csv", 'w').close()
            # just open new file to delete the old one
            for i in range(clusters):
                open(f"{self.folder_path}/cluster_{i}.csv", 'w').close()
           
    def insert_records(self, rows: List[ Annotated[List[float], 71]],dic=True):
        embeddings = [0]*len(rows)
        ids = [0]*len(rows)
        index=0
        for row in rows:
            id, embed = row[0], row[1:]
            embeddings[index]=embed
            ids[index]=id
            index+=1
        if(len(ids)==0):
            return
        self.create_clusters(ids, embeddings, k=min(len(ids),clusters))
        self._build_index()
        
    def retrive_centers(self,path):
        centroid_path=path
        total_bytes=os.path.getsize(centroid_path)
        centers=[]
        with open(centroid_path, "rb") as centroids:
            data = centroids.read(total_bytes)
            records=(len(data)//4)//(total_dim) 
            current_byte=0
            for i in range (records):
                row=[0]*total_dim
                # Read the id (integer) from file (4 bytes)
                cluster = struct.unpack('>i', data[current_byte:current_byte+4])[0]
                current_byte+=4
                embed = struct.unpack('>' + 'f' * 70, data[current_byte:current_byte+vector_dim*4])
                current_byte+=vector_dim*4
                row[0]=cluster
                row[1:]=embed
                centers.append(row)
        return centers
    
    def insert_centers(self,path,centers):
        file_index = 0
        # with open(f"{self.folder_path}/centroids.csv", 'w') as centroids:
        #     for centroid in kmeans.cluster_centers_:
        #         row_str = f"{file_index}," + \
        #             ",".join([str(c) for c in centroid])
        #         centroids.write(f"{row_str}\n")
        #         file_index += 1
        
        
        
        
        # write centers
        with open(path, 'wb') as centroids:
            for centroid in centers:
                centroids.write(struct.pack('>i', file_index))
                for c in centroid:
                    float_bytes = struct.pack('>f', float(c))
                    centroids.write(float_bytes)
                file_index += 1
    
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

        # for each cluster save the vectors with the same label to the correct file
        # there should be k files 
        for i in range(k):
            with open(f"{self.folder_path}/cluster_{i}.csv", 'ab') as cluster:
                labels_indices = np.where(kmeans.labels_ == i)[0]   # returns the indices of labels of a specific cluster
                for index in labels_indices:    # loops on all the label indices to get the embedding that matches the index label  
                    embed = embeddings[index]   
                    id = ids[index]
                    cluster.write(struct.pack('>i', id))
                    for num in embed:
                        float_bytes = struct.pack('>f', float(num))
                        cluster.write(float_bytes)
    
    def retrive(self, query: Annotated[List[float], 70], top_k=5):
        # read all the centroids from the file
        # calculates the cosine similarity between the query and all the centroids and takes top 3 centroids to search in
        centroids=self.retrive_centers(path=f"{self.folder_path}/centroids.csv")
        centroids_scores = []
        for center in centroids:
            c_score = self._cal_score(query, center[1:])
            centroids_scores.append((c_score, center[0]))
            
        centroids_scores = sorted(centroids_scores, reverse=True)
        target_clusters = [centroids_scores[i][1] for i in range(len(centroids_scores))] 

        kmeans_scores = []
        # calculate the cosine similarty in the files of similar centroids
        for i in range(len(target_clusters)):
            if(i>=taken_cluster and len(kmeans_scores)>=top_k):
                break
            scores=[]
            file_path=f"{self.folder_path}/cluster_{target_clusters[i]}.csv"
            total_bytes=os.path.getsize(file_path)
            with open(file_path, "rb") as fcluster:
                data = fcluster.read(total_bytes)
                records=(len(data)//4)//(total_dim)
                current_byte=0
                for i in range (records):
                    row=[0]*total_dim
                    # Read the id (integer) from file (4 bytes)
                    id = struct.unpack('>i', data[current_byte:current_byte+4])[0]
                    current_byte+=4
                    nums = struct.unpack('>' + 'f' * 70, data[current_byte:current_byte+vector_dim*4])
                    current_byte+=vector_dim*4
                    row[0]=id
                    index=1
                    for num in nums:
                        row[index]=num
                        index+=1
                    kmeans_score = self._cal_score(query, row[1:])
                    scores.append((kmeans_score, id))
            kmeans_scores.extend(sorted(scores, reverse=True)[:min(len(scores),top_k)])
            del scores
            # gc.collect()
        # sort the scores descendingly to get the top k similar vectors
        kmeans_scores = sorted(kmeans_scores, reverse=True)[:min(len(kmeans_scores),top_k)]
        top_k_kmeans = [(ks[0],ks[1]) for ks in kmeans_scores]
        return top_k_kmeans

    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def _build_index(self):
        pass
