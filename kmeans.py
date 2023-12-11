from typing import Dict, List, Annotated
from sklearn.cluster import KMeans
import numpy as np
import struct
import os
taken_cluster = 20
clusters = 50
vector_dim = 70
total_dim = 71

class VecDBKmeans:
    def __init__(self,index=0, file_path="saved_db.csv", new_db=True) -> None:
        self.file_path = file_path
        self.index=index
        folder_path = os.path.join(os.getcwd(), str(self.index)+"kmeans_files")

        # Check if the folder already exists
        if not os.path.exists(folder_path):
            # Create the folder
            os.makedirs(folder_path)

        if new_db:
            # just open new file to delete the old one
            with open(self.file_path, "w") as fout:
                # if you need to add any head to the file
                pass

    def insert_records(self, rows: List[Dict[int, Annotated[List[float], 70]]]):

        embeddings = []
        ids = []
        for row in rows:
            id, embed = row["id"], row["embed"]
            embeddings.append(embed)
            ids.append(id)

        self.create_clusters(ids, embeddings, k=clusters)
        self._build_index()

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
        file_index = 0
        with open("./"+str(self.index)+"kmeans_files/centroids.csv", 'w') as centroids:
            for centroid in kmeans.cluster_centers_:
                row_str = f"{file_index}," + \
                    ",".join([str(c) for c in centroid])
                centroids.write(f"{row_str}\n")
                file_index += 1

        # for each cluster save the vectors with the same label to the correct file
        # there should be k files 
        for i in range(k):
            with open(f"./"+str(self.index)+f"kmeans_files/cluster_{i}.csv", 'wb') as cluster:
                labels_indices = np.where(kmeans.labels_ == i)[0]   # returns the indices of labels of a specific cluster
                for index in labels_indices:    # loops on all the label indices to get the embedding that matches the index label  
                    embed = embeddings[index]   
                    id = ids[index]
                    # row_str = f"{ids[index]}," + \
                    #     ",".join([str(e) for e in embed])
                    # cluster.write(f"{row_str}\n")
                    cluster.write(struct.pack('>i', id))
                    for num in embed:
                        float_bytes = struct.pack('>f', float(num))
                        cluster.write(float_bytes)

    def retrive(self, query: Annotated[List[float], 70], top_k=5):

        # read all the centroids from the file
        # calculates the cosine similarity between the query and all the centroids and takes top 3 centroids to search in
        centroids_scores = []
        with open(f"./"+str(self.index)+"kmeans_files/centroids.csv", "r") as centroids:
            for row in centroids.readlines():
                row_splits = row.split(",")
                cluster = int(row_splits[0])
                embed = [float(e) for e in row_splits[1:]]
                c_score = self._cal_score(query, embed)
                centroids_scores.append((c_score, cluster))
            centroids_scores = sorted(centroids_scores, reverse=True)

        target_clusters = [centroids_scores[i][1] for i in range(taken_cluster)] # top 3 similary centroids

        kmeans_scores = []
        # calculate the cosine similarty in the files of similar centroids
        for i in range(len(target_clusters)):
            file_path=f"./"+str(self.index)+f"kmeans_files/cluster_{target_clusters[i]}.csv"
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
                    kmeans_scores.append((kmeans_score, id))
                # for row in fcluster.readlines():
                #     row_splits = row.split(",")
                #     id = int(row_splits[0])
                #     embed = [float(e) for e in row_splits[1:]]
                #     kmeans_score = self._cal_score(query, embed)
                #     kmeans_scores.append((kmeans_score, id))
        # sort the scores descendingly to get the top k similar vectors
        kmeans_scores = sorted(kmeans_scores, reverse=True)[:top_k]
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
