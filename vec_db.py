from typing import Dict, List, Annotated
import numpy as np
import struct
import csv
import os
import threading
import gc
from kmeans import VecDBKmeans
import time
vector_dim = 70
total_dim = 71
num_random_vectors = 3
taken_buckets = 6
similarity_threshold = 0.75
num_threads=1



def _cal_score( vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        # calc the euclidean norm of vec1 and vec2
        # norm_vec1 = np.sqrt(np.sum(np.square(vec1)))
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity
def retrive_thred(query,top_k, data,restored_matrix):
    records=(len(data)//4)//(total_dim)
    current_byte=0
    temp_matrix=[]
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
        temp_matrix.append(row)
    scores = []
    for row in temp_matrix:
            id = row[0]
            embed = row[1:]
            score = _cal_score(query, embed)
            scores.append((score,row))
    scores = sorted(scores, reverse=True)[:top_k]
    restored_matrix.extend([s[1] for s in scores])

class VecDB:
    def __init__(self, file_path = "saved_db.csv",meta_data_path="meta.csv", new_db = True) -> None:
        self.file_path = file_path
        self.file_paths=[]
        self.kmeans=[]
        random_vectors = []
        self.folder_path = os.path.join(os.getcwd(),file_path.split('.')[0] )
        self.meta_data_path='./'+file_path.split('.')[0]+'/'+meta_data_path

        # Check if the folder already exists
        if not os.path.exists(self.folder_path):
            # Create the folder
            os.makedirs(self.folder_path)
        if new_db:
            # just open new file to delete the old one
            # with open(self.file_path, "w") as fout:
            #     # if you need to add any head to the file
            #     pass
            with open(self.meta_data_path, "w") as file:
                random_vectors = self.generate_random_vectors(num_random_vectors)
                for vector in random_vectors:
                    row_str = ",".join([str(e) for e in vector])
                    file.write(f"{row_str}\n")

                    # file.write(str(vector))
                    # file.write("\n")
                pass
            
        else:
            with open(self.meta_data_path, "r") as file:
                for line in file:
                    row_splits = line.split(",")
                    # float_values = [float(value) for value in line[1:-2].split()]
                    random_vectors.append([float(e) for e in row_splits[:]])
        # # create 2**num_random_vectors files
        # for i in range(2**num_random_vectors):
        #     self.file_paths.append(str(i+1)+"_"+self.file_path)
        #     if new_db :
        #         with open(self.file_paths[i], "w") as fout:
        #             # store fout in a list to use it later
        #             pass
        self.random_vectors = random_vectors
        for i in range(2**num_random_vectors):
            self.kmeans.append(VecDBKmeans(index=i,folder_path=file_path.split('.')[0],new_db=new_db))
    
    def generate_random_vectors(self,num_vectors):
        vectors = []
        for i in range(num_vectors):
            vectors.append(np.random.random( vector_dim).astype(np.float32))

        return vectors
    def insert_level_1(self,rows_list):
        bucket_indices = self.find_bucket_indces(rows_list)
        bucket_indices_list = [np.where(bucket_indices == i)[0] for i in range(2**len(self.random_vectors))]
        for bucket_index in range(2**len(self.random_vectors)):
            records_for_bucket = rows_list[bucket_indices_list[bucket_index]]
            memmap_array = np.memmap(f"./temp/data{bucket_index}.csv", dtype=np.float32, mode='w+', shape=(len(records_for_bucket),total_dim) )
            memmap_array[:,0] = bucket_indices_list[bucket_index]
            memmap_array[:,1:] = records_for_bucket[:]
            memmap_array.flush()
            del memmap_array
    def insert_level_2(self,index):
        path=f"./temp/data{index}.csv"
        total_bytes=os.path.getsize(path)
        records=(total_bytes//4)//(total_dim) 
        memmap_array_read = np.memmap(path, dtype=np.float32, mode='r', shape=(records,total_dim) )
        records_for_bucket=memmap_array_read[:,1:]
        ids=memmap_array_read[:,0]
        self.kmeans[index].insert_records(rows=records_for_bucket,ids=ids) 
         
    def insert_records(self, rows: List[Dict[int, Annotated[List[float], 70]]],dic=True,rows_list=[]):
        # rows has the following shape [
        # {
        #     "id": the actual id,
        # },   "embed" : [70 dim vector]
        # 
        # ]
        records_list=[]
        if(dic):
            records_list=np.empty((len(rows),vector_dim) , dtype=np.float32)
            for row in rows:
                id, embed = row["id"], row["embed"]
                records_list[id]=embed
        else :
            records_list=rows_list
        bucket_indices = self.find_bucket_indces(records_list)
        bucket_indices_list = [np.where(bucket_indices == i)[0] for i in range(2**len(self.random_vectors))]
        for bucket_index in range(2**len(self.random_vectors)):
            records_for_bucket = records_list[bucket_indices_list[bucket_index]]
            if(len(records_for_bucket)>0):
                self.kmeans[bucket_index].insert_records(rows=records_for_bucket,ids=bucket_indices_list[bucket_index])
        self._build_index()
         
    def hamming_distance(self ,a, b):
        return bin(a ^ b).count('1')

    def custom_sort(self,element):
        return self.hamming_distance(element, self.target_bucket)

    def retrive(self, query: Annotated[List[float], 70], top_k = 5):        
        bucket_index=self.find_bucket_index(query)
        self.target_bucket=bucket_index
        global_scores=[]
        buckets=[ i for i in range(2**num_random_vectors)]
        sorted_buckets = sorted(buckets, key=self.custom_sort)
        for i in range (len(sorted_buckets)):
            if(i>=taken_buckets and len(global_scores)>top_k):
                break
            temp_bucket_index=sorted_buckets[i]
            kmeans=self.kmeans[temp_bucket_index]
            global_scores.extend(kmeans.retrive(query,top_k))
        global_scores=sorted(global_scores, reverse=True)[:min(top_k,len(global_scores))]
        return [s[1] for s in global_scores]
    
    def find_bucket_index(self, query):
        bucket=0
        for j in range(len(self.random_vectors)):
            temp_score=self._cal_score(query,self.random_vectors[j])
            bucket=bucket<<1
            if(temp_score>similarity_threshold):
                bucket=bucket+1
        return bucket
    
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        # calc the euclidean norm of vec1 and vec2
        # norm_vec1 = np.sqrt(np.sum(np.square(vec1)))
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def find_bucket_indces(self, vec_list):
        bucket_indices = np.zeros(len(vec_list), dtype=int)
        
        for j in range(len(self.random_vectors)):
            temp_scores = self._cal_scores(vec_list, self.random_vectors[j]) 
            bucket_indices = (bucket_indices << 1) + (temp_scores > 0.75).astype(int)
        return bucket_indices

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

