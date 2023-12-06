from typing import Dict, List, Annotated
import numpy as np
import struct
import csv
import os

vector_dim = 70
total_dim = 71
num_random_vectors = 2
similarity_threshold = 0.75
num_bytes = 8*70
class VecDBWorst:
    def __init__(self, file_path = "saved_db.csv",meta_data_path="meta.csv", new_db = True) -> None:
        self.file_path = file_path
        self.meta_data_path=meta_data_path
        self.file_paths=[]

        random_vectors = []
        if new_db:
            # just open new file to delete the old one
            # with open(self.file_path, "w") as fout:
            #     # if you need to add any head to the file
            #     pass
            with open(meta_data_path, "w") as file:
                random_vectors = self.generate_random_vectors(num_random_vectors)
                for vector in random_vectors:
                    row_str = ",".join([str(e) for e in vector])
                    file.write(f"{row_str}\n")

                    # file.write(str(vector))
                    # file.write("\n")
                pass
            
        else:
            with open(meta_data_path, "r") as file:
                for line in file:
                    row_splits = line.split(",")
                    # float_values = [float(value) for value in line[1:-2].split()]
                    random_vectors.append([float(e) for e in row_splits[:]])
        # create 2**num_random_vectors files
        for i in range(2**num_random_vectors):
            self.file_paths.append(str(i+1)+"_"+self.file_path)
            if new_db :
                with open(self.file_paths[i], "w") as fout:
                    # store fout in a list to use it later
                    pass
        self.random_vectors = random_vectors
    
    
    def gram_schmidt(self,vectors):
        basis = []
        for vector in vectors:
            for existing_vector in basis:
                vector -= np.dot(vector, existing_vector) / np.dot(existing_vector, existing_vector) * existing_vector
            basis.append(vector)
        return basis
    def generate_random_vectors(self,num_vectors):
        vectors = []
        for i in range(num_vectors):
            vectors.append(np.random.random( vector_dim))
        # orthogonalized_vectors = self.gram_schmidt(vectors)

        return vectors

    def insert_records(self, rows: List[Dict[int, Annotated[List[float], 70]]]):
        # rows has the following shape [
        # {
        #     "id": the actual id,
        # },   "embed" : [70 dim vector]
        # 
        # ]
        files=[]
        for path in self.file_paths:
            fout = open(path, 'ab')
            files.append(fout)
        for row in rows:
            id, embed = row["id"], row["embed"]
            bucket_index=self.find_bucket_index(embed)
            # with open( self.file_paths[bucket_index] , "ab") as fout:
            fout=files[bucket_index]
            fout.write(struct.pack('>i', id))
            for num in embed:
                # Convert each integer to bytes (using 4 bytes in this example) and write to file
                float_bytes = struct.pack('>f', float(num))
                fout.write(float_bytes)
        for file in files:
            file.close()
        self._build_index()
    def find_bucket_index(self, query):
        bucket=0
        # 11

        # bucket = 1 
        # 1
        # bucket << 1 ---> 10 --> 11
        for j in range(len(self.random_vectors)):
            temp_score=self._cal_score(query,self.random_vectors[j])
            bucket=bucket<<1
            if(temp_score>similarity_threshold):
                bucket=bucket+1
        return bucket
    def retrive(self, query: Annotated[List[float], 70], top_k = 5):        
        bucket_index=self.find_bucket_index(query)
        global_scores=[]
        for i in range (num_random_vectors+1):
            scores = []
            restored_matrix = []
            temp_bucket_index=bucket_index
            if(i!=0):
                temp_bucket_index=bucket_index^(1<<(i-1))
            file_path =self.file_paths[temp_bucket_index] 
            total_bytes=os.path.getsize(file_path)
            with open(file_path, 'rb') as file:
                data = file.read(total_bytes)
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
                    restored_matrix.append(row)
            # it is 2d matrix
            # each elemnt represent a row
            # the first element of each row is the id, the rest is the embed

            for row in restored_matrix:
                id = row[0]
                embed = row[1:]
                score = self._cal_score(query, embed)
                scores.append((score, id))

            # here we assume that if two rows have the same score, return the lowest ID
            scores = sorted(scores, reverse=True)[:top_k]#sort in decreasing order , the best choice is the biggest one
            global_scores.extend(scores)
        
        global_scores=sorted(global_scores, reverse=True)[:top_k]
        return [s[1] for s in global_scores]
    
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        # calc the euclidean norm of vec1 and vec2
        # norm_vec1 = np.sqrt(np.sum(np.square(vec1)))
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def _build_index(self):
        pass


# db = VecDBWorst(new_db=True)
# records_np = np.random.random((1000, 3))
# records_dict = [{"id": i, "embed": list(row)} for i, row in enumerate(records_np)]
# db.insert_records(records_dict)

