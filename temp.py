
# # import numpy as np
# import h5py
# # Create an HDF5 file
# with h5py.File('example.hdf5', 'a') as file:
#     # Create a group
#     group=None
#     group_name = 'my_group'
#     if group_name in file:
#         group = file[group_name]
#     else:
#         group = file.create_group(group_name)

#     # Create a dataset
#     dataset=None
#     if 'my_dataset' in group:
#         dataset = group['my_dataset']
#     else:
#         dataset = group.create_dataset('my_dataset', data=data)


#     data = [ [2,4.0]]

#     # Add an attribute
#     dataset.attrs['units'] = 'meters'

#     # Read data from the dataset
#     read_data = group['my_dataset'][:]
#     print("Read Data:", read_data)

# data = np.array([[1.0,1.0], [2.0,2.0], [3.0,3.0], [2,4.0]], dtype=np.float32)
# np.save('my_data.npy', data)
# loaded_data = np.load('my_data.npy')
# print(loaded_data)
# # Create sample data
# record1 = {'id': 1, 'values': np.array([1.0, 2.0, 3.0], dtype=np.float32)}
# record2 = {'id': 2, 'values': np.array([4.0, 5.0, 6.0], dtype=np.float32)}
# record3 = {'id': 3, 'values': np.array([7.0, 8.0, 9.0], dtype=np.float32)}

# # Store records in a dictionary
# records_dict = {'record1': record1, 'record2': record2, 'record3': record3}

# # Save the dictionary to a .npy file
# np.save('multiple_records.npy', records_dict)
# Integer value
# import struct
# import numpy as np
# vector_dim=2
# total_dim=3
# records_np = np.random.random((3, vector_dim))
# matrix = [{"id": i, "embed": list(row)} for i, row in enumerate(records_np)]

# print(matrix)
# # Convert each sub-array to bytes and write to a file
# with open('output_file.txt', 'wb') as file:
#     for row in matrix:
#         id, embed = row["id"], row["embed"]
#         file.write(struct.pack('>i', id))

#         for num in embed:
#             # Convert each integer to bytes (using 4 bytes in this example) and write to file
#             float_bytes = struct.pack('>d', num)
#             file.write(float_bytes)
#         # file.write(b'\n')


# restored_matrix = []
# with open('output_file.txt', 'rb') as file:
#     while True:
#         # Read the id (integer) from file (4 bytes)
#         id_bytes = file.read(4)
#         if not id_bytes:
#             break
#         id = struct.unpack('>i', id_bytes)[0]
#         restored_matrix.append(id)
#         for i in range(vector_dim):
#             num_bytes = file.read(8)
#             if not num_bytes:
#                 print("Error: End of file reached")
#                 break
#             num = struct.unpack('>d', num_bytes)[0]
#             restored_matrix.append(num)

# # Reshape the 1D array into a 2D array
# restored_matrix = np.reshape(restored_matrix, (len(restored_matrix)//(total_dim), total_dim))
# # for row in restored_matrix:
# #     row[0]=int(row[0])
# # convert numpy array to list
# # restored_matrix = restored_matrix.tolist()
# # Print the restored 2D array
# print("Restored 2D Array:", restored_matrix)





# generate 2 random vectors with 70 dim
# LSH
# import numpy as np
# import random

# vector_dim=70
# total_dim=71
# top_k=10
# threshold=0.8
# def gram_schmidt(vectors):
#     basis = []
#     for vector in vectors:
#         for existing_vector in basis:
#             vector -= np.dot(vector, existing_vector) / np.dot(existing_vector, existing_vector) * existing_vector
#         basis.append(vector)
#     return basis
# def generate_random_vectors2(num_vectors):
#     vectors = []
#     for i in range(num_vectors):
#         vectors.append(np.random.random( vector_dim))
#     orthogonalized_vectors = gram_schmidt(vectors)
#     return orthogonalized_vectors
# def generate_random_vectors(num_vectors):
#     vectors = []
#     for i in range(num_vectors):
#         vector=np.zeros( vector_dim)
#         vector[i]=1.0
#         vectors.append(vector)
#     return vectors

# def _cal_score( vec1, vec2):
#         dot_product = np.dot(vec1, vec2)
#         # calc the euclidean norm of vec1 and vec2
#         # norm_vec1 = np.sqrt(np.sum(np.square(vec1)))
#         norm_vec1 = np.linalg.norm(vec1)
#         norm_vec2 = np.linalg.norm(vec2)
#         cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
#         return cosine_similarity

# rand=generate_random_vectors(5)
# mapping=[0]*(2**len(rand))
# data=[]
# for i in range(len(mapping)):
#     data.append([])

# for i in range(100000):
#     vec=np.random.random( vector_dim)
#     score=0
#     for j in range(len(rand)):
#         temp_score=_cal_score(vec,rand[j])
#         score=score<<1
#         if(temp_score>threshold):
#              score=score+1
#     mapping[score-1]+=1
#     data[score-1].append(vec)
# # print(mapping)
# index=0
# for x in mapping:
#     # print binarry represenation of index
#     print(index,x)
#     index+=1
# print(data)
# for _ in range(5): 
#     query = np.random.random( vector_dim)
#     # we need to fine the actual top k
#     # all we need is to find for wich buckets belong to the top k simmilar vectors
#     scores = []
#     actual=[0]*len(data)
#     index=0
#     for bucket in data:
#         for vec in bucket:
#             score = _cal_score(query, vec)
#             scores.append((score, index+1))
#         index+=1
#     out=sorted(scores, reverse=True)[:top_k]
#     # print(out)
#     for i in range(len(out)):
#         actual[out[i][1]-1]+=1

#     score=0
#     for j in range(len(rand)):
#         temp_score=_cal_score(query,rand[j])
#         score=score<<1
#         if(temp_score>threshold):
#             score=score+1
#     print(actual)
#     print("pretected bucket is : ",score)


# ########################################### end LSH
import time
import struct
import os
# import nanopq
# import numpy as np
# import nanopq
# import numpy as np

# N, Nt, D = 100000, 2000, 70
# X = np.random.random((N, D)).astype(np.float32)  # 10,000 128-dim vectors to be indexed
# Xt = np.random.random((Nt, D)).astype(np.float32)  # 2,000 128-dim vectors for training
# query = np.random.random((D,)).astype(np.float32)  # a 128-dim query vector

# # Instantiate with M=8 sub-spaces
# pq = nanopq.PQ(M=70,Ks=1024, verbose=True)
# # Train codewords
# pq.fit(Xt,)
# print(pq.verbose)
# print(pq.M)
# # we need to print the centroids itself
# print(len(pq.codewords))
# print(len(pq.codewords[0][0]))
# # Encode to PQ-codes
# X_code = pq.encode(X)  # (10000, 8) with dtype=np.uint8
# with open('test_compresed.csv', "w") as fout:
#             for row in X_code:
#                 row_str = ",".join([str(e) for e in row])
#                 fout.write(f"{row_str}\n")# Results: create a distance table online, and compute Asymmetric Distance to each PQ-code 
# dists = pq.dtable(query).adist(X_code)  # (10000, ) 

# rows=np.random.random((N, 70))
# with open('test.csv', "w") as fout:
#             for row in rows:
#                 row_str = ",".join([str(e) for e in row])
#                 fout.write(f"{row_str}\n")
def insertion():
    file_count=1
    fils=[]
    data=[[]]*10
    for i in range(file_count):
        fout = open(str(i)+'_.csv', 'wb')
        fils.append(fout)
    for i in range(1 * 70):
        for j in range(file_count):
            fils[j].write(struct.pack('>i', i))
            # data[j].append(str(i)+'\n')
    for i in range(file_count):
        # fils[i].write(''.join(data[i]))
        fils[i].close()
chunk_size = 70*8
def retrive():
    data=[]
    with open('0_.csv', "rb") as file:
        total_bytes = os.path.getsize('0_.csv')

        while True:
            # ebyte=file.read(4)
            data_chunk = file.read(chunk_size)


            if not data_chunk:
                break
            numbers = struct.unpack('>' + 'I' * (chunk_size // 4), data_chunk)
            for number in numbers:
                
                data.append(number)
        # for row in file.readlines():
        #     data.append(int(row[:-1]))
    return data

# tic = time.time()
# insertion()
# toc = time.time()
# run_time = toc - tic
# print("insertion time", run_time)

# tic = time.time()
# data=retrive()
# toc = time.time()
# run_time = toc - tic
# print("read time", run_time)
# print(len(data))
import numpy as np
import time

def generate_random_vectors(num_vectors):
    vectors = []
    for i in range(num_vectors):
        vectors.append(np.random.random(70))

    return vectors

random_vec = generate_random_vectors(2)

def _cal_score(vec_list, vec2):
    # Calculate dot product and norms
    dot_products = np.dot(vec_list, vec2)
    norm_vec1 = np.linalg.norm(vec_list, axis=1)
    norm_vec2 = np.linalg.norm(vec2)
    # Calculate cosine similarity for each pair
    cosine_similarities = dot_products / (norm_vec1 * norm_vec2)
    return cosine_similarities

def find_bucket_indices(vec_list):
    global random_vec
    num_buckets = 2**len(random_vec)
    bucket_indices = np.zeros(len(vec_list), dtype=int)

    for j in range(len(random_vec)):
        temp_scores = _cal_score(vec_list, random_vec[j]) 
        bucket_indices = (bucket_indices << 1) + (temp_scores > 0.75).astype(int)

    return bucket_indices

# Parameters
record_size = (70,)  # Shape of each record
total_records = 100000
filename = 'large_data.npy'

# Generate a list of random records
rng = np.random.default_rng(50)
records_list = rng.random((total_records,) + record_size, dtype=np.float32)

tic = time.time()

# Calculate bucket indices for the entire records_list
bucket_indices = find_bucket_indices(records_list)
tooc=time.time()
print('time =', tooc - tic)
print(len(bucket_indices),bucket_indices)
# Create a memory-mapped array for writing
memmap_array = np.memmap(filename, dtype=np.float32, mode='w+', shape=(total_records,) + record_size)

# Create a list to store the indices for each bucket
bucket_indices_list = [np.where(bucket_indices == i)[0] for i in range(2**len(random_vec))]

# Concatenate the records for each bucket and write them to the memory-mapped array
start_index = 0
for bucket_index in range(2**len(random_vec)):
    records_for_bucket = records_list[bucket_indices_list[bucket_index]]
    memmap_array[start_index: start_index + len(records_for_bucket)] = records_for_bucket
    start_index += len(records_for_bucket)

# Flush changes to disk
memmap_array.flush()

# Optionally, close the memmap array
del memmap_array

toc = time.time()

print('time =', toc - tic)
