
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
import numpy as np
import random

vector_dim=2
total_dim=71
top_k=10
def generate_random_vectors(num_vectors):
    vectors = []
    for i in range(num_vectors):
        vectors.append(np.random.random( vector_dim))
    return vectors

def _cal_score( vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        # calc the euclidean norm of vec1 and vec2
        # norm_vec1 = np.sqrt(np.sum(np.square(vec1)))
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

rand=generate_random_vectors(5)
mapping=[0]*(2**len(rand))
data=[]
for i in range(len(mapping)):
    data.append([])

for i in range(10000):
    vec=np.random.random( vector_dim)
    score=0
    for j in range(len(rand)):
        temp_score=_cal_score(vec,rand[j])
        score=score<<1
        if(temp_score>.75):
             score=score+1
    mapping[score-1]+=1
    data[score-1].append(vec)
print(mapping)
# print(data)
query = np.random.random( vector_dim)
# we need to fine the actual top k
# all we need is to find for wich buckets belong to the top k simmilar vectors
scores = []
actual=[0]*len(data)
index=0
for bucket in data:
    for vec in bucket:
        score = _cal_score(query, vec)
        scores.append((score, index+1))
    index+=1
out=sorted(scores, reverse=True)[:top_k]
# print(out)
for i in range(len(out)):
    actual[out[i][1]-1]+=1
print(actual)
score=0
for j in range(len(rand)):
    temp_score=_cal_score(query,rand[j])
    score=score<<1
    if(temp_score>.75):
        score=score+1
print("pretected bucket is : ",score)
