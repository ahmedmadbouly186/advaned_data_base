from typing import Dict, List, Annotated
import numpy as np
import struct
import csv


vector_dim=70
total_dim=71
class VecDBWorst:
    def __init__(self, file_path = "saved_db.csv", new_db = True) -> None:
        self.file_path = file_path
        if new_db:
            # just open new file to delete the old one
            with open(self.file_path, "w") as fout:
                # if you need to add any head to the file
                pass
    
    def insert_records(self, rows: List[Dict[int, Annotated[List[float], 70]]]):
        # rows has the following shape [
        # {
        #     "id": [70 dim vector],
        # },
        # {
        #     "id": [70 dim vector],
        # },.....
        # 
        # ]
        with open(self.file_path, "ab") as fout:
            for row in rows:
                id, embed = row["id"], row["embed"]
                fout.write(struct.pack('>i', id))
                for num in embed:
                    # Convert each integer to bytes (using 4 bytes in this example) and write to file
                    float_bytes = struct.pack('>d', num)
                    fout.write(float_bytes)

                # row_str = f"{id}," + ",".join([str(e) for e in embed])
                # fout.write(f"{row_str}\n")
                # print(type(id),id)
                # print(type(embed))
                # print(type(embed[0]),embed[0])
                # for nustn(num) 


                # id, embed = row["id"], row["embed"]
                # binary_embed = struct.pack(f"{len(embed)}f", *embed)
                # row_str = f"{id},".encode() + binary_embed + "\n".encode()
                # fout.write(f"{row_str}\n".encode())
            # writer.writerows(marks)
        self._build_index()

    def retrive(self, query: Annotated[List[float], 70], top_k = 5):
        scores = []
        restored_matrix = []
        with open(self.file_path, 'rb') as file:
            while True:
                # Read the id (integer) from file (4 bytes)
                id_bytes = file.read(4)
                if not id_bytes:
                    break
                id = struct.unpack('>i', id_bytes)[0]
                restored_matrix.append(id)
                for i in range(vector_dim):
                    num_bytes = file.read(8)
                    if not num_bytes:
                        print("Error: End of file reached")
                        break
                    num = struct.unpack('>d', num_bytes)[0]
                    restored_matrix.append(num)
        restored_matrix = np.reshape(restored_matrix, (len(restored_matrix)//(total_dim), total_dim))
        for row in restored_matrix:
            id = row[0]
            embed = row[1:]
            score = self._cal_score(query, embed)
            scores.append((score, id))

        # with open(self.file_path, "rb") as fin:
        #     for row in fin.readlines():
        #         row_splits = row.split(",")
        #         id = int(row_splits[0])
        #         embed = [float(e) for e in row_splits[1:]]
        #         score = self._cal_score(query, embed)
        #         scores.append((score, id))
        # here we assume that if two rows have the same score, return the lowest ID
        scores = sorted(scores, reverse=True)[:top_k]#sort in decreasing order , the best choice is the biggest one
        return [s[1] for s in scores]
    
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


