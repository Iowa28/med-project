import operator

import numpy as np
import pandas as pd


class AnemiaService(object):

    def __init__(self):
        self.data = pd.read_csv('cvs/anemia_dataset.csv')
        self.mean = self.data.mean(numeric_only = True)
        self.std = self.data.std(numeric_only = True)
        self.k = int(np.sqrt(len(self.data) + 1))
        self.__load_dataset()

    def __load_dataset(self):
        pass

    def __knn(self, employee_data):
        distances = {}

        for index, row in self.data.iterrows():
            distances[index + 1] = self.__euclidean_distance(employee_data, row)

        sort_distances = sorted(distances.items(), key=operator.itemgetter(1))
        neighbors = []

        for x in range(1, self.k + 1):
            neighbors.append(sort_distances[x - 1][0])

        counts = {"Iris-setosa": 0, "Iris-versicolor": 0, "Iris-virginica": 0}

        for x in range(len(neighbors)):
            neighbor = neighbors[x]
            species = self.data.iloc[neighbor - 1][-1]
            if species in counts:
                counts[species] += 1
            else:
                counts[species] = 1

        sort_counts = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)
        return sort_counts[0][0]

    def __euclidean_distance(self, employee_data, data_row):
        n_dist = 0
        for i in range(1, len(employee_data)):
            n_dist += np.square(
                (employee_data[i] - self.mean[i]) / self.std[i] - (data_row[i] - self.mean[i]) / self.std[i]
            )
        return np.sqrt(n_dist)

    def __row_list(self):
        row_list = []
        for index, rows in self.data.iterrows():
            row_list.append(
                [rows.Id, rows.SepalLengthCm, rows.SepalWidthCm, rows.PetalLengthCm, rows.PetalWidthCm]
            )
        return row_list

    def calculate_hgb(self, employee):
        pass
