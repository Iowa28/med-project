import operator
import os

import numpy as np
import pandas as pd


class AnemiaService(object):

    def __init__(self):
        file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cvs/anemia_dataset.csv')
        self.data = pd.read_csv(file)
        self.mean = (self.data.mean(numeric_only=True))
        self.std = self.data.std(numeric_only=True)
        self.k = int(np.sqrt(len(self.data))) - 1

    def __euclidean_distance(self, employee_data, dataset_row):
        n_dist = 0
        for i in range(3, len(employee_data) - 1):
            n_dist += np.square(
                (float(employee_data[i]) - self.mean[i]) / self.std[i] - (dataset_row[i] - self.mean[i]) / self.std[i]
            )
        return np.sqrt(n_dist)

    def __knn(self, employee_data):
        distances = {}

        for index, row in self.data.iterrows():
            distances[index + 1] = self.__euclidean_distance(employee_data, row)

        sort_distances = sorted(distances.items(), key=operator.itemgetter(1))
        neighbors = []

        for x in range(1, self.k + 1):
            neighbors.append(sort_distances[x - 1][0])

        average_hgb = 0.0
        for x in range(len(neighbors)):
            neighbor = neighbors[x]
            average_hgb += self.data.iloc[neighbor - 1][-1]
        average_hgb /= len(neighbors)

        return average_hgb

    def calculate_hgb(self, employee):
        employee_data = pd.DataFrame([
            employee.id,
            employee.age,
            employee.sex,
            employee.rbc,
            employee.pcv,
            employee.mcv,
            employee.mch,
            employee.mchc,
            employee.rdw,
            employee.tlc,
            employee.plt
        ])

        return self.__knn(employee_data[0])
