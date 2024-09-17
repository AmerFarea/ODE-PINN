import csv
import numpy as np

def load_data(filename):
    timepoints, population = [], []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            timepoints.append(float(row[0]))
            population.append(float(row[1]))
    return np.array(timepoints), np.array(population)
