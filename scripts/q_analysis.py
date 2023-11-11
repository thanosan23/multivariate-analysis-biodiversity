import numpy as np
import matplotlib.pyplot as plt

Q_table = np.load('q_table.npy')

counts = {}
for i in range(160):
    prev = counts.get(np.argmax(Q_table[i]), 0)
    counts[np.argmax(Q_table[i])] = prev + 1

plt.bar(list(counts.keys()), list(counts.values()))

plt.show()
