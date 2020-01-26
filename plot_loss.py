import matplotlib.pyplot as plt
import numpy as np

infile = open('slurm-41901121.out', 'r')
loss = []
accu = []
for line in infile:
    if 'Loss' in line and 'Acc' in line:
        loss.append(float(line.split()[3]))
        accu.append(float(line.split()[6][:-1]))

x = np.linspace(0, 10, len(loss))
np_loss = np.array(loss)
np_accu = np.array(accu)
fig, (ax1, ax2)  = plt.subplots(1,2)
ax1.plot(x, np_loss)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax2.plot(x, np_accu)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
plt.show()
