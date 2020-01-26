import matplotlib.pyplot as plt
import numpy as np
import pickle

output = pickle.load( open( "save_3mm.p", "rb" ) )
train_loss = np.zeros(30)
train_acc = np.zeros(30)
valid_loss = np.zeros(30)
valid_acc = np.zeros(30)
x = np.linspace(0, 30, 30)
for iter in range(30):
    train_loss[iter] = output[0][iter]
    train_acc[iter] = output[1][iter]
    valid_loss[iter] = output[2][iter]
    valid_acc[iter] = output[3][iter]

fig,axs = plt.subplots(1,2)
axs[0].set_xlabel('epoch', fontsize=14)
axs[0].set_ylabel('loss', fontsize=14)
axs[0].set_ylim(0.,0.5)
#axs[0].set_yscale("log")

axs[1].set_xlabel('epoch', fontsize=14)
axs[1].set_ylabel('accuracy (%)', fontsize=14)
axs[1].set_ylim(83.0,95.0)
# plot up to current iteration
axs[0].plot(x[:],train_loss[:],'b')
axs[0].plot(x[:],valid_loss[:],'r')

# plot up to current iteration
axs[1].plot(x[:],train_acc[:],'b')
axs[1].plot(x[:],valid_acc[:],'r')
fig.canvas.draw()
plt.subplots_adjust(wspace=.3)
plt.savefig('acc_loss_3mm.png')
