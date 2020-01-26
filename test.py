import pickle
output = pickle.load( open( "save_9mm.p", "rb" ) )
train_loss = output[0]
train_acc  = output[1]
valid_loss = output[2]
valid_acc  = output[3]

