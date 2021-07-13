import sys
import numpy as np
sys.path.append('../../../nn')
from mynn import *
from datetime import datetime

import matplotlib as mpl 
from matplotlib.lines import Line2D 
# mpl.use('TkAgg')
import matplotlib.pyplot as plt

plt.rc("figure", dpi=300)           # High-quality figure ("dots-per-inch")
plt.rc("text", usetex=True)         # Crisp axis ticks
plt.rc("font", family="serif")      # Crisp axis labels
plt.rc("legend", edgecolor='none')  # No boxes around legends

plt.rc("figure",facecolor="#ffffff")
plt.rc("axes",facecolor="#ffffff",edgecolor="#000000",labelcolor="#000000")
plt.rc("savefig",facecolor="#ffffff")
plt.rc("text",color="#000000")
plt.rc("xtick",color="#000000")
plt.rc("ytick",color="#000000")

color1 = 'tab:blue'
color2 = 'tab:green'
color3 = 'tab:orange'


data    = np.load('../../data/data.npz')
inputs  = np.concatenate((data["theta"], data["u"][np.newaxis, :]), axis=0)
r_f, _ = inputs.shape
outputs = data["aT"]
N, M = outputs.shape

xgrid = np.linspace(0,1,N+1)
xgrid = xgrid[:-1]
dx    = xgrid[1] - xgrid[0]



train_inputs = inputs[:,0::2]
test_inputs  = inputs[:,1::2]

train_outputs = outputs[:,0::2]
test_outputs  = outputs[:,1::2]


f_hat = train_inputs.T
fx = np.concatenate((np.repeat(f_hat,N,axis=0), np.tile(xgrid,M//2)[:,np.newaxis]),axis=1)
gx = np.reshape(train_outputs,((M//2)*N,),order='F')

Ndata = N*(M//2)

# train on random 1/256 of the data
# tr_i = np.random.randint(0,Ndata,(1024,))
# x_train = torch.from_numpy(fx[tr_i,:].astype(np.float32))
# y_train = torch.from_numpy(gx[tr_i,np.newaxis].astype(np.float32))
x_train = torch.from_numpy(fx.astype(np.float32))
y_train = torch.from_numpy(gx[:,np.newaxis].astype(np.float32))

ds = DirectData(X=x_train, y=y_train)
ds = DataLoader(ds, batch_size=1024, shuffle=True)


N_neurons = 50

if N_neurons == 20:
    DirectNet = DirectNet_20
elif N_neurons == 50:
    DirectNet = DirectNet_50

model = DirectNet(r_f+1,1)
# model = torch.load("fxNet_"+str(N_neurons)+".model")

loss_fn = torch.nn.MSELoss(reduction='sum')
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=1e-4)

loss_scale = 1000
n_epochs = 100000
for epoch in range(n_epochs):
	y_pred = model(x_train)
	loss = loss_fn(y_pred,y_train)*loss_scale

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	if epoch % 100 == 0:
		print("[{}/{}], loss: {}, time {}".format(epoch, n_epochs, np.round(loss.item(), 3),datetime.now()))
		torch.save(model, "fxNet_"+str(N_neurons)+".model")

	
# save the model
torch.save(model, "fxNet_"+str(N_neurons)+".model")
# np.save('tr_i.npy',tr_i)