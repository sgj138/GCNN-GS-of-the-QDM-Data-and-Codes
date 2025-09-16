import pickle
import os
import sys
import jax.flatten_util
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import jax
jax.config.update('jax_platform_name', 'cpu')
import glob
import sys
import time
import json
import jax
import jax.numpy as jnp
import netket as nk
import optax
from dimer_utils_chunked import *
import equivariant_p4m as p4m 
import copy
import pickle
import random
from mpi4py import MPI


# Get the MPI communicator
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


# Take command line arguments
L = int(sys.argv[1])            #linear system size
t = -1.0
V = float(sys.argv[2])
n_layers = int(sys.argv[3])     #number of layers in GCNN

n_iter = int(sys.argv[4])       #number of iterations in the optimisation
save_state_interval = int(sys.argv[5]) #number of iterations between saving the state
chunk_size = int(sys.argv[6])
num_intervals = int(n_iter / save_state_interval)
jobc = None


n_samples =2**12
n_chains = 2**6
sweep_size = 4
n_discard_per_chain = 1000



if nk.utils.mpi.rank == 0:
    jobc = random.randint(100, 999)
    print(" L : ", L, " t : ", t, " V : ", V, " jobc ", jobc)

jobc = comm.bcast(jobc, root=0)     # Broadcast the jobc value to all processes

# Now, jobc is available across all processes
print(f"Rank {rank}: jobc = {jobc}")


lattice = nk.graph.Square(L, max_neighbor_order=1)
hi = Dimer(N=lattice.n_nodes)
H = DimerHamiltonian(hi, V=V, t=t)

# Channels in the GCNN
channel_options = np.empty(n_layers, dtype=int)
channel_options[-1] = 2  #last layer will have 2 channels (user defined)

for i in range(len(channel_options)-2,-1,-1):   #if nLayers = 5, them it will be (5th layer, 0th layer, step_size =-1)
    channel_options[i]= channel_options[i+1] + 2

# Set up the machine
machine = p4m.GCNN(
    symmetries=lattice, 
    product_table=None,
    layers=n_layers, 
    mode="p4m", 
    features=tuple(channel_options), 
    param_dtype=np.complex128)
print(" Model Initialization done ")

sampler = nk.sampler.MetropolisSamplerNumpy(hi, WormRule(), n_chains=n_chains, sweep_size = sweep_size)

opt = nk.optimizer.Sgd(learning_rate=0.02)
sr = nk.optimizer.SR(diag_shift=0.01)
vstate = nk.vqs.MCState(
    sampler=sampler,
    model=machine,
    n_samples=n_samples,
    n_discard_per_chain=n_discard_per_chain,
    chunk_size=chunk_size,)

#save the details of the network in a pickle file
if nk.utils.mpi.rank == 0:
    machine_symmetries_shape = machine.symmetries.shape
    network_details = {
    'machine': 
            {'layers': machine.layers, 
            'features': machine.features, 
            'numOfSymmetries' : machine_symmetries_shape, 
            'param_dtype' : machine.param_dtype}, 
      'vstate' : 
            {'numOfParams': vstate.n_parameters, 
             'numOfSamples': vstate.n_samples, 
             'n_discard_per_chain' : vstate.n_discard_per_chain}
    }

    network_filename = (
		    str(jobc)
		    + "_L_"
		    + str(L)
		    + "_t_"
		    + str(t)
		    + "_V_"
		    + str(V)
		    + "_nLayers_"
		    + str(n_layers)
		    + "_numSymmetries_"
		    + str(machine_symmetries_shape[0])
            + "_chunk_size_"
            + str(chunk_size)
		    + ".pickle"
		)
		
    with open(network_filename, "wb") as file:
	    pickle.dump(network_details, file)


#printing details of machine and 
if nk.utils.mpi.rank == 0:
    print("machine.layers = ", machine.layers)
    print("machine.features = ", machine.features)
    print(" vstate.n_samples ", vstate.n_samples, "\n")
    print(" vstate.n_samples_per_rank ", vstate.n_samples_per_rank, "\n")
    print(" vstate.chain_length ", vstate.chain_length, "\n")
    print(" sampler.sweep_size ", sampler.sweep_size, "\n")
    print(" vstate.chunk_size ", vstate.chunk_size, "\n")

gs = nk.driver.VMC(H, opt, variational_state=vstate, preconditioner=sr)

# Store the data
json_data_filename = (
    str(jobc) + "_jsondata_t_" + str(t) + "_V_" + str(V) + "_L_" + str(L) + "_nLayers_" + str(n_layers) + "_chunkSize_" + str(chunk_size)
)
text_data_filename = (
    str(jobc) + "_textdata_t_" + str(t) + "_V_" + str(V) + "_L_" + str(L) + "_nLayers_"	+ str(n_layers) + "_chunkSize_" + str(chunk_size) + ".txt"
)


for interval in range(1, num_intervals + 1):
    start_time = time.time()
    gs.run(n_iter=save_state_interval, out=json_data_filename)
    finish_time = time.time()
    # Use only one rank to save the state
    if nk.utils.mpi.rank == 0:
        state_filename = (
            str(jobc)
            + "_state_t_"
            + str(t)
            + "_V_"
            + str(V)
            + "_L_"
            + str(L)
            + "_iters_passed_"
            + str(interval * save_state_interval)
            + "_chunkSize_"
            + str(chunk_size)
            + ".pickle"
        )
        print(" Time ", finish_time - start_time, " Interval ", interval)

        with open(state_filename, "wb") as file:
            file.write(pickle.dumps(vstate.parameters))

        with open(text_data_filename, "a") as file:
            data = json.load(open(json_data_filename + ".log"))
            for key, value in data.items():
                file.write("%s\n" % (data[key]))
