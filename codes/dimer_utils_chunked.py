from copy import deepcopy

import jax.numpy as jnp
import matplotlib.pyplot as plt
import netket as nk
import numpy as np
from netket.hilbert.homogeneous import HomogeneousHilbert
from netket.operator._abstract_operator import AbstractOperator
from netket.sampler.metropolis_numpy import *
from netket.sampler.rules.exchange import *
from netket.sampler.rules.local_numpy import *
from netket.utils import StaticRange
from netket.utils.dispatch import dispatch
from netket.utils.types import DType
from netket.vqs.mc.kernels import *


def basis(dir):
    """
    A function that returns a specific direction based on the input parameter dir.
    Parameters:
        dir (int): An integer representing a specific direction.
    Returns:
        tuple: A tuple representing a  vector corresponding to the input direction.
    """
    if dir == 1:
        return (0, -1)
    elif dir == 2:
        return (0, 1)
    elif dir == 3:
        return (1, 0)
    elif dir == 4:
        return (-1, 0)
    else:
        raise ("ValueError: Direction must be an integer between 1 and 4")


############################################################################


def coordinate(i, L):
    """
    A function that returns the coordinates (x, y) based on the input linear index i.
    Returns:
        tuple: A tuple representing the coordinates (x, y) derived from the input index i.
    """
    return (i % L, i // L)


############################################################################


def getIndex(x, y, L):
    """
    A function that returns the linear index i based on the input coordinates (x, y).
    """
    return y * L + x


############################################################################


def columnarState(L):
    """
    Generate a columnar state (See below) for a square lattice with side length L .

    o o o o o o o o o o o
    | | | | | | | | | | |
    o o o o o o o o o o o

    o o o o o o o o o o o
    | | | | | | | | | | |
    o o o o o o o o o o o


    Parameters:
        L (int): The side length of the square lattice.

    Returns:
        np.ndarray: A 2D array representing the columnar state of the lattice.
    """
    state = np.zeros((L, L), dtype=int)
    for i in range(L**2):
        x, y = coordinate(i, L)
        if x % 2 == 0:
            state[x, y] = 4
        else:
            state[x, y] = 3

    return np.ravel(state, order="F")


############################################################################


def get_random_state(L, key, σ=None):
    """
    Generates a random state for a system with side length L starting with a columnar state (vertical).

    Parameters:
    L (int): The side length of the system.
    key (int): The random seed.

    Returns:
    np.ndarray: The random state as a flattened 1D array.
    """
    # columns left, right, straight, back
    # rows left, right, top, bottom
    # This array gives the orientation of the new dimer. It is a 2D array.
    directionArray = [
        [0, 0, 0, 0],
        [4, 3, 1, 2],
        [3, 4, 2, 1],
        [1, 2, 3, 4],
        [2, 1, 4, 3],
    ]
    oppositeDir = [0, 2, 1, 4, 3]

    rng = np.random.default_rng(key)

    initialSite = rng.integers(0, L * L)

    if σ is None:
        # print("σ is None")
        state = np.reshape(columnarState(L), (L, L), order="F")
    else:
        state = np.reshape(σ.astype(int), (L, L), order="F")
        # print(" Inside σ:", σ)
        # print("Shape of σ:", σ.shape)

    x, y = coordinate(initialSite, L)
    localstate = state[x, y]
    b = basis(localstate)
    monomer = {
        "site-behind": (x, y),
        "dir": localstate,
        "site-ahead": (((x + L + b[0]) % L, (y + L + b[1]) % L)),
        "parity": -1,
    }

    state[x, y] = -1

    while True:
        if monomer["parity"] == -1:
            newDir = directionArray[monomer["dir"]][rng.integers(0, 3)]
            a = monomer["site-ahead"]
            b = basis(newDir)
            monomer = {
                "site-behind": a,
                "dir": newDir,
                "site-ahead": (((a[0] + L + b[0]) % L, (a[1] + L + b[1]) % L)),
                "parity": 1,
            }
            temp = monomer["site-behind"]
            state[temp[0], temp[1]] = newDir

        elif monomer["parity"] == 1:
            stateAhead = state[monomer["site-ahead"]]
            if stateAhead != -1:
                oldDir = monomer["dir"]
                newDir = stateAhead
                a = monomer["site-ahead"]
                b = basis(newDir)
                monomer = {
                    "site-behind": a,
                    "dir": newDir,
                    "site-ahead": (((a[0] + L + b[0]) % L, (a[1] + L + b[1]) % L)),
                    "parity": -1,
                }
                temp = monomer["site-behind"]
                state[temp[0], temp[1]] = oppositeDir[oldDir]
            else:
                state[monomer["site-ahead"]] = oppositeDir[monomer["dir"]]
                break
        else:
            print("parity error")

    return np.ravel(state, order="F")


############################################################################
def detectFlippablePlaq(state, x, y):
    L = (np.sqrt(len(jnp.asarray(state)))).astype(int)
    state = jnp.reshape(state, (L, L), order="F")
    s1 = state[x, y]

    def check_horizontal():
        b1 = basis(4)
        s2 = state[(x + b1[0]) % L, (y + b1[1]) % L]
        return ((s2 == 2).astype(int), 0)

    def check_vertical():
        b1 = basis(2)
        s2 = state[(x + b1[0]) % L, (y + b1[1]) % L]
        return (0, (s2 == 4).astype(int))

    branches = [check_horizontal, check_vertical, lambda: (0, 0)]
    conditions = jnp.array([s1 == 2, s1 == 4, True])
    result = lax.switch(jnp.argmax(conditions), branches)
    return tuple(x for x in result)


def countNumOfFlippablePlaq(state):
    L = (jnp.sqrt(len(jnp.asarray(state)))).astype(int)
    data = [detectFlippablePlaq(state, x, y) for x in range(L) for y in range(L)]
    x_plaqts = [element[0] for element in data]
    y_plaqts = [element[1] for element in data]
    return sum(x_plaqts) + sum(y_plaqts)


from jax import lax

############################################################################


def drawState(state):
    L = int(jnp.sqrt(len(state)))
    state = jnp.reshape(state, (L, L), order="F")

    for x in range(L):
        for y in range(L):
            q = state[x, y]
            if q != -1:
                s = basis(q)
                plt.plot([y, y + s[1] / 2.0], [x, x + s[0] / 2.0], color="r")
                plt.plot(y, x, "o", color="black")
            else:
                plt.plot(y, x, "o", color="green")

            if detectFlippablePlaq(jnp.ravel(state, order="F"), x, y) == (1, 0):
                plt.plot(y + 0.5, x - 0.5, "s", color="blue")
            elif detectFlippablePlaq(jnp.ravel(state, order="F"), x, y) == (0, 1):
                plt.plot(y + 0.5, x - 0.5, "s", color="green")

        plt.axis("off")
        plt.axis("equal")
        plt.xlim([-1, L + 1])
        plt.ylim([-1, L + 1])


###########################################################################################################################################

# --------------------------SAMPLER-------------------------------------------------------------------------------------------------------

###########################################################################################################################################


class WormRule(MetropolisRule):
    r"""
    A Rule that generates a valid dimer configuration.
    """

    def __init__(
        self,
        *,
        graph= None,
    ):
        r"""
        Constructs the Worm Rule.

        You can pass either a list of clusters or a netket graph object to
        determine the clusters to exchange.

        Args:
            clusters: The list of clusters that can be exchanged. This should be
                a list of 2-tuples containing two integers. Every tuple is an edge,
                or cluster of sites to be exchanged.
            graph: A graph, from which the edges determine the clusters
                that can be exchanged.
            d_max: Only valid if a graph is passed in. The maximum distance
                between two sites
        """

    def transition(rule, sampler, machine, parameters, state, rng, σ):
        σ = state.σ  # sigma and sigma1 are from "state" class
        σ1 = state.σ1  # sigma is the current state of the sampler and sigma1 is the next proposed state
        n_chains = σ.shape[0]
        # print("-----------------------", n_chains)
        """
        #print("state.rgen",state.rng)
        #print("σ ",σ )
        #print("σ1 ",σ1) 
        """
        key_value = rng.integers(0, 1263456666, size=(n_chains))
        # print("key_value",key_value)

        _kernel(σ, σ1, key_value)

    def __repr__(self):
        return "WormRule"


def _kernel(
    σ, σ1, key_value
):  # gives an array of states (because we may run it n_chains (>1) number of times parallely) --> for each markov chain, we need an initial state and key for each M. chain
    n_chains = σ.shape[0]
    σ1[:] = (
        σ  # : means all the elements ex. sigma1[0] .... rthis is a deepcopy ----> alt.: sigma1 = copy.deepcopy(sigma)
    )
    L = int(np.sqrt(σ.shape[-1]))
    for i in range(n_chains):  # sigma1 has dimensions (n_chain, 16)
        # print(i,"-----σ ",σ,np.shape(σ) )
        σ1[i, :] = get_random_state(L, key_value[i], σ[i, :])

        # print("count= ",L)
        # print(i," key_rule ",key_rule[i])
        # print(i,"-----σ ",σ,np.shape(σ) )
        # print(i,"-----σ1 ",σ1,np.shape(σ1) )
        # σ1[:]=get_random_state(L, key_rule,σ)


############################################################################


def DimerMetropolisSampler(
    hilbert, *, clusters=None, graph=None, d_max=1, **kwargs
) -> MetropolisSamplerNumpy:
    rule = WormRule(graph=graph)
    return MetropolisSamplerNumpy(hilbert, rule, **kwargs)


###########################################################################################################################################

# --------------------------DIMER HILBERT SPACE--------------------------------------------------------------------------------------------

###########################################################################################################################################


class Dimer(HomogeneousHilbert):
    r"""Hilbert space obtained as tensor product of local dimer state o lattice nodes."""

    def __init__(
        self,
        N: int = 1,
    ):
        r"""Hilbert space obtained as tensor product of local dimer states.

        Args:

           N: Number of sites (default=1)


        Examples:
           Simple dimer hilbert space.

           >>> import netket as nk
           >>> hi = nk.hilbert.Dimer(s=1/2, N=4)
           >>> print(hi.size)
           4
        """
        local_size = 4
        local_states = StaticRange(1, 1, 4, dtype=np.int8)

        super().__init__(local_states, N)

    def __pow__(self, n):
        if not self.constrained:
            return Dimer(self.size * n)

        return NotImplemented

    def _mul_sametype_(self, other):
        assert type(self) == type(other)
        if self._s == other._s:
            if not self.constrained and not other.constrained:
                return Dimer(N=self.size + other.size)

        return NotImplemented

    def __repr__(self):
        return f"Dimer(N={self.size})"

    @property
    def _attrs(self):
        return self.size


############################################################################


@dispatch
def random_state(hilb: Dimer, key, batches: int, *, dtype):
    if not hilb.is_finite or hilb.constrained:
        raise NotImplementedError()
    L = int(jnp.sqrt(hilb.size))

    # Generate integer seeds using numpy RNG

    keys = np.asarray(jax.random.randint(key, (batches,), 0, np.iinfo(np.int64).max))
    #print(keys)
    random_state = []

    for i in range(len(keys)):
        if i != 0:
            random_state.append(get_random_state(L, keys[i], random_state[i - 1]))
        else:
            random_state.append(get_random_state(L, keys[i]))

    #print("dtype : ", dtype)
    return jnp.asarray(random_state, dtype=dtype)


###########################################################################################################################################

# --------------------------DIMER Hamiltonian --------------------------------------------------------------------------------------------

###########################################################################################################################################


class DimerHamiltonian(AbstractOperator):
    def __init__(
        self,
        hilbert: HomogeneousHilbert,
        V: float,
        t: float,
        dtype= float,
    ):
        super().__init__(hilbert)

        self._t = jnp.array(t, dtype=dtype)
        self._V = jnp.array(V, dtype=dtype)

    @property
    def t(self) -> float:
        """The magnitude of the hopping term"""
        return self._t

    @property
    def V(self) -> float:
        """The magnitude of the potential term"""
        return self._V

    @property
    def dtype(self):
        return float

    @property
    def is_hermitian(self):
        return True


def get_conn_elements(state):
    L_numpy = (np.sqrt(len(state))).astype(int)
    # print("\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\   "+ str(len(state)) )
    # state = jnp.reshape(state, (L, L), order="F")
    # num_flippable_plaq=countNumOfFlippablePlaq(state).astype(int)
    # print(" ////////////////////num_flippable_plaq ",num_flippable_plaq)
    # Preallocate for connected states & corresponding matrix elemets
    # print("--------------------------------",state,"   ----- ",type(state))
    list_connected_states = jnp.tile(state, (L_numpy * L_numpy, 1))
    list_returned_weights = jnp.zeros((L_numpy * L_numpy + 1))

    def update_state_horizontal(state_list_args):
        # L = (jnp.sqrt(len(state))).astype(int)
        state = state_list_args[0]
        x = state_list_args[1]
        y = state_list_args[2]
        L = L_numpy
        # print("Horizontal ------",x,"  ",y)
        state1 = deepcopy(state)
        state1 = jnp.array(jnp.reshape(state1, (L, L), order="F"))
        state1 = state1.at[x, y].set(4)
        state1 = state1.at[(x - 1) % L, y].set(3)
        state1 = state1.at[x, (y + 1) % L].set(4)
        state1 = state1.at[(x - 1) % L, (y + 1) % L].set(3)
        return (jnp.ravel(state1, order="F"), 1)

    def update_state_vertical(state_list_args):
        # L = (jnp.sqrt(len(state))).astype(int)
        state = state_list_args[0]
        x = state_list_args[1]
        y = state_list_args[2]
        L = L_numpy
        # print("Vertical ------",x,"  ",y)
        state1 = deepcopy(state)
        state1 = jnp.array(jnp.reshape(state1, (L, L), order="F"))
        state1 = state1.at[x, y].set(2)
        state1 = state1.at[(x - 1) % L, y].set(2)
        state1 = state1.at[x, (y + 1) % L].set(1)
        state1 = state1.at[(x - 1) % L, (y + 1) % L].set(1)
        return (jnp.ravel(state1, order="F"), 1)

    def no_update_state(state_list_args):
        state = state_list_args[0]
        x = state_list_args[1]
        y = state_list_args[2]
        state1 = deepcopy(state)

        state1 = jnp.ravel(state1, order="F")
        # list_scattered_states.append(jnp.ravel(state, order="F"))
        return (jnp.ravel(state1, order="F"), 0)

    count = 0
    for x in range(L_numpy):
        for y in range(L_numpy):
            condition = jnp.array(detectFlippablePlaq(state, x, y))

            conditions = jnp.array(
                [
                    jnp.array_equal(condition, jnp.array((1, 0))),
                    jnp.array_equal(condition, jnp.array((0, 1))),
                    jnp.array_equal(condition, jnp.array((0, 0))),
                ]
            )
            branches = [update_state_horizontal, update_state_vertical, no_update_state]
            count_old = count
            new_state, cond = lax.switch(
                jnp.argmax(conditions), branches, operand=[state, x, y]
            )
            count = lax.cond(cond == 1, lambda x: x + 1, lambda x: x, count)
            # print(count," ",cond)
            # First state is laways the input state with weight 0
            # Then numflippableplaquettes tates are conndcted states with weight 1
            # Rest are padded with input state with weight 0
            list_connected_states = list_connected_states.at[
                count * (count - count_old)
            ].set(new_state)
            list_returned_weights = list_returned_weights.at[
                count * (count - count_old)
            ].set(cond)

    return (list_connected_states, list_returned_weights)


@partial(jax.vmap, in_axes=(0, None, None))
def get_conns_and_mels(sigma, t, V):
    # this code only works if sigma is a single bitstring
    # sigma=np.array(sigma)
    # print(sigma.ndim)
    # print(sigma)
    # print(type(sigma))
    assert sigma.ndim == 1
    # print("-----",t,V)
    # print(op)
    # get number of spins
    N = sigma.shape[-1]
    beta, array_wt = get_conn_elements(sigma)
    beta = jnp.asarray(beta)

    # Get Number of connected elemets
    # This includes number of flippable plaquets and the state itself(first elements in beta is the state itself)
    num_conn_elements = len(array_wt)
    num_flippable_plaquettes = countNumOfFlippablePlaq(sigma)
    # print("num_conn_elements",num_conn_elements)
    # repeat eta num_conn_elements times
    eta = jnp.tile(sigma, (num_conn_elements, 1))

    for idx in range(num_conn_elements):
        eta = eta.at[idx].set(beta.at[idx].get())

    # Store the matrix elements
    array_wt = t * array_wt  # The first element is set to 0
    array_wt = array_wt.at[0].set(num_flippable_plaquettes * V)
    # print(num_flippable_plaquettes)
    # print(array_wt)

    return (eta, array_wt)


def e_loc(logpsi, pars, sigma, extra_args):
    eta, mels = extra_args
    # check that sigma has been reshaped to 2D, eta is 3D
    # sigma is (Nsamples, Nsites)
    assert sigma.ndim == 2
    # eta is (Nsamples, Nconnected, Nsites)
    assert eta.ndim == 3

    # let's write the local energy assuming a single sample, and vmap it
    @partial(jax.vmap, in_axes=(0, 0, 0))
    def _loc_vals(sigma, eta, mels):
        return jnp.sum(mels * jnp.exp(logpsi(pars, eta) - logpsi(pars, sigma)), axis=-1)

    return _loc_vals(sigma, eta, mels)


@nk.vqs.get_local_kernel.dispatch
def get_local_kernel(vstate: nk.vqs.MCState, op: DimerHamiltonian):
    print(
        "---------------------------------------- This is not chunked local kernel ##########################"
    )
    return e_loc


@dispatch
def get_local_kernel(vstate: nk.vqs.MCState, Ô: DimerHamiltonian, chunk_size: int):
#    print(
 #       "############################  This is chunked local kernel ##########################"
  #  )
    return local_value_kernel_chunked


@nk.vqs.get_local_kernel_arguments.dispatch
def get_local_kernel_arguments(vstate: nk.vqs.MCState, op: DimerHamiltonian):
    sigma = vstate.samples
    # get the connected elements. Reshape the samples because that code only works
    # if the input is a 2D matrix
    extra_args = get_conns_and_mels(sigma.reshape(-1, vstate.hilbert.size), op.t, op.V)
    return sigma, extra_args


@nk.vqs.expect.dispatch
def expect(vstate: nk.vqs.MCState, op: DimerHamiltonian, chunk_size: None):
    print(
        "---------------------------------------- This is not chunked expect ##########################"
    )

    return _expect(vstate._apply_fun, vstate.variables, vstate.samples, op.t, op.V)


@expect.dispatch
def expect_mcstate_operator_chunked(
    vstate: nk.vqs.MCState, Ô: DimerHamiltonian, chunk_size: int
):
    print(
        "############################ This is chunked expect ##########################"
    )
    σ, args = get_local_kernel_arguments(vstate, Ô)

    local_estimator_fun = get_local_kernel(vstate, Ô, chunk_size)

    return _expect_chunking(
        chunk_size,
        local_estimator_fun,
        vstate._apply_fun,
        vstate.sampler.machine_pow,
        vstate.parameters,
        vstate.model_state,
        σ,
        args,
    )


@partial(jax.jit, static_argnums=(0, 1, 2))
def _expect_chunking(
    chunk_size: int,
    local_value_kernel: Callable,
    model_apply_fun: Callable,
    machine_pow: int,
    parameters: PyTree,
    model_state: PyTree,
    σ: jnp.ndarray,
    args: PyTree,
):
    σ_shape = σ.shape

    if jnp.ndim(σ) != 2:
        σ = σ.reshape((-1, σ_shape[-1]))

    def logpsi(w, σ):
        return model_apply_fun({"params": w, **model_state}, σ)

    def log_pdf(w, σ):
        return machine_pow * model_apply_fun({"params": w, **model_state}, σ).real

    _, Ō_stats = nkjax.expect(
        log_pdf,
        partial(local_value_kernel, logpsi, chunk_size=chunk_size),
        parameters,
        σ,
        args,
        n_chains=σ_shape[0],
    )

    return Ō_stats


@partial(jax.jit, static_argnums=0)
def _expect(logpsi, variables, sigma, t_val, V_val):
    n_chains = sigma.shape[-2]
    N = sigma.shape[-1]
    # flatten all batches
    sigma = sigma.reshape(-1, N)

    eta, mels = get_conns_and_mels(sigma, t_val, V_val)

    E_loc = e_loc(logpsi, variables, sigma, [eta, mels])

    # reshape back into chains to compute statistical information
    E_loc = E_loc.reshape(-1, n_chains)

    # this function computes things like variance and convergence information.
    return nk.stats.statistics(E_loc)

