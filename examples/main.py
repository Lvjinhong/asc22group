import time

from load_circuits import QuantumCircuit
from artensor import (
    AbstractTensorNetwork,
    ContractionTree,
    find_order,
    contraction_scheme,
    tensor_contraction,
    contraction_scheme_sparse,
    tensor_contraction_sparse
)
from copy import deepcopy
import numpy as np
import torch
import cirq
torch.backends.cuda.matmul.allow_tf32 = False

#计时区
aT=time.time()

n, m, seq, device, sc_target, seed = 30, 14, 'EFGH', 'cuda', 30, 0
qc = QuantumCircuit(n, m, seq=seq)
edges = []
for i in range(len(qc.neighbors)):
    for j in qc.neighbors[i]:
        if i < j:
            edges.append((i, j))
neighbors = list(qc.neighbors)
final_qubits = set(range(len(neighbors) - n, len(neighbors)))
tensor_bonds = {
    i: [edges.index((min(i, j), max(i, j))) for j in neighbors[i]]
    for i in range(len(neighbors)) if i not in final_qubits
}  # open tensor network without final state
bond_dims = {i: 2.0 for i in range(len(edges))}
order_slicing, slicing_bonds, ctree_new = find_order(
    tensor_bonds, bond_dims, seed,
    sc_target=sc_target, trials=5,
    iters=10, slicing_repeat=1, betas=np.linspace(3.0, 21.0, 61)
)
print('order_slicing =', order_slicing)
print('slicing_bonds =', slicing_bonds)

tensors = []
for x in range(len(qc.tensors)):
    if x not in final_qubits:
        tensors.append(qc.tensors[x].to(device))

scheme, bonds_final = contraction_scheme(ctree_new)

final_qubits = sorted(final_qubits)
permute_dims = [0] * len(final_qubits)
for x in range(len(bonds_final)):
    _, y = edges[bonds_final[x]]
    permute_dims[list(final_qubits).index(y)] = x
full_amps = tensor_contraction(
    deepcopy(tensors), scheme
).permute(permute_dims).reshape(-1).cpu()

slicing_edges_manually_select = [
    (44, 66), (46, 66), (69, 94), (81, 94),
    (88, 99), (94, 116), (111, 127), (112, 122)
]

tensors_slicing = deepcopy(tensors)
slicing_indices = {}
neighbors_copy = deepcopy(neighbors)
tensor_network = AbstractTensorNetwork(
    deepcopy(tensor_bonds),
    deepcopy(bond_dims))

while len(slicing_edges_manually_select):
    slicing_edge = slicing_edges_manually_select.pop(0)
    x, y = slicing_edge
    idx_x_y = neighbors_copy[x].index(y)
    idx_y_x = neighbors_copy[y].index(x)
    neighbors_copy[x].pop(idx_x_y)
    neighbors_copy[y].pop(idx_y_x)
    slicing_indices[(x, y)] = (idx_x_y, idx_y_x)
    tensors_slicing[x] = tensors_slicing[x].select(idx_x_y, 0)
    tensors_slicing[y] = tensors_slicing[y].select(idx_y_x, 0)

    tensor_network.slicing(edges.index(slicing_edge))
    ctree_appro = ContractionTree(deepcopy(tensor_network), order_slicing, 0)
    scheme, _ = contraction_scheme(ctree_appro)
    appro_amps = tensor_contraction(
        deepcopy(tensors_slicing), scheme
    ).permute(permute_dims).reshape(-1).cpu()
    fidelity = (
            (full_amps.conj() @ appro_amps.reshape(-1)).abs() /
            (full_amps.abs().square().sum().sqrt() * appro_amps.abs().square().sum().sqrt())
    ).square().item()

    print(
        'after slicing {} edges, fidelity now is {:.5f} (estimated value {})'.format(
            len(slicing_indices), fidelity, 1 / 2 ** (len(slicing_indices))
        )
    )


def read_samples(filename):
    import os
    if os.path.exists(filename):
        samples_data = []
        with open(filename, 'r') as f:
            l = f.readlines()
        f.close()
        for line in l:
            ll = line.split()
            samples_data.append((ll[0], float(ll[1]) + 1j * float(ll[2])))
        return samples_data
    else:
        raise ValueError("{} does not exist".format(filename))


data = read_samples('amplitudes_n30_m14_s0_e0_pEFGH_10000.txt')
max_bitstrings = 1_000
bitstrings = [data[i][0] for i in range(max_bitstrings)]
amplitude_google = np.array([data[i][1] for i in range(max_bitstrings)])

tensors_sparsestate = []
for i in range(len(qc.tensors)):
    if i in final_qubits:
        tensors_sparsestate.append(
            torch.tensor([[1, 0], [0, 1]], dtype=torch.complex64, device=device)
        )
    else:
        tensors_sparsestate.append(qc.tensors[i].to(device))

tensor_bonds = {
    i: [edges.index((min(i, j), max(i, j))) for j in neighbors[i]]
    for i in range(len(neighbors))
}  # now all tensors will be included

order_slicing, slicing_bonds, ctree_new = find_order(
    tensor_bonds, bond_dims, final_qubits, seed, max_bitstrings,
    sc_target=sc_target, trials=5, iters=10, slicing_repeat=1,
    betas=np.linspace(3.0, 21.0, 61)
)

scheme_sparsestate, _, bitstrings_sorted = contraction_scheme_sparse(
    ctree_new, bitstrings, sc_target=sc_target
)

slicing_edges = [edges[i] for i in slicing_bonds]
slicing_indices = {}.fromkeys(slicing_edges)
neighbors_copy = deepcopy(neighbors)
for i, j in slicing_edges:
    idxi_j = neighbors_copy[i].index(j)
    idxj_i = neighbors_copy[j].index(i)
    neighbors_copy[i].pop(idxi_j)
    neighbors_copy[j].pop(idxj_i)
    slicing_indices[(i, j)] = (idxi_j, idxj_i)

amplitude_sparsestate = torch.zeros(
    (len(bitstrings),), dtype=torch.complex64, device=device
)
for s in range(2 ** len(slicing_edges)):
    configs = list(map(int, np.binary_repr(s, len(slicing_edges))))
    sliced_tensors = tensors_sparsestate.copy()
    for i in range(len(slicing_edges)):
        m, n = slicing_edges[i]
        idxm_n, idxn_m = slicing_indices[(m, n)]
        sliced_tensors[m] = sliced_tensors[m].select(idxm_n, configs[i]).clone()
        sliced_tensors[n] = sliced_tensors[n].select(idxn_m, configs[i]).clone()
    amplitude_sparsestate += tensor_contraction_sparse(
        sliced_tensors, scheme_sparsestate
    )

correct_num = 0
for i in range(len(bitstrings_sorted)):
    ind_google = bitstrings.index(bitstrings_sorted[i])
    relative_error = abs(
        amplitude_sparsestate[i].item() - amplitude_google[ind_google]
    ) / abs(amplitude_google[ind_google])
    if relative_error <= 0.05:
        correct_num += 1
print(f'bitstring amplitude correct ratio:{correct_num / max_bitstrings}')

bT=time.time()

print('一共用时： '+str(bT-aT))