import torch
import numpy as np
from copy import deepcopy
from os.path import exists, dirname, abspath
from os import makedirs
import multiprocessing as mp
from math import ceil
import time


def run_bash_command(cmd: str):
    import subprocess

    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    if error:
        raise Exception(error)
    else:
        return output
    
def get_gpu_info():
    gpu_info = []
    bash_command = "nvidia-smi --query-gpu=index,name,uuid --format=csv"
    output = run_bash_command(bash_command).decode()
    lines = output.split("\n")
    lines.pop(0)
    for l in lines:
        tokens = l.split(", ")
        print(tokens)
        if len(tokens) >= 3:
            gpu_info.append({'id': tokens[0], 'name': tokens[1], 'uuid': tokens[2]})

    return gpu_info 

def tensor_contraction_sparse(tensors, contraction_scheme):
    '''
    contraction the tensor network according to contraction scheme

    :param tensors: numerical tensors of the tensor network
    :param contraction_scheme: 
        list of contraction step, defintion of entries in each step:
        step[0]: locations of tensors to be contracted
        step[1]: einsum equation of this tensor contraction
        step[2]: batch dimension of the contraction
        step[3]: optional, if the second tensor has batch dimension, 
            then here is the reshape sequence
        step[4]: optional, if the second tensor has batch dimension, 
            then here is the correct reshape sequence for validation

    :return tensors[i]: the final resulting amplitudes
    '''
    for step in contraction_scheme:
        i, j = step[0]
        batch_i, batch_j = step[2]
        if len(batch_i) > 1:
            tensors[i] = [tensors[i]]
            for k in range(len(batch_i)-1, -1, -1):
                if k != 0:
                    if step[3]:
                        tensors[i].insert(
                            1, 
                            torch.einsum(
                                step[1],
                                tensors[i][0][batch_i[k]], 
                                tensors[j][batch_j[k]], 
                            ).reshape(step[3])
                        )
                    else:
                        tensors[i].insert(
                            1, 
                            torch.einsum(
                                step[1],
                                tensors[i][0][batch_i[k]], 
                                tensors[j][batch_j[k]])
                        )
                else:
                    if step[3]:
                        tensors[i][0] = torch.einsum(
                            step[1],
                            tensors[i][0][batch_i[k]], 
                            tensors[j][batch_j[k]], 
                        ).reshape(step[3])
                    else:
                        tensors[i][0] = torch.einsum(
                            step[1],
                            tensors[i][0][batch_i[k]], 
                            tensors[j][batch_j[k]], 
                        )
            tensors[j] = []
            tensors[i] = torch.cat(tensors[i], dim=0)
        elif len(step) > 3 and len(batch_i) == len(batch_j) == 1:
            tensors[i] = tensors[i][batch_i[0]]
            tensors[j] = tensors[j][batch_j[0]]
            tensors[i] = torch.einsum(step[1], tensors[i], tensors[j])
        elif len(step) > 3:
            tensors[i] = torch.einsum(
                step[1],
                tensors[i],
                tensors[j],
            ).reshape(step[3])
            if len(batch_i) == 1:
                tensors[i] = tensors[i][batch_i[0]]
            tensors[j] = []
        else:
            tensors[i] = torch.einsum(step[1], tensors[i], tensors[j])
            tensors[j] = []

    return tensors[i]


def contraction_collective(
        tensors:list, scheme:list, slicing_indices:dict, 
        task_ids:list, task_num:int, device='cuda:0'
    ):
    """
    By default, there will be 2^16 sub-tasks since there are 16 slicing indices.
    For a single task, it will complete 2^12 sub-tasks, so task id will range from [0, 2^4)
    """
    if max(task_ids) >= task_num or min(task_ids) < 0:
        raise ValueError(
            "The task_id argument is too large, the range of it should be [0, 2^4)."
        )
    for task_id in task_ids:
        contraction_single_task(tensors, scheme, slicing_indices, task_id, device)


def contraction_single_task(
        tensors:list, scheme:list, slicing_indices:dict, 
        task_id:int, device='cuda:0'
    ):
    store_path = abspath(dirname(__file__)) + '/results/'
    if not exists(store_path):
        try:
            makedirs(store_path)
        except:
            pass
    file_path = store_path + f'partial_contraction_results_{task_id}.pt'
    time_path = store_path + f'result_time.txt'
    gpunum_path = store_path + f'result_gpunum.txt'
    if not exists(file_path):
        t0 = time.perf_counter()
        slicing_edges = list(slicing_indices.keys())
        tensors_gpu = [tensor.to(device) for tensor in tensors]
        for s in range(task_id * 2 ** 12, (task_id + 1) * 2 ** 12):
            configs = list(map(int, np.binary_repr(s, len(slicing_edges))))
            sliced_tensors = tensors_gpu.copy()
            for x in range(len(slicing_edges)):
                m, n = slicing_edges[x]
                idxm_n, idxn_m = slicing_indices[(m, n)]
                sliced_tensors[m] = sliced_tensors[m].select(idxm_n, configs[x]).clone()
                sliced_tensors[n] = sliced_tensors[n].select(idxn_m, configs[x]).clone()
            if s == task_id * 2 ** 12:
                collect_tensor = tensor_contraction_sparse(sliced_tensors, scheme)
            else:
                collect_tensor += tensor_contraction_sparse(sliced_tensors, scheme)
        t1 = time.perf_counter()
        torch.save(collect_tensor.cpu(), file_path)
        with open(time_path, 'a') as f:
            f.write(f'task id {task_id} running time: {t1-t0:.4f} seconds\n')
        with open(gpunum_path, 'a') as f:
            gpu_info = get_gpu_info()
            assert len(gpu_info) == 1
            uuid = gpu_info[0]['uuid']
            f.write(f'{uuid}\n')
        print(f'subtask {task_id} done, the partial result file has been written into results/partial_contraction_results_{task_id}.pt')
    else:
        print(f'subtask {task_id} has already been calculated, skip to another one.')


def collect_results(task_num):
    for task_id in range(task_num):
        file_path = abspath(dirname(__file__)) + f'/results/partial_contraction_results_{task_id}.pt'
        if task_id == 0:
            collect_result = torch.load(file_path)
        else:
            collect_result += torch.load(file_path)
    
    return collect_result


def write_result(bitstrings, results):
    amplitude_filename = abspath(dirname(__file__)) + f'/results/result_amplitudes.txt'
    xeb_filename = abspath(dirname(__file__)) + f'/results/result_xeb.txt'
    time_filename = abspath(dirname(__file__)) + f'/results/result_time.txt'
    gpunum_filename = abspath(dirname(__file__)) + f'/results/result_gpunum.txt'
    with open(amplitude_filename, 'w') as f:
        for bitstring, amplitude in zip(bitstrings, results):
            f.write(f'{bitstring} {np.real(amplitude)} {np.imag(amplitude)}j\n')
    with open(xeb_filename, 'w') as f:
        f.write(f'{results.abs().square().mean().item() * 2 ** 53 - 1:.4f}')
    with open(gpunum_filename, 'r') as f:
        lines = f.readlines()
    num_gpu = len(set(lines))
    with open(gpunum_filename, 'a') as f:
        f.write(f'{num_gpu} gpus are used.\n')
    with open(time_filename, 'r') as f:
        lines = f.readlines()
    time_all = sum([float(line.split()[5]) for line in lines])
    with open(time_filename, 'a') as f:
        f.write(f'overall running time: {time_all:.2f} seconds.\n')
        f.write(f'{num_gpu} gpus running time: {time_all/num_gpu:.2f} seconds.\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        "-taskid", type=int, default=0, 
        help="tensor network contraction id"
    )
    parser.add_argument(
        "-device", type=int, default=-1, 
        help="using which device, -1 for cpu, otherwise use cuda:device"
    )
    args = parser.parse_args()
    assert args.taskid >=0 and args.taskid <= 15
    assert args.device >= -1
    args.device = f'cuda:{args.device}' if args.device >= 0 else 'cpu'
    contraction_filename = abspath(dirname(__file__)) + '/contraction.pt'
    if not exists(contraction_filename):
        assert ValueError('No contraction data!')

    """
    There will be four objects in the contraction scheme:
        tensors: Numerical tensors in the tensor network
        scheme: Contraction scheme to guide the contraction of the tensor network
        slicing_indices: Indices to be sliced, the whole tensor network will be
            divided into 2**(num_slicing_indices) sub-pieces and the contraction of
            all of them returns the overall result. The indices is sliced to avoid
            large intermediate tensors during the contraction.
        bitstrings: bitstrings of interest, the contraction result will be amplitudes
            of these bitstrings
    """
    tensors, scheme, slicing_indices, bitstrings = torch.load(contraction_filename)
    task_num = 2 ** (len(slicing_indices) - 12) # each subroutine runs 2^12 sub-tasks

    contraction_single_task(tensors, scheme, slicing_indices, args.taskid, args.device)

    file_exist_flag = True
    for i in range(task_num):
        if not exists(abspath(dirname(__file__)) + f'/results/partial_contraction_results_{i}.pt'):
            file_exist_flag = False
    if file_exist_flag:
        print('collecting results, results will be written into results/result_*.txt')
        results = collect_results(task_num)
        write_result(bitstrings, results)