import argparse
import datetime
from deepsnap.batch import Batch as deepsnap_Batch
from deepsnap.dataset import GraphDataset
from deepsnap.graph import Graph
from deepsnap.hetero_graph import HeteroGraph
import numpy as np
import pdb
import pickle
import scipy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from collections.abc import Mapping, Sequence

import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..', '..'))
from datasets.mppde1d_dataset import MPPDE1D
from datasets.arcsimmesh_dataset import ArcsimMesh
from datasets.tetramesh_dataset import TetraMesh
from pytorch_net.util import ddeepcopy as deepcopy, Batch, make_dir, Attr_Dict, My_Tuple
from utils import p, PDE_PATH, get_elements, is_diagnose, get_keys_values, loss_op, to_tuple_shape, parse_string_idx_to_list, parse_multi_step, get_device, get_activation, get_normalization, to_cpu, add_data_noise


def load_data(args, **kwargs):
    """Load data."""

    def get_train_val_idx(num_train_val, chunk_size=200):
        train_idx = (np.arange(0, num_train_val, chunk_size)[:, None] + np.arange(int(chunk_size * 0.8))[None, :]).reshape(-1)
        train_idx = train_idx[train_idx < num_train_val]
        val_idx = np.sort(np.array(list(set(np.arange(num_train_val)) - set(train_idx))))
        return torch.LongTensor(train_idx), torch.LongTensor(val_idx)

    def get_train_val_idx_random(num_train_val, train_fraction):
        train_idx = np.random.choice(num_train_val, size=int(num_train_val * train_fraction), replace=False)
        val_idx = np.sort(np.array(list(set(np.arange(num_train_val)) - set(train_idx))))
        return torch.LongTensor(train_idx), torch.LongTensor(val_idx)

    train_val_fraction = 0.9
    train_fraction = args.train_fraction
    multi_step_dict = parse_multi_step(args.multi_step)
    max_pred_steps = max(list(multi_step_dict.keys()) + [1]) * args.temporal_bundle_steps
    filename_train_val = os.path.join(PDE_PATH, "deepsnap", "{}_train_val_in_{}_out_{}{}{}{}.p".format(
        args.dataset, args.input_steps * args.temporal_bundle_steps, max_pred_steps, 
        "_itv_{}".format(args.time_interval) if args.time_interval > 1 else "",
        "_yvar_{}".format(args.is_y_variable_length) if args.is_y_variable_length is True else "",
        "_noise_{}".format(args.data_noise_amp) if args.data_noise_amp > 0 else "",
    ))
    filename_test = os.path.join(PDE_PATH, "deepsnap", "{}_test_in_{}_out_{}{}{}{}.p".format(
        args.dataset, args.input_steps * args.temporal_bundle_steps, max_pred_steps, 
        "_itv_{}".format(args.time_interval) if args.time_interval > 1 else "",
        "_yvar_{}".format(args.is_y_variable_length) if args.is_y_variable_length is True else "",
        "_noise_{}".format(args.data_noise_amp) if args.data_noise_amp > 0 else "",
    ))
    make_dir(filename_train_val)
    p.print("{} does not exist. Generating...".format(filename_test))
    is_save = True  # If True, will save generated deepsnap dataset.

    if args.dataset.startswith("mppde1d"):
        if not args.is_test_only:
            pyg_dataset_train = MPPDE1D(
                dataset=args.dataset,
                input_steps=args.input_steps * args.temporal_bundle_steps,
                output_steps=max_pred_steps,
                time_interval=args.time_interval,
                is_y_diff=args.is_y_diff,
                split="train",
                args = args
            )
            pyg_dataset_val = MPPDE1D(
                dataset=args.dataset,
                input_steps=args.input_steps * args.temporal_bundle_steps,
                output_steps=max_pred_steps,
                time_interval=args.time_interval,
                is_y_diff=args.is_y_diff,
                split="valid",
                args = args
            )
        pyg_dataset_test = MPPDE1D(
            dataset=args.dataset,
            input_steps=args.input_steps * args.temporal_bundle_steps,
            output_steps=max_pred_steps,
            time_interval=args.time_interval,
            is_y_diff=args.is_y_diff,
            split="test",
            args = args
        )
        is_to_deepsnap = True
    elif args.dataset.startswith("arcsimmesh"):
        if args.algo.startswith("rlgnnremesher"):
            max_pred_steps = args.rl_horizon
        if not args.is_test_only:
            pyg_dataset_train_val = ArcsimMesh(
                dataset=args.dataset,
                input_steps=args.input_steps * args.temporal_bundle_steps,
                output_steps=max_pred_steps,
                time_interval=args.time_interval,
                use_fineres_data=args.use_fineres_data,
                is_train=True,
                stress = args.stress
                #is_y_diff=args.is_y_diff,
                #split="train",
            )
        pyg_dataset_test = ArcsimMesh(
            dataset=args.dataset,
            input_steps=args.input_steps * args.temporal_bundle_steps,
            output_steps=max_pred_steps,
            time_interval=args.time_interval,
            is_shifted_data=args.is_shifted_data,
            use_fineres_data=args.use_fineres_data,
            is_train=False,
            stress = args.stress
            #is_y_diff=args.is_y_diff,
            #split="test",
        )
        is_to_deepsnap = False
    elif args.dataset.startswith("tetramesh"):
        if args.algo.startswith("rlgnnremesher"):
            max_pred_steps = args.rl_horizon
        if not args.is_test_only:
            pyg_dataset_train = TetraMesh(
                dataset=args.dataset,
                input_steps=args.input_steps * args.temporal_bundle_steps,
                output_steps=max_pred_steps,
                time_interval=args.time_interval,
                use_fineres_data=args.use_fineres_data,
                stress = args.stress,
                split="train",
                args = args
            )
            pyg_dataset_val = TetraMesh(
                dataset=args.dataset,
                input_steps=args.input_steps * args.temporal_bundle_steps,
                output_steps=max_pred_steps,
                time_interval=args.time_interval,
                use_fineres_data=args.use_fineres_data,
                stress = args.stress,
                split="valid",
                args = args
            )
        pyg_dataset_test = TetraMesh(
            dataset=args.dataset,
            input_steps=args.input_steps * args.temporal_bundle_steps,
            output_steps=max_pred_steps,
            time_interval=args.time_interval,
            use_fineres_data=args.use_fineres_data,
            stress = args.stress,
            split="test",
            args = args
        )
        is_to_deepsnap = False
    else:
        raise
   
    if args.n_train != "-1":
        # Test overfitting:
        is_save = False
        if not args.is_test_only:
            if "pyg_dataset_train_val" in locals():
                pyg_dataset_train_val = get_elements(pyg_dataset_train_val, args.n_train)
            else:
                pyg_dataset_train_val = get_elements(pyg_dataset_train, args.n_train)
        pyg_dataset_test = pyg_dataset_train_val
        p.print(", using the following elements {}.".format(args.n_train))
    else:
        p.print(":")

    # Transform :
    if  "pyg_dataset_train" in locals() and  "pyg_dataset_val" in locals():
        if not args.is_test_only:
            if "pyg_dataset_train_val" in locals():
                dataset_train_val = pyg_dataset_train_val
            else:
                dataset_train = pyg_dataset_train
                dataset_val = pyg_dataset_val
        dataset_test = pyg_dataset_test
    else:
        if not args.is_test_only:
            dataset_train_val = pyg_dataset_train_val
        dataset_test = pyg_dataset_test

    # Save pre-processed dataset into file:
    if is_save:
        if not args.is_test_only:
            if "pyg_dataset_train_val" in locals():
                if not os.path.isfile(filename_train_val):
                    pickle.dump(dataset_train_val, open(filename_train_val, "wb"))
            else:
                pickle.dump((dataset_train, dataset_val), open(filename_train_val, "wb"))
        try:
            pickle.dump(dataset_test, open(filename_test, "wb"))
            p.print("saved generated deepsnap dataset to {}".format(filename_test))
        except Exception as e:
            p.print(f"Cannot save dataset object. Reason: {e}")


    # Split into train, val and test:
    collate_fn = deepsnap_Batch.collate() if is_to_deepsnap else MeshBatch(is_absorb_batch=True, is_collate_tuple=True).collate()
    if not args.is_test_only:
        if args.n_train == "-1":
            if "dataset_train" in locals() and "dataset_val" in locals():
                dataset_train_val = (dataset_train, dataset_val)
            elif args.dataset_split_type == "standard":
                if args.dataset.startswith("VL") or args.dataset.startswith("PL") or args.dataset.startswith("PIL"):
                    train_idx, val_idx = get_train_val_idx(len(dataset_train_val), chunk_size=200)
                    dataset_train = dataset_train_val[train_idx]
                    dataset_val = dataset_train_val[val_idx]
                else:
                    num_train = int(len(dataset_train_val) * train_fraction)
                    dataset_train, dataset_val = dataset_train_val[:num_train], dataset_train_val[num_train:]
            elif args.dataset_split_type == "random":
                train_idx, val_idx = get_train_val_idx_random(len(dataset_train_val), train_fraction=train_fraction)
                dataset_train, dataset_val = dataset_train_val[train_idx], dataset_train_val[val_idx]
            elif args.dataset_split_type == "order":
                n_train = int(len(dataset_train_val) * train_fraction)
                dataset_train, dataset_val = dataset_train_val[:n_train], dataset_train_val[n_train:]
            else:
                raise Exception("dataset_split_type '{}' is not valid!".format(args.dataset_split_type))
        else:
            # train, val, test are all the same as designated by args.n_train:
            dataset_train =  dataset_train_val
            dataset_val =  dataset_train_val
        train_loader = DataLoader(dataset_train, num_workers=args.n_workers, collate_fn=collate_fn,
                                  batch_size=args.batch_size, shuffle=True if args.dataset_split_type!="order" else False, drop_last=True)
        val_loader = DataLoader(dataset_val, num_workers=args.n_workers, collate_fn=collate_fn,
                                batch_size=args.val_batch_size if not args.algo.startswith("supn") else 1, shuffle=False, drop_last=False)
    else:
        dataset_train_val = None
        train_loader, val_loader = None, None
    test_loader = DataLoader(dataset_test, num_workers=args.n_workers, collate_fn=collate_fn,
                             batch_size=args.val_batch_size if not args.algo.startswith("supn") else 1, shuffle=False, drop_last=False)
    return (dataset_train_val, dataset_test), (train_loader, val_loader, test_loader)

class MeshBatch(object):
    def __init__(self, is_absorb_batch=False, is_collate_tuple=False):
        """

        Args:
            is_collate_tuple: if True, will collate inside the tuple.
        """
        self.is_absorb_batch = is_absorb_batch
        self.is_collate_tuple = is_collate_tuple

    def collate(self):
        import re
        # if torch.__version__.startswith("1.9") or torch.__version__.startswith("1.10") or torch.__version__.startswith("1.11"):
        #     from torch._six import string_classes
        #     from collections import abc as container_abcs
        # else:
        #     from torch._six import container_abcs, string_classes, int_classes
            
        from pstar import pdict, plist
        default_collate_err_msg_format = (
            "collate_fn: batch must contain tensors, numpy arrays, numbers, "
            "dicts or lists; found {}")
        np_str_obj_array_pattern = re.compile(r'[SaUO]')
        def default_convert(data):
            r"""Converts each NumPy array data field into a tensor"""
            elem_type = type(data)
            if isinstance(data, torch.Tensor):
                return data
            elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                    and elem_type.__name__ != 'string_':
                # array of string classes and object
                if elem_type.__name__ == 'ndarray' \
                        and np_str_obj_array_pattern.search(data.dtype.str) is not None:
                    return data
                return torch.as_tensor(data)
            elif isinstance(data, Mapping):
                return {key: default_convert(data[key]) for key in data}
            elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
                return elem_type(*(default_convert(d) for d in data))
            elif isinstance(data, Sequence) and not isinstance(data, str):
                return [default_convert(d) for d in data]
            else:
                return data

        def collate_fn(batch):
            r"""Puts each data field into a tensor with outer dimension batch size, adapted from PyTorch's default_collate."""
            # pdb.set_trace()
            elem = batch[0]
            elem_type = type(elem)
            if isinstance(elem, torch.Tensor):
                # out = None
                # if torch.utils.data.get_worker_info() is not None:
                    # If we're in a background process, concatenate directly into a
                    # shared memory tensor to avoid an extra copy
                    # numel = sum([x.numel() for x in batch])
                    # storage = elem.storage()._new_shared(numel)
                    # out = elem.new(storage)
                tensor = torch.cat(batch, dim = 0)
                if self.is_absorb_batch:
                    # pdb.set_trace()
                    if tensor.shape[1] == 0:
                        tensor = tensor.view(tensor.shape[0], 0)
                    else:
                        tensor = tensor.view(-1, *tensor.shape[2:])
                return tensor
            elif elem is None:
                return None
            elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
                if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
                    # array of string classes and object
                    if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                        raise TypeError(default_collate_err_msg_format.format(elem.dtype))
                    return collate_fn([torch.as_tensor(b) for b in batch])
                elif elem.shape == ():  # scalars
                    return torch.as_tensor(batch)
            elif isinstance(elem, float):
                return torch.tensor(batch, dtype=torch.float64)
            elif isinstance(elem, int):
                return torch.tensor(batch)
            elif isinstance(elem, str):
                return batch
            elif isinstance(elem, Mapping):
                Dict = {}
                for key in elem:
                    if key == "node_feature":
                        Dict["vers"] = collate_trans_fn([d[key] for d in batch])
                        Dict[key] = collate_fn([d[key] for d in batch])
                        Dict["batch"] = {"n0": []}
                        batch_nodes = [d[key]["n0"] for d in batch]
                        for i in range(len(batch_nodes)):
                            item = torch.full((batch_nodes[i].shape[0],), i, dtype=torch.long)
                            Dict["batch"]["n0"].append(item)
                        Dict["batch"]["n0"] = torch.cat(Dict["batch"]["n0"])
                    elif key in ["y_tar","y_tar_stress", "reind_yfeatures",
                                 "reind_yfeatures_finegrain","reind_yfeatures_stress",
                                 "reind_yfeatures_finegrain_finegrain"]:
                        # pdb.set_trace()
                        Dict[key] = collate_ytar_trans_fn([d[key] for d in batch])
                    elif key in ["history","history_stress"]:
                        Dict[key] = collate_fn([d[key] for d in batch])
                        Dict["batch_history"] = {"n0": []}
                        batch_nodes = [d[key]["n0"] for d in batch]
                        for i in range(len(batch_nodes)):
                            item = torch.full((batch_nodes[i][0].shape[0],), i, dtype=torch.long)
                            Dict["batch_history"]["n0"].append(item)
                        Dict["batch_history"]["n0"] = torch.cat(Dict["batch_history"]["n0"])
                    elif key == "edge_index":
                        Dict[key] = collate_edgeidshift_fn([d[key] for d in batch])
                    elif key == "yedge_index":
                        # pdb.set_trace()
                        Dict[key] = collate_y_edgeidshift_fn([d[key] for d in batch])
                    elif key == "xfaces":
                        Dict[key] = collate_xfaceshift_fn([d[key] for d in batch])
                    elif key in ["bary_indices", "hist_indices"]:
                        #pdb.set_trace()
                        Dict[key] = collate_bary_indices_fn([d[key] for d in batch])
                    elif key in ["yface_list", "xface_list","yface_list_finegrain"]:
                         Dict[key] = collate_fn([d[key] for d in batch])
                    else:
                        Dict[key] = collate_fn([d[key] for d in batch])
                if isinstance(elem, pdict) or isinstance(elem, Attr_Dict):
                    Dict = elem.__class__(**Dict)
                return Dict
            elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple:
                return elem_type(*(collate_fn(samples) for samples in zip(*batch)))
            elif isinstance(elem, My_Tuple):
                it = iter(batch)
                elem_size = len(next(it))
                if not all(len(elem) == elem_size for elem in it):
                    raise RuntimeError('each element in list of batch should be of equal size')
                transposed = zip(*batch)
                return elem.__class__([collate_fn(samples) for samples in transposed])
            elif isinstance(elem, tuple):
                # pdb.set_trace()
                if self.is_collate_tuple:
                    #pdb.set_trace()
                    if len(elem) == 0:
                        return batch[0]
                    elif isinstance(elem[0], torch.Tensor):
                        newbatch = ()
                        for i in range(len(elem)):
                            newbatch = newbatch + tuple([torch.cat([tup[i] for tup in batch], dim=0)])
                        return newbatch
                    elif type(elem[0]) == list:
                        newbatch = ()
                        for i in range(len(elem)):
                            cumsum = 0
                            templist = []
                            for k in range(len(batch)):
                                shiftbatch = np.array(batch[k][i]) + cumsum
                                cumsum = shiftbatch.max() + 1
                                templist.extend(shiftbatch.tolist())
                            newbatch = newbatch + (templist,)
                    elif type(elem[0]).__module__ == np.__name__:
                        newbatch = ()
                        for i in range(len(elem)):
                            cumsum = 0
                            templist = []
                            for k in range(len(batch)):
                                shiftbatch = batch[k][i] + cumsum
                                cumsum = shiftbatch.max() + 1
                                templist.extend(shiftbatch.tolist())
                            newbatch = newbatch + (templist,)
                    else:
                        newbatch = batch[0]
                    return newbatch
                else:
                    return batch
            elif isinstance(elem, Sequence):
                # check to make sure that the elements in batch have consistent size
                it = iter(batch)
                elem_size = len(next(it))
                if not all(len(elem) == elem_size for elem in it):
                    raise RuntimeError('each element in list of batch should be of equal size')
                transposed = zip(*batch)
                return  [collate_fn(samples) for samples in transposed]
            elif elem.__class__.__name__ == 'Dictionary':
                return batch
            elif elem.__class__.__name__ == 'DGLHeteroGraph':
                import dgl
                return dgl.batch(batch)
            raise TypeError(default_collate_err_msg_format.format(elem_type))

        def collate_bary_indices_fn(batch):
            r"""Puts each data field into a tensor with outer dimension batch size, adapted from PyTorch's default_collate."""
            # pdb.set_trace()
            elem = batch[0]
            elem_type = type(elem)
            if isinstance(elem, Mapping):
                Dict = {key: collate_bary_indices_fn([d[key] for d in batch]) for key in elem}
                if isinstance(elem, pdict) or isinstance(elem, Attr_Dict):
                    Dict = elem.__class__(**Dict)
                return Dict
            elif isinstance(elem, tuple):
                # pdb.set_trace()
                if self.is_collate_tuple:
                    #pdb.set_trace()
                    if type(elem[0]).__module__ == np.__name__:
                        newbatch = ()
                        for i in range(len(elem)):
                            cumsum = 0
                            templist = []
                            for k in range(len(batch)):
                                shiftbatch = batch[k][i] + cumsum
                                cumsum = shiftbatch.max() + 1
                                templist.extend(shiftbatch)
                            newbatch = newbatch + (templist,)
                        return newbatch
                else:
                    return batch
            raise TypeError(default_collate_err_msg_format.format(elem_type))

        def collate_y_edgeidshift_fn(batch):
            r"""Puts each data field into a tensor with outer dimension batch size, adapted from PyTorch's default_collate."""
            # pdb.set_trace()
            elem = batch[0]
            elem_type = type(elem)
            if isinstance(elem, torch.Tensor):
                # out = None
                # if torch.utils.data.get_worker_info() is not None:
                    # If we're in a background process, concatenate directly into a
                    # shared memory tensor to avoid an extra copy
                    # numel = sum([x.numel() for x in batch])
                    # storage = elem.storage()._new_shared(numel)
                    # out = elem.new(storage)
                cumsum = 0
                newbatch = []
                for i in range(len(batch)):
                    shiftbatch = batch[i] + cumsum
                    newbatch.append(shiftbatch)
                    cumsum = shiftbatch.max().item() + 1
                tensor = torch.cat(newbatch, dim=1)
                return tensor
            elif isinstance(elem, Mapping):
                Dict = {key: collate_y_edgeidshift_fn([d[key] for d in batch]) for key in elem}
                if isinstance(elem, pdict) or isinstance(elem, Attr_Dict):
                    Dict = elem.__class__(**Dict)
                return Dict
            elif isinstance(elem, tuple):
                if self.is_collate_tuple:
                    if isinstance(elem[0], torch.Tensor):
                        newbatch = ()
                        for i in range(len(elem)):
                            cumsum = 0
                            tempbatch = []
                            for tup in batch:
                                shiftbatch = tup[i] + cumsum
                                tempbatch.append(shiftbatch)
                                cumsum = shiftbatch.max().item() + 1
                            newbatch = newbatch + tuple([torch.cat(tempbatch, dim=-1)])
                        return newbatch
                else:
                    return batch
            raise TypeError(default_collate_err_msg_format.format(elem_type))

        def collate_edgeidshift_fn(batch):
            r"""Puts each data field into a tensor with outer dimension batch size, adapted from PyTorch's default_collate."""
            # pdb.set_trace()
            elem = batch[0]
            elem_type = type(elem)
            if isinstance(elem, torch.Tensor):
                # out = None
                # if torch.utils.data.get_worker_info() is not None:
                    # If we're in a background process, concatenate directly into a
                    # shared memory tensor to avoid an extra copy
                    # numel = sum([x.numel() for x in batch])
                    # storage = elem.storage()._new_shared(numel)
                    # out = elem.new(storage)
                cumsum = 0
                newbatch = []
                for i in range(len(batch)):
                    shiftbatch = batch[i] + cumsum
                    newbatch.append(shiftbatch)
                    cumsum = shiftbatch.max().item() + 1
                tensor = torch.cat(newbatch, dim=1)
                return tensor
            elif isinstance(elem, Mapping):
                Dict = {key: collate_edgeidshift_fn([d[key] for d in batch]) for key in elem}
                if isinstance(elem, pdict) or isinstance(elem, Attr_Dict):
                    Dict = elem.__class__(**Dict)
                return Dict
            raise TypeError(default_collate_err_msg_format.format(elem_type))

        def collate_xfaceshift_fn(batch):
            r"""Puts each data field into a tensor with outer dimension batch size, adapted from PyTorch's default_collate."""
            # pdb.set_trace()
            elem = batch[0]
            elem_type = type(elem)
            if isinstance(elem, torch.Tensor):
                # out = None
                # if torch.utils.data.get_worker_info() is not None:
                    # If we're in a background process, concatenate directly into a
                    # shared memory tensor to avoid an extra copy
                    # numel = sum([x.numel() for x in batch])
                    # storage = elem.storage()._new_shared(numel)
                    # out = elem.new(storage)
                cumsum = 0
                newbatch = []
                for i in range(len(batch)):
                    shiftbatch = batch[i] + cumsum
                    newbatch.append(shiftbatch)
                    cumsum = shiftbatch.max().item() + 1
                tensor = torch.cat(newbatch, dim=-1)
                return tensor
            elif isinstance(elem, Mapping):
                Dict = {key: collate_xfaceshift_fn([d[key] for d in batch]) for key in elem}
                if isinstance(elem, pdict) or isinstance(elem, Attr_Dict):
                    Dict = elem.__class__(**Dict)
                return Dict
            raise TypeError(default_collate_err_msg_format.format(elem_type))

        def collate_trans_fn(batch):
            r"""Puts each data field into a tensor with outer dimension batch size, adapted from PyTorch's default_collate."""
            # pdb.set_trace()
            elem = batch[0]
            elem_type = type(elem)
            if isinstance(elem, torch.Tensor):
                # out = None
                # if torch.utils.data.get_worker_info() is not None:
                #     # If we're in a background process, concatenate directly into a
                #     # shared memory tensor to avoid an extra copy
                #     numel = sum([x.numel() for x in batch])
                #     storage = elem.untyped_storage()._new_shared(numel)                    
                #     out = elem.new(storage)
                batch = [batch[i] + 2*i for i in range(len(batch))]
                try:
                    tensor = torch.cat(batch, dim = 0)
                except:
                    pdb.set_trace()
                if self.is_absorb_batch:
                    # pdb.set_trace()
                    # if tensor.shape[1] == 0:
                    #     tensor = tensor.view(tensor.shape[0]*tensor.shape[1], 0)
                    # else:
                    tensor = tensor.view(-1, *tensor.shape[2:])
                return tensor
            elif isinstance(elem, Mapping):
                Dict = {key: collate_trans_fn([d[key] for d in batch]) for key in elem}
                if isinstance(elem, pdict) or isinstance(elem, Attr_Dict):
                    Dict = elem.__class__(**Dict)
                return Dict
            raise TypeError(default_collate_err_msg_format.format(elem_type))

        def collate_ytar_trans_fn(batch):
            r"""Puts each data field into a tensor with outer dimension batch size, adapted from PyTorch's default_collate."""
            # pdb.set_trace()
            elem = batch[0]
            elem_type = type(elem)
            if isinstance(elem, tuple):
                # pdb.set_trace()
                if self.is_collate_tuple:
                    #pdb.set_trace()
                    if len(elem) == 0:
                        return batch[0]
                    elif isinstance(elem[0], torch.Tensor):
                        newbatch = ()
                        for i in range(len(elem)):
                            templist = [batch[j][i] + 2*j for j in range(len(batch))]
                            newbatch = newbatch + (torch.cat(templist, dim=0),)
                        return newbatch
            elif isinstance(elem, Mapping):
                Dict = {key: collate_ytar_trans_fn([d[key] for d in batch]) for key in elem}
                if isinstance(elem, pdict) or isinstance(elem, Attr_Dict):
                    Dict = elem.__class__(**Dict)
                return Dict
            raise TypeError(default_collate_err_msg_format.format(elem_type))
        return collate_fn
