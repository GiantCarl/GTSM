#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import scipy.io
import numpy as np
import h5py
import pickle
import torch
from torch_geometric.data import Dataset, Data
import pdb
import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from utils import MPPDE1D_PATH, PDE_PATH
from pytorch_net.util import Attr_Dict, plot_matrices
from utils import get_root_dir, to_tuple_shape
from MP_Neural_PDE_Solvers.common.utils import HDF5Dataset
from MP_Neural_PDE_Solvers.equations.PDEs import *
from deepsnap.hetero_graph import HeteroGraph
from pytorch_net.util import ddeepcopy as deepcopy
from utils import add_data_noise

# In[ ]:
def get_data_infor(path, dataset_index):
    """ Generate numpy array of mesh data as 2d supspace in 3d space.
    Args:
        traj_id: index of trajectory folder.            
    Returns:
        `numpy.struct`.
    """
    file_name = os.path.join(path, dataset_index ,"infor.npz")
    data_struct = np.load(file_name)
    return data_struct

def get_data(path, dataset_index,traj_id):
    """ Generate numpy array of mesh data as 2d supspace in 3d space.
    Args:
        traj_id: index of trajectory folder.            
    Returns:
        `numpy.struct`.
    """
    file_name = os.path.join(path,dataset_index, "traj_{:06d}.npz".format(traj_id))
    data_struct = np.load(file_name)
    return data_struct

class MPPDE1D(Dataset):
    def __init__(
        self,
        dataset="mppde1d-E1-100",
        input_steps=1,
        output_steps=1,
        time_interval=1,
        is_y_diff=False,
        split="train",
        transform=None,
        pre_transform=None,
        verbose=False,
        is_return_super = False,
        uniform_sample: int=-1,
        args = None,
    ):
        assert dataset.startswith("mppde1d")
        self.dataset = dataset
        self.dirname = MPPDE1D_PATH
        self.root = PDE_PATH
        
        if len(dataset.split("-")) == 3:
            _, self.mode, self.nx = dataset.split("-")
            self.nt_total, self.nx_total = 250, 200
        else:
            assert len(dataset.split("-")) == 7
            _, self.mode, self.nx, _, self.nt_total, _, self.nx_total = dataset.split("-")
            self.nt_total, self.nx_total = int(self.nt_total), int(self.nx_total)
        self.nx = int(self.nx)
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.time_interval = time_interval
        self.is_y_diff = is_y_diff
        self.split = split
        assert self.split in ["train", "valid", "test"]
        self.verbose = verbose
        self.args = args

        self.t_cushion_input = self.input_steps * self.time_interval if self.input_steps * self.time_interval > 1 else 1
        self.t_cushion_output = self.output_steps * self.time_interval if self.output_steps * self.time_interval > 1 else 1

        self.original_shape = (self.nx,)
        self.dyn_dims = 1  # density
        
        self.pde=CE(device="cpu")
        if (self.nt_total, self.nx_total) == (250, 200):
            path = os.path.join(PDE_PATH, MPPDE1D_PATH) + f'{self.pde}_{self.split}_{self.mode}'
        else:
            path = os.path.join(PDE_PATH, MPPDE1D_PATH) + f'{self.pde}_{self.split}_{self.mode}_nt_{self.nt_total}_nx_{self.nx_total}'
        self.path = path
        print(f"Load dataset {path}")
        self.time_stamps = self.nt_total
        self.base_resolution=[self.nt_total, self.nx]
        self.super_resolution=[self.nt_total, self.nx_total]
        self.is_return_super = is_return_super
        self.uniform_sample = uniform_sample
        self.dataset_base = f'pde_{self.base_resolution[0]}-{self.base_resolution[1]}'
        self.dataset_super = f'pde_{self.super_resolution[0]}-{self.super_resolution[1]}'

        if self.base_resolution[1] not in [34, 25]:
            ratio_nt = get_data_infor(self.path,self.dataset_super)['grid_size'][0] / get_data_infor(self.path,self.dataset_base)['grid_size'][0]
            ratio_nx = get_data_infor(self.path,self.dataset_super)['grid_size'][1] / get_data_infor(self.path,self.dataset_base)['grid_size'][1]
        else:
            ratio_nt = 1.0
            if self.base_resolution[1] == 34:
                ratio_nx = int(200 / 33)
            elif self.base_resolution[1] == 25:
                ratio_nx = int(200 / 25)
            else:
                raise
        # assert (ratio_nt.is_integer())
        # assert (ratio_nx.is_integer())
        self.ratio_nt = int(ratio_nt)
        self.ratio_nx = int(ratio_nx)

        if self.base_resolution[1] not in [34, 25]:
            self.nt = get_data_infor(self.path,self.dataset_base)['nt']
            self.dt = get_data_infor(self.path,self.dataset_base)['dt']
            self.dx = get_data_infor(self.path,self.dataset_base)['dx']
            self.x = get_data_infor(self.path,self.dataset_base)['x']
            self.tmin = get_data_infor(self.path,self.dataset_base)['tmin']
            self.tmax = get_data_infor(self.path,self.dataset_base)['tmax']
        else:
            self.nt = 250
            self.dt = get_data_infor(self.path,'pde_250-50')['dt']
            if self.base_resolution[1] == 34:
                self.dx = 0.16 * 3
                self.x = get_data_infor(self.path,'pde_250-100')['x'][::3]
            elif self.base_resolution[1] == 25:
                self.dx = 0.16 * 4
                self.x = get_data_infor[self.path,'pde_250-100']['x'][::4]
            else:
                raise
            self.tmin = get_data_infor(self.path,'pde_250-50')['tmin']
            self.tmax = get_data_infor(self.path,'pde_250-50')['tmax']
        self.x_ori = get_data_infor(self.path,'pde_250-100')['x']
        
        self.n_simu = 2048 if self.split == 'train' else 128
        self.time_stamps_effective = (self.time_stamps - self.t_cushion_input - self.t_cushion_output + self.time_interval) // self.time_interval
        super(MPPDE1D, self).__init__(self.root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.dirname)

    @property
    def processed_file_names(self):
        return []

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _process(self):
        import warnings
        from typing import Any, List
        from torch_geometric.data.makedirs import makedirs
        def _repr(obj: Any) -> str:
            if obj is None:
                return 'None'
                return re.sub('(<.*?)\\s.*(>)', r'\1\2', obj.__repr__())

        def files_exist(files: List[str]) -> bool:
            # NOTE: We return `False` in case `files` is empty, leading to a
            # re-processing of files on every instantiation.
            return len(files) != 0 and all([os.path.exists(f) for f in files])

        f = os.path.join(self.processed_dir, 'pre_transform.pt')
        if os.path.exists(f) and torch.load(f) != _repr(self.pre_transform):
            warnings.warn(
                f"The `pre_transform` argument differs from the one used in "
                f"the pre-processed version of this dataset. If you want to "
                f"make use of another pre-processing technique, make sure to "
                f"sure to delete '{self.processed_dir}' first")

        f = os.path.join(self.processed_dir, 'pre_filter.pt')
        if os.path.exists(f) and torch.load(f) != _repr(self.pre_filter):
            warnings.warn(
                "The `pre_filter` argument differs from the one used in the "
                "pre-processed version of this dataset. If you want to make "
                "use of another pre-fitering technique, make sure to delete "
                "'{self.processed_dir}' first")

        if files_exist(self.processed_paths):  # pragma: no cover
            return

        makedirs(self.processed_dir)
        self.process()

        path = os.path.join(self.processed_dir, 'pre_transform.pt')
        if not os.path.isfile(path):
            torch.save(_repr(self.pre_transform), path)
        path = os.path.join(self.processed_dir, 'pre_filter.pt')
        if not os.path.isfile(path):
            torch.save(_repr(self.pre_filter), path)

    def get_edge_index(self):
        edge_index_filename = os.path.join(self.processed_dir, f"{self.dataset}_edge_index.p")
        mask_valid_filename = os.path.join(self.root, self.dirname, f"{self.dataset}_mask_index.p")
        if os.path.isfile(edge_index_filename) and os.path.isfile(mask_valid_filename):
            edge_index = pickle.load(open(edge_index_filename, "rb"))
            mask_valid = pickle.load(open(mask_valid_filename, "rb"))
            return edge_index.to('cpu'), mask_valid.to('cpu')
        mask_valid = torch.ones(self.original_shape).bool()
        #velo_invalid_ids = np.where(velo_invalid_mask.flatten())[0]
        rows, cols = (*self.original_shape, 1)
        cube = np.arange(rows * cols).reshape(rows, cols)
        edge_list = []
        for i in range(rows):
            for j in range(cols):
                if i + 1 < rows: #and cube[i, j] not in velo_invalid_ids and cube[i+1, j] not in velo_invalid_ids:
                    edge_list.append([cube[i, j], cube[i+1, j]])
                    edge_list.append([cube[i+1, j], cube[i, j]])
                if j + 1 < cols: #and cube[i, j]: #not in velo_invalid_ids and cube[i, j+1] not in velo_invalid_ids:
                    edge_list.append([cube[i, j], cube[i, j+1]])
                    edge_list.append([cube[i, j+1], cube[i, j]])
        edge_index = torch.LongTensor(edge_list).T
        pickle.dump(edge_index, open(edge_index_filename, "wb"))
        pickle.dump(mask_valid, open(mask_valid_filename, "wb"))
        return edge_index, mask_valid

    def process(self):
        pass
    
    def to_deepsnap(self,data):
        """Transform the PyG dataset into deepsnap format."""
        original_shape = to_tuple_shape(data.original_shape)
        #pdb.set_trace()
        Graph = HeteroGraph(
            edge_index={("n0", "0", "n0"): data.edge_index},
            node_feature={"n0": add_data_noise(data.x, self.args.data_noise_amp)},
            node_label={"n0": add_data_noise(data.y, self.args.data_noise_amp)},
            mask={"n0": data.mask},
            directed=True,
            original_shape=(("n0", original_shape),),
            dyn_dims=(("n0", to_tuple_shape(data.dyn_dims)),),
            compute_func=(("n0", to_tuple_shape(data.compute_func)),),
            param={"n0": data.param[None]} if hasattr(data, "param") else {"n0": torch.ones(1,1)},
            grid_keys=("n0",),
            part_keys=(),
            params=to_tuple_shape(data.params) if hasattr(data, "params") else (),
            )
        
        if hasattr(data, "x_pos"):
            x_pos = deepcopy(data.x_pos)
            Graph["node_pos"] = {"n0": x_pos}
        if hasattr(data, "dataset"):
            Graph["dataset"] = data.dataset
        if hasattr(data, "xfaces"):
            Graph["xfaces"] = {"n0": data.xfaces}
        if hasattr(data, "y_tar"):
            Graph["y_tar"] = {"n0": data.y_tar}
        if hasattr(data, "y_back"):
            Graph["y_back"] = {"n0": data.y_back}
        if hasattr(data, "yface_list"):
            Graph["yface_list"] = {"n0": data.yface_list}
        if hasattr(data, "yedge_index"):
            Graph["yedge_index"] = {"n0": data.yedge_index}
        if hasattr(data, "x_bdd"):
            Graph["x_bdd"] = {"n0": data.x_bdd}
        if hasattr(data, "mask_non_badpoints"):
            Graph["mask_non_badpoints"] = data.mask_non_badpoints
        if hasattr(data, "edge_attr"):
            Graph["edge_attr"] = {("n0", "0", "n0"): data.edge_attr}
            
        return Graph
    
    def len(self):
        return self.time_stamps_effective * self.n_simu

    def get(self, idx):
        # assert self.time_interval == 1
        sim_id, time_id = divmod(idx, self.time_stamps_effective)

        if(f'{self.pde}' == 'CE'):
            # Super resolution trajectories are downprojected via kernel which averages of neighboring cell values
            data_super_struct = get_data(self.path,self.dataset_super,sim_id)  
            u_super = data_super_struct['f_u'][::self.ratio_nt]
            u_super = np.reshape(u_super,(u_super.shape[0],-1,u_super.shape[-1]))
            left = u_super[..., -3:-1]
            right = u_super[..., 1:3]
            u_super_padded = torch.DoubleTensor(np.concatenate((left, u_super, right), -1))
            weights = torch.DoubleTensor([[[0.2]*5]])
            if self.uniform_sample == -1:
                u_super = F.conv1d(u_super_padded, weights, stride= self.ratio_nx ).squeeze().numpy()
            else:
                u_super = F.conv1d(u_super_padded, weights, stride=(1, 2)).squeeze().numpy()
            x_pos = self.x

            # pdb.set_trace()
            # Base resolution trajectories (numerical baseline) and equation specific parameters
            if self.uniform_sample != -1:
                data_base_struct = get_data(self.path,f'pde_{self.base_resolution[0]}-{100}',sim_id)
                u_base = data_base_struct['f_u'][...,::self.uniform_sample]
                data_traj = u_super[...,::self.uniform_sample]
            else:
                u_base = get_data(self.path,self.dataset_base,sim_id)['f_u']
                data_traj = u_super
            param = {}
            param['alpha'] = data_super_struct['alpha']
            param['beta'] = data_super_struct['beta']
            param['gamma'] = data_super_struct['gamma']            

        elif(f'{self.pde}' == 'WE'):
            # Super resolution trajectories are downprojected via kernel which averages of neighboring cell values
            # No padding is possible due to non-periodic boundary conditions
            weights = torch.tensor([[[[1./self.ratio_nx]*self.ratio_nx]]])
            u_super = self.data[self.dataset_super][idx][::self.ratio_nt][None, None, ...]
            u_super = F.conv1d(torch.tensor(u_super), weights, stride=(1, self.ratio_nx)).squeeze().numpy()

            # To match the downprojected trajectories, also coordinates need to be downprojected
            x_super = torch.tensor(self.data[self.dataset_super].attrs['x'][None, None, None, :])
            x = F.conv1d(x_super, weights, stride=(1, self.ratio_nx)).squeeze().numpy()

            # Base resolution trajectories (numerical baseline) and equation specific parameters
            u_base = self.data[self.dataset_base][idx]
            variables = {}
            variables['bc_left'] = self.data['bc_left'][idx]
            variables['bc_right'] = self.data['bc_right'][idx]
            variables['c'] = self.data['c'][idx]
            # return u_base, u_super, x, variables

        else:
            raise Exception("Wrong experiment")
        
        if self.verbose:
            print(f"sim_id: {sim_id}   time_id: {time_id}   input: ({time_id * self.time_interval + self.t_cushion_input -self.input_steps * self.time_interval}, {time_id * self.time_interval + self.t_cushion_input})  output: ({time_id * self.time_interval + self.t_cushion_input}, {time_id * self.time_interval + self.t_cushion_input + self.output_steps * self.time_interval})")

        x_dens = torch.FloatTensor(np.stack([data_traj[time_id * self.time_interval + self.t_cushion_input + j] for j in range(-self.input_steps * self.time_interval, 0, self. time_interval)], -1))
        y_dens = torch.FloatTensor(np.stack([data_traj[time_id * self.time_interval + self.t_cushion_input + j] for j in range(0, self.output_steps * self.time_interval, self.time_interval)], -1))
        edge_index, mask_valid = self.get_edge_index()
        param = torch.cat([torch.FloatTensor(ele) for key, ele in param.items()]).squeeze()
        x_bdd = torch.ones(x_dens.shape[0])
        x_bdd[0] = 0
        x_bdd[-1] = 0
        x_pos = torch.FloatTensor(x_pos)[...,None]
        for dim in range(len(self.original_shape)):
            x_pos[..., dim] /= self.original_shape[dim]

        data = Data(
            x=x_dens.reshape(-1, *x_dens.shape[-1:], 1).clone(),       # [number_nodes: 64 * 64, input_steps, 1]
            x_pos=x_pos,  # [number_nodes: 128 * 128, 2]
            x_bdd=x_bdd[...,None],
            xfaces=torch.tensor([]),
            y=y_dens.reshape(-1, *y_dens.shape[-1:], 1).clone(),       # [number_nodes: 64 * 64, input_steps, 1]
            edge_index=edge_index,
            mask=mask_valid,
            param=param,
            original_shape=self.original_shape,
            dyn_dims=self.dyn_dims,
            compute_func=(0, None),
            dataset=self.dataset,
        )
        # data = Attr_Dict(
        #     node_feature={"n0": x_dens.reshape(-1, *x_dens.shape[-1:], 1).clone()},
        #     node_label={"n0": y_dens.reshape(-1, *y_dens.shape[-1:], 1).clone()},
        #     node_pos={"n0": x_pos},
        #     x_bdd={"n0": x_bdd[...,None]},
        #     xfaces={"n0": torch.tensor([])},
        #     edge_index={("n0","0","n0"): edge_index},
        #     mask={"n0": mask_valid},
        #     param=((("n0", param),)),
        #     original_shape=((("n0", self.original_shape),)),
        #     dyn_dims=((("n0", self.dyn_dims),)),
        #     compute_func=((("n0", (0, None)),)),
        #     grid_keys=("n0",),
        #     part_keys=(),
        #     time_step=time_id,
        #     sim_id=sim_id,
        #     dataset=self.dataset,
        # )

        update_edge_attr_1d(data)
        data = self.to_deepsnap(data)
        return data


def update_edge_attr_1d(data):
    dataset_str = to_tuple_shape(data.dataset)
    if dataset_str.split("-")[0] != "mppde1d":
        if hasattr(data, "node_feature"):
            relative_mesh_pos = data.node_pos["n0"][data.edge_index[("n0","0","n0")][0]] - data.node_pos["n0"][data.edge_index[("n0","0","n0")][1]]
            relative_world_pos = data.node_feature["n0"][data.edge_index[("n0","0","n0")][0]] - data.node_feature["n0"][data.edge_index[("n0","0","n0")][1]]
            edge_attr = torch.cat([relative_mesh_pos,relative_world_pos.reshape(relative_world_pos.size(0),-1)],-1)
            if dataset_str.split("-")[0] in ["mppde1de"]:
                data.edge_attr = {("n0","0","n0"): edge_attr}
            elif dataset_str.split("-")[0] in ["mppde1df", "mppde1dg", "mppde1dh"]:
                data.edge_attr = {("n0","0","n0"): torch.cat([edge_attr, edge_attr.abs()], -1)}
            else:
                raise
            if dataset_str.split("-")[0] in ["mppde1dg"]:
                data.x_bdd = {"n0": 1 - data.x_bdd["n0"]}
        else:
            relative_mesh_pos = data.x_pos[data.edge_index[0].to(data.x_pos.device)] - data.x_pos[data.edge_index[1].to(data.x_pos.device)] 
            relative_world_pos = data.x[data.edge_index[0].to(data.x_pos.device)] - data.x[data.edge_index[1].to(data.x_pos.device)] 
            edge_attr = torch.cat([relative_mesh_pos,relative_world_pos.reshape(relative_world_pos.size(0),-1)],-1)           
            if dataset_str.split("-")[0] in ["mppde1de"]:
                data.edge_attr = relative_mesh_pos
            elif dataset_str.split("-")[0] in ["mppde1df", "mppde1dg", "mppde1dh"]:
                data.edge_attr = torch.cat([edge_attr, edge_attr.abs()], -1)
            else:
                raise
            if dataset_str.split("-")[0] in ["mppde1dg"]:
                data.x_bdd = 1 - data.x_bdd
    return data


def get_data_pred(state_preds, step, data, **kwargs):
    """Get a new mppde1d Data from the state_preds at step "step" (e.g. 0,1,2....).
    Here we assume that the mesh does not change.

    Args:
        state_preds: has shape of [n_nodes, n_steps, feature_size:temporal_bundle_steps]
        data: a Deepsnap Data object
        **kwargs: keys for which need to use the value instead of data's.
    """
    is_deepsnap = hasattr(data, "node_feature")
    is_list = isinstance(state_preds, list)
    if is_list:
        assert len(state_preds[0].shape) == 3
        state_pred = state_preds[step].reshape(state_preds[step].shape[0], -1, data.node_feature["n0"].shape[-1])
    else:
        assert isinstance(state_preds, torch.Tensor)
        state_pred = state_preds[...,step,:].reshape(state_preds.shape[0], state_preds.shape[-1], 1)
    data_pred = Attr_Dict(
        node_feature={"n0": state_pred},
        node_label={"n0": None},
        node_pos={"n0": kwargs["x_pos"] if "x_pos" in kwargs else data.node_pos["n0"] if is_deepsnap else data.x_pos},
        x_bdd={"n0": kwargs["x_bdd"] if "x_bdd" in kwargs else data.x_bdd["n0"] if is_deepsnap else data.x_bdd},
        xfaces={"n0": torch.tensor([])},
        edge_index={("n0","0","n0"): kwargs["edge_index"] if "edge_index" in kwargs else data.edge_index[("n0","0","n0")] if is_deepsnap else data.edge_index},
        mask={"n0": data.mask["n0"] if is_deepsnap else data.mask},
        param={"n0": data.param["n0"][:1] if is_deepsnap else data.param[:1]},
        original_shape=to_tuple_shape(data.original_shape),
        dyn_dims=to_tuple_shape(data.dyn_dims),
        compute_func=to_tuple_shape(data.compute_func),
        grid_keys=("n0",),
        part_keys=(),
        dataset=to_tuple_shape(data.dataset),
        batch=kwargs["batch"] if "batch" in kwargs else data.batch,
    )
    update_edge_attr_1d(data_pred)
    return data_pred


# In[ ]:


if __name__ == "__main__":
    dataset = MPPDE1D(
        dataset="mppde1dh-E2-100",
        input_steps=1,
        output_steps=1,
        time_interval=1,
        is_y_diff=False,
        split="valid",
        transform=None,
        pre_transform=None,
        verbose=False,
    )
    # dataset2 = MPPDE1D(
    #     dataset="mppde1de-E2-400-nt-1000-nx-400",
    #     input_steps=1,
    #     output_steps=1,
    #     time_interval=5,
    #     is_y_diff=False,
    #     split="valid",
    #     transform=None,
    #     pre_transform=None,
    #     verbose=True,
    # )
    import matplotlib.pylab as plt
    plt.plot(dataset[198].x.squeeze())

