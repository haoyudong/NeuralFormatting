import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.fetch_B = (opt.phase == 'test' and opt.show_ground_truth)

        ### input A (label maps)
        dir_A = '_A' if self.opt.label_nc == 0 else '_label'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        ### input B (real images)
        if opt.isTrain or self.fetch_B:
            dir_B = '_B' if self.opt.label_nc == 0 else '_img'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)  
            self.B_paths = sorted(make_dataset(self.dir_B))

        ### instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### load precomputed instance-wise encoded features
        if opt.load_features:                              
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))

        self.dataset_size = len(self.A_paths) 


    def __getitem__(self, index):
        def data_reshape(img_path, dim, mean=0.5, std=0.5):
            array = np.array(Image.open(img_path).convert('RGB'))
            (h, nw, k) = array.shape
            w = self.opt.image_width
            n = nw // w
            reshaped = array.reshape(h, n, w, k).transpose(0, 2, 1, 3).reshape(h, w, n * k)[:, :, :dim]

            transform_list = [transforms.ToTensor(), transforms.Normalize(tuple([mean] * dim), tuple([std] * dim))]
            transform = transforms.Compose(transform_list)
            tensor = transform(reshaped)
            return tensor

        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        A_tensor = data_reshape(A_path, self.opt.input_nc)
        B_tensor = data_reshape(B_path, self.opt.output_nc)

        f_tensor = 0
        if self.opt.load_features:
            f_path = self.feat_paths[index]
            f_tensor = data_reshape(f_path, self.opt.feat_num, mean=0, std=0.05)

        input_dict = {'label': A_tensor, 'image': B_tensor, 'inst': 0, 'feat': f_tensor, 'path': A_path}
        return input_dict


    """
    def __getitem__(self, index):        
        ### input A (label maps)
        A_path = self.A_paths[index]              
        A = Image.open(A_path)        
        params = get_params(self.opt, A.size)
        if self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params)
            A_tensor = transform_A(A.convert('RGB'))
        else:
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            A_tensor = transform_A(A) * 255.0

        B_tensor = inst_tensor = feat_tensor = 0
        ### input B (real images)
        if self.opt.isTrain or self.fetch_B:
            B_path = self.B_paths[index]   
            B = Image.open(B_path).convert('RGB')
            transform_B = get_transform(self.opt, params)      
            B_tensor = transform_B(B)

        ### if using instance maps        
        if not self.opt.no_instance:
            inst_path = self.inst_paths[index]
            inst = Image.open(inst_path)
            inst_tensor = transform_A(inst)

            if self.opt.load_features:
                feat_path = self.feat_paths[index]            
                feat = Image.open(feat_path).convert('RGB')
                norm = normalize()
                feat_tensor = norm(transform_A(feat))                            

        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 
                      'feat': feat_tensor, 'path': A_path}

        return input_dict
    """

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'