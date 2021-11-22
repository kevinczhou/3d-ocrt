import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import h5py
from scipy.ndimage import zoom


class paraOCRT:
    def __init__(self, recon_shape, RI_shape_scale, dxyz, hdf5_path, batch_size, xyz_offset=np.zeros(3), scale=1,
                 momentum=.5):
        # recon_shape: the pixel dimensions of the 3D reconstruction;
        # RI_shape_scale: number between 0 and 1 (or can be >1) specifying what fraction of the size the RI map is
        # relative to the recon;
        # dxyz: pixel size of recon;
        # batch_size is the stratified batch size, meaning every angle will have the same exact spatial positions;
        # scale: recon_shape * scale is the actual shape;

        self.recon_shape_base = recon_shape  # base shape is full shape, as you could use multi-resolution;
        self.RI_shape_scale = RI_shape_scale  # however, RI will always be the same shape, since it's a tf.Variable;
        self.dxyz = dxyz  # recon pixel size in um;
        self.xyz_offset = xyz_offset
        self.step = 1.083639771532828  # axial spacing between pixels in air in um in the OCT system;
        self.batch_size = batch_size
        self.scale = scale
        self.momentum = tf.Variable(momentum, dtype=tf.float32, name='momentum', trainable=False)  # make this a
        # variable so we can change it via .assign() after optimization;
        self.sig_proj = .42465  # for the intepolation kernel width;
        self.optimizer = tf.keras.optimizers.Adam
        self.reuse_batch = None  # how many times to reuse the same batch in a row; if None, don't reuse;
        self.hdf5_path = hdf5_path  # for batching from an hdf5 file;
        self.shuffle_size = None  # when creating a tf dataset, what's the shuffle buffer size?
        self.prefetch = 1  # how many batches to prefetch?
        self.stop_gradient_projection = True  # stop gradient on the recon after backprojection;
        self.shuffle_unit = None  # for shuffling the hdf5 dataset; probably should keep this None because it's slow;
        self.th_range = None  # which angle indices to use (None means use all); note that normally you would just pass
        # the subsetted dataset, but the issue is that th hdf5 generator contains all the data; therefore, index out th
        # probe and galvo scan data, but pass the hdf5 path with the appropriate th_range here;
        self.n_back = 1
        self.n_dome = 1.453  # RI of dome, if applicable; fused silica;
        self.optimize_RI = True  # whether to optimize RI distribution;
        self.use_intersection_loss = True  # still have to supply the regularization coefficient;
        self.use_first_reflection_RI_loss = False  # still have to supply the regularization coefficient;
        self.use_first_reflection_RI_stop_gradient = False  # an alternate method; if you want, you can use both;
        self.z_downsamp = None  # an integer specifying how many times to downsamp rayprop in z dimension;
        self.filter_delta = .01  # filter = sqrt(f^2+delta);
        self.correct_momentum_bias = False  # adam-like bias correction; should probably be used only when
        # self.momentum is very small (note that self.momentum corresponds to 1-momentum in the paper);
        self.correct_momentum_bias_in_loss = False  # similar to correct_momentum_bias, but don't alter how the
        # reconstruction is made, but rather alter how the loss is computed;

        # OCT volume dimensions:
        self.data_num_x = 400
        self.data_num_y = 400
        self.data_num_z = 2048
        self.z_start = None  # if you want to specify a sub-range of the A-scans to backproject;
        self.z_end = None

        self.ckpt = None  # for checkpointing models

    def generate_dataset(self):
        # batching is always done, and is done in a stratified manner

        dataset = tf.data.Dataset.from_generator(self.hdf5_generator, (tf.float32, tf.int32))
        assert self.shuffle_size == 1  # shuffling is handled by generator; dataset is larger than RAM;
        dataset = dataset.shuffle(self.shuffle_size)

        # batch goes before repeat to enforce clear boundary between epochs; otherwise, loading from hdf5 file could be
        # slowed down because the A-scans are not contiguous in memory (this would be worse if you're shuffling, which
        # is not handled here by tf.data, but by the generator);
        # https://www.tensorflow.org/guide/data#training_workflows
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.repeat(None)  # go forever; shuffle before repeat means every element of an epoch
        # is shown before moving to the next epoch (see above link);
        if self.reuse_batch is not None:
            dataset = dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensors((x, y)).repeat(self.reuse_batch))
        dataset = dataset.prefetch(self.prefetch)

        return dataset  # need to iterate over this

    def _generate_generator(self):
        # takes the hdf5 file and outputs a generator that yields a stratified batch of A-scans and the corresponding
        # indices;
        # hdf5 file has as many datasets as there are angles;
        # assumes the dataset has been preshuffled per angle, specified by shuffle indices;
        # th is a list of indices of different probe positions to use;

        with h5py.File(self.hdf5_path, 'r') as f:
            num_x = f.attrs['num_x']
            num_y = f.attrs['num_y']
            num_th = f.attrs['num_th']
            stack_shape = (num_th, num_x, num_y)
            flattened = f.attrs['flattened']
            assert flattened  # for now, since this function always assumes preshuffled stacks;
            global_Ascan_flat_inds = np.array(f['global_Ascan_flat_inds'])
            global_Ascan_flat_inds = [[i for i in j] for j in
                                      global_Ascan_flat_inds]  # convert to list, slightly faster looping;

        if self.th_range is None:
            th_list = range(num_th)
        else:
            th_list = list(self.th_range)  # lists loop faster than numpy arrays?

        xy_inds = range(num_x * num_y)  # if flattened, read in sequence (blocked shuffling happens inside generator);

        if flattened:
            if self.shuffle_unit is None:  # don't shuffle
                def generator():
                    with h5py.File(self.hdf5_path, 'r') as f:
                        for xy_i in xy_inds:
                            Ascans = [f[str(th_i)][xy_i] for th_i in th_list]
                            indices = [global_Ascan_flat_inds[th_i][xy_i] for th_i in th_list]
                            # yielding these indices^ doesn't add that much overhead;
                            yield Ascans, indices
            else:
                # shuffle unit must divide evenly into the number of A-scans per angle;
                assert (num_x * num_y) % self.shuffle_unit == 0

                def generator():
                    # block shuffling of flattened spatial inds:
                    # do this inside the generator so after each epoch, you get a new order:
                    xy_inds_shuffled = np.reshape(xy_inds, (-1, self.shuffle_unit))
                    xy_inds_shuffled = np.random.permutation(xy_inds_shuffled)
                    xy_inds_shuffled = xy_inds_shuffled.flatten()
                    with h5py.File(self.hdf5_path, 'r') as f:
                        for xy_i in xy_inds_shuffled:
                            Ascans = [f[str(th_i)][xy_i] for th_i in th_list]
                            indices = [global_Ascan_flat_inds[th_i][xy_i] for th_i in th_list]
                            # yielding these indices^ doesn't add that much overhead;
                            yield Ascans, indices
        else:
            raise Exception('generator not implemented for non-flattened datasets')

        self.hdf5_generator = generator

    def create_variables(self, nominal_probe_xy, nominal_galvo_xy, propagation_model='parabolic',
                         learning_rates=None, variable_initial_values=None, recon=None):
        # nominal_probe_xy: _ by 2 array of the programmed xy coordinate trajectory of the probe;
        # nominal_galvo_xy: _ by 2 array of the programmed xy scan amplitudes for each probe position;
        # recon and normalize are optional initial estimates;

        self.recon_shape = np.int32(np.array(self.recon_shape_base) * self.scale)
        self.propagation_model = propagation_model
        self.nominal_probe_xy = nominal_probe_xy
        self.nominal_galvo_xy = nominal_galvo_xy
        self.data_num_th = len(nominal_galvo_xy)  # number of volumes acquired;
        if self.th_range is None:
            # called 'gathered' because we will use tf.gather later;
            self.data_num_th_gathered = len(nominal_galvo_xy)
        else:
            self.data_num_th_gathered = len(self.th_range)  # number of volumes acquired;
        self.RI_shape = tf.cast(self.recon_shape * self.RI_shape_scale, dtype=tf.int32)
        self.dxyz /= self.scale
        self.recon_fov = self.dxyz * self.recon_shape

        # dictionaries of tf.Variables and their corresponding optimizers;
        self.train_var_dict = dict()
        self.optimizer_dict = dict()
        self.non_train_dict = dict()  # dict of variables that aren't trained (probably .assigned()'d; for checkpoints);
        self.tensors_to_track = dict()  # intermediate tensors to track; have a tf.function return the contents;

        def use_default_for_missing(input_dict, default_dict):
            # to be used directly below; allows for dictionaries in which not all keys are specified; if not specified,
            # then use default_dict's value;

            if input_dict is None:  # if nothing given, then use the default;
                return default_dict
            else:
                for key in default_dict:
                    if key in input_dict:
                        if input_dict[key] is None:  # if the key is present, but None is specified;
                            input_dict[key] = default_dict[key]
                        else:  # i.e., use the value given;
                            pass
                    else:  # if key is not even present;
                        input_dict[key] = default_dict[key]
                return input_dict

        # define variables to be optimized, based on a dictionary of variable names and learning rates/initial values;
        # these variables are always used; otherwise, add additional variables via the propagation_model:
        # (negative learning rates mean that later we will not update these variables)
        default_learning_rates = {'f_mirror': -1e-3, 'f_lens': 1e-3,
                                  'galvo_xy': 1e-3, 'galvo_normal': 1e-3, 'galvo_theta': 1e-3,
                                  'probe_dx': 1e-3, 'probe_dy': 1e-3, 'probe_z': 1e-3, 'probe_normal': 1e-3,
                                  'probe_theta': 1e-3, 'd_before_f': 1e-3, 'RI': -1e-3, 'Ascan_background': 1e-3}
        default_variable_initial_values = {'f_mirror': 12.5,  # focal length  of parabolic mirror in mm;
                                           'f_lens': 30,  # effective focal length of lens before the mirror in mm;
                                           'galvo_xy': .5 * np.ones(2, dtype=np.float32),  # scan ampitude at lens
                                           # principal plane in x in mm;
                                           'galvo_normal': np.array((1e-7, 1e-7, -1), dtype=np.float32),  # direction
                                           # of the center ray;
                                           'galvo_theta': 0,  # angle-vector representation of rotation;
                                           'probe_dx': 0,  # global x shift in the nominal probe trajectories;
                                           'probe_dy': 0,  # global y shift in the nominal probe trajectories;
                                           'probe_z': 25,  # distance between the lens and mirror foci (the origin);
                                           'probe_normal': np.array((1e-7, 1e-7, -1), dtype=np.float32),  # normal of
                                           # the probe translation plane, in case there's any relative tilt;
                                           'probe_theta': 0,  # angle-vector representation of probe translation plane;
                                           'd_before_f': 1,  # where the A-scans start relative to the focus;
                                           # positive number means before focus; in mm;
                                           'RI': self.n_back + np.zeros(self.RI_shape, dtype=np.float32),  # RI
                                           # distribution;
                                           'Ascan_background': np.zeros(self.data_num_z, dtype=np.float32),  # infer
                                           # background OCT noise;
                                           }

        # if there are additional variables, define here (modify the two dictionaries):
        if 'nonparametric' in propagation_model:
            # allow the final boundary conditions to vary arbitrarily:
            default_learning_rates = {**default_learning_rates, 'delta_r': 1e-3, 'delta_u': 1e-3, 'galvo_xy_per': 1e-3,
                                      'galvo_theta_in_plane_per': 1e-3, 'probe_dxyz_per': 1e-3, 'r_2nd_order': 1e-3,
                                      'u_2nd_order': 1e-3}
            default_variable_initial_values = {**default_variable_initial_values,
                                               'delta_r': np.zeros((self.data_num_th, 3), dtype=np.float32),  # position
                                               'delta_u': np.zeros((self.data_num_th, 3), dtype=np.float32),  # orient;
                                               'galvo_xy_per': np.ones((self.data_num_th, 2), dtype=np.float32),  # per
                                               # probe position amplitude;
                                               'galvo_theta_in_plane_per': np.zeros(self.data_num_th, dtype=np.float32),
                                               # per probe position in-plane angle;
                                               'probe_dxyz_per': np.zeros((self.data_num_th, 3), dtype=np.float32),
                                               # per probe position xyz shift;
                                               'r_2nd_order': np.zeros((self.data_num_th, 5, 3)),  # 2nd order
                                               # correction of A-scan position; 3 sets of 5 coefficients:
                                               # a0*x + a1*y + a2*x**2 + a3*y**2 + a4*x*y; (one for x,y,z); allowing
                                               # this for xy allows for nonlinearly spaced A-scans;
                                               'u_2nd_order': np.zeros((self.data_num_th, 5, 3)),  # likewise, for
                                               # angular fanning;
                                               }

        if 'higher_order_correction' in propagation_model:
            # try 3rd and 4th order correction on top of the 2nd order
            assert 'nonparametric' in propagation_model
            default_learning_rates = {**default_learning_rates,'r_higher_order': 1e-3, 'u_higher_order': 1e-3}
            default_variable_initial_values = {**default_variable_initial_values,
                                               'r_higher_order': np.zeros((self.data_num_th, 9, 3)),
                                               # 3rd and 4th order correction of A-scan position; 3 sets of 9 coeffs:
                                               # a0*x**3 + a1*y**3 + a2*x**2*y + a3*x*y**2 + ...
                                               # a4*x**4 + a5*y**4 + a6*x**3*y + a7*x**2*y**2 + a8*x*y**3;
                                               'u_higher_order': np.zeros((self.data_num_th, 9, 3)),  # likewise, for
                                               # angular fanning;
                                               }

        if 'dome' in propagation_model:
            # allow dome parameters to update:
            # by default, don't allow the radii to vary, and use the manufacturer nominal values:
            default_learning_rates = {**default_learning_rates, 'dome_outer_radius': -1e-3,
                                      'dome_inner_radius': -1e-3, 'dome_center': 1e-3}
            default_variable_initial_values = {**default_variable_initial_values,
                                               'dome_outer_radius': np.float32(8.128),  # in mm;
                                               'dome_inner_radius': np.float32(6.858),  # in mm;
                                               'dome_center': np.zeros(3, dtype=np.float32)  # parabolic center origin;
                                               }

        if self.correct_momentum_bias_in_loss:
            # the effective momentum will be smaller; by how much depends on the sparsity of the update, which we learn;
            default_learning_rates = {**default_learning_rates, 'effective_inverse_momentum': 1000}
            default_variable_initial_values = {**default_variable_initial_values,
                                               'effective_inverse_momentum': np.float32(1 / self.momentum)
                                               }

        learning_rates = use_default_for_missing(learning_rates, default_learning_rates)
        variable_initial_values = use_default_for_missing(variable_initial_values, default_variable_initial_values)

        if 'delta_r' in variable_initial_values:
            # for keeping the mean shift 0, if optimizing this;
            self.delta_r_initial = np.mean(variable_initial_values['delta_r'], axis=0)[None]

        # create variables and train ops based on the initial values:
        for key in learning_rates.keys():  # learning_rates and variable_initial_values should have the same keys;
            var = tf.Variable(variable_initial_values[key], dtype=tf.float32, name=key)
            opt = self.optimizer(learning_rate=learning_rates[key])
            self.train_var_dict[key] = var
            self.optimizer_dict[key] = opt

        # non-trainable variables (which are updated with .assign()):
        if recon is None:  # if none supplied, initialize with 0s;
            recon_previous = np.zeros(self.recon_shape)
        else:  # otherwise, upsample to the current shape (trilinear interpolation);
            recon_previous = zoom(recon, self.recon_shape / np.array(recon.shape), order=2)

        # initialize first recon and normalize tensors for momentum; use the scaled recon shape, not the base shape;
        recon_previous = tf.Variable(recon_previous, dtype=tf.float32, trainable=False)
        self.non_train_dict['recon_previous'] = recon_previous

        # coordinates of the IDEAL full boundary conditions (i.e., for all A-scans); to be indexed with batch_inds;
        # they will be transformed according to the tf.Variables describing misalignment;
        self.probe_xy_BC = self.nominal_probe_xy[self.th_range].squeeze()  # squeeze if self.th_range is None;
        self.probe_xy_BC = np.tile(self.probe_xy_BC[None, :, :], [self.batch_size, 1, 1]).reshape(-1, 2)
        # since the batching is stratified by angle, just need to tile the nominal_probe_xy;

        x = np.linspace(-1, 1, self.data_num_x)
        y = np.linspace(-1, 1, self.data_num_y)
        x, y = np.meshgrid(x, y, indexing='ij')
        self.galvo_xy_BC = np.stack([x.flatten(), y.flatten()], axis=1)
        self.galvo_xy_BC = np.float32(self.galvo_xy_BC[None, :, :] * self.nominal_galvo_xy[:, None, :])
        # shape: data_num_th, data_num_x*data_num_y, 2;
        self.galvo_xy_BC = self.galvo_xy_BC.reshape(-1, 2)  # flatten out all but last dim for batch_inds gathering;

        # NOTE 1 regarding probe_xy_BC and galvo_xy_BC: galvo_xy_BC is NOT indexed by th_range because th indices
        # generated by the generator are global indices, so all of galvo_xy_BC need to present when using tf.gather; on
        # the other hand, probe_xy_BC is not gathered using batch indices, but rather we just use tiling because it's a
        # stratified batching scheme;

        # NOTE 2: the dim order is typically spatial xy, theta (probe positions), 2/3 -- that is, the first dimension is
        # the batch dimension; HOWEVER, for things that use "batch_inds" (galvo_xy_BC, xy_BC), the first two dimensions
        # should be swapped, because when I saved the hdf5 files, the A-scans were packed with the theta dimension first

        if 'nonparametric' in propagation_model:
            # make another copy for 2nd order polynomial correction;
            self.xy_BC = np.stack([x.flatten(), y.flatten()], axis=1)
            self.xy_BC = np.tile(self.xy_BC[None, :, :], [self.data_num_th, 1, 1])  # shape: data_num_th, num_x*num_y,2;
            # note: don't tile with self.data_num_th_gathered, because xy_BC will be subject to tf.gather later, and
            # requires participation of all angles, regardless of angle downsampling;
            self.xy_BC = np.float32(self.xy_BC.reshape(-1, 2))

        # create a list of booleans to accompany self.train_var_list and self.optimizer_list to specify whether to train
        # those variables (as specified by the whether the user-specified learning rates are negative); doing this so
        # that autograph doesn't traverse all branches of the conditionals; if the user ever wants to turn off
        # optimization of a variable mid-optimization, then just do .assign(0) to the learning rate, such that the
        # update is still happening, but the change is 0;
        self.trainable_or_not = list()
        for var in self.train_var_dict:
            name = self.train_var_dict[var].name[:-2]
            flag = learning_rates[name] > 0
            self.trainable_or_not.append(flag)

        assert (self.correct_momentum_bias + self.correct_momentum_bias_in_loss) < 2  # only exactly one can be true;
        if self.correct_momentum_bias or self.correct_momentum_bias_in_loss:
            # for keeping track of iteration number; note that we can't necessarily use an optimizer's iteration counter
            # because we may update the reconstruction with momentum without doing a gradient update;
            assert recon is None  # the bias correction is only valid when recon = 0;
            self.iter = tf.Variable(0, dtype=tf.float32, trainable=False)

        # create hdf5 generator:
        self._generate_generator()

        # if available, load center Ascan global coordinates:
        with h5py.File(self.hdf5_path, 'r') as f:
            if 'center_Ascan_inds' in list(f.keys()) and self.use_intersection_loss:
                self.center_Ascan_inds = np.array(f['center_Ascan_inds'])

                # center_Ascan_inds, as stored, operates on non-scrambled-order A-scans; need to unshuffle;
                shuffle_inds = np.array(f['preshuffle_inds'])
                unshuffle_inds = np.argsort(shuffle_inds)  # want inverse indices;
                # unravel from 3D then ravel into 2D:
                th_list, x_list, y_list = np.unravel_index(self.center_Ascan_inds,
                                                           (self.data_num_th, self.data_num_x, self.data_num_y))
                xy_inds = np.ravel_multi_index((x_list, y_list), (self.data_num_x, self.data_num_y))
                xy_inds_unshuffled = unshuffle_inds[np.arange(len(unshuffle_inds)), xy_inds]
                # reravel so that center_Ascan_inds_ operates on scrambled coordinates:
                x_list_, y_list_ = np.unravel_index(xy_inds_unshuffled, (self.data_num_x, self.data_num_y))
                center_Ascan_inds_ = np.ravel_multi_index((th_list, x_list_, y_list_),
                                                          (self.data_num_th, self.data_num_x, self.data_num_y))
                self.center_Ascan_inds = center_Ascan_inds_  # replace;

                if self.th_range is not None:
                    self.center_Ascan_inds = self.center_Ascan_inds[self.th_range]
            else:
                self.center_Ascan_inds = None

        # if available, load first reflection index:
        with h5py.File(self.hdf5_path, 'r') as f:
            if 'first_reflection_index' in list(f.keys()):
                self.first_reflection_index = np.array(f['first_reflection_index']).flatten().astype(np.int32)
                # flatten for gathering with batch_inds (just like self.galvo_xy_BC above);
            else:
                self.first_reflection_index = None

        # define filter for FBP (ramp filter with correction at origin):
        self.FBP_filter = np.sqrt(np.fft.fftfreq(self.data_num_z, d=1) ** 2 + self.filter_delta).astype(np.complex64)

    def _gather_by_th_range(self, var, axis=None):
        # if th_range consists of fewer angles than the total saved in the hdf5 file, then use tf.gather on the
        # variables, which are always defined based on the original total length; we do this because we can load old
        # models that optimized with a different number of angles;
        if self.data_num_th == self.data_num_th_gathered:
            return var  # do nothing
        else:
            return tf.gather(var, self.th_range, axis)

    def _axis_angle_rotmat(self, axis, angle):
        # return 3D rotation matrix given axis and angle;
        # axis is of shape (3) and angle is a single number;

        axis_unit, _ = tf.linalg.normalize(axis)  # convert to unit vector
        cos = tf.cos(angle)
        sin = tf.sin(angle)
        ux = axis_unit[0]
        uy = axis_unit[1]
        uz = axis_unit[2]

        r00 = cos + ux ** 2 * (1 - cos)
        r01 = ux * uy * (1 - cos) - uz * sin
        r02 = ux * uz * (1 - cos) + uy * sin
        r10 = ux * uy * (1 - cos) + uz * sin
        r11 = cos + uy ** 2 * (1 - cos)
        r12 = uy * uz * (1 - cos) - ux * sin
        r20 = ux * uz * (1 - cos) - uy * sin
        r21 = uy * uz * (1 - cos) + ux * sin
        r22 = cos + uz ** 2 * (1 - cos)

        rotmat = tf.stack([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
        return rotmat

    def _generate_2D_rotmat(self, angle):
        # simple, 2D rotation; angle is a 1D vector;

        cos = tf.cos(angle)
        sin = tf.sin(angle)

        rotmat = tf.stack([[cos, -sin],
                           [sin, cos]])
        return rotmat  # shape: 2, 2, _;

    def _propagate_to_parabolic_focus(self, batch_inds, mode='all_rays'):
        # generate boundary conditions from parabolic mirror parameters; need position and direction;
        # then, propagate to just before the parabolic mirror's focus, and return the final position and direction,
        # which will serve as the boundary conditions for ray propagation through the sample;
        # mode can be 'all_rays' or 'central_ray', where the latter is for computing ray intersection loss;

        if mode == 'all_rays':
            batch_size = self.batch_size
        elif mode == 'central_ray':
            batch_size = 1
        else:
            raise Exception('invalid mode: ' + mode)

        batch_inds = tf.reshape(batch_inds, [-1])

        # unpack:
        f_mirror = self.train_var_dict['f_mirror']
        f_lens = self.train_var_dict['f_lens']
        galvo_xy = self.train_var_dict['galvo_xy']
        galvo_theta = self.train_var_dict['galvo_theta']
        galvo_normal = self.train_var_dict['galvo_normal']
        probe_dx = self.train_var_dict['probe_dx']
        probe_dy = self.train_var_dict['probe_dy']
        probe_z = self.train_var_dict['probe_z']
        probe_normal = self.train_var_dict['probe_normal']
        probe_theta = self.train_var_dict['probe_theta']
        d_before_f = self.train_var_dict['d_before_f']
        probe_xy = self.nominal_probe_xy

        # gather the relevant rays to propagate, corresponding to the batch:
        galvo_xy_batch = tf.gather(self.galvo_xy_BC, batch_inds)  # data_num_th * batch_size by 2;
        galvo_xy_batch = galvo_xy_batch * galvo_xy[None, :]  # scale by scan amplitude, converting to mm?;
        if 'nonparametric' in self.propagation_model:
            galvo_xy_batch = tf.reshape(galvo_xy_batch, (-1, self.data_num_th_gathered, 2))  # unflatten;

            # scale by scan amplitude:
            galvo_xy_per = self.train_var_dict['galvo_xy_per']
            galvo_xy_per = self._gather_by_th_range(galvo_xy_per, axis=0)
            galvo_xy_batch = galvo_xy_batch * galvo_xy_per[None, :, :]

            # rotate by in-plane angle:
            galvo_theta_in_plane_per = self.train_var_dict['galvo_theta_in_plane_per']
            galvo_theta_in_plane_per = self._gather_by_th_range(galvo_theta_in_plane_per, axis=0)
            rotmat2D = self._generate_2D_rotmat(galvo_theta_in_plane_per)
            galvo_xy_batch = tf.einsum('bac,cda->bad', galvo_xy_batch, rotmat2D)

            # reflatten:
            galvo_xy_batch = tf.reshape(galvo_xy_batch, (-1, 2))

        galvo_xyz_batch = tf.concat([galvo_xy_batch, tf.broadcast_to(
            -f_lens, (self.data_num_th_gathered * batch_size, 1))], axis=1)  # add the uniform z coordinate; this
        # resulting list of vectors should be normalized; data_num_th * batch_size by 3;

        if mode == 'all_rays':
            probe_xy_batch = self.probe_xy_BC  # don't need to gather, because we already tiled;
            # data_num_th*batch_size by 2;
        elif mode == 'central_ray':
            probe_xy_batch = self.nominal_probe_xy[self.th_range].squeeze()  # don't need the tiled version;
            # data_num_th by 2
        probe_xyz_batch = tf.concat([probe_xy_batch, tf.broadcast_to(probe_z,
                                                                     (self.data_num_th_gathered * batch_size, 1))],
                                    axis=1)
        # augment with probe_z, because rotation will be done wrt the absolute origin; data_num_th by 3;

        if 'nonparametric' in self.propagation_model:
            # change xyz position of probe:
            probe_dxyz_per = self.train_var_dict['probe_dxyz_per']
            probe_dxyz_per = self._gather_by_th_range(probe_dxyz_per, axis=0)
            probe_xyz_batch = tf.reshape(probe_xyz_batch,
                                         (-1, self.data_num_th_gathered, 3)) + probe_dxyz_per[None, :, :]
            probe_xyz_batch = tf.reshape(probe_xyz_batch, (-1, 3))

        # initial ray direction:

        # first, rotate according to galvo_normal/galvo_theta;
        # have to rotate rays; fortunately, they're all origin-centered at this stage, so it's the same as rotating
        # points;
        galvo_rotmat = self._axis_angle_rotmat(galvo_normal, galvo_theta)
        galvo_xyz_batch @= galvo_rotmat  # rotate vectors; (vectors start at origin,so just need to rotate tips);

        # next, account for probe translational plane tilt:
        # to do this, we rotate both ends of each ray:
        ray_start = probe_xyz_batch
        ray_end = ray_start + galvo_xyz_batch  # shape: data_num_th * batch_size, 3;
        probe_rotmat = self._axis_angle_rotmat(probe_normal, probe_theta)
        ray_start @= probe_rotmat
        ray_end @= probe_rotmat

        # final boundary conditions:
        ray_directions = ray_end - ray_start  # shape: data_num_th * batch_size, 3;
        ray_directions, _ = tf.linalg.normalize(ray_directions, axis=1)  # normalize to unit vectors;
        ray_positions = ray_start  # shape: data_num_th * batch_size, 3;
        ray_positions = ray_positions + tf.stack([probe_dx, probe_dy, 0])[None, :]

        x0 = ray_positions[:, 0]
        y0 = ray_positions[:, 1]
        z0 = ray_positions[:, 2]
        r0 = ray_positions  # pseudonym

        ux = ray_directions[:, 0]
        uy = ray_directions[:, 1]
        uz = ray_directions[:, 2]
        u0 = ray_directions  # pseudonym

        # propagate to mirror:
        # coefficients of quadratic equation a*d^2 + b*d + c, where d is the distance from r0 to the mirror
        a = -ux ** 2 - uy ** 2
        b = 4 * f_mirror * uz - 2 * ux * x0 - 2 * uy * y0
        c = 4 * f_mirror * z0 - x0 ** 2 - y0 ** 2 + 4 * f_mirror ** 2
        d = 2 * c / (-b + tf.sqrt(b ** 2 - 4 * a * c))

        r_at_mirror = r0 + d[:, None] * u0  # position at mirror

        # propagate to near focus:
        n = tf.concat([-r_at_mirror[:, :2] / 2 / f_mirror,
                       np.ones((self.data_num_th_gathered * batch_size, 1))], 1)  # surface normals
        n, _ = tf.linalg.normalize(n, axis=1)
        u1 = u0 - 2 * tf.reduce_sum(u0 * n, 1, keepdims=True) * n  # the sum is a dot product with broadcasting;
        d_remain = probe_z + 2 * f_mirror - d - d_before_f  # remaining distance to propagate

        r_near_focus = r_at_mirror + u1 * d_remain[:, None]  # position near focus

        # the next boundary conditions for ray prop through sample:
        r_before_sample = r_near_focus * 1000  # convert to um;
        u_before_sample = u1
        # shape: data_num_th * batch_size, 3;

        return r_before_sample, u_before_sample

    def _propagate_through_dome(self, r, u):
        # after running _propagate_to_parabolic_focus, refract through two boundaries of the dome;
        # r and u are of shape _ x 3;

        r /= 1000  # _propagate_to_parabolic_focus converted to um;

        assert 'dome' in self.propagation_model
        rc = self.train_var_dict['dome_center']
        r1 = self.train_var_dict['dome_outer_radius']
        r2 = self.train_var_dict['dome_inner_radius']
        d_before_f = self.train_var_dict['d_before_f']
        delta_r = r - rc[None, :]  # for convenience;

        # _propagate_to_parabolic_focus propagates to just before the focus, but that's okay; that just means the first
        # propagation distance will be negative, and the lost distance will be added back; this should also undo any
        # effects of d_before_f in _propagate_to_parabolic_focus;
        u_dot_dr = tf.einsum('ij,ij->i', u, delta_r)
        dist_to_dome1 = - u_dot_dr - tf.sqrt(r1 ** 2 - tf.norm(delta_r, axis=1) ** 2 + u_dot_dr ** 2)  # take neg sol;
        r_at_dome = r + u * dist_to_dome1[:, None]  # position at dome

        # refract using snell's law:
        n_hat = r_at_dome - rc[None, :]
        ri_ratio = 1 / self.n_dome
        _n_dot_u = - tf.einsum('ij,ij->i', n_hat, u)  # negative n dot u;
        u_at_dome = ri_ratio * u + (ri_ratio * _n_dot_u -
                                    tf.sqrt(1 - ri_ratio ** 2 * (1 - _n_dot_u ** 2)))[:, None] * n_hat

        # propagate to the next surface:
        delta_r = r_at_dome - rc
        u_dot_dr = tf.einsum('ij,ij->i', u_at_dome, delta_r)
        dist_to_dome2 = - u_dot_dr - tf.sqrt(r2 ** 2 - tf.norm(delta_r, axis=1) ** 2 + u_dot_dr ** 2)
        r_at_dome = r_at_dome + u_at_dome * dist_to_dome2[:, None]  # position at inner surface of dome;

        # refract again using snell's law:
        n_hat = r_at_dome - rc[None, :]
        ri_ratio = self.n_dome
        _n_dot_u = - tf.einsum('ij,ij->i', n_hat, u_at_dome)
        u_at_dome = ri_ratio * u_at_dome + (ri_ratio * _n_dot_u -
                                            tf.sqrt(1 - ri_ratio ** 2 * (1 - _n_dot_u ** 2)))[:, None] * n_hat

        # make sure each ray travels the same overall OPL; since the ray was originally close to the focus, let's do
        # a net OPL of 0, adjusted by d_before_f (reminder that the d_before_f in _propagate_to_parabolic_focus no
        # longer does anything);
        opl_remain = 0 - dist_to_dome1 - dist_to_dome2 * self.n_dome - d_before_f  # subtract out opl already
        # accumulated, and d_before_f;
        r_before_focus = r_at_dome + u_at_dome * opl_remain[:, None]
        return r_before_focus * 1000, u_at_dome

    def _adjust_boundary_conditions(self, r, u, batch_inds):
        # if using the nonparametric model, then you should apply this function to the output of
        # propagate_to_parabolic_focus;
        assert 'nonparametric' in self.propagation_model

        r = tf.reshape(r, [-1, self.data_num_th_gathered, 3])  # unflatten;
        u = tf.reshape(u, [-1, self.data_num_th_gathered, 3])  # unflatten;

        # subtract out mean from delta_r:
        delta_r = (self.train_var_dict['delta_r'] + self.delta_r_initial  # make sure the mean stays the same;
                   - tf.reduce_mean(self.train_var_dict['delta_r'], axis=0, keepdims=True))

        r_new = r + self._gather_by_th_range(delta_r, axis=0)[None, :, :]
        u_new = u + self._gather_by_th_range(self.train_var_dict['delta_u'], axis=0)[None, :, :]
        u_new, _ = tf.linalg.normalize(u_new, axis=2)  # renormalize to unit vector;

        # 2nd order correction:
        # adjust position (r):
        xy_batch = tf.gather(self.xy_BC, batch_inds)
        xy_batch = tf.reshape(xy_batch, (-1, self.data_num_th_gathered, 2))  # unflatten;
        coefs = self.train_var_dict['r_2nd_order']  # shape: data_num_th, 5, 3;
        coefs = self._gather_by_th_range(coefs, axis=0)
        x = xy_batch[:, :, 0:1]  # shape: _, data_num_th, 1;
        y = xy_batch[:, :, 1:]
        dr = (x * coefs[None, :, 0, :] + y * coefs[None, :, 1, :] + x ** 2 * coefs[None, :, 2, :]
              + y ** 2 * coefs[None, :, 3, :] + x * y * coefs[None, :, 4, :])
        # shape^: _, data_num_th, 3
        r_new = r_new + dr
        # adjust ray fans (u):
        coefs = self.train_var_dict['u_2nd_order']  # shape: data_num_th, 5, 3;
        coefs = self._gather_by_th_range(coefs, axis=0)
        du = (x * coefs[None, :, 0, :] + y * coefs[None, :, 1, :] + x ** 2 * coefs[None, :, 2, :]
              + y ** 2 * coefs[None, :, 3, :] + x * y * coefs[None, :, 4, :])
        u_new = u_new + du

        if 'higher_order_correction' in self.propagation_model:
            coefs = self.train_var_dict['r_higher_order']
            coefs = self._gather_by_th_range(coefs, axis=0)
            dr = (x ** 3 * coefs[None, :, 0, :] + y ** 3 * coefs[None, :, 1, :] + x ** 2 * y * coefs[None, :, 2, :]
                  + y ** 2 * x * coefs[None, :, 3, :] + x ** 4 * coefs[None, :, 4, :] + y ** 4 * coefs[None, :, 5, :]
                  + y ** 3 * x * coefs[None, :, 6, :] + x ** 2 * y ** 2 * coefs[None, :, 7, :]
                  + y * x ** 3 * coefs[None, :, 8, :]
                  )
            # shape^: _, data_num_th, 3
            r_new = r_new + dr

            coefs = self.train_var_dict['u_higher_order']
            coefs = self._gather_by_th_range(coefs, axis=0)
            du = (x ** 3 * coefs[None, :, 0, :] + y ** 3 * coefs[None, :, 1, :] + x ** 2 * y * coefs[None, :, 2, :]
                  + y ** 2 * x * coefs[None, :, 3, :] + x ** 4 * coefs[None, :, 4, :] + y ** 4 * coefs[None, :, 5, :]
                  + y ** 3 * x * coefs[None, :, 6, :] + x ** 2 * y ** 2 * coefs[None, :, 7, :]
                  + y * x ** 3 * coefs[None, :, 8, :]
                  )
            # shape^: _, data_num_th, 3
            u_new = u_new + du

        u_new, _ = tf.linalg.normalize(u_new, axis=2)  # renormalize to unit vector;
        r_new = tf.reshape(r_new, [-1, 3])
        u_new = tf.reshape(u_new, [-1, 3])

        return r_new, u_new

    def _gather_RI(self, R, xyz, recon_or_RI, round_or_floor='round'):
        # get RI or recon value at xyz coordinates;
        # recon_or_RI specifies what R is;
        # multichannel, if R is multiple channels;

        if round_or_floor == 'round':
            discretize = tf.round
        elif round_or_floor == 'floor':
            discretize = tf.floor
        elif round_or_floor == 'average_neighbors':
            raise Exception('average_neighbors not yet implemented')
        else:
            raise Exception('invalid round_or_floor')

        y, z, x = tf.split(xyz, 3, axis=-1)
        # convert to pixels:
        if recon_or_RI == 'recon':
            shape = self.recon_shape  # rescaled version; not self.recon_shape_base
        elif recon_or_RI == 'RI':
            shape = self.RI_shape
        else:
            raise Exception('invalid recon_or_RI')
        size_dims_float = tf.cast(shape, tf.float32)
        x_c = tf.cast(discretize(((x / self.recon_fov[0]) + .5) * size_dims_float[0]), dtype=tf.int32)
        y_c = tf.cast(discretize(((y / self.recon_fov[1]) + .5) * size_dims_float[1]), dtype=tf.int32)
        z_c = tf.cast(discretize(((z / self.recon_fov[2]) + .5) * size_dims_float[2]), dtype=tf.int32)
        xyz_c = tf.concat([x_c, y_c, z_c], axis=-1)

        RI = tf.gather_nd(R, xyz_c)  # on a gpu, no error is returned if out of bounds;

        # handle points that are out of the fov -- fill them in with the medium RI:
        # in_fov = (0 <= xyz_c < shape[None, :])  # _ by 3 (broadcasting); OperatorNotAllowedInGraphError
        in_fov = tf.logical_and(tf.greater_equal(xyz_c - 1, 0), tf.less(xyz_c + 1, shape[None, :]))
        in_fov = tf.math.reduce_all(in_fov, axis=-1)  # all 3 dims must be in range;
        RI = tf.where(in_fov, RI, self.n_back)  # if in FOV, pick RI, else use background;

        return RI

    def _propagate_rays(self, r_before_sample, u_before_sample):
        # given the initial positions and directions of the rays (after propagating thru parabolic mirror), and an RI
        # map, propagate the rays, output ray paths tensors of shape;
        # data_num_th, num_z, num_x, num_y, 3

        def linear_step(current_step, i):
            # takes one linear step; i is the step number, used for indexing the Gaussian beam, if applicable;
            # wondering if we should use the ri at the current,or the next step? or incorporate some sort of momentum?

            xyz_0 = current_step[:, :3]  # the ':3' handles the case of using first reflection loss;
            # shape: data_num_th * batch_size, 3
            ri_i = self._gather_RI(self.train_var_dict['RI'],
                                   xyz_0, recon_or_RI='RI')[:, None]  # get the RI at current location
            # shape: data_num_th * batch_size, 1

            # take a step in the propagation direction:
            xyz_i = xyz_0 + self.step / ri_i * u_before_sample

            next_step = xyz_i
            # shape: data_num_th * batch_size, 3

            if self.use_first_reflection_RI_loss:
                # if regularizing the RI based on the first reflection index, then need to keep track of the RI along
                # the trajectories;
                next_step = tf.concat([next_step, ri_i], axis=1)
                # shape: data_num_th * batch_size, 4

            return next_step

        dummy = tf.range(self.data_num_z + 1)  # acts as a counter for the for-loop;
        self.propped = tf.scan(linear_step, dummy, r_before_sample, swap_memory=True)
        self.propped = tf.transpose(self.propped, (1, 0, 2))
        # shape: # data_num_th * batch_size, numz+1, 3/4
        self.propped = self.propped[:, :-1, :]  # strip one position along prop dim (or should it be the 1st?)

        # global shift:
        if self.use_first_reflection_RI_loss:
            self.RI_path = self.propped[:, :, 3]
            self.propped = self.propped[:, :, :3]

        # unpack for convenience:
        self.x_path, self.y_path, self.z_path = [unpacked[..., 0] for unpacked in tf.split(self.propped, 3, axis=2)]

    def _propagate_rays_downsampled(self, r_before_sample, u_before_sample):
        # same as _propagate_rays, except his function is called when each ray propagation step covers more than one
        # A-scan pixel;

        # z_downsamp will be the number of A-scan pixels in one step;
        assert self.z_downsamp is not None
        assert isinstance(self.z_downsamp, (int, np.integer))

        scan_step = np.arange(1, self.z_downsamp + 1, dtype=np.float32)[:, None, None]

        if self.use_first_reflection_RI_stop_gradient:
            first_reflection = tf.gather(self.first_reflection_index, self.batch_inds_flat)
            # shape: data_num_th * batch_size

        def linear_step(current_step, i):
            # takes one linear step; i is the step number, used for indexing the Gaussian beam, if applicable;
            # wondering if we should use the ri at the current,or the next step? or incorporate some sort of momentum?

            xyz_0 = current_step[-1:, :, :3]  # the ':3' handles the case of using first reflection loss;
            # shape: (z_downsamp --> 1), data_num_th * batch_size, 3
            ri_i = self._gather_RI(self.train_var_dict['RI'],  # get the RI at most recent position;
                                   xyz_0[0], recon_or_RI='RI')[None, :, None]
            # shape: 1, data_num_th * batch_size, 1

            if self.use_first_reflection_RI_stop_gradient:
                ri_i_ = tf.where(tf.math.less(i, first_reflection)[None, :, None],
                                 tf.stop_gradient(ri_i), ri_i)
            else:
                ri_i_ = ri_i

            # take a step in the propagation direction:
            xyz_i = xyz_0 + (self.step / ri_i_ * u_before_sample[None, :, :] * scan_step)

            next_step = xyz_i
            # shape: z_downsamp, data_num_th * batch_size, 3

            if self.use_first_reflection_RI_loss:
                # if regularizing the RI based on the first reflection index, then need to keep track of the RI along
                # the trajectories;
                next_step = tf.concat([next_step, tf.broadcast_to(ri_i, [self.z_downsamp,
                                                                         self.data_num_th_gathered * self.batch_size,
                                                                         1])], axis=2)
                # shape: z_downsamp, data_num_th * batch_size, 4

            return next_step

        num_step = self.data_num_z // self.z_downsamp + 1
        dummy = tf.range(num_step, dtype=np.int32)
        scan_step_first = scan_step - 1  # make sure first step starts at 0;
        if self.use_first_reflection_RI_loss:
            # the initial value needs a 4th channel, the RI;
            r_before_sample = tf.concat([r_before_sample,
                                         self.n_back * tf.ones((self.data_num_th_gathered * self.batch_size, 1))],
                                        axis=1)
        initial_value = (r_before_sample[None, :, :] +   # need to expand r_before_sample for initial value;
                         0 * scan_step_first)  # initializer not included in output of tf.scan, but shape needs to match
        # ^assume background RI;
        self.propped = tf.scan(linear_step, dummy, initial_value, swap_memory=True)
        # shape: # num_step, z_downsamp, data_num_th * batch_size, 3/4
        if self.use_first_reflection_RI_loss:
            self.propped = tf.reshape(self.propped, [num_step * self.z_downsamp, -1, 4])  # reshape is interleaving;
        else:
            self.propped = tf.reshape(self.propped, [num_step * self.z_downsamp, -1, 3])  # reshape is interleaving;
        # shape: # ~numz, data_num_th * batch_size, 3/4
        self.propped = tf.transpose(self.propped, (1, 0, 2))
        # shape: # data_num_th * batch_size, ~numz, 3/4
        self.propped = self.propped[:, :self.data_num_z, :]  # let number of points in path match pixels in A-scan;

        # global shift:
        if self.use_first_reflection_RI_loss:
            self.RI_path = self.propped[:, :, 3]
            self.propped = self.propped[:, :, :3]

        # unpack for convenience:
        self.x_path, self.y_path, self.z_path = [unpacked[..., 0] for unpacked in tf.split(self.propped, 3, axis=2)]

    def _propagate_rays_constant_RI(self, r_before_sample, u_before_sample):
        # this can be used for registering a constant-index phantom;
        # avoids the more expensive tf.scan loop, which would be overkill if your RI map is flat;
        # due to accumulation of numerical error, this function doesn't give exactly the same result as propagate_rays,
        # but since the result is rounded to pixels in backprojection, it's usually fine;

        # scaled np.arange does the straightline rayprop through homogeneous medium:
        paths = self.step / self.n_back * (np.arange(self.data_num_z, dtype=np.float32)[None, :, None] + 1)
        self.propped = r_before_sample[:, None, :] + paths * u_before_sample[:, None, :]
        # shape: # data_num_th * batch_size, numz, 3

        # unpack for convenience:
        self.x_path, self.y_path, self.z_path = [unpacked[..., 0] for unpacked in tf.split(self.propped, 3, axis=2)]

    def _backproject_and_predict(self, Ascan_batch, dither_coords, assign_update_recon=True, use_FBP=False,
                                 only_backproject=False):
        # assign_update_recon: controls whether to use the .assign() mechanism to update the reconstruction (specified
        # via update_gradient option in the gradient_update function);
        # only_backproject: if true, don't bother with forward prediction and just return recon and normalize tensors;

        # swap axes because it's more convenient to think of z as normal to the sample surface:
        self.x_path, self.y_path, self.z_path = self.z_path, self.x_path, self.y_path
        self.tensors_to_track['x_path'] = self.x_path
        self.tensors_to_track['y_path'] = self.y_path
        self.tensors_to_track['z_path'] = self.z_path

        # stratified sampling, so flatten out;
        Ascan_batch = Ascan_batch - self.train_var_dict['Ascan_background'][None, None, :]

        if use_FBP:
            A_fft = tf.signal.fft(tf.cast(Ascan_batch, dtype=tf.complex64))  # tf.fft operates on last dim
            Ascan_batch_filtered = tf.math.real(tf.signal.ifft(A_fft * self.FBP_filter[None, None, :]))
        else:
            Ascan_batch_filtered = Ascan_batch

        # if we want to reduce the A-scan range:
        if self.z_start is not None:
            if self.z_end is not None:
                self.data_num_z_sliced = self.z_end - self.z_start
                self.x_path = self.x_path[:, self.z_start:self.z_end]
                self.y_path = self.y_path[:, self.z_start:self.z_end]
                self.z_path = self.z_path[:, self.z_start:self.z_end]
                Ascan_batch_filtered = Ascan_batch_filtered[:, :, self.z_start:self.z_end]
                if use_FBP:  # Ascan_batch and Ascan_batch_filtered are different;
                    Ascan_batch = Ascan_batch[:, :, self.z_start:self.z_end]
                else:  # here, they're the same;
                    Ascan_batch = Ascan_batch_filtered
            else:
                self.data_num_z_sliced = self.data_num_z - self.z_start
                self.x_path = self.x_path[:, self.z_start:]
                self.y_path = self.y_path[:, self.z_start:]
                self.z_path = self.z_path[:, self.z_start:]
                Ascan_batch_filtered = Ascan_batch_filtered[:, :, self.z_start:]
                if use_FBP:  # Ascan_batch and Ascan_batch_filtered are different;
                    Ascan_batch = Ascan_batch[:, :, self.z_start:]
                else:  # here, they're the same;
                    Ascan_batch = Ascan_batch_filtered
        else:
            if self.z_end is not None:
                self.data_num_z_sliced = self.z_end
                self.x_path = self.x_path[:, :self.z_end]
                self.y_path = self.y_path[:, :self.z_end]
                self.z_path = self.z_path[:, :self.z_end]
                Ascan_batch_filtered = Ascan_batch_filtered[:, :, :self.z_end]
                if use_FBP:  # Ascan_batch and Ascan_batch_filtered are different;
                    Ascan_batch = Ascan_batch[:, :, :self.z_end]
                else:  # here, they're the same;
                    Ascan_batch = Ascan_batch_filtered
            else:
                # use full;
                self.data_num_z_sliced = self.data_num_z

        Ascan_batch_filtered = tf.reshape(Ascan_batch_filtered, [-1])
        if use_FBP:  # Ascan_batch and Ascan_batch_filtered are different;
            Ascan_batch = tf.reshape(Ascan_batch, [-1])
        else:  # here, they're the same;
            Ascan_batch = Ascan_batch_filtered

        # convert to pixel units from physical spatial units:
        recon_size_dims = tf.cast(self.recon_shape, tf.float32)

        x_float = ((self.x_path / self.recon_fov[0]) + .5) * recon_size_dims[0]
        y_float = ((self.y_path / self.recon_fov[1]) + .5) * recon_size_dims[1]
        z_float = ((self.z_path / self.recon_fov[2]) + .5) * recon_size_dims[2]
        x_float = tf.reshape(x_float, [-1])
        y_float = tf.reshape(y_float, [-1])
        z_float = tf.reshape(z_float, [-1])

        # remove points that don't fall in the FOV:
        in_fov_x = tf.logical_and(tf.greater_equal(x_float, 0), tf.less(x_float, recon_size_dims[0]))
        in_fov_y = tf.logical_and(tf.greater_equal(y_float, 0), tf.less(y_float, recon_size_dims[0]))
        in_fov_z = tf.logical_and(tf.greater_equal(z_float, 0), tf.less(z_float, recon_size_dims[0]))
        in_fov = tf.logical_and(tf.logical_and(in_fov_x, in_fov_y), in_fov_z)
        x_float = tf.boolean_mask(x_float, in_fov)
        y_float = tf.boolean_mask(y_float, in_fov)
        z_float = tf.boolean_mask(z_float, in_fov)
        Ascan_batch_filtered = tf.boolean_mask(Ascan_batch_filtered, in_fov)
        if use_FBP:  # Ascan_batch and Ascan_batch_filtered are different;
            Ascan_batch = tf.boolean_mask(Ascan_batch, in_fov)
        else:  # here, they're the same;
            Ascan_batch = Ascan_batch_filtered

        if dither_coords:
            raise Exception('dither_coords not yet implemented')

        # trilinear interp (for backprojection/scattering and gathering):
        x_floor = tf.floor(x_float)
        x_ceil = x_floor + 1
        z_floor = tf.floor(z_float)
        z_ceil = z_floor + 1
        y_floor = tf.floor(y_float)
        y_ceil = y_floor + 1

        fx = x_float - x_floor
        cx = x_ceil - x_float
        fz = z_float - z_floor
        cz = z_ceil - z_float
        fy = y_float - y_floor
        cy = y_ceil - y_float

        # cast into integers:
        x_floor = tf.cast(x_floor, dtype=tf.int32)
        x_ceil = tf.cast(x_ceil, dtype=tf.int32)
        z_floor = tf.cast(z_floor, dtype=tf.int32)
        z_ceil = tf.cast(z_ceil, dtype=tf.int32)
        y_floor = tf.cast(y_floor, dtype=tf.int32)
        y_ceil = tf.cast(y_ceil, dtype=tf.int32)

        # generate the coordinates of the projection cells:
        xyzfff = tf.stack([x_floor, y_floor, z_floor], 1)
        xyzfcf = tf.stack([x_floor, y_ceil, z_floor], 1)
        xyzcff = tf.stack([x_ceil, y_floor, z_floor], 1)
        xyzccf = tf.stack([x_ceil, y_ceil, z_floor], 1)
        xyzffc = tf.stack([x_floor, y_floor, z_ceil], 1)
        xyzfcc = tf.stack([x_floor, y_ceil, z_ceil], 1)
        xyzcfc = tf.stack([x_ceil, y_floor, z_ceil], 1)
        xyzccc = tf.stack([x_ceil, y_ceil, z_ceil], 1)

        # gaussian-weighted factors (these are for interp_project and for the gathering stage after projection):
        fx = tf.exp(-fx ** 2 / 2. / self.sig_proj ** 2)
        fy = tf.exp(-fy ** 2 / 2. / self.sig_proj ** 2)
        fz = tf.exp(-fz ** 2 / 2. / self.sig_proj ** 2)
        cx = tf.exp(-cx ** 2 / 2. / self.sig_proj ** 2)
        cy = tf.exp(-cy ** 2 / 2. / self.sig_proj ** 2)
        cz = tf.exp(-cz ** 2 / 2. / self.sig_proj ** 2)

        # reconstruct:
        # compute the interpolated normalize tensor here:
        # _8 is used because for 3D, trilinear interpolation uses 8 cubes
        xyz_8 = tf.concat([xyzfff, xyzfcf, xyzcff, xyzccf, xyzffc, xyzfcc, xyzcfc, xyzccc], 0)

        # compute the interpolated backprojection:
        # it might be more efficient to use broadcasting for this:
        Ascan_8 = tf.concat([
            Ascan_batch_filtered * fx * fy * fz,
            Ascan_batch_filtered * fx * cy * fz,
            Ascan_batch_filtered * cx * fy * fz,
            Ascan_batch_filtered * cx * cy * fz,
            Ascan_batch_filtered * fx * fy * cz,
            Ascan_batch_filtered * fx * cy * cz,
            Ascan_batch_filtered * cx * fy * cz,
            Ascan_batch_filtered * cx * cy * cz
        ], 0)
        # projection weights (perhaps could be more efficient to create Ascans_8 from this):
        w_8 = tf.concat([
            fx * fy * fz,
            fx * cy * fz,
            cx * fy * fz,
            cx * cy * fz,
            fx * fy * cz,
            fx * cy * cz,
            cx * fy * cz,
            cx * cy * cz
        ], 0)

        if only_backproject:
            # this is for generating a reconstruction without momentum, so that each A-scan can contribute equally
            normalize = tf.scatter_nd(xyz_8, w_8, self.recon_shape)
            recon = tf.scatter_nd(xyz_8, Ascan_8, self.recon_shape)
            return recon, normalize

        # update recon with moving average:
        # gather values at recon_previous using the NEW coordinates:
        recon_previous = self.non_train_dict['recon_previous']
        self.Ascan_8_previous = tf.gather_nd(recon_previous, xyz_8) * w_8  # with appropriate weighting by w_8;
        if self.correct_momentum_bias:
            assert not self.correct_momentum_bias_in_loss
            self.Ascan_8_previous *= (1 - (1 - self.momentum) ** self.iter)  # the reconstruction is unbiased, but we
            # want to generate the next *biased* reconstruction, so add back the bias; after updating, the bias will
            # be re-removed, below;
            self.Ascan_8_updated = (Ascan_8 * self.momentum + self.Ascan_8_previous * (1 - self.momentum))  # biased;
            self.iter.assign_add(1)
            self.Ascan_8_updated /= (1 - (1 - self.momentum) ** self.iter)  # unbiased;
        else:
            self.Ascan_8_updated = (Ascan_8 * self.momentum + self.Ascan_8_previous * (1 - self.momentum))

        # because we are batching, it is more efficient to generate self.normalize, regather using the same
        # coordinates, and divide out im_updated (because im_updated is smaller than the reconstruction);
        normalize = tf.scatter_nd(xyz_8, w_8, self.recon_shape)  # can we avoid creating this ...
        self.norm_updated_regathered = tf.gather_nd(normalize, xyz_8)
        self.Ascan_8_updated_norm = tf.math.divide_no_nan(self.Ascan_8_updated, self.norm_updated_regathered)
        # since tensor_scatter_nd_update doesn't accumulate values, but tensor_scatter_nd_add does,first zero
        # out the regions to be updated and then just add them:
        recon_zeroed = tf.tensor_scatter_nd_update(recon_previous, xyz_8,
                                                   tf.zeros_like(self.Ascan_8_updated_norm))
        self.recon = tf.tensor_scatter_nd_add(recon_zeroed, xyz_8, self.Ascan_8_updated_norm)

        if assign_update_recon:
            with tf.device('/CPU:0'):
                recon_previous.assign(self.recon)

        if self.stop_gradient_projection:
            self.recon = tf.stop_gradient(self.recon)  # this might save some computation

        # forward prediction:
        # gathering stage for computing the loss
        fff = tf.gather_nd(self.recon, xyzfff)
        fcf = tf.gather_nd(self.recon, xyzfcf)
        cff = tf.gather_nd(self.recon, xyzcff)
        ccf = tf.gather_nd(self.recon, xyzccf)
        ffc = tf.gather_nd(self.recon, xyzffc)
        fcc = tf.gather_nd(self.recon, xyzfcc)
        cfc = tf.gather_nd(self.recon, xyzcfc)
        ccc = tf.gather_nd(self.recon, xyzccc)

        forward = (ccc * cx * cy * cz +
                   ccf * cx * cy * fz +
                   cff * cx * fy * fz +
                   cfc * cx * fy * cz +
                   fcc * fx * cy * cz +
                   fcf * fx * cy * fz +
                   fff * fx * fy * fz +
                   ffc * fx * fy * cz)

        forward /= (cx * cy * cz +
                    cx * cy * fz +
                    cx * fy * fz +
                    cx * fy * cz +
                    fx * cy * cz +
                    fx * cy * fz +
                    fx * fy * fz +
                    fx * fy * cz)

        if self.correct_momentum_bias_in_loss:
            assert not self.correct_momentum_bias
            self.iter.assign_add(1)
            forward /= (1 - (1 - 1 / self.train_var_dict['effective_inverse_momentum']) ** self.iter)  # unbiased;

        self.error = Ascan_batch - forward
        self.MSE = tf.reduce_mean(self.error ** 2)

    def _add_regularization_loss(self, reg_coefs):
        # create loss_list of all the los terms;

        # always have data-dependent loss:
        self.loss_list = [self.MSE]
        self.loss_list_names = ['MSE']

        if 'intersection' in reg_coefs:
            assert self.center_Ascan_inds is not None  # must be provided by the hdf5 file;
            self.loss_list.append(reg_coefs['intersection'] * self.intersection_MSE)
            self.loss_list_names.append('intersection')
        if 'first_reflection_RI' in reg_coefs:
            assert self.use_first_reflection_RI_loss is not None
            assert self.first_reflection_index is not None
            self.loss_list.append(reg_coefs['first_reflection_RI'] * self.first_reflection_RI_loss)
            self.loss_list_names.append('first_reflection_RI')
        if 'first_reflection_RI_weighted' in reg_coefs:
            assert self.use_first_reflection_RI_loss is not None
            assert self.first_reflection_index is not None
            self.loss_list.append(reg_coefs['first_reflection_RI_weighted'] * self.first_reflection_RI_loss_weighted)
            self.loss_list_names.append('first_reflection_RI_weighted')
        if 'RI_TV' in reg_coefs:
            self._RI_TV_loss()
            self.loss_list.append(reg_coefs['RI_TV'] * self.RI_TV_loss)
            self.loss_list_names.append('RI_TV')
        if 'RI_TV2' in reg_coefs:
            self._RI_TV2_loss()
            self.loss_list.append(reg_coefs['RI_TV2'] * self.RI_TV2_loss)
            self.loss_list_names.append('RI_TV2')
        if '+' in reg_coefs:
            self._positivity_loss()
            self.loss_list.append(reg_coefs['+'] * self.positivity_loss)
            self.loss_list_names.append('+')

    def _positivity_loss(self):
        RI = self.train_var_dict['RI']
        negative_components = tf.minimum(RI - self.n_back, 0)
        self.positivity_loss = tf.reduce_sum(negative_components ** 2)

    def _first_reflection_RI_loss(self, batch_inds):
        batch_inds = tf.reshape(batch_inds, [-1])
        first_reflection = tf.gather(self.first_reflection_index, batch_inds)  # shape: data_num_th * batch_size

        # mask that's 1 for pixels before the first reflection, 0 after;
        medium_mask = tf.math.less(tf.range(self.data_num_z, dtype=tf.int32)[None, :],
                                   first_reflection[:, None])  # data_num_th * batch_size, numz
        medium_mask = tf.cast(medium_mask, dtype=tf.float32)
        self.first_reflection_RI_loss = tf.reduce_sum(medium_mask * (self.RI_path - self.n_back) ** 2)

    def _first_reflection_RI_loss_weighted(self, batch_inds):
        # linear ramp weight, decreasing until it hits the first surface;
        batch_inds = tf.reshape(batch_inds, [-1])
        first_reflection = tf.gather(self.first_reflection_index, batch_inds)  # shape: data_num_th * batch_size

        # mask that's decreasing to 0 for pixels before the first reflection, 0 after;
        medium_mask = tf.math.maximum(first_reflection[:, None] -
                                      tf.range(self.data_num_z, dtype=tf.int32)[None, :], 0)
        medium_mask = tf.cast(medium_mask, dtype=tf.float32)
        # ^data_num_th * batch_size, numz
        self.first_reflection_RI_loss_weighted = tf.reduce_sum(medium_mask * (self.RI_path - self.n_back) ** 2)

    def _RI_TV_loss(self):
        # total variation;
        RI = self.train_var_dict['RI']
        RI_ = RI[:-1, :-1, :-1]
        d0 = RI[1:, :-1, :-1] - RI_
        d1 = RI[:-1, 1:, :-1] - RI_
        d2 = RI[:-1, :-1, 1:] - RI_
        self.RI_TV_loss = tf.reduce_sum(tf.sqrt(d0 ** 2 + d1 ** 2 + d2 ** 2 + 1e-7))

    def _RI_TV2_loss(self):
        # total variation squared;
        RI = self.train_var_dict['RI']
        RI_ = RI[:-1, :-1, :-1]
        d0 = RI[1:, :-1, :-1] - RI_
        d1 = RI[:-1, 1:, :-1] - RI_
        d2 = RI[:-1, :-1, 1:] - RI_
        self.RI_TV2_loss = tf.reduce_sum(d0 ** 2 + d1 ** 2 + d2 ** 2)

    def _ray_intersection_loss(self, r, u):
        # r, u - positions and orientations of rays;
        # compute current intersection point, whether you use it or not, for monitoring:
        uxx = u[:, 0] ** 2 - 1
        uyy = u[:, 1] ** 2 - 1
        uzz = u[:, 2] ** 2 - 1
        uxy = u[:, 0] * u[:, 1]
        uyz = u[:, 1] * u[:, 2]
        uxz = u[:, 0] * u[:, 2]
        C = tf.stack([tf.reduce_sum(uxx * r[:, 0] + uxy * r[:, 1] + uxz * r[:, 2]),
                      tf.reduce_sum(uxy * r[:, 0] + uyy * r[:, 1] + uyz * r[:, 2]),
                      tf.reduce_sum(uxz * r[:, 0] + uyz * r[:, 1] + uzz * r[:, 2])])

        M = tf.stack([[tf.reduce_sum(uxx), tf.reduce_sum(uxy), tf.reduce_sum(uxz)],
                      [tf.reduce_sum(uxy), tf.reduce_sum(uyy), tf.reduce_sum(uyz)],
                      [tf.reduce_sum(uxz), tf.reduce_sum(uyz), tf.reduce_sum(uzz)]])

        R = tf.linalg.solve(M, C[:, None])  # the best intersection point;
        self.current_intersection_point = R[:, 0]  # remove singleton dim;

        # if optimizing dome center, then you already have the desired intersection; otherwise, compute best guess;
        if 'dome' in self.propagation_model:
            self.intersection_point = self.train_var_dict['dome_center']
        else:
            self.intersection_point = self.current_intersection_point

        intersection_errors = (r + tf.einsum('ni,ni->n', self.intersection_point[None, :] - r, u)[:, None] * u -
                               self.intersection_point)  # length n
        self.intersection_MSE = tf.reduce_mean(intersection_errors ** 2) * 3  # *3 for MSE

        self.tensors_to_track['intersection_MSE'] = self.intersection_MSE
        self.tensors_to_track['current_intersection_point'] = self.current_intersection_point

    def checkpoint_all_variables(self, path=None, skip_saving=False, skip_saving_optimizer=True,
                                 save_non_train_var=False, var_ignore=['RI']):
        # save variables so that they can be used on a different dataset;
        # skip_saving if you just want to create the ckpt and manager;
        # skip_saving_optimizer, so you don't reuse optimizer state when optimizing for a new dataset;
        # var_ignore: if there are variables that you don't want to store;
        # if save_non_train_var, then also save the recon running average, for initializing the next optimization;
        # just be sure, when running this function with skip_saving=False to create the manager, to set
        # save_non_train_var=False so it doesn't try to load it into a differently sized recon_previous!

        if path is None:
            # if none supplied, save in the directory where the dataset is;
            path = os.path.join(os.path.dirname(self.hdf5_path), 'tf_ckpts/')

        if self.ckpt is None:
            self.ckpt = tf.train.Checkpoint()
            self.ckpt.var = {key:self.train_var_dict[key] for key in self.train_var_dict if key not in var_ignore}
            if not skip_saving_optimizer:
                self.ckpt.opt = self.optimizer_dict
            if save_non_train_var:
                self.ckpt.non_train_var = self.non_train_dict
                # ^ this just contains the recon running average;
            self.manager = tf.train.CheckpointManager(self.ckpt, path, max_to_keep=1)
        if not skip_saving:
            self.manager.save()

    def restore_all_variables(self):
        self.ckpt.restore(self.manager.checkpoints[0])

    def _propagate(self, batch_inds):
        # this function is a convenience function that bundles all of the propagate function calls in one;

        r_before_sample, u_before_sample = self._propagate_to_parabolic_focus(batch_inds)
        if 'dome' in self.propagation_model:
            r_before_sample, u_before_sample = self._propagate_through_dome(r_before_sample, u_before_sample)
        if 'nonparametric' in self.propagation_model:
            r_before_sample, u_before_sample = self._adjust_boundary_conditions(r_before_sample, u_before_sample,
                                                                                batch_inds)
        # global shift:
        r_before_sample = r_before_sample + self.xyz_offset[None, :]

        if self.optimize_RI:
            if self.z_downsamp is not None:
                self._propagate_rays_downsampled(r_before_sample, u_before_sample)
            else:
                self._propagate_rays(r_before_sample, u_before_sample)
        else:
            self._propagate_rays_constant_RI(r_before_sample, u_before_sample)

    @tf.function
    def gradient_update(self, Ascan_batch, batch_inds, update_gradient=True, reg_coefs=None, dither_coords=False,
                        update_recon_running_average=True, use_FBP=False, return_tracked_tensors=False,
                        return_grads=False, return_loss_only=False):
        # Ascan_batch and batch_inds are obtained by iterating over the dataset, created from the generator, which in
        # turn was created from the hdf5 file;
        # shape of Ascan_batch is stratified batch_size by total number of angles by num_z; shape of batch_inds is the
        # same except without the last dimension;
        # reg_coefs: dictionary of regularization coefficients;
        # use_FBP: whether to apply backprojection filter before backprojecting;
        # return_tracked_tensors: if True, will return tracked tensors from tf graph (that are not tf.Variables);

        self.batch_inds_flat = tf.reshape(batch_inds, [-1])

        with tf.GradientTape() as tape:
            self._propagate(batch_inds)
            self._backproject_and_predict(Ascan_batch, dither_coords=dither_coords,
                                          assign_update_recon=update_recon_running_average, use_FBP=use_FBP)

            if self.center_Ascan_inds is not None:
                r, u = self._propagate_to_parabolic_focus(self.center_Ascan_inds, mode='central_ray')
                self._ray_intersection_loss(r / 1000, u)  # convert r back to mm;
                self.tensors_to_track['r_center_Ascan'] = r / 1000
                self.tensors_to_track['u_center_Ascan'] = u

            if self.use_first_reflection_RI_loss:
                if 'first_reflection_RI' in reg_coefs:
                    self._first_reflection_RI_loss(batch_inds)
                if 'first_reflection_RI_weighted' in reg_coefs:
                    self._first_reflection_RI_loss_weighted(batch_inds)

            if reg_coefs is not None:
                self._add_regularization_loss(reg_coefs)
                loss = tf.reduce_sum(self.loss_list)
            else:
                loss = self.MSE

        trainables = [(var, optim) for var, optim, train in  # avoid computing gradients when not used;
                      zip(self.train_var_dict.values(),
                          self.optimizer_dict.values(),
                          self.trainable_or_not) if train]
        grads = tape.gradient(loss, [pair[0] for pair in trainables])

        # apply gradient update for each optimizer:
        # note that this assumes that the dictionaries are ordered!
        if update_gradient:
            for grad, (var, optimizer) in zip(grads, trainables):
                optimizer.apply_gradients([(grad, var)])

        if return_loss_only:
            if reg_coefs is not None:
                return_list = [self.loss_list]
            else:
                return_list = [self.MSE]
        else:
            if reg_coefs is not None:
                return_list = [self.loss_list, self.recon]
            else:
                return_list = [self.MSE, self.recon]

            if return_tracked_tensors:
                return_list.append(self.tensors_to_track)
            if return_grads:
                return_list.append(grads)

        return return_list


def reshape_shuffled_stack(flattened_stack, shuffle_inds, new_shape):
    # since the hdf5 files are save scrambled, this helper function takes in the flattened stack and the shuffle_inds
    # and returns a nicely shaped stack;

    shuffle_inds_inv = np.argsort(shuffle_inds)  # want the "inverse" indices;
    reordered = flattened_stack[shuffle_inds_inv]
    reshaped = np.reshape(reordered, new_shape)
    return reshaped


def summarize_recon(R, cmap='gray_r', colorbar=False):
    # for summarizing a 3D reconstruction: x, y, and z slices and max projections;
    recon_size_x = R.shape[0]
    recon_size_y = R.shape[1]
    recon_size_z = R.shape[2]

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    R_ = np.copy(R).max(0)
    plt.imshow(R_, cmap=cmap)
    clim = np.percentile(R_, [.1, 99.9])
    plt.clim(clim)
    plt.title('projection across x')
    if colorbar:
        plt.colorbar()

    plt.subplot(2, 3, 4)
    R_ = R[recon_size_x // 2]
    plt.imshow(R_, cmap=cmap)
    clim = np.percentile(R_, [.1, 99.9])
    plt.clim(clim)
    plt.title('yz slice')
    if colorbar:
        plt.colorbar()

    plt.subplot(2, 3, 2)
    R_ = np.copy(R).T.max(1)
    plt.imshow(R_, cmap=cmap)
    clim = np.percentile(R_, [.1, 99.9])
    plt.clim(clim)
    plt.title('projection across y')
    if colorbar:
        plt.colorbar()

    plt.subplot(2, 3, 5)
    R_ = R[:, recon_size_y // 2, :].T
    plt.imshow(R_, cmap=cmap)
    clim = np.percentile(R_, [.1, 99.9])
    plt.clim(clim)
    plt.title('xz slice')
    if colorbar:
        plt.colorbar()

    plt.subplot(2, 3, 3)
    R_ = R.max(2)
    plt.imshow(R_, cmap=cmap)
    clim = np.percentile(R_, [.1, 99.9])
    plt.clim(clim)
    plt.title('en face projection')
    if colorbar:
        plt.colorbar()

    plt.subplot(2, 3, 6)
    ind_z = np.sum(R, (0, 1)).argmax()
    R_ = R[:, :, ind_z]
    plt.imshow(R_, cmap=cmap)
    clim = np.percentile(R_, [.1, 99.9])
    plt.clim(clim)
    plt.title('brightest z slice')
    if colorbar:
        plt.colorbar()
    plt.show()


def get_Bscan(path, th_ind, x_ind=None, y_ind=None):
    # exactly one of x_ind and y_ind must be defined;
    # returns the B-scan given by the angular specification and one of the spatial dims;
    # also return the indices corresponding to that B-scan;
    # NOTE: slicing with an x_ind (i.e., so that y_ind is None) is more efficient because those A-scans are
    # contiguous!

    assert ((x_ind is None) and (y_ind is not None)) or ((x_ind is not None) and (y_ind is None))  # xor

    with h5py.File(path, 'r') as f:
        shape = f['default'].shape
        if x_ind is not None:
            return (f['default'][th_ind, x_ind],
                    np.ravel_multi_index((th_ind, x_ind, np.arange(shape[3])), shape[:-1]))
        elif y_ind is not None:
            return (f['default'][th_ind, :, y_ind],
                    np.ravel_multi_index((th_ind, np.arange(shape[2]), y_ind), shape[:-1]))
        