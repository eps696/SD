# This code is built from the Stable Diffusion repository: https://github.com/CompVis/stable-diffusion.
# Adobe’s modifications are licensed under the Adobe Research License
# edited by Vadim Epstein
import os, sys
import datetime
import argparse, glob
import numpy as np
from packaging import version
from omegaconf import OmegaConf
from functools import partial
from PIL import Image

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only

from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config

import logging
logging.basicConfig(level=logging.ERROR) # stop xformers warnings
import transformers
transformers.logging.set_verbosity_error() # stop transformers warning
import warnings
warnings.filterwarnings("ignore")

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool): return v
        if v.lower() in ("yes", "true", "t", "y", "1"): return True
        elif v.lower() in ("no", "false", "f", "n", "0"): return False
        else: raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument('-md', '--maindir', default='./', help='Main SD directory')
    parser.add_argument('-o',  '--outdir',  default="train", help="training directory")
    parser.add_argument('-r',   '--resume', default="", nargs="?", help="resume previous run - from outdir or checkpoint in outdir")
    parser.add_argument('-rc', '--resume_ckpt', default="models/sd-v15-512-fp16.ckpt", nargs="?", help="resume from outdir or checkpoint in outdir")
    parser.add_argument('-c',  '--configs', default=None, nargs="*", help="paths to base configs (can be overwritten by args)")
    # input
    parser.add_argument("--token",  default='', help="special word to invoke the embedding")
    parser.add_argument("--term",   default='', help="generic word(s), associating with that object or style")
    parser.add_argument("--data",   default='', help="path to target images")
    parser.add_argument("--style",  action='store_true', help="is it style? (otherwise object)")
    # embeddings
    parser.add_argument('-ec', '--emb_man_ckpt', default=None, help="Initialize embedding manager from a checkpoint")
    # custom
    parser.add_argument("--reg_data",   default=None, help="path to regularization images")
    parser.add_argument("--delta_ckpt", default=None, nargs="?", help="resume from outdir or checkpoint in outdir")
    parser.add_argument("--freeze_model", default=None, help="set 'crossattn' to enable fine-tuning of all key, value, query matrices")
    parser.add_argument("--repeat",     default=0, type=int, help="repeat the target dataset by how many times. for training without regularization?")
    parser.add_argument("--face",       action='store_true', help="are we training face?")
    parser.add_argument("--token2",     default=None, help="special word to invoke the embedding")
    parser.add_argument("--term2",      default=None, help="generic word(s), associating with that object or style")
    parser.add_argument("--data2",      default=None, help="path to target images")
    parser.add_argument("--reg_data2",  default=None, help="path to regularization images")
    parser.add_argument("--style2",     action='store_true', help="is it style? (otherwise object)")
    # misc
    parser.add_argument("--scale_lr",   default=True, type=str2bool, help="scale base-lr by gpus * batch_size * n_accumulate")
    parser.add_argument("--save_step",  default=250, type=int, help="how often to save models and samples")
    parser.add_argument('-b', "--batch_size", default=1, type=int, help="overwrite batch size")
    parser.add_argument("--num_gpus",   default=1, type=int, help="amount of gpus to be used")
    parser.add_argument("-s", "--seed", type=int, default=None, help="seed for seed_everything")
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser

def nondefault_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))

def load_model_from_config(config, ckpt):
    pl_sd = torch.load(ckpt, map_location="cuda")
    sd = pl_sd["state_dict"]
    config.model.params.ckpt_path = ckpt
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.cuda()
    return model

class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""
    def __init__(self, dataset):
        self.data = dataset
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

# for custom
class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
    def __getitem__(self, idx):
        return tuple(d[idx] for d in self.datasets)
    def __len__(self):
        return min(len(d) for d in self.datasets)

def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    worker_id = worker_info.id
    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)

class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, train2=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False, shuffle_val_dataloader=False):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        if train2 is not None and all(att in train2['params'] for att in ['datapath', 'reg_datapath', 'caption', 'reg_caption']): # for custom
            self.dataset_configs["train2"] = train2
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)
        self.datasets = dict((k, instantiate_from_config(self.dataset_configs[k])) for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        if "train2" in self.dataset_configs and self.dataset_configs["train2"]['params']["caption"] is not None: # for custom
            train_set = self.datasets["train"]
            train2_set = self.datasets["train2"]
            concat_dataset = ConcatDataset(train_set, train2_set)
            return DataLoader(concat_dataset, batch_size=self.batch_size, num_workers=self.num_workers, 
                              shuffle=False if is_iterable_dataset else True, worker_init_fn=init_fn)
        else:
            return DataLoader(self.datasets["train"], batch_size=self.batch_size, num_workers=self.num_workers, 
                              shuffle=False if is_iterable_dataset else True, worker_init_fn=init_fn)

    def _val_dataloader(self, shuffle=False):
        if isinstance(self.datasets['validation'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["validation"], batch_size=self.batch_size, num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle)

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)
        return DataLoader(self.datasets["test"], batch_size=self.batch_size, num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle)

    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets['predict'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size, num_workers=self.num_workers, worker_init_fn=init_fn)

class SetupCallback(Callback):
    def __init__(self, resume, outdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.outdir = outdir
        self.cfgdir = os.path.join(outdir, "configs")
        self.config = config
        self.lightning_config = lightning_config

    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            ckpt_path = os.path.join(self.outdir, "last.ckpt")
            print(".. saving checkpoint at", ckpt_path)
            trainer.save_checkpoint(ckpt_path)

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            os.makedirs(self.outdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)
            OmegaConf.save(self.config, os.path.join(self.cfgdir, "project.yaml"))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}), os.path.join(self.cfgdir, "lightning.yaml"))
        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.outdir):
                dst, name = os.path.split(self.outdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.outdir, dst)
                except FileNotFoundError:
                    pass

class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True, rescale=True, disabled=False, 
                 log_on_batch_idx=False, log_first_step=False, log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {pl.loggers.TestTubeLogger: self._testtube,}
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(tag, grid, global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}{:04}-e{:03}-b{:03}.jpg".format(k, global_step, current_epoch, batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        # batch_idx % self.batch_freq == 0
        if self.check_frequency(check_idx) and hasattr(pl_module, "log_images") and callable(pl_module.log_images) and self.max_images > 0:
            logger = type(pl_module.logger)
            is_train = pl_module.training
            if is_train:
                pl_module.eval()
            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)
            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)
            self.log_local(pl_module.logger.save_dir, split, images, pl_module.global_step, pl_module.current_epoch, batch_idx)
            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)
            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (check_idx > 0 or self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                print(e)
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")

    # not in custom ?
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and pl_module.global_step > 0:
            self.log_img(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)

# not used yet, for text inversion?
class ModeSwapCallback(Callback):
    def __init__(self, swap_step=2000):
        super().__init__()
        self.is_frozen = False
        self.swap_step = swap_step

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.global_step < self.swap_step and not self.is_frozen:
            self.is_frozen = True
            trainer.optimizers = [pl_module.configure_opt_embedding()]
        if trainer.global_step > self.swap_step and self.is_frozen:
            self.is_frozen = False
            trainer.optimizers = [pl_module.configure_opt_model()]


# configs are merged from left-to-right followed by command line parameters
def main():
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    opt, unknown = parser.parse_known_args()

    # what to train - custom diffusion or textual inversion
    CUSTOM = True if opt.reg_data is not None and len(opt.reg_data) > 0 else False

    if opt.configs is None:
        opt.configs = 'v1-finetune-custom.yaml' if CUSTOM else 'v1-finetune.yaml'
        opt.configs = os.path.join(opt.maindir, 'src/yaml', opt.configs)
    print('.. training %s [%s]' % ('Custom Diffusion token' if CUSTOM else 'Textual Inversion embedding', os.path.basename(opt.configs)))
    if not isinstance(opt.configs, list): opt.configs = [opt.configs]
    
    seed_everything(opt.seed)
    if opt.verbose is not True:
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

    if opt.resume: # only for custom? test!
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            outdir = "/".join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            outdir = opt.resume.rstrip("/")
            ckpt = os.path.join(outdir, "checkpoints", "last.ckpt")
        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(outdir, "configs/*.yaml")))
        opt.configs = base_configs + opt.configs
        _tmp = outdir.split("/")
        nowname = _tmp[-1]
    else:
        now = datetime.datetime.now().strftime("%d_%H%M%S")
        nowname = '-'.join([now, opt.token, 'custom' if CUSTOM else 'textinv'])
        outdir = os.path.join(opt.outdir, nowname)

    # init and save configs
    configs = [OmegaConf.load(cfg) for cfg in opt.configs]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    lightning_config = config.pop("lightning", OmegaConf.create())

    # data
    if CUSTOM:
        data_dict = {'datapath': opt.data, 'reg_datapath': opt.reg_data, 'caption': "<%s> %s" % (opt.token, opt.term), 'reg_caption': opt.term, 'style': opt.style}
        for k in data_dict:
            config.data.params.train.params[k] = data_dict[k]
        mod_token = "<%s>" % opt.token
        if opt.repeat > 0:
            config.data.params.train.params.repeat = opt.repeat
        if opt.face is True:
            lightning_config.trainer.max_steps *= 2
        if not None in [opt.token2, opt.term2, opt.data2, opt.reg_data2]:
            data_dict2 = {'datapath': opt.data2, 'reg_datapath': opt.reg_data2, 'caption': "<%s> %s" % (opt.token2, opt.term2), 'reg_caption': opt.term2, 
                          'style': opt.style2}
            for k in data_dict2:
                config.data.params.train2.params[k] = data_dict2[k]
            mod_token += "+<%s>" % opt.token2
            lightning_config.trainer.max_steps *= 2
    else:
        data_dict = {'data_root': opt.data, 'style': opt.style, 'placeholder_token': "<%s>" % opt.token, 'coarse_class_text': opt.term}
        for k in data_dict:
            config.data.params.train.params[k] = data_dict[k]

    config.data.params.validation = config.data.params.train
    config.data.params.batch_size = opt.batch_size
    data = instantiate_from_config(config.data)
    data.prepare_data() # need to call it?  https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html 

    condcfg = config.model.params.cond_stage_config # FrozenCLIPEmbedder / FrozenCLIPEmbedderWrapper / FrozenOpenCLIPEmbedder / ..
    if not hasattr(condcfg, 'params'): condcfg.params = {}
    condcfg.params.model_dir = os.path.join(opt.maindir, 'models')

    # model
    if CUSTOM:
        condcfg.params.modifier_token = mod_token 
        if opt.resume_ckpt:
            config.model.params.ckpt_path = None
        if opt.freeze_model is not None:
            config.model.params.freeze_model = opt.freeze_model

        # custom model load
        model = instantiate_from_config(config.model)
        if opt.resume_ckpt:
            print(f".. model {opt.resume_ckpt}")
            st = torch.load(opt.resume_ckpt, map_location='cpu')["state_dict"]
            token_ws = st["cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"]
            del st["cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"]
            model.load_state_dict(st, strict=False)
            model.cond_stage_model.transformer.text_model.embeddings.token_embedding.weight.data[:token_ws.shape[0]] = token_ws 
        if opt.delta_ckpt is not None:
            st = torch.load(opt.delta_ckpt)
            embed = None
            if 'embed' in st:
                embed = st['embed'].reshape(-1, 768)
            if 'state_dict' in st:
                st = st['state_dict']
            print(".. restoring from delta model from previous version")
            st1 = model.state_dict()
            for each in st1.keys():
                if each in st.keys():
                    print("found common", each)
            model.load_state_dict(st, strict=False)
            if embed is not None:
                print(".. restoring embedding")
                model.cond_stage_model.transformer.text_model.embeddings.token_embedding.weight.data[token_ws.shape[0]: token_ws.shape[0] + embed.shape[0]] = embed
    else:
        # embeddings
        if opt.emb_man_ckpt is not None:
            config.model.params.personalization_config.params.embedding_manager_ckpt = opt.emb_man_ckpt
        config.model.params.personalization_config.params.placeholder_strings = ["<%s>" % opt.token]
        if opt.term:
            config.model.params.personalization_config.params.initializer_words[0] = opt.term
        model = load_model_from_config(config, opt.resume_ckpt)
        model.create_embedding_manager()

    trainer_kwargs = dict()

    # logger
    logger_cfg = {"target": "pytorch_lightning.loggers.TestTubeLogger", "params": {"name": "testtube", "save_dir": outdir}}
    if "logger" in lightning_config:
        logger_cfg = OmegaConf.merge(lightning_config.logger, logger_cfg)
    trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

    # checkpoints
    modelckpt_cfg = {"target": "pytorch_lightning.callbacks.ModelCheckpoint", "params": {"dirpath": outdir, 'verbose': opt.verbose}}
    modelckpt_cfg["params"].update({'every_n_train_steps': opt.save_step, "filename": "{global_step:04}"})
    # modelckpt_cfg["params"].update({'every_n_epochs': 1, "filename": "{epoch:02}"})
    if hasattr(model, "monitor"):
        print(f".. checkpoint metric: {model.monitor}")
        modelckpt_cfg["params"].update({'monitor': model.monitor, 'save_top_k': -1})
    if "modelcheckpoint" in lightning_config:
        modelckpt_cfg = OmegaConf.merge(lightning_config.modelcheckpoint, modelckpt_cfg)
    if version.parse(pl.__version__) < version.parse('1.4.0'):
        trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

    # learning rate
    acc_grad_bs = 2
    model.learning_rate, bs = config.model.base_learning_rate, config.data.params.batch_size
    if opt.scale_lr is True: model.learning_rate *= acc_grad_bs * opt.num_gpus * bs
    if opt.face is True:     model.learning_rate /= 2. # 5.0e-06

    # callbacks & trainer
    lightning_config.trainer.update({'accumulate_grad_batches': acc_grad_bs, 'accelerator': "gpu", 'gpus': '0,', 'precision': 16})
    for k in nondefault_args(opt): # extra command line args => directly to trainer
        lightning_config.trainer[k] = getattr(opt, k)
    callbacks_cfg = {
        "setup_callback": {"target": "train.SetupCallback", 
                           "params": {"resume": opt.resume, "outdir": outdir, "config": config, "lightning_config": lightning_config}},
        # "image_logger":   {"target": "train.ImageLogger", "params": {"batch_frequency": opt.save_step}},
    }
    if version.parse(pl.__version__) >= version.parse('1.4.0'):
        callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg})
    if "callbacks" in lightning_config:
        callbacks_cfg = OmegaConf.merge(lightning_config.callbacks, callbacks_cfg)
    trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
    trainer_opt = argparse.Namespace(**lightning_config.trainer)
    trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)

    trainer.fit(model, data)
    
    if CUSTOM:
        from custom.get_deltas import save_delta
        save_delta(outdir, 2 if opt.term2 is not None else 1)

if __name__ == "__main__":
    main()
