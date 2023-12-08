import ml_collections
import torch
from torch import multiprocessing as mp
from datasets import get_dataset
from torchvision.utils import make_grid, save_image
import utils
import einops
from torch.utils._pytree import tree_map
import accelerate
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from dpm_solver_pp import NoiseScheduleVP, DPM_Solver
import tempfile
from fid_score import calculate_fid_given_paths
from absl import logging
import builtins
import os
import wandb
import libs.autoencoder
import numpy as np
import ml_collections

def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)
    
config = ml_collections.ConfigDict()
config.seed = 1234
config.z_shape = (3, 1296, 4)
config.config_name = 'test'
config.ckpt_root = '/home/users/jihyeonj/jointdiffuser_protein/test/'
config.sample_dir = '/home/users/jihyeonj/jointdiffuser_protein/test/'
config.workdir = '/home/users/jihyeonj/jointdiffuser_protein/test/'
config.hparams = 'default'

config.autoencoder = d(
    pretrained_path='assets/stable-diffusion/autoencoder_kl.pth',
    scale_factor=0.23010
)
config.train = d(
    n_steps=50,
    batch_size=2,
    log_interval=10,
    eval_interval=5,
    save_interval=5
)

config.optimizer = d(
    name='adamw',
    lr=0.0002,
    weight_decay=0.03,
    betas=(0.9, 0.9)
)

config.lr_scheduler = d(
    name='customized',
    warmup_steps=5000
)

config.nnet = d(
    name='uvit_t2i',
    img_size=1296,
    in_chans=3,
    patch_size=4,
    embed_dim=100,
    depth=12,
    num_heads=10,
    mlp_ratio=4,
    qkv_bias=False,
    mlp_time_embed=False,
    clip_dim=19,
    num_clip_token=77
)

config.dataset = d(
    name='ligprot_features',
    path='/scratch/users/jihyeonj/processesd/',
    cfg=False,
    p_uncond=0.1
)

config.sample = d(
    sample_steps=2,
    n_samples=10,
    mini_batch_size=1,
    cfg=False,
    scale=1.,
    path='/home/jihyeonj/unidiffuser/test/res/'
)

def stable_diffusion_beta_schedule(linear_start=0.00085, linear_end=0.0120, n_timestep=1000):
    _betas = (torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2)
    return _betas.numpy()


def get_skip(alphas, betas):
    N = len(betas) - 1
    skip_alphas = np.ones([N + 1, N + 1], dtype=betas.dtype)
    for s in range(N + 1):
        skip_alphas[s, s + 1:] = alphas[s + 1:].cumprod()
    skip_betas = np.zeros([N + 1, N + 1], dtype=betas.dtype)
    for t in range(N + 1):
        prod = betas[1: t + 1] * skip_alphas[1: t + 1, t]
        skip_betas[:t, t] = (prod[::-1].cumsum())[::-1]
    return skip_alphas, skip_betas


def stp(s, ts: torch.Tensor):  # scalar tensor product
    if isinstance(s, np.ndarray):
        s = torch.from_numpy(s).type_as(ts)
    extra_dims = (1,) * (ts.dim() - 1)
    return s.view(-1, *extra_dims) * ts


def mos(a, start_dim=1):  # mean of square
    return a.pow(2).flatten(start_dim=start_dim).mean(dim=-1)

def LSimple(x0, y0, nnet, schedule, **kwargs):
    n, eps_x, eps_y, xn, yn = schedule.sample(x0, y0)  # n in {1, ..., 1000}
    #n= timestep
    eps_pred_prot, eps_pred_lig = nnet(xn, n, n, yn)

    mos_p = mos(eps_x - eps_pred_prot)
    mos_l = mos(eps_y - eps_pred_lig)
    return mos_p + mos_l


class Schedule(object):  # discrete time
    def __init__(self, _betas):
        r""" _betas[0...999] = betas[1...1000]
             for n>=1, betas[n] is the variance of q(xn|xn-1)
             for n=0,  betas[0]=0
        """

        self._betas = _betas
        self.betas = np.append(0., _betas)
        self.alphas = 1. - self.betas
        self.N = len(_betas)

        assert isinstance(self.betas, np.ndarray) and self.betas[0] == 0
        assert isinstance(self.alphas, np.ndarray) and self.alphas[0] == 1
        assert len(self.betas) == len(self.alphas)

        # skip_alphas[s, t] = alphas[s + 1: t + 1].prod()
        self.skip_alphas, self.skip_betas = get_skip(self.alphas, self.betas)
        self.cum_alphas = self.skip_alphas[0]  # cum_alphas = alphas.cumprod()
        self.cum_betas = self.skip_betas[0]
        self.snr = self.cum_alphas / self.cum_betas

    def tilde_beta(self, s, t):
        return self.skip_betas[s, t] * self.cum_betas[s] / self.cum_betas[t]

    def sample(self, x0, y0):  # 
        n = np.random.choice(list(range(1, self.N + 1)), (len(x0),))
        eps_x = torch.randn_like(x0)
        eps_y = torch.randn_like(y0)
        xn = stp(self.cum_alphas[n] ** 0.5, x0) + stp(self.cum_betas[n] ** 0.5, eps_x)
        yn = stp(self.cum_alphas[n] ** 0.5, y0) + stp(self.cum_betas[n] ** 0.5, eps_y)
        return torch.tensor(n, device=x0.device), eps_x, eps_y, xn, yn

    def __repr__(self):
        return f'Schedule({self.betas[:10]}..., {self.N})'

def combine_joint(z, text):
    z = einops.rearrange(z, 'B C H W -> B (C H W)')
    text = einops.rearrange(text, 'B L D -> B (L D)')
    return torch.concat([z, text], dim=-1)

def split_joint(x):
    C, H, W = 3, 1296, 4
    z_dim = C * H * W
    z, text = x.split([z_dim, 100 * 19], dim=1)
    z = einops.rearrange(z, 'B (C H W) -> B C H W', C=C, H=H, W=W)
    text = einops.rearrange(text, 'B (L D) -> B L D', L=100, D=19)
    return z, text


def main(config):
#    if config.get('benchmark', False):
#        torch.backends.cudnn.benchmark = True
#        torch.backends.cudnn.deterministic = False

    #mp.set_start_method('spawn')
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f'Process {accelerator.process_index} using device: {device}')

    config.mixed_precision = accelerator.mixed_precision
    config = ml_collections.FrozenConfigDict(config)

    assert config.train.batch_size % accelerator.num_processes == 0
    mini_batch_size = config.train.batch_size // accelerator.num_processes

    if accelerator.is_main_process:
        os.makedirs(config.ckpt_root, exist_ok=True)
        os.makedirs(config.sample_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        wandb.login()

        wandb.init(project='jointdiff-train', config=config.to_dict(),
                   name=config.hparams, job_type='train', mode='offline')
        utils.set_logger(log_level='info', fname=os.path.join(config.workdir, 'output.log'))
        logging.info(config)
    else:
        utils.set_logger(log_level='error')
        builtins.print = lambda *args: None
    logging.info(f'Run on {accelerator.num_processes} devices')

    dataset = get_dataset(**config.dataset)
    #assert os.path.exists(dataset.fid_stat)
    train_dataset = dataset.get_split(split='train', labeled=True)
    train_dataset_loader = DataLoader(train_dataset, batch_size=mini_batch_size, shuffle=True, drop_last=True,
                                      num_workers=1, pin_memory=True, persistent_workers=True)
    test_dataset = dataset.get_split(split='test', labeled=True)  # for sampling
    test_dataset_loader = DataLoader(test_dataset, batch_size=config.sample.mini_batch_size, shuffle=True, drop_last=True,
                                     num_workers=1, pin_memory=True, persistent_workers=True)

    train_state = utils.initialize_train_state(config, device)
    nnet, nnet_ema, optimizer, train_dataset_loader, test_dataset_loader = accelerator.prepare(
        train_state.nnet, train_state.nnet_ema, train_state.optimizer, train_dataset_loader, test_dataset_loader)
    lr_scheduler = train_state.lr_scheduler
    train_state.resume(config.ckpt_root)


    def get_data_generator():
        while True:
            for data in tqdm(train_dataset_loader, disable=not accelerator.is_main_process, desc='epoch'):
                yield data

    data_generator = get_data_generator()

    def get_context_generator():
        while True:
            for data in test_dataset_loader:
                yield data[0], data[1]

    context_generator = get_context_generator()
    
    _betas = stable_diffusion_beta_schedule()
    _schedule = Schedule(_betas)
    logging.info(f'use {_schedule}')

    def joint_nnet(x, timesteps):
        z, text = split_joint(x)
        z_out, text_out = nnet(z, t_prot=timesteps, t_lig=timesteps, context = text)
        if len(z_out.shape)==3:
            z_out = torch.unsqueeze(z_out, 0)
            text_out = torch.unsqueeze(text_out, 0)
        x_out = combine_joint(z_out, text_out)

        return x_out


    def train_step(_batch):
        _metrics = dict()
        optimizer.zero_grad()
        
        loss = LSimple(_batch[0], _batch[1], nnet, _schedule)  # currently only support the extracted feature version
        _metrics['loss'] = accelerator.gather(loss.detach()).mean()
        accelerator.backward(loss.mean())
        optimizer.step()
        lr_scheduler.step()
        train_state.ema_update(config.get('ema_rate', 0.9999))
        train_state.step += 1
        return dict(lr=train_state.optimizer.param_groups[0]['lr'], **_metrics)
    

    def sample_fn(_n_samples, sample_steps):
        _z_init = torch.randn(_n_samples, *config.z_shape, device=device)
        _t_init = torch.randn(_n_samples, *(100,19), device=device)
        _x_init = combine_joint(_z_init, _t_init)
        noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(_betas, device=device).float())

        def model_fn(x, t_continuous):
            t = t_continuous * _schedule.N
            return joint_nnet(_x_init, t)

        dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
        _z = dpm_solver.sample(_x_init, steps=sample_steps, eps=1. / _schedule.N, T=1.)
        prot, lig = split_joint(_z)
        return prot, lig

    
    def eval_step(n_samples, sample_steps):
        logging.info(f'eval_step: n_samples={n_samples}, sample_steps={sample_steps}, algorithm=dpm_solver, '
                     f'mini_batch_size={config.sample.mini_batch_size}')

        _z, _text = sample_fn(n_samples, sample_steps)
        with tempfile.TemporaryDirectory() as temp_path:
            path = config.sample.path or temp_path
            if accelerator.is_main_process:
                os.makedirs(path, exist_ok=True)
            utils.sample2dir(accelerator, path, n_samples, config.sample.mini_batch_size, sample_fn, sample_steps, dataset.unpreprocess)

            score = 0
            if accelerator.is_main_process:
                score = calculate_fid_given_paths((path))
                logging.info(f'step={train_state.step} eval{n_samples}={score}')
                with open(os.path.join(config.workdir, 'eval.log'), 'a') as f:
                    print(f'step={train_state.step} score{n_samples}={score}', file=f)
                wandb.log({f'score{n_samples}': score}, step=train_state.step)
            score = torch.tensor(score, device=device)
            #_fid = accelerator.reduce(_fid, reduction='sum')

        return score

    logging.info(f'Start fitting, step={train_state.step}, mixed_precision={config.mixed_precision}')

    step_score = []
    while train_state.step < config.train.n_steps:
        nnet.train()
        batch = tree_map(lambda x: x.to(device), next(data_generator))
        metrics = train_step(batch)

        nnet.eval()
        if accelerator.is_main_process and train_state.step % config.train.log_interval == 0:
            logging.info(utils.dct2str(dict(step=train_state.step, **metrics)))
            logging.info(config.workdir)
            wandb.log(metrics, step=train_state.step)

        if accelerator.is_main_process and train_state.step % config.train.eval_interval == 0:
            torch.cuda.empty_cache()
            #contexts = torch.tensor(dataset.contexts, device=device)[: 2 * 5]
            #samples = dpm_solver_sample(_n_samples=2 * 5, _sample_steps=50, context=contexts)
            #samples = make_grid(dataset.unpreprocess(samples), 5)
            #save_image(samples, os.path.join(config.sample_dir, f'{train_state.step}.png'))
            #wandb.log({'samples': wandb.Image(samples)}, step=train_state.step)
            torch.cuda.empty_cache()
        #accelerator.wait_for_everyone()

        if train_state.step % config.train.save_interval == 0 or train_state.step == config.train.n_steps:
            torch.cuda.empty_cache()
            logging.info(f'Save and eval checkpoint {train_state.step}...')
            if accelerator.local_process_index == 0:
                train_state.save(os.path.join(config.ckpt_root, f'{train_state.step}.ckpt'))
            accelerator.wait_for_everyone()
            score = eval_step(n_samples=5, sample_steps=5)  # calculate fid of the saved checkpoint
            step_score.append((train_state.step, score))
            torch.cuda.empty_cache()
        #accelerator.wait_for_everyone()

    logging.info(f'Finish fitting, step={train_state.step}')
    logging.info(f'step_score: {step_score}')
    step_best = sorted(step_score, key=lambda x: x[1])[0][0]
    logging.info(f'step_best: {step_best}')
    train_state.load(os.path.join(config.ckpt_root, f'{step_best}.ckpt'))
    del metrics
    #accelerator.wait_for_everyone()
    eval_step(n_samples=config.sample.n_samples, sample_steps=config.sample.sample_steps)



if __name__ == "__main__":
    main(config)