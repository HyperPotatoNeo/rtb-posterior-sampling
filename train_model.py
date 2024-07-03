from functools import partial
import os
import argparse
import yaml
import wandb
from distutils.util import strtobool

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.distributions import Normal, kl
import matplotlib.pyplot as plt

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color, mask_generator
from util.logger import get_logger


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=100000)
    parser.add_argument('--log_image_freq', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--diffusion_config', type=str)
    parser.add_argument('--task_config', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--wandb-project-name", type=str, default="advantage-diffusion",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default='swish',
        help="the entity (team) of wandb's project")
    parser.add_argument('--run_name', type=str, default='test')
    args = parser.parse_args()
   
    # logger
    logger = get_logger()
    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=args.run_name
        )
    
    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  
    
    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)
   
    #assert model_config['learn_sigma'] == diffusion_config['learn_sigma'], \
    #"learn_sigma must be the same for model and diffusion configuartion."
    
    # Load model
    model = create_model(**model_config)
    model = model.to(device)
    model.train()

    # Prepare Operator and noise
    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    # Prepare conditioning method
    cond_config = task_config['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning method : {task_config['conditioning']['method']}")
   
    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config) 
    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)

    # Prepare dataloader
    data_config = task_config['data']
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((256, 256)),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = get_dataset(**data_config, transforms=transform)
    loader = get_dataloader(dataset, batch_size=args.batch_size, num_workers=1, train=True)
    
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Exception) In case of inpainting, we need to generate a mask 
    if measure_config['operator']['name'] == 'inpainting':
        mask_gen = mask_generator(
           **measure_config['mask_opt']
        )
        
    # Training
    loss_epoch = 0.0
    step = 0
    for epoch in range(args.n_epochs):
        for i, ref_img in enumerate(loader):
            opt.zero_grad()
            ref_img = ref_img.to(device)

            # Exception) In case of inpainging,
            if measure_config['operator'] ['name'] == 'inpainting':
                mask = mask_gen(ref_img)
                #mask = mask[:, 0, :, :].unsqueeze(dim=0)
                measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
                sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn)

                # Forward measurement model (Ax + n)
                y = operator.forward(ref_img, mask=mask)
                y_n = noiser(y)
            else: 
                # Forward measurement model (Ax + n)
                y = operator.forward(ref_img)
                y_n = noiser(y).detach()
            
            if measure_config['operator'] ['name'] == 'super_resolution':
                y_n = F.interpolate(y_n, (256, 256), mode="bilinear")
            
            t = torch.randint(low=0, high=sampler.num_timesteps, size=(ref_img.shape[0],)).to(device)
            x_t, epsilon = sampler.q_sample(ref_img, t, return_noise=True)
            output = model(torch.cat([x_t,y_n],dim=1), sampler._scale_timesteps(t))

            loss = torch.sum((output-epsilon)**2)/ref_img.shape[0]
            loss.backward()
            opt.step()
            
            loss_epoch += loss.item()
            
            if (step+1) % 200 == 0:
                loss_epoch = loss_epoch/200
                torch.save(model.state_dict(), 'models/'+args.run_name+'.pt')
                # Sampling
                if (step+1) % args.log_image_freq == 0 and args.track:
                    x_start = torch.randn((5,3,256,256), device=device)
                    samples = sample_fn(x_start=x_start, y=y_n[:5])
                    samples = samples.permute(0, 2, 3, 1).cpu().numpy()
                    y = y_n[:5].permute(0, 2, 3, 1).cpu().numpy()
                    x = ref_img[:5].permute(0, 2, 3, 1).cpu().numpy()
                    fig1, axs1 = plt.subplots(1, 5, figsize=(15, 3))
                    for i in range(5):
                        axs1[i].imshow((samples[i] - samples[i].min()) / (samples[i].max() - samples[i].min()))  # Normalize the images to [0, 1]
                        axs1[i].axis('off')
                    fig2, axs2 = plt.subplots(1, 5, figsize=(15, 3))
                    for i in range(5):
                        axs2[i].imshow((x[i] - x[i].min()) / (x[i].max() - x[i].min()))  # Normalize the images to [0, 1]
                        axs2[i].axis('off')
                    fig3, axs3 = plt.subplots(1, 5, figsize=(15, 3))
                    for i in range(5):
                        axs3[i].imshow((y[i] - y[i].min()) / (y[i].max() - y[i].min()))  # Normalize the images to [0, 1]
                        axs3[i].axis('off')
                    
                    wandb.log({"loss": loss_epoch, "x_true": wandb.Image(fig2), "y": wandb.Image(fig3), "x_sampled": wandb.Image(fig1), "step": step})
                elif args.track:
                    wandb.log({"loss": loss_epoch, "step": step})
                loss_epoch = 0.0
            step += 1

if __name__ == '__main__':
    main()
