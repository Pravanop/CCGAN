from typing import Tuple
from torch import nn, optim
import torch
from dataproc.dataloader import CrystalVoxelDataset as CVD
from torch.utils.data import DataLoader
from model.generator import Generator
from model.discriminator import Discriminator
from model.constraint import Constraint
from tqdm import tqdm
import wandb


def trainer(parameters_dict: dict = None) -> Tuple[Generator, Discriminator, Constraint]:
    """
    The main training loop for the full model.
    :param parameters_dict: A dictionary for holding all useful parameters. An example would be:
                parameters = {
                            'data': {
                                'element_system': "**O3",
                                'property': 'band_gap',
                                'stability_factor': 'energy_above_hull',
                                'sigmaGaus': 0.3,
                                'voxel_grid_size': 20
                            },
                            'model': {
                                'batch_size': 64,
                                'device': "cpu",
                                'learning_rate': 0.001,
                                'noise_dimension': 64,
                                'epochs': 100
                            },
                            'loss': {
                                'weight_generator': 0.5,
                                'weight_constraint': 0.5
                            }
                    }
    :return: The trained generator, discriminator and constraint modules
    """
    dataset = CVD(pool=parameters_dict['data']['element_system'],
                  target=parameters_dict['data']['property'],
                  stability=parameters_dict['data']['stability_factor'],
                  sigma=parameters_dict['data']['sigmaGaus'],
                  grid_size=parameters_dict['data']['voxel_grid_size'])  # create pytorch dataset

    dataloader = DataLoader(dataset, batch_size=parameters_dict['model']['batch_size'],
                            shuffle=True, num_workers=0)  # create pytorch dataloader

    device = parameters_dict['model']['device']

    criterion_GAN = nn.BCELoss()  # loss function for classification
    criterion_con = nn.MSELoss()  # loss function for stability factor prediction

    lr = parameters_dict['model']['learning_rate']
    nz = parameters_dict['model']['noise_dimension']

    # instantiate modules of network
    netG = Generator(latent_dim=nz,
                     label_dim=1).to(device)
    netD = Discriminator(label_dim=1).to(device)
    netC = Constraint().to(device)

    # instantiate optimizers
    optDis = optim.Adam(netD.parameters(), lr=lr, betas=(0.001, 0.999))
    optGen = optim.Adam(netG.parameters(), lr=lr, betas=(0.001, 0.999))
    optCon = optim.Adam(netC.parameters(), lr=lr, betas=(0.001, 0.999))

    # define real and fake label values
    real_label = 1
    fake_label = 0

    # obtain weights for weighted loss
    Wg = parameters_dict['loss']['weight_generator']
    Wc = parameters_dict['loss']['weight_constraint']

    # initiliazing wandb logging
    wandb.init(project="CCGAN", entity="pravanop")
    wandb.config = {
        "learning_rate": lr,
        "epochs": parameters_dict['model']['epochs'],
        "batch_size": parameters_dict['model']['batch_size']
    }

    for epoch in range(parameters_dict['model']['epochs']):

        errG_epoch, errC_epoch, errD_epoch = 0, 0, 0  # trackers for epoch-wise loss

        for i, data in tqdm(enumerate(dataloader)):
            structures = data['voxel'].to(device)
            structures = structures[:, None, :, :, :]  # add channel dimension

            target = data['property'].to(device)
            constraint = data['stability'].to(device)
            constraint = constraint[:, None]

            # discriminator
            netD.zero_grad()
            batch_size = structures.size(0)

            # first lets predict with real structures
            label = torch.full((batch_size, 1), real_label, dtype=torch.float32, device=device)
            output = netD((structures, target))
            errD_real = criterion_GAN(output, label)
            errD_real.backward()

            # then lets predict with generated structures
            noise = torch.randn(batch_size, nz, dtype=torch.float32, device=device)
            fake = netG((noise, target))
            label.fill_(fake_label)
            output = netD((fake.detach(), target))
            errD_fake = criterion_GAN(output, label)

            errD = errD_fake + errD_real  # total discriminator loss
            errD_fake.backward()
            optDis.step()

            errD_epoch += errD
            wandb.log({"Discriminator Loss": errD})

            # constraint
            netC.zero_grad()
            output = netC(fake.detach())

            errC = criterion_con(output, constraint)
            errC.backward()
            optCon.step()

            errC_epoch += errC
            wandb.log({"Constraint Loss": errC})

            # generator
            netG.zero_grad()
            label.fill_(real_label)  # what is fake label, is real from generator perspective
            output = netD((fake.detach(), target))

            errG = Wg * criterion_GAN(output, label) + Wc * errC  # weighted loss function
            errG.backward()
            optGen.step()

            errG_epoch += errG
            wandb.log({"Generator Loss": errG})

            #  end of batch

        print("\n")
        print(f"Epoch: {epoch}, Generator Loss: {errG_epoch / batch_size}, Constraint Loss: {errC_epoch / batch_size}, "
              f"Discriminator Loss: {errD_epoch / batch_size}")

    return netG, netD, netC
