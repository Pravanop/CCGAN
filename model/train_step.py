from torch import nn, optim
import torch
from dataproc.dataloader import CrystalVoxelDataset as CVD
from torch.utils.data import DataLoader
from model.generator import Generator
from model.discriminator import Discriminator
from model.constraint import Constraint
from tqdm import tqdm

dataset = CVD(pool=['Ba', 'Ti', 'O'],
              property='band_gap',
              stability='energy_above_hull',
              sigma=0.3,
              grid_size=20)
dataloader = DataLoader(dataset, batch_size=32,
                        shuffle=True, num_workers=0)

#device = "mps" if torch.backends.mps.is_available() else "cpu"
device = "cpu"

criterion_GAN = nn.BCELoss()
criterion_con = nn.MSELoss()

lr = 0.001
nz = 64

netG = Generator(latent_dim=nz,
                 label_dim=1).to(device)
netD = Discriminator(label_dim=1).to(device)
netC = Constraint(label_dim=1).to(device)

optDis = optim.Adam(netD.parameters(), lr=lr, betas=(0.001, 0.999))
optGen = optim.Adam(netG.parameters(), lr=lr, betas=(0.001, 0.999))
optCon = optim.Adam(netC.parameters(), lr=lr, betas=(0.001, 0.999))

real_label = 1
fake_label = 0

for epochs in range(1000):

    errG_epoch, errC_epoch, errD_epoch = 0, 0, 0

    for i, data in tqdm(enumerate(dataloader)):

        structures= data['voxel'].to(device)
        structures = structures[:, None, :, :, :]

        target = data['property'].to(device)
        constraint = data['stability'].to(device)
        constraint = constraint[:, None]
        names = data['name']

        # discriminator
        netD.zero_grad()
        batch_size = structures.size(0)
        label = torch.full((batch_size, 1), real_label, dtype=torch.float32, device=device)

        output = netD((structures, target))
        errD_real = criterion_GAN(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(batch_size, nz, dtype = torch.float32, device = device)
        fake = netG((noise, target))
        label.fill_(fake_label)
        output = netD((fake.detach(), target))
        errD_fake = criterion_GAN(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        errD = errD_fake + errD_real
        optDis.step()

        errD_epoch += errD

        # constraint
        netC.zero_grad()
        output = netC((fake.detach(), target))

        errC = criterion_con(output, constraint)
        errC.backward()
        G_C = output.mean().item()
        optCon.step()

        errC_epoch += errC

        # generator
        netG.zero_grad()
        label.fill_(real_label)  # what is fake label, is real for generator
        output = netD((fake.detach(), target))
        errG = criterion_GAN(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optGen.step()

        errG_epoch += errG

    print("\n")
    print(errG_epoch / batch_size, errC_epoch / batch_size, errD_epoch / batch_size)
