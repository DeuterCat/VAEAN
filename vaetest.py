import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
from torch import optim
# hyperparameters
c = 1
d = 64
h = 10
batch_size = 64
device = 'cuda'
num_epochs = 50
lr = 0.0001
img_size = 84
need_scheduler = False
kld_weight = 0.5 # calculated below
size = torch.Size([batch_size, c, img_size, img_size])
# 
def show_device():
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name())
    print(torch.cuda.current_device())
    return 

dst = datasets.FashionMNIST(
    root = "./",
    train=True,
    download=False,
    transform=transforms.Compose([
        transforms.RandomCrop(img_size, padding=28),
        transforms.ToTensor(),
    ]) 
    )


# kld_weight = batch_size/len(dst)
print("kld_weight:", kld_weight)

dld = torch.utils.data.DataLoader(dst, batch_size=batch_size,
                                    shuffle=True)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Vae(nn.Module):
    def __init__(self):
        super(Vae, self).__init__()
        self.encoder = nn.Sequential(
            # input is (nc) x 84*84
            nn.Linear(img_size**2, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.LeakyReLU(inplace=True),
            # output is (nc) x 256
        )

        self.decoder = nn.Sequential(
            # input is (nc) x 256
            nn.Linear(256, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, img_size**2),
            nn.Tanh()
        )


        self.mu = nn.Sequential(
            nn.Linear(256, h),
            nn.LeakyReLU(inplace=True),
        )
        self.log_var = nn.Sequential(
            nn.Linear(256, h),
            nn.LeakyReLU(inplace=True),
        )
        self.de = nn.Sequential(
            nn.Linear(h, 256),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = x.view(batch_size, c, -1)
        encoded = self.encoder(x)
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        # reparameterization
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps*std + mu
        transz = self.de(z)
        output = self.decoder(transz)
        result = output.view(batch_size, c, img_size, img_size)
        return result, mu, log_var


def criterion(output, input, mu, logvar):
    MSE = nn.MSELoss(reduction='sum')
    recons_loss = MSE(output, input)
    kld_loss = -0.5 * torch.sum(1+logvar-torch.exp(logvar)-mu**2)
    return recons_loss + kld_loss

vae = Vae().to(device)
vae.apply(weights_init)
optimizer = optim.Adam(vae.parameters(), lr=lr, betas=(0.5, 0.999))
scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer,
                                                             gamma=0.95)






plotimg = torch.Tensor()
show_device()
for epoch in range(1,num_epochs+1):
    print("##############epoch:",epoch,"##################")
    for i, data in enumerate(dld):
        vae.zero_grad()
        iterbatch = data[0].to(device)
        # print(iterbatch.size())
        if iterbatch.size() != size:
            continue
        result, mu, log_sigma = vae(iterbatch)
        loss = criterion(result, iterbatch, mu, log_sigma)
        loss.backward()
        optimizer.step()
        if need_scheduler == True:
            scheduler.step()

        if i%50 == 0:
            print('[{}/{}][{}/{}]\tloss:{}'.format(epoch,
             num_epochs, i+1, len(dld), loss.item()))
        
        if epoch == num_epochs and i+2 == len(dld): 
            # note down last batch
            with torch.no_grad():
                plotimg = result.detach()







# Plot the fake images from the last epoch


real_batch = next(iter(dld))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0][:64].to(device), padding=2, normalize=True).cpu(),(1,2,0)))
plt.savefig('origin_fig.png')
plt.show()
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(vutils.make_grid(plotimg, padding=2, normalize=True).cpu(),(1,2,0)))
plt.savefig('gen_fig.png')
plt.show()