import torch
import torchvision.datasets as vdst
from torchvision import transforms as T
from conv64 import Conv64F
from conv64 import SingleLinear
from conv64 import ReparaDecoder
from conv64 import AutoEncoder

import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
import copy
from torch import nn
import torch.optim as optim
from sampler import FewshotSampler
# Hyperparameters
data_root = r'.\new_miniimagenet'
img_size = 84
batch_size = 64 # set
transforms =  T.Compose([
    T.Resize(img_size),
    T.RandomCrop(img_size),
    T.ToTensor()
])
train_epoch = 1
hidden_size = 64
vae_lr = 0.0001
emb_lr = 0.001
class_lr = 0.001
inner_lr = 0.001
way_num = 5
shot_num = 1
query_num = 15
inner_iter = 100
episode_num = 100
vae_weight = 0.0001
recon_rate = 0.5
image_num = shot_num + query_num
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mode = 'vaean' # baseline, aean, vaean
print('device is :', device)
print('%'*10,mode,'%'*10)


# prepare train test and val datasets and dataloaders

train_dst = vdst.ImageFolder(
    root=data_root+r'\train', 
    transform=transforms
    )

train_dld = torch.utils.data.DataLoader(train_dst, batch_size=batch_size,
                                    shuffle=True)


'''
during the validation or the test stage the sampling strategy is changed
as you have to take 'way_num' categories of data
and each way has 'shot_num' of images(image if the 'shot_num' is 1) to form support set
and 'query_num' of images to form query set
thus we have to restate sampler
the sampler is build in sampler as class FewshotSampler 
'''
val_dst = vdst.ImageFolder(
    root=data_root+r'\val', 
    transform=transforms
    )

sampler = FewshotSampler(
    dataset=val_dst,
    total_cat=len(val_dst.class_to_idx),
    episode_num=episode_num,
    way_num=way_num,
    image_num=image_num,
)

val_dld = torch.utils.data.DataLoader(
    val_dst,
    batch_sampler =sampler
    )


test_dst = vdst.ImageFolder(
    root=data_root+r'\test', 
    transform=transforms
    )

test_dld = torch.utils.data.DataLoader(
    val_dst,
    batch_sampler =sampler
    )

# define embedding model and classifier
emb_model = Conv64F().to(device)
classifier = SingleLinear().to(device)
decoder = ReparaDecoder().to(device)

# loss functions to use
CELoss = nn.CrossEntropyLoss()

# optimizers&schedulers
optimizer_emb = optim.Adam(emb_model.parameters(), lr=emb_lr, betas=(0.5, 0.999))
optimizer_class = optim.Adam(classifier.parameters(), lr=class_lr, betas=(0.5, 0.999))
optimizer_decoder = optim.Adam(decoder.parameters(), lr = vae_lr, betas=(0.5, 0.999))
scheduler_emb = optim.lr_scheduler.StepLR(optimizer_emb, step_size=10, gamma=0.5)
scheduler_class = optim.lr_scheduler.StepLR(optimizer_class, step_size=10, gamma=0.5)
scheduler_decoder = optim.lr_scheduler.StepLR(optimizer_decoder,step_size=10, gamma=0.5)

# pretraining
# pretrain-stage

h = 10
c = 3
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
        x = x.view(1, c, -1)
        encoded = self.encoder(x)
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        # reparameterization
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps*std + mu
        transz = self.de(z)
        output = self.decoder(transz)
        result = output.view(1, c, img_size, img_size)
        return result, mu, log_var




def plot_img(input):
    plt.figure(figsize=(4,2))
    plt.axis("off")
    plt.imshow(np.transpose(vutils.make_grid(input.to(device), padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()

    return


def pre_train():
    print('='*10,'pretraining on auxiliary set','='*10)
    loss_epoch = 0
    cor_epoch = 0
    for i, data in enumerate(train_dld):
        
        # data to model
        emb_model.zero_grad()
        classifier.zero_grad()
        image, label = data
        image = image.to(device)
        label = label.to(device)

        # forward
        embedded = emb_model(image)
        resized = embedded.view(batch_size, -1)
        class_output = classifier(resized)

        # calculate loss
        class_loss = CELoss(class_output, label)
        
        loss = class_loss

        # backward
        loss.backward()
        optimizer_emb.step()
        optimizer_class.step()

        loss_epoch += loss.item()
        # calculate accuracy of training batch
        with torch.no_grad():
            cor = (class_output.argmax(dim=1).to(device)==label).float().sum().item()
            cor = cor/batch_size
            cor_epoch += cor
        print(i, cor)
    #calculate training accuracy and loss of the poch
    cor_epoch = cor_epoch/(i+1)
    loss_epoch = loss_epoch/len(train_dst)
    return cor_epoch, loss_epoch


def split_data(image, label):
    support_image = []
    query_image = []
    support_label = []
    query_label = []
    for i in range(way_num):
        support_image.append(image[i*image_num])
        support_label.append(torch.tensor([i]))
        for j in range(image_num-1):
            query_image.append(image[i*image_num+j+1])        
            query_label.append(torch.tensor([i]))        


    support_image = torch.stack(support_image).to(device)
    query_image = torch.stack(query_image).to(device)
    support_label = torch.stack(support_label).view(-1).to(device)
    query_label = torch.stack(query_label).view(-1).to(device)

    return  support_image, query_image, support_label, query_label


def fine_tuning(dld):
    loss_epoch = 0
    cor_epoch = 0
    for i, data in enumerate(dld):
        # image size [80, 3, 84, 84]
        # label size [80] 
        image, label = data
        image = image.to(device)
        label = label.to(device)

        # split data into support and query
        # and relabel the data
        support_image, query_image, support_label, query_label = split_data(image, label)


        new_classifier = nn.Linear(64*5*5, way_num)
        new_classifier = new_classifier.to(device)
        inner_optimizer = torch.optim.Adam(new_classifier.parameters(), lr=inner_lr, betas=(0.5, 0.999))
        # inner_scheduler = optim.lr_scheduler.StepLR(inner_optimizer, step_size=10, gamma=0.5)

        # train new_classifier on support set
        print("="*10,'train on support set',"="*10, i)
        inner_loss = 0
        new_classifier.train()
        for inner_epoch in range(inner_iter):
            with torch.no_grad():
                embedded_support = emb_model(support_image)
                resized  = embedded_support.view(way_num,-1)    
            
            class_output = new_classifier(resized)
            loss = CELoss(class_output, support_label)
            inner_loss += loss.item()
            
            inner_optimizer.zero_grad()
            loss.backward(retain_graph=True)
            inner_optimizer.step()
        print('pretraining inner loss', inner_loss/inner_iter)
        loss_epoch += inner_loss/inner_iter
        
        # test on query set
        print('='*10,'test on query set','='*10, i)
        with torch.no_grad():
            embedded_test = emb_model(query_image)
            resized  = embedded_test.view(way_num * query_num,-1) 
            test_output = new_classifier(resized)
            cor = (test_output.argmax(dim=1).to(device)==query_label).float().sum().item()
            cor = cor/(way_num * query_num)
            print('query_accuracy',cor)
            cor_epoch += cor

    cor_epoch = cor_epoch/(i+1)
    loss_epoch = loss_epoch/(i+1)
    return cor_epoch, loss_epoch


def criterion(output, input, mu, logvar, recon_rate):
    MSE = nn.MSELoss(reduction='sum')
    recons_loss = MSE(output, input)
    kld_loss = -0.5 * torch.sum(1+logvar-torch.exp(logvar)-mu**2)
    return recon_rate*recons_loss + (1-recon_rate)*kld_loss

def vae_pre_train(epoch):
    print('='*10,'pretraining on auxiliary set','='*10)
    loss_epoch = 0
    cor_epoch = 0
    for i, data in enumerate(train_dld):
        
        # data to model
        emb_model.zero_grad()
        classifier.zero_grad()
        decoder.zero_grad()
        image, label = data
        image = image.to(device)
        label = label.to(device)

        # forward
        embedded = emb_model(image)
        resized = embedded.view(batch_size, -1)
        class_output = classifier(resized)
        

        mu, log_var, vae_output = decoder(resized)
        vae_output = vae_output.view(-1, 3, 84, 84)
        # calculate loss
        class_loss = CELoss(class_output, label)
        vae_loss = criterion(vae_output, image, mu, log_var, recon_rate)
        loss =(1 - vae_weight)*class_loss + vae_weight*vae_loss
        print(i, vae_weight, class_loss.item(), vae_loss.item(), loss.item())

        # backward
        loss.backward()
        optimizer_emb.step()
        optimizer_class.step()
        optimizer_decoder.step()
        loss_epoch += loss.item()
        # calculate accuracy of training batch
        with torch.no_grad():
            cor = (class_output.argmax(dim=1).to(device)==label).float().sum().item()
            cor = cor/batch_size
            cor_epoch += cor
        print(i, cor)
    #calculate training accuracy and loss of the poch
    cor_epoch = cor_epoch/(i+1)
    loss_epoch = loss_epoch/len(train_dst)
    return cor_epoch, loss_epoch

def vae_fine_tuning(dld):
    loss_epoch = 0
    cor_epoch = 0
    for i, data in enumerate(dld):
        # image size [80, 3, 84, 84]
        # label size [80] 
        image, label = data
        image = image.to(device)
        label = label.to(device)

        # split data into support and query
        # and relabel the data
        support_image, query_image, support_label, query_label = split_data(image, label)
        
        # augmentation
        # 5 -> 40
        aug_list = []
        label_list = []
        for aug_way in range(way_num):
            under_aug = support_image[aug_way]
            under_aug = under_aug.view(-1, 3, img_size, img_size)
            label_list.append(torch.tensor([aug_way]))
            aug_list.append(under_aug)
            # decoder finetuning
            copy_de = copy.deepcopy(decoder)
            vaeop = optim.Adam(copy_de.parameters(), lr=emb_lr, betas=(0.5, 0.999))




            for j in range(20):
                with torch.no_grad():
                    out = emb_model(under_aug)
                resized = out.view(1, -1)
                mu, log_var, aug = copy_de(resized)
                aug = aug.view(-1, 3, 84, 84)
                train_loss = criterion(aug, under_aug, mu, log_var, recon_rate)
                vaeop.zero_grad()
                train_loss.backward()
                vaeop.step()



            for aug_num in range(7):
                with torch.no_grad():
                    out = emb_model(under_aug)
                    resize = out.view(1, -1)
                    mu, log_var, aug = copy_de(resize)
                    aug = aug.view(-1, 3, img_size, img_size)
                
                aug_list.append(aug)
                label_list.append(torch.tensor([aug_way]))

        support_image = torch.cat(aug_list).to(device)
        plot_img(support_image)
        support_label = torch.cat(label_list).to(device)

        new_classifier = nn.Linear(64*5*5, way_num)
        new_classifier = new_classifier.to(device)
        inner_optimizer = torch.optim.Adam(new_classifier.parameters(), lr=inner_lr, betas=(0.5, 0.999))
        # inner_scheduler = optim.lr_scheduler.StepLR(inner_optimizer, step_size=10, gamma=0.5)

        # train new_classifier on support set
        print("="*10,'train on support set',"="*10, i)
        inner_loss = 0
        new_classifier.train()
        for inner_epoch in range(inner_iter):
            with torch.no_grad():
                embedded_support = emb_model(support_image)
                resized  = embedded_support.view(40,-1)    
            
            class_output = new_classifier(resized)
            loss = CELoss(class_output, support_label)
            inner_loss += loss.item()
            
            inner_optimizer.zero_grad()
            loss.backward(retain_graph=True)
            inner_optimizer.step()
            
        print('pretraining inner loss', inner_loss/inner_iter)
        loss_epoch += inner_loss/inner_iter
        
        # test on query set
        print('='*10,'test on query set','='*10, i)
        with torch.no_grad():
            embedded_test = emb_model(query_image)
            resized  = embedded_test.view(way_num * query_num,-1) 
            test_output = new_classifier(resized)
            cor = (test_output.argmax(dim=1).to(device)==query_label).float().sum().item()
            cor = cor/(way_num * query_num)
            print('query_accuracy',cor)
            cor_epoch += cor

    cor_epoch = cor_epoch/(i+1)
    loss_epoch = loss_epoch/(i+1)
    return cor_epoch, loss_epoch

def ae_fine_tuning(dld):
    loss_epoch = 0
    cor_epoch = 0
    for i, data in enumerate(dld):
        # image size [80, 3, 84, 84]
        # label size [80] 
        image, label = data
        image = image.to(device)
        label = label.to(device)

        # split data into support and query
        # and relabel the data
        support_image, query_image, support_label, query_label = split_data(image, label)
        
        # augmentation
        # 5 -> 40
        aug_list = []
        label_list = []
        for aug_way in range(way_num):
            under_aug = support_image[aug_way]
            
            vae = Vae().to(device)
            vaeop = optim.Adam(vae.parameters(), lr=0.001, betas=(0.5, 0.999))
            # inner_scheduler = optim.lr_scheduler.StepLR(inner_optimizer, step_size=10, gamma=0.5)

            # vae train
        
            for aeepoch in range(20):
                train_out, mu, log_var = vae(under_aug)
                vaeloss = criterion(train_out, under_aug, mu, log_var, recon_rate)
                vaeop.zero_grad()
                vaeloss.backward()
                vaeop.step()
            
            for aug_num in range(7):
                # train autoencoder
                with torch.no_grad():
                    aug,mu,log_var = vae(under_aug)

                aug = aug.view(-1, 3, img_size, img_size)
                aug_list.append(aug)
                label_list.append(torch.tensor([aug_way]))

            under_aug = under_aug.view(-1, 3, img_size, img_size)
            label_list.append(torch.tensor([aug_way]))
            aug_list.append(under_aug)

        support_image = torch.cat(aug_list).to(device)
        # plot_img(support_image)
        support_label = torch.cat(label_list).to(device)

        new_classifier = nn.Linear(64*5*5, way_num)
        new_classifier = new_classifier.to(device)
        inner_optimizer = torch.optim.Adam(new_classifier.parameters(), lr=inner_lr, betas=(0.5, 0.999))

        # train new_classifier on support set
        print("="*10,'train on support set',"="*10, i)
        inner_loss = 0
        new_classifier.train()
        for inner_epoch in range(inner_iter):
            with torch.no_grad():
                embedded_support = emb_model(support_image)
                resized  = embedded_support.view(40,-1)    
            
            class_output = new_classifier(resized)
            loss = CELoss(class_output, support_label)
            inner_loss += loss.item()
            
            inner_optimizer.zero_grad()
            loss.backward(retain_graph=True)
            inner_optimizer.step()
            
        print('pretraining inner loss', inner_loss/inner_iter)
        loss_epoch += inner_loss/inner_iter
        
        # test on query set
        print('='*10,'test on query set','='*10, i)
        with torch.no_grad():
            embedded_test = emb_model(query_image)
            resized  = embedded_test.view(way_num * query_num,-1) 
            test_output = new_classifier(resized)
            cor = (test_output.argmax(dim=1).to(device)==query_label).float().sum().item()
            cor = cor/(way_num * query_num)
            print('query_accuracy',cor)
            cor_epoch += cor

    cor_epoch = cor_epoch/(i+1)
    loss_epoch = loss_epoch/(i+1)
    return cor_epoch, loss_epoch







preloss_list = []
precor_list = []
valcor_list = []
valloss_list = []
# test and val epochs
for epoch in range(train_epoch):
    if mode == 'baseline':
        precor, preloss = pre_train()
        valcor, valloss = fine_tuning(val_dld)
    
    if mode == 'vaean':
        precor, preloss = vae_pre_train(epoch)
        valcor, valloss = vae_fine_tuning(val_dld)
        
    if mode == 'aean':
        precor, preloss = pre_train()
        valcor, valloss = ae_fine_tuning(val_dld)

    precor_list.append(precor)
    preloss_list.append(preloss)
    valcor_list.append(valcor)
    valloss_list.append(valloss)


    scheduler_emb.step()
    scheduler_class.step()
    if mode == 'vaean':
        scheduler_decoder.step()
        
    print('epoch:[{}/{}]\t pretraing loss:{}\t pretraining accuracy:{}\t validation loss:{}\t validation accuracy:{}'.format(epoch+1, train_epoch, preloss, precor, valloss, valcor))



# test epochs
max_acc = 0 

# baseline test
if mode == 'baseline':
    for i in range(5):
        test_cor, testloss = fine_tuning(test_dld)
        if test_cor > max_acc:
            max_acc = test_cor


# vaean test
if mode == 'vaean':
    for i in range(5):
        test_cor, testloss = vae_fine_tuning(test_dld)
        if test_cor> max_acc:
            max_acc = test_cor

# aean test
if mode == 'aean':
    for i in range(5):
        test_cor, testloss = ae_fine_tuning(test_dld)
        if test_cor> max_acc:
            max_acc = test_cor

print('max_acc',max_acc)
print('max_val_cor:', max(valcor_list))

# visualization

fig, axs = plt.subplots(2, 2, figsize=(10,8))
axs[0, 0].plot(preloss_list)
axs[0, 0].set_title('preloss_list')
axs[0, 1].plot(precor_list)
axs[0, 1].set_title('precor_list')
axs[1, 0].plot(valloss_list)
axs[1, 0].set_title('valloss_list')
axs[1, 1].plot(valcor_list)
axs[1, 1].set_title('valcor_list')
if mode == 'baseline':
    plt.savefig('baseline.png')
if mode == 'vaean':
    plt.savefig('vaean.png')
if mode == 'aean':
    plt.savefig('aean.png')
plt.show()