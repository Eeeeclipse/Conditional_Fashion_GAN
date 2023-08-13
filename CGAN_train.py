import CGAN_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import data_loader

device  =  torch.device('cuda:0')
dis = CGAN_model.discriminator().to(device) 
gen = CGAN_model.generator().to(device)
g_optimizer = optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
d_optimizer = optim.Adam(dis.parameters(), lr=0.0002, betas=(0.5, 0.999))
batch_size = 50
trainloader, testloader = data_loader.load(batch_size)
train_data = list(trainloader)
criterion = nn.BCELoss()
epochs = 200
for epoch in range(epochs):
    epoch += 1
    for data in train_data:
        inputs = data[0].to(device)
        label = (data[1]).to(device)
        real_number_label = F.one_hot(label, num_classes = 10).to(device)
        real_number_label = real_number_label.type(torch.FloatTensor).to(device)
        outputs = dis(inputs, real_number_label)
        real_label  =  torch.ones(batch_size, 1).to(device)
        noise  =  ((torch.rand(batch_size, 128) - 0.5 ) / 0.5).to(device)
        fake_number  = torch.randint(10, (batch_size, 1)).to(device)
        fake_number_label = F.one_hot(fake_number, num_classes = 10).to(device)
        fake_number_label = torch.reshape(fake_number_label, (batch_size, 10))
        fake_number_label = fake_number_label.type(torch.FloatTensor).to(device)
        fake_img = gen(noise, fake_number_label)
        fake_outputs = dis(fake_img, fake_number_label)
        fake_label = torch.zeros(batch_size, 1).to(device)
        outputs = torch.cat((outputs, fake_outputs), 0)
        labels  =  torch.cat((real_label, fake_label), 0)
        d_optimizer.zero_grad()
        d_loss = criterion(outputs, labels)
        d_loss.backward()
        d_optimizer.step()
        g_optimizer.zero_grad()
        new_noise  =  ((torch.rand(batch_size, 128) - 0.5 ) / 0.5).to(device)
        new_fake_number  = torch.randint(10, (batch_size, 1)).to(device)
        new_fake_number_label = F.one_hot(new_fake_number, num_classes = 10).to(device)
        new_fake_number_label = torch.reshape(new_fake_number_label, (batch_size, 10))
        new_fake_number_label = new_fake_number_label.type(torch.FloatTensor).to(device)
        new_fake_img = gen(new_noise, new_fake_number_label)
        new_fake_outputs = dis(new_fake_img, new_fake_number_label)
        new_fake_label = torch.ones(batch_size, 1).to(device)
        g_loss = criterion(new_fake_outputs, new_fake_label)
        g_loss.backward()
        g_optimizer.step()
    print(str(epoch) + "     G:" + str(g_loss.item()) + "     D:" + str(d_loss.item()))
torch.save(gen, 'CGAN_epoch_generator.pth')