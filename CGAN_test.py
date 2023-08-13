import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

device = torch.device('cuda:0')
gen = torch.load('CGAN_epoch_generator.pth')
gen.eval()
new_noise  =  ((torch.rand(25, 128) - 0.5 ) / 0.5).to(device)
array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4]).astype(np.int64)
new_fake_number  = torch.from_numpy(array)
new_fake_number = new_fake_number.to(device)
new_fake_number = F.one_hot(new_fake_number, num_classes = 10).to(device)
new_fake_number = torch.reshape(new_fake_number, (25, 10))
new_fake_number_label = new_fake_number.type(torch.FloatTensor).to(device)
test_out = gen(new_noise, new_fake_number_label).cpu()
test_img = (torch.reshape(test_out, (25, 28, 28))).detach().numpy()
plt.suptitle("CGAN")
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.axis('off')
    plt.imshow(test_img[i], cmap='gray_r')
plt.show()