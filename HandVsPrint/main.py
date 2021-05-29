import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models.shufflenetv2 import shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0
from torchvision.models.mnasnet import mnasnet0_5, mnasnet0_75, mnasnet1_0, mnasnet1_3
from torchvision.models.resnet import resnet18

from data import dataloader, test_loader
from torchvision.transforms.functional import to_pil_image

save_path = './shufflenetv2_05.pth'
# save_path = './mnasnet_05.pth'
# save_path = './mnasnet_10.pth'
# save_path = './resnet18.pth'

network = shufflenet_v2_x0_5(num_classes=3)
network.conv1[0] = nn.Conv2d(1, 24, 3, 2, 1, bias=False)

# network = mnasnet0_5(num_classes=3)  # Acc. 76 %
# network.layers[0] = nn.Conv2d(1, 16, 3, padding=1, stride=2, bias=False)  # 05
# network = mnasnet1_0(num_classes=3)  # Acc. 81.5%
# network.layers[0] = nn.Conv2d(1, 32, 3, padding=1, stride=2, bias=False)  # 10
# network = resnet18(num_classes=3)  # Acc. 100 %
# network.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)


def train(dataloader):
    network.cuda()
    n_epoch = 20
    loss_fn = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(network.parameters(), lr=1e-3)

    for epoch in range(n_epoch):
        for i, data in enumerate(dataloader):
            img, label = data

            # save samples
            # print(label)
            # n = img.shape[0]
            # for i in range(n):
            #     ex = img[i]
            #     ex = to_pil_image(ex)
            #     ex.save('./temp/{}.png'.format(i))
            # quit()

            logit = network(img.cuda())
            loss = loss_fn(logit, label[:, 0].cuda())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 250 == 0:
                print(i, loss.item())
                torch.save(network.state_dict(), save_path)
    # print(loss.item())
    torch.save(network.state_dict(), save_path)


def validate(dataloader):
    network.eval().cuda()
    n_correct = 0
    total_n = 0
    for i, data in enumerate(dataloader):
        img, label = data
        with torch.no_grad():
            logit = network(img.cuda())
            b = logit.shape[0]
            _, pred = logit.topk(1, 1, True, True)
            pred = pred.t()
            correct = pred.eq(label.cuda().view(1, -1).expand_as(pred))

            correct_k = correct[:1].reshape(-1).float().sum(0, keepdim=True)
            n_correct = n_correct + correct_k
            total_n = total_n + b

    acc = (n_correct / total_n) * 100
    network.train()

    return acc.item()


if __name__ == '__main__':
    # train
    train(dataloader)

    # validate
    network.load_state_dict(torch.load(save_path))
    acc = validate(test_loader)
    print('Acc: {}%'.format(acc))
