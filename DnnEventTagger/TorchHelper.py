from __future__ import print_function
import torch

def init_tagger(filename):
    global net, device
    import torch
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    import DnnEventTagger.ResNet as ResNet
    print('==> Building model..')
    net = ResNet.resnet18(pretrained=False, num_classes=2, input_channels=3)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        torch.backends.cudnn.benchmark = True
    net = net.to(device)
    return True # I don't have data
    print('==> Load the network from checkpoint..')
    checkpoint = torch.load(filename)
    # Use trained network to tag event.
    net.load_state_dict(checkpoint['net'])
    return True

def fini_tagger():
    global net, device
    return True

def tag_event(image_np):
    global net, device
    print('tag net=', net)
    #return 0. # skip actual work
    from torchvision import transforms
    import torch.nn as nn
    net.eval()
    #image_np = np.array(Image.open(filename), dtype=np.float32)
    image_tensor = torch.unsqueeze(transforms.functional.to_tensor(image_np), 0)
    softmax = nn.Softmax()
    with torch.no_grad():
        output = net(image_tensor.to(device))
        return softmax(output)[0][1].item()
    return 0.

def save_image(image, filename):
    from PIL import Image
    Image.fromarray(image,'RGB').save(filename)

