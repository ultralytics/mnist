# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os

import torch


def init_seeds(seed=0):
    """Initialize random seeds for reproducibility and set CUDA deterministic behavior."""
    torch.manual_seed(seed)

    # Remove randomness (may be slower on Tesla GPUs) # https://pytorch.org/docs/stable/notes/randomness.html
    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_device(device="", apex=False, batch_size=None):
    """Select and configure the computational device (CPU or CUDA GPUs) for PyTorch operations."""
    cpu_request = device.lower() == "cpu"
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ["CUDA_VISIBLE_DEVICES"] = device  # set environment variable
        assert torch.cuda.is_available(), f"CUDA unavailable, invalid device {device} requested"

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024**2  # bytes to MB
        ng = torch.cuda.device_count()
        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % ng == 0, f"batch-size {batch_size:g} not multiple of GPU count {ng:g}"
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = "Using CUDA " + ("Apex " if apex else "")  # apex for mixed precision https://github.com/NVIDIA/apex
        for i in range(ng):
            if i == 1:
                s = " " * len(s)
            print(
                "%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)"
                % (s, i, x[i].name, x[i].total_memory / c)
            )
    else:
        print("Using CPU")

    print("")  # skip a line
    return torch.device("cuda:0" if cuda else "cpu")


def fuse_conv_and_bn(conv, bn):
    """
    Fuses a convolutional layer and a batch normalization layer into a single convolutional layer.

    https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    """
    with torch.no_grad():
        # init
        fusedconv = torch.nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=True,
        )

        # prepare filters
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.size()))

        # prepare spatial bias
        b_conv = torch.zeros(conv.weight.size(0)) if conv.bias is None else conv.bias
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

        return fusedconv


def model_info(model, report="summary"):
    """Prints a summary or full report of a PyTorch model's layers, parameters, gradients, shapes, mean, and standard
    deviation.
    """
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if report == "full":
        print("%5s %40s %9s %12s %20s %10s %10s" % ("layer", "name", "gradient", "parameters", "shape", "mu", "sigma"))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace("module_list.", "")
            print(
                "%5g %40s %9s %12g %20s %10.3g %10.3g"
                % (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std())
            )
    print(f"Model Summary: {len(list(model.parameters())):g} layers, {n_p:g} parameters, {n_g:g} gradients")


def load_classifier(name="resnet101", n=2):
    """Loads a pretrained model reshaped to n-class output using the specified model architecture."""
    import pretrainedmodels  # https://github.com/Cadene/pretrained-models.pytorch#torchvision

    model = pretrainedmodels.__dict__[name](num_classes=1000, pretrained="imagenet")

    # Display model properties
    for x in ["model.input_size", "model.input_space", "model.input_range", "model.mean", "model.std"]:
        print(f"{x} =", eval(x))

    # Reshape output to n classes
    filters = model.last_linear.weight.shape[1]
    model.last_linear.bias = torch.nn.Parameter(torch.zeros(n))
    model.last_linear.weight = torch.nn.Parameter(torch.zeros(n, filters))
    model.last_linear.out_features = n
    return model
