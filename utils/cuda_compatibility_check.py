import torch
from torch.cuda import device_count, get_device_capability

def is_cuda_compatible():
    """
    Checks for the compatibility of the system GPU architecture capability (a.k.a sm-6x, sm-7x, sm-8x) with the 
    current installation of torch.
    Output:
    - True: torch installation compatible
    - False: incompatible torch installation, refer to https://pytorch.org/get-started/locally/
    """
    compatible_device_count = 0
    if torch.version.cuda is not None:
        for d in range(device_count()):
            capability = get_device_capability(d)
            major = capability[0]
            minor = capability[1]
            current_arch = major * 10 + minor
            min_arch = min((int(arch.split("_")[1]) for arch in torch.cuda.get_arch_list()), default=35)
            if (not current_arch < min_arch
                    and not torch._C._cuda_getCompiledVersion() <= 9000):
                compatible_device_count += 1

    if compatible_device_count > 0:
        return True
    return False

is_cuda_compatible()
