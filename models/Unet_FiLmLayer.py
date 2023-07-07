
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Sequence, Dict, Union, Callable
import torchvision


def one_param(m):
    "get model first parameter"
    return next(iter(m.parameters()))

# Padding for images to be divisible by 2^depth
def pad_to(x, stride):
    h, w = x.shape[-2:]

    if h % stride > 0:
        new_h = h + stride - h % stride
    else:
        new_h = h
    if w % stride > 0:
        new_w = w + stride - w % stride
    else:
        new_w = w
    lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
    pads = (lw, uw, lh, uh)

    # zero-padding by default.
    # See others at https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.pad
    out = F.pad(x, pads, "constant", 0)

    return out, pads

def unpad(x, pad):
    if pad[2]+pad[3] > 0:
        x = x[:,:,pad[2]:-pad[3],:]
    if pad[0]+pad[1] > 0:
        x = x[:,:,:,pad[0]:-pad[1]]
    return x


class SelfAttention(nn.Module):
    """
    Transformer Structure:
    
    Attention is all you need paper (https://arxiv.org/abs/1706.03762): 
        See the diagram of the transformer architecture (example: the encoder)

    1. Multihead Attention 
    2-  Normalization
    3- Feed Forward Network 
    4-  Normalization
    """
    def __init__(self, channels):
        super().__init__()
        self.label_size = 1
        self.channels = channels        
        self.attention = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )



    def forward(self, x: torch.Tensor):
        signal_len = x.shape[-1] * x.shape[-2] # Sequence length

        x_tmp = x.contiguous().view(-1, self.channels, signal_len).swapaxes(1, 2) # View(): reshapes the tensor to the desired shape
            # -1: infer this dimension from the other given dimension; Preserve number of batches
            # swapaxes(1, 2): swap the second and third dimension -> (B, C, len) -> (B, len, C)

        x_ln = self.ln(x_tmp) # Normalize input
        attention_value, _ = self.attention(x_ln, x_ln, x_ln) #Multihead attention: Pytorch Implementation
        attention_value = attention_value + x_tmp #Add residual connection (See paper; we add the input to the output of the multihead attention)
        attention_value = self.ff_self(attention_value) + attention_value #Second residual connection (see paper)
        return attention_value.swapaxes(2, 1).contiguous().view(-1, self.channels, x.shape[-2], x.shape[-1]) #Swap back the second and third dimension and reshape to original image


class DoubleConvolution(nn.Module):
    """
    Structure taken from original UNet paper (https://arxiv.org/abs/1505.04597)
    Adjusted to fit implementation of DDPM (https://arxiv.org/abs/2006.11239) 

    Removed internal residual connections, coud not be bothered to implement them
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        :param in_channels: is the number of input channels
        :param out_channels: is the number of output channels
        """
        super().__init__()

        # First 3x3 convolutional layer
        self.first = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False) #Takes inputs of (B,Cin,H,W) where B is batch size, Cin is input channels, H is height, W is width  
        # Second 3x3 convolutional layer
        self.second = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.act = nn.GELU()
        self.norm = nn.GroupNorm(1, out_channels)


    def forward(self, x: torch.Tensor):

        # Apply the two convolution layers and activations
        x = self.first(x)   # (B,Cin,H,W) -> (B,Cout,H,W)
        x = self.norm(x)    # Group normalization
        x = self.act(x)     # GELU activation function (https://arxiv.org/abs/1606.08415)
        x = self.second(x)  # (B,Cin,H,W) -> (B,Cout,H,W)
        return self.norm(x) # Group normalization Final output shape (B,Cout,H,W)


class DownSample(nn.Module):
    """
    ### Down-sample

    Each step in the contracting path down-samples the feature map with
    a 2x2 max pooling operation with stride 2. 
    Two Double Convolution layers are applied to the feature map before each down-sampling operation. 
        (effectively doubling the number of channels)
    
    """

    def __init__(self, in_channels: int, out_channels: int, embeddedTime_dim=256, cond_dim = None):
        super().__init__()
        # Max pooling layer
        self.pool = nn.MaxPool2d(2, ) #2x2 max pooling windows -> reduce size of feature map by 2
        self.doubleConv1 = DoubleConvolution(in_channels, in_channels)
        self.doubleConv2 = DoubleConvolution(in_channels, out_channels)

        self.emb_layer = nn.Sequential( # Brutally make dimensions match using a linear layer
            nn.SiLU(),
            nn.Linear(
                embeddedTime_dim, # IN: Dimension of embedded "denoising timestep" (B, 256)
                out_channels # OUT: Number of channels of the image (B, Channels_out)
            ),
        ) # Trainable layer: Not sure how okay this is, some repo's do it, some don't

        if cond_dim is not None:
            # FiLM modulation https://arxiv.org/abs/1709.07871
            # predicts per-channel scale and bias
            cond_channels = out_channels * 2
            self.out_channels = out_channels
            self.cond_encoder = nn.Sequential( # This is the FiLM modulation
                nn.Mish(),
                nn.Flatten(1, -1), # Flatten all dimensions except batch dimension (B, cond_dim
                nn.Linear(cond_dim, cond_channels), # cond_dim is the dimension of the global conditioning vector -- becomes 2*output_channels
                nn.Unflatten(-1, (-1, 1)) # Unflatten the last dimension to be (batch_size, 1, cond_channels)
            )

        

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor = None):
        x = self.pool(x)
        x = self.doubleConv1(x)
        x = self.doubleConv2(x) # (B, C_out, signal_height, signal_width)



        emb_t = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1]) # self.emb_layer(t) -> (B, C_out, 1, 1) 
                                                                                            #-> repeat to match image dimensions (B, C_out, img_s, img_s)
                                                                                            # -> same "time" value for all pixels
        x = x + emb_t
        if cond is not None:
            # cond in as (B, obs_horizon * obs_dim)
            emb_c = self.cond_encoder(cond)
            emb_c = emb_c.view(emb_c.shape[0], 2, -1) # (B, 2, C_out) -- Halfed channel back to original c_out

            scale = emb_c[:,0,...].view(emb_c.shape[0], self.out_channels, 1, 1)
            bias = emb_c[:,1,...].view(emb_c.shape[0], self.out_channels, 1, 1)

            x = scale * x + bias
        
        return x  # (B, C_out, signal_height, signal_width)
        


class UpSample(nn.Module):
    """
    ### Up-sample
    """
    def __init__(self, in_channels: int, out_channels: int, embeddedTime_dim=256, cond_dim = None):
        super().__init__()

        # Up-convolution
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.doubleConv1 = DoubleConvolution(in_channels, in_channels)
        self.doubleConv2 = DoubleConvolution(in_channels, out_channels)

        self.emb_layer = nn.Sequential( # Brutally make dimensions match unsing a linear layer
            nn.SiLU(),
            nn.Linear(
                embeddedTime_dim,
                out_channels
            ),
        )

        if cond_dim is not None:
            # FiLM modulation https://arxiv.org/abs/1709.07871
            # predicts per-channel scale and bias
            cond_channels = out_channels * 2
            self.out_channels = out_channels
            self.cond_encoder = nn.Sequential( # This is the FiLM modulation
                nn.Mish(),
                nn.Flatten(1, -1), # Flatten all dimensions except batch dimension (B, cond_dim
                nn.Linear(cond_dim, cond_channels), # cond_dim is the dimension of the global conditioning vector -- becomes 2*output_channels
                nn.Unflatten(-1, (-1, 1)) # Unflatten the last dimension to be (batch_size, 1, cond_channels)
            )


    def forward(self, x: torch.Tensor, x_res: torch.Tensor, t: torch.Tensor, cond: torch.Tensor = None):
        x = self.up(x)
        x = torch.cat([x, x_res], dim=1)# Concatenate along the channel dimension; kept previous feature map and upsampled feature map

        x = self.doubleConv1(x)
        x = self.doubleConv2(x)
        emb_t = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])

        x = x + emb_t # (B, C_out, signal_height, signal_width)
        
        if cond is not None:
            # cond in as (B, obs_horizon * obs_dim)
            emb_c = self.cond_encoder(cond)
            emb_c = emb_c.view(emb_c.shape[0], 2, -1) # (B, 2, C_out) -- Halfed channel back to original c_out

            scale = emb_c[:,0,...].view(emb_c.shape[0], self.out_channels, 1, 1)
            bias = emb_c[:,1,...].view(emb_c.shape[0], self.out_channels, 1, 1)

            x = scale * x + bias

        
        return x
    

class UNet_Film(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, noise_steps: int, time_dim=256, global_cond_dim=None):
        super(UNet_Film, self).__init__()
        self.time_dim = time_dim

        # Define all layers used by U-net
        self.inc = DoubleConvolution(in_channels, 64)
        self.down1 = DownSample(64, 128, cond_dim=global_cond_dim) #set time_dim to 256 for all up and down sampling layers in init()
        self.sa1 = SelfAttention(128)
        self.down2 = DownSample(128, 256 ,cond_dim=global_cond_dim)
        self.sa2 = SelfAttention(256)
        self.down3 = DownSample(256, 256 ,cond_dim=global_cond_dim)
        self.sa3 = SelfAttention(256)

        self.bot1 = DoubleConvolution(256, 512)
        self.bot2 = DoubleConvolution(512, 512)
        self.bot3 = DoubleConvolution(512, 256)

        self.up1 = UpSample(512, 128 ,cond_dim=global_cond_dim)
        self.sa4 = SelfAttention(128)
        self.up2 = UpSample(256, 64 ,cond_dim=global_cond_dim)
        self.sa5 = SelfAttention(64)
        self.up3 = UpSample(128, 64 ,cond_dim=global_cond_dim)
        self.sa6 = SelfAttention(64)
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2,device=t.device) / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc


    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor = None):

        # y is a optional conditioning input

        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim) # Do the encoding
        
        # Check if the input tensor has the 2^3 divisible image size (ie downsampling 3 times)
        # include residual connections
        x, padding = pad_to(x, 2**3)

        x1 = self.inc(x)
        
        x2 = self.down1(x1, t, y)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t, y)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t, y)
        x4 = self.sa3(x4)

        x5 = self.bot1(x4)
        x5 = self.bot2(x5)
        x5 = self.bot3(x5)

        x = self.up1(x5, x3, t, y ) # include residual connections
        x = self.sa4(x)
        x = self.up2(x, x2, t , y)
        x = self.sa5(x)
        x = self.up3(x, x1, t , y)
        x = self.sa6(x)

        x = self.outc(x)

        x = unpad(x , padding)

        return x


# Vision Encoder:
def get_resnet(name:str, weights=None, **kwargs) -> nn.Module:
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", None
    """
    # Use standard ResNet implementation from torchvision
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)

    # remove the final fully connected layer
    # for resnet18, the output dim should be 512
    resnet.fc = torch.nn.Identity()
    return resnet


def replace_submodules(
        root_module: nn.Module, 
        predicate: Callable[[nn.Module], bool], 
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module

def replace_bn_with_gn(
    root_module: nn.Module, 
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group, 
            num_channels=x.num_features)
    )
    return root_module

def VisionEncoder():
    vision_encoder = get_resnet('resnet18')
    vision_encoder = replace_bn_with_gn(vision_encoder)
    return vision_encoder