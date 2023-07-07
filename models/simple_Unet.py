
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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



    def forward(self, x: torch.Tensor, y: torch.Tensor = None):
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

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None, residual: bool = False):
        """
        :param in_channels: is the number of input channels
        :param out_channels: is the number of output channels
        """
        super(DoubleConvolution, self).__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels

        # First 3x3 convolutional layer
        self.first = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False) #Takes inputs of (B,Cin,H,W) where B is batch size, Cin is input channels, H is height, W is width  
        # Second 3x3 convolutional layer
        self.second = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.act = nn.GELU()
        self.norm = nn.GroupNorm(1, out_channels)


    def forward(self, x: torch.Tensor):
        
        x_res = x
        # Apply the two convolution layers and activations
        x = self.first(x)   # (B,Cin,H,W) -> (B,Cout,H,W)
        x = self.norm(x)    # Group normalization
        x = self.act(x)     # GELU activation function (https://arxiv.org/abs/1606.08415)
        x = self.second(x)  # (B,Cin,H,W) -> (B,Cout,H,W)
        x =  self.norm(x) # Group normalization Final output shape (B,Cout,H,W)

        if self.residual:
            return F.gelu(x + x_res)
        return F.gelu(x)


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
        self.cond_dim = cond_dim
        self.pool = nn.MaxPool2d(2,2) #2x2 max pooling windows -> reduce size of feature map by 2
        self.doubleConv1 = DoubleConvolution(in_channels, in_channels, residual=True)
        self.doubleConv2 = DoubleConvolution(in_channels, out_channels)


        self.emb_layer = nn.Sequential( # Brutally make dimensions match using a linear layer
            nn.SiLU(),
            nn.Linear(
                embeddedTime_dim, # IN: Dimension of embedded "denoising timestep" (B, 256)
                out_channels # OUT: Number of channels of the image (B, Channels_out)
            ),
        ) # Trainable layer: Not sure how okay this is, some repo's do it, some don't

        if cond_dim is not None:
            self.cond_emb_layer = nn.Sequential(
                nn.SiLU(),
                nn.Linear(in_features= cond_dim, out_features=32),
            )

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond = None):
        x = self.pool(x)
        x = self.doubleConv1(x)
        x = self.doubleConv2(x)

        if len(t.shape) == 1:
            t = t[None, :]
        emb_t = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1]) # self.emb_layer(t) -> (B, C_out, 1, 1) 
                                                                                            #-> repeat to match image dimensions (B, C_out, img_s, img_s)
                                                                                            # -> same "time" value for all pixels
        x =  x + emb_t
        
        if cond is not None:
            # Add conditional embedding
            cond_emb = self.cond_emb_layer(cond.view(cond.shape[0], -1))
            cond_emb = cond_emb.view(cond_emb.shape[0], cond_emb.shape[1], 1, 1).repeat(1, 1, x.shape[-2], x.shape[-1])
            x = torch.cat([x,cond_emb],dim=1)
        return x



class UpSample(nn.Module):
    """
    ### Up-sample
    """
    def __init__(self, in_channels: int, out_channels: int, embeddedTime_dim=256, cond_dim = None):
        super().__init__()

        # Up-convolution
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.doubleConv1 = DoubleConvolution(in_channels=in_channels, out_channels=in_channels, residual=True)
        self.doubleConv2 = DoubleConvolution(in_channels=in_channels, out_channels=out_channels)


        self.emb_layer = nn.Sequential( # Brutally make dimensions match unsing a linear layer
            nn.SiLU(),
            nn.Linear(
                embeddedTime_dim,
                out_channels
            ),
        )

        if cond_dim is not None:
            self.cond_emb_layer = nn.Sequential(
                nn.SiLU(),
                nn.Linear(in_features= cond_dim, out_features=32),
            )

    def forward(self, x: torch.Tensor, x_res: torch.Tensor, t: torch.Tensor, cond = None):
        x = self.up(x)
        x = torch.cat([x, x_res], dim=1)# Concatenate along the channel dimension; kept previous feature map and upsampled feature map

        x = self.doubleConv1(x)
        x = self.doubleConv2(x)
        
        if len(t.shape) == 1:
            t = t[None, :]
        emb_t = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        x =  x + emb_t
        
        if cond is not None:
            # Add conditional embedding
            cond_emb = self.cond_emb_layer(cond.view(cond.shape[0], -1))
            cond_emb = cond_emb.view(cond_emb.shape[0], cond_emb.shape[1], 1, 1).repeat(1, 1, x.shape[-2], x.shape[-1])
            x = torch.cat([x,cond_emb],dim=1)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, dropout: float = 0.1, max_len: int = 1000, apply_dropout: bool = True):
        """Section 3.5 of attention is all you need paper.

        Extended slicing method is used to fill even and odd position of sin, cos with increment of 2.
        Ex, `[sin, cos, sin, cos, sin, cos]` for `embedding_dim = 6`.

        `max_len` is equivalent to number of noise steps or patches. `embedding_dim` must same as image
        embedding dimension of the model.

        Args:
            embedding_dim: `d_model` in given positional encoding formula.
            dropout: Dropout amount.
            max_len: Number of embeddings to generate. Here, equivalent to total noise steps.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.apply_dropout = apply_dropout

        pos_encoding = torch.zeros(max_len, embedding_dim)
        position = torch.arange(start=0, end=max_len).unsqueeze(1)
        div_term = torch.exp(-math.log(10000.0) * torch.arange(0, embedding_dim, 2).float() / embedding_dim)

        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer(name='pos_encoding', tensor=pos_encoding)

    def forward(self, t: torch.LongTensor) -> torch.Tensor:
        positional_encoding = self.pos_encoding[t].squeeze(-1)
        if self.apply_dropout:
            return self.dropout(positional_encoding)
        return positional_encoding


class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, noise_steps: 1000,  time_dim=256, global_cond_dim = None):
        super(UNet, self).__init__()

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.noise_steps  = noise_steps
        self.time_dim     = time_dim
        self.global_cond_dim = global_cond_dim
        self.pos_encoding = PositionalEncoding(embedding_dim=time_dim, max_len=self.noise_steps+1)

        self.input_conv     = DoubleConvolution(in_channels, 16) # 16
        self.down1          = DownSample(16, 32, cond_dim=global_cond_dim) # 32 + 32 = 64
        self.down2          = DownSample(64, 128, cond_dim=global_cond_dim) # 128 + 32 = 160
        self.down3          = DownSample(160 , 256, cond_dim=global_cond_dim) # 256 + 32 = 288
        #self.down4          = DownSample(64+32, 128 // factor, cond_dim=global_cond_dim)
        self.up1            = UpSample (288 + 160, 128,cond_dim=global_cond_dim) #128 + 32 = 160
        self.up2            = UpSample (160 + 64, 64,cond_dim=global_cond_dim) # 64 + 32 = 96
        self.up3            = UpSample (96+16, 32 , cond_dim=global_cond_dim) # 32 + 32 = 64
        #self.up4            = UpSample (32+32, 16, cond_dim=global_cond_dim)
        self.outc           = nn.Conv2d(in_channels = 64, out_channels = out_channels, kernel_size=(1, 1))


    #     self.in_channels = in_channels
    #     self.out_channels = out_channels
    #     self.time_dim = time_dim

    #     # Define all layers used by U-net
    #     self.inc = DoubleConvolution(in_channels, 64)
    #     self.down1 = DownSample(64, 128) #set time_dim to 256 for all up and down sampling layers in init()
    #     self.sa1 = SelfAttention(128)
    #     self.down2 = DownSample(128, 256)
    #     self.sa2 = SelfAttention(256)
    #     self.down3 = DownSample(256, 256)
    #     self.sa3 = SelfAttention(256)

    #     self.bot1 = DoubleConvolution(256, 512)
    #     self.bot2 = DoubleConvolution(512, 512)
    #     self.bot3 = DoubleConvolution(512, 256)

    #     self.up1 = UpSample(512, 128)
    #     self.sa4 = SelfAttention(128)
    #     self.up2 = UpSample(256, 64)
    #     self.sa5 = SelfAttention(64)
    #     self.up3 = UpSample(128, 64)
    #     self.sa6 = SelfAttention(64)
    #     self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    # def pos_encoding(self, t, channels):
    #     inv_freq = 1.0 / (
    #         10000
    #         ** (torch.arange(0, channels, 2,device=t.device) / channels)
    #     )
    #     pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
    #     pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
    #     pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
    #     return pos_enc


    def forward(self, x: torch.Tensor, t:  torch.LongTensor, y: torch.Tensor = None):

        x, padding = pad_to(x, 2**3)


        t = self.pos_encoding(t)

        x1 = self.input_conv(x)
        x2 = self.down1(x1, t, y)
        x3 = self.down2(x2, t, y)
        x4 = self.down3(x3, t, y) 
        #x5 = self.down4(x4, t, y)
        x = self.up1(x4, x3, t, y)
        x = self.up2(x, x2, t, y)
        x = self.up3(x, x1, t, y)
        #x = self.up4(x, x1, t, y)

        logits = self.outc(x)
        logits = unpad(logits , padding)

        return logits

        # y is a optional conditioning input
        # t = t.unsqueeze(-1).type(torch.float)
        # t = self.pos_encoding(t, self.time_dim) # Do the encoding
        
        # # Check if the input tensor has the 2^3 divisible image size (ie downsampling 3 times)
        # # include residual connections
        # x, padding = pad_to(x, 2**3)

        # x1 = self.inc(x)
        
        # x2 = self.down1(x1, t)
        # x2 = self.sa1(x2, y)
        # x3 = self.down2(x2, t)
        # x3 = self.sa2(x3, y)
        # x4 = self.down3(x3, t)
        # x4 = self.sa3(x4, y)

        # x5 = self.bot1(x4)
        # x5 = self.bot2(x5)
        # x5 = self.bot3(x5)

        # x = self.up1(x5, x3, t) # include residual connections
        # x = self.sa4(x, y)
        # x = self.up2(x, x2, t)
        # x = self.sa5(x, y)
        # x = self.up3(x, x1, t)
        # x = self.sa6(x, y)

        # x = self.outc(x)

        # x = unpad(x , padding)

        #return x

