import torch
from torch import nn
import functools, operator

class SpectralConv_1D(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super().__init__()
        self.modes1 = modes1
        self.out_channels = out_channels

        self.scale = (1 / (in_channels * out_channels))

        self.weights = nn.Parameter(self.scale * torch.randn((in_channels, out_channels, modes1), dtype=torch.cfloat)) # we don't initialize it directly as complex matrix (i.e. in cfloats) because it is not compatible with DataParallel.

    # Complex multiplication 2d
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("B I L, I O L-> B O L", input, weights)
    
    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x, dim=-1)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x_ft.size(-1), dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights)

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1), dim=-1)
        return x
    
class SpectralConv_2D(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.modes1, self.modes2 = modes1, modes2
        self.out_channels = out_channels

        self.scale = (1 / (in_channels * out_channels))

        self.weights = nn.Parameter(self.scale * torch.randn((in_channels, out_channels, modes1, modes2), dtype=torch.cfloat)) # we don't initialize it directly as complex matrix (i.e. in cfloats) because it is not compatible with DataParallel.
    # Complex multiplication 2d
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("B I H W, I O H W-> B O H W", input, weights)
    
    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x = x.to(torch.float32)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x_ft.size(-2), x_ft.size(-1), dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class SpectralConv_2D_diag(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        assert in_channels == out_channels, "Channel-diagonal spectral conv requires in_channels == out_channels"
        self.modes1, self.modes2 = modes1, modes2
        self.out_channels = out_channels

        self.scale = (1 / (in_channels))

        self.weights1 = nn.Parameter(self.scale * torch.randn((in_channels, modes1, modes2), dtype=torch.cfloat)) 
        self.weights2 = nn.Parameter(self.scale * torch.randn((in_channels, modes1, modes2), dtype=torch.cfloat))

    # Complex multiplication 2d
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("B C H W, C H W-> B C H W", input, weights)
    
    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x = x.to(torch.float32)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x_ft.size(-2), x_ft.size(-1), dtype=torch.cfloat, device=x.device)

        tmp1 = torch.zeros_like(out_ft)
        tmp2 = torch.zeros_like(out_ft)

        tmp1[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        tmp2[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        out_ft = tmp1 + tmp2
        
        # out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        # out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))

        # with torch.no_grad():
        #     s1 = x_ft[:, :, :self.modes1, :self.modes2]
        #     s2 = x_ft[:, :, -self.modes1:, :self.modes2]
        #     print("slice1 max", s1.abs().max().item(), "shape", tuple(s1.shape))
        #     print("slice2 max", s2.abs().max().item(), "shape", tuple(s2.shape))
        return x

class SpectralConv_3D_diag(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super().__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        assert in_channels == out_channels, "Depthwise spectral conv requires in_channels == out_channels" # To make it depthwise

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        #return torch.einsum("bixyz,ioxyz->boxyz", input, weights)
        return torch.einsum("bcxyz,cxyz->bcxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x
    
class HO_SpectralConv_2D(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, order):
        super().__init__()
        self.modes1, self.modes2 = modes1, modes2
        self.out_channels = out_channels

        self.scale = (1 / (in_channels * out_channels))

        self.weights = nn.Parameter(self.scale * torch.randn((in_channels, out_channels, modes1, modes2), dtype=torch.cfloat)) # we don't initialize it directly as complex matrix (i.e. in cfloats) because it is not compatible with DataParallel.

        self.As = nn.ModuleList(nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)
                                for _ in range(order))
        
        # self.A = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)
        # self.B = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)
        # self.C = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)

    # Complex multiplication 2d
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("B I H W, I O H W-> B O H W", input, weights)
    
    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        # x1 = self.A(x)
        # x2 = self.B(x)
        # Layer norm per channel to normalize the products?? Let's try it one day!!!
        
        #xs = [A(x) for A in self.As]

        # RMS prop
        xs = []

        for A in self.As:
            z = A(x)  # [B, C, H, W]
            rms = z.pow(2).mean(dim=(-2, -1), keepdim=True).sqrt() + 1e-9
            z = z / rms
            xs.append(z)
        

        # rms = x.pow(2).mean(dim=(-2, -1), keepdim=True).sqrt() + 1e-6
        # x = x / rms
        # xs = [A(x) for A in self.As]
 
        # x_prod = x1 * x2 
        x_prod = functools.reduce(operator.mul, xs)

        x_ft = torch.fft.rfft2(x_prod)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x_ft.size(-2), x_ft.size(-1), dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))

        return x
    
class HO_SpectralConv_1D(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, order):
        super().__init__()
        self.modes1 = modes1
        self.out_channels = out_channels

        self.scale = (1 / (in_channels * out_channels))

        self.weights = nn.Parameter(self.scale * torch.randn((in_channels, out_channels, modes1), dtype=torch.cfloat)) # we don't initialize it directly as complex matrix (i.e. in cfloats) because it is not compatible with DataParallel.

        self.As = nn.ModuleList(nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=True)
                                for _ in range(order))
        
        # self.A = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)
        # self.B = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)
        # self.C = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)

    # Complex multiplication 2d
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("B I L, I O L-> B O L", input, weights)
    
    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        # x1 = self.A(x)
        # x2 = self.B(x)
        # Layer norm per channel to normalize the products?? Let's try it one day!!!
        
        #xs = [A(x) for A in self.As]

        xs = []
        for A in self.As:
            z = A(x)  # [B, C, H, W]
            rms = z.pow(2).mean(dim=(-2, -1), keepdim=True).sqrt() #+ 1e-6
            z = z / rms
            xs.append(z)
        

        # rms = x.pow(2).mean(dim=(-2, -1), keepdim=True).sqrt() + 1e-6
        # x = x / rms
        # xs = [A(x) for A in self.As]
 
        # x_prod = x1 * x2 
        x_prod = functools.reduce(operator.mul, xs)

        x_ft = torch.fft.rfft(x_prod)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x_ft.size(-1), dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :self.modes1] = self.compl_mul2d(x_ft[:, :, :self.modes1], self.weights)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))

        return x
    



# class SpectralConv_2D(nn.Module):
#     def __init__(self, in_channels, out_channels, modes1, modes2):
#         super().__init__()
#         self.modes1, self.modes2 = modes1, modes2
#         self.out_channels = out_channels

#         self.scale = (1 / (in_channels * out_channels))

#         self.weights_real = nn.Parameter(self.scale * torch.randn((in_channels, out_channels, modes1, modes2))) #, dtype=torch.cfloat)) # we don't initialize it directly as complex matrix (i.e. in cfloats) because it is not compatible with DataParallel.
#         self.weights_imag = nn.Parameter(self.scale * torch.randn((in_channels, out_channels, modes1, modes2))) #, dtype=torch.cfloat))
#     # Complex multiplication 2d
#     def compl_mul2d(self, input, weights_real, weights_imag):
#         weights = torch.complex(weights_real, weights_imag)
#         # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
#         return torch.einsum("B I H W, I O H W-> B O H W", input, weights)
    
#     def forward(self, x):
#         orig_dtype = x.dtype
#         with torch.amp.autocast(device_type='cuda', enabled=False):
#             batchsize = x.shape[0]
#             # Compute Fourier coeffcients up to factor of e^(- something constant)
#             x = x.to(torch.float32)
#             x_ft = torch.fft.rfft2(x)

#             # Multiply relevant Fourier modes
#             out_ft = torch.zeros(batchsize, self.out_channels, x_ft.size(-2), x_ft.size(-1), dtype=torch.cfloat, device=x.device)
            
#             out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights_real, self.weights_imag)

#             # Return to physical space
#             x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))

#             x = x.to(orig_dtype)
#         return x
    
# class HO_SpectralConv_2D(nn.Module):
#     def __init__(self, in_channels, out_channels, modes1, modes2, order):
#         super().__init__()
#         self.modes1, self.modes2 = modes1, modes2
#         self.out_channels = out_channels

#         self.scale = (1 / (in_channels * out_channels))

#         self.weights_real = nn.Parameter(self.scale * torch.randn((in_channels, out_channels, modes1, modes2))) #, dtype=torch.cfloat)) # we don't initialize it directly as complex matrix (i.e. in cfloats) because it is not compatible with DataParallel.
#         self.weights_imag = nn.Parameter(self.scale * torch.randn((in_channels, out_channels, modes1, modes2)))

#         self.As = nn.ModuleList(nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)
#                                 for _ in range(order))
        
#         # self.A = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)
#         # self.B = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)
#         # self.C = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)

#     # Complex multiplication 2d
#     def compl_mul2d(self, input, weights_real, weights_imag):
#         weights = torch.complex(weights_real, weights_imag)
#         # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
#         return torch.einsum("B I H W, I O H W-> B O H W", input, weights)
    
#     def forward(self, x):
#         batchsize = x.shape[0]
#         # Compute Fourier coeffcients up to factor of e^(- something constant)
#         # x1 = self.A(x)
#         # x2 = self.B(x)
#         # Layer norm per channel to normalize the products?? Let's try it one day!!!
        
#         #xs = [A(x) for A in self.As]

#         xs = []
#         for A in self.As:
#             z = A(x)  # [B, C, H, W]
#             rms = z.pow(2).mean(dim=(-2, -1), keepdim=True).sqrt() + 1e-6
#             z = z / rms
#             xs.append(z)
        

#         # rms = x.pow(2).mean(dim=(-2, -1), keepdim=True).sqrt() + 1e-6
#         # x = x / rms
#         # xs = [A(x) for A in self.As]
 
#         # x_prod = x1 * x2 
#         x_prod = functools.reduce(operator.mul, xs)
#         orig_dtype = x_prod.dtype
#         with torch.amp.autocast(device_type='cuda', enabled=False):
#             x_prod.to(torch.float32)
#             x_ft = torch.fft.rfft2(x_prod)

#             # Multiply relevant Fourier modes
#             out_ft = torch.zeros(batchsize, self.out_channels, x_ft.size(-2), x_ft.size(-1), dtype=torch.cfloat, device=x.device)
            
#             out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights_real, self.weights_imag)

#             # Return to physical space
#             x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
#             x.to(orig_dtype)
#         return x
