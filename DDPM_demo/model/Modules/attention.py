import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(Attention, ch = 128) -> None:
        super(Attention, self).__init__()
        self.linear1 = nn.Linear(ch, ch)
        self.linear2 = nn.Linear(ch, ch)
        self.linear3 = nn.Linear(ch, ch)
        
        self.linearfinal = nn.Linear(ch, ch)
    
    def forward(self, x) -> torch.Tensor:
        b,c,h,w = x.size()
        """
        b 是批量大小（batch size）。
        c 是通道数（channels）。
        h 和 w 是输入特征图的高度和宽度。
        """
        xt = x.view(b,c,h*w).transpose(1,2)  # [b, h*w, c]
        K = self.linear1(xt)                 # [b, h*w, c]
        Q = self.linear2(xt)
        V = self.linear3(xt)
        Q = Q.view(b,-1,1,c).transpose(1,2)  # [b, 1, h*w, c]
        K = K.view(b,-1,1,c).transpose(1,2)
        V = V.view(b,-1,1,c).transpose(1,2)
        
        a = nn.functional.scaled_dot_product_attention(Q, K, V)
        a = a.transpose(1,2).reshape(b,-1,c)
        a = self.linearfinal(a)
        a = a.transpose(-1,-2).reshape(b,c,h,w)
        
        return a+x