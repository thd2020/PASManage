import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    2D image patch embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, norm_layer=None, flatten=True):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.embed_dim = patch_size**2 * in_chans
        self.flatten = flatten

        self.project = nn.Conv2d(in_chans, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.norm = norm_layer(self.embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size does not match the model!"
        x = self.project(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4., drop_rate=0.):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=drop_rate)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(drop_rate)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.mlp(self.ln2(x))
        return x


if __name__ == "__main__":
    patchify = PatchEmbedding(img_size=448, in_chans=3)
    dummy_img = torch.rand(3, 3, 448, 448)
    patch = patchify.forward(dummy_img)
    print(patch.shape)
