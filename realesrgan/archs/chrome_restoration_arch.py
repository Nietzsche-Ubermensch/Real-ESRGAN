from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple

import torch
from basicsr.utils.registry import ARCH_REGISTRY
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm


@dataclass
class ChromeRestorationConfig:
    in_channels: int = 3
    out_channels: int = 3
    base_channels: int = 64
    growth_channels: int = 32
    num_rrdb_blocks: int = 8
    num_holo_layers: int = 4
    holographic_dim: int = 32
    scratch_channels: int = 16
    edge_channels: int = 16
    upscale_factor: int = 2
    use_spectral_norm: bool = False
    use_perceptual_loss: bool = False
    perceptual_pretrained: bool = False
    chrome_aware_loss_weight: float = 2.0
    holo_loss_weight: float = 1.5
    edge_loss_weight: float = 1.0


def default_conv(in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 bias=True,
                 use_sn=False):
    padding = (kernel_size // 2) * dilation
    layer = nn.Conv2d(
        in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
    return spectral_norm(layer) if use_sn else layer


class ResidualDenseBlock(nn.Module):

    def __init__(self, channels: int, growth_channels: int = 32, use_sn: bool = False):
        super().__init__()
        self.conv1 = default_conv(channels, growth_channels, 3, use_sn=use_sn)
        self.conv2 = default_conv(channels + growth_channels, growth_channels, 3, use_sn=use_sn)
        self.conv3 = default_conv(channels + 2 * growth_channels, growth_channels, 3, use_sn=use_sn)
        self.conv4 = default_conv(channels + 3 * growth_channels, growth_channels, 3, use_sn=use_sn)
        self.conv5 = default_conv(channels + 4 * growth_channels, channels, 3, use_sn=use_sn)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.beta = 0.2

    def forward(self, x):
        c1 = self.lrelu(self.conv1(x))
        c2 = self.lrelu(self.conv2(torch.cat([x, c1], dim=1)))
        c3 = self.lrelu(self.conv3(torch.cat([x, c1, c2], dim=1)))
        c4 = self.lrelu(self.conv4(torch.cat([x, c1, c2, c3], dim=1)))
        c5 = self.conv5(torch.cat([x, c1, c2, c3, c4], dim=1))
        return x + c5 * self.beta


class RRDB(nn.Module):

    def __init__(self, channels: int, growth_channels: int = 32, use_sn: bool = False):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(channels, growth_channels=growth_channels, use_sn=use_sn)
        self.rdb2 = ResidualDenseBlock(channels, growth_channels=growth_channels, use_sn=use_sn)
        self.rdb3 = ResidualDenseBlock(channels, growth_channels=growth_channels, use_sn=use_sn)
        self.beta = 0.2

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + out * self.beta


class ChromaticAberrationCorrection(nn.Module):

    def __init__(self, channels: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            default_conv(3, channels, 3),
            nn.LeakyReLU(0.2, inplace=True),
            default_conv(channels, channels, 3),
            nn.LeakyReLU(0.2, inplace=True),
            default_conv(channels, channels, 3, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.flow_r = nn.Sequential(
            default_conv(channels, channels // 2, 3),
            nn.LeakyReLU(0.2, inplace=True),
            default_conv(channels // 2, 2, 3),
        )
        self.flow_b = nn.Sequential(
            default_conv(channels, channels // 2, 3),
            nn.LeakyReLU(0.2, inplace=True),
            default_conv(channels // 2, 2, 3),
        )

    def warp(self, x: Tensor, flow: Tensor) -> Tensor:
        b, _, h, w = x.shape
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype),
            torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype),
            indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(b, -1, -1, -1)

        flow_norm = flow.clone()
        flow_norm[:, 0] = flow_norm[:, 0] / max(w / 2, 1)
        flow_norm[:, 1] = flow_norm[:, 1] / max(h / 2, 1)
        grid = grid + flow_norm.permute(0, 2, 3, 1)
        return F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        _, _, h, w = x.shape
        feat = self.encoder(x)
        flow_r = F.interpolate(self.flow_r(feat), size=(h, w), mode='bilinear', align_corners=False)
        flow_b = F.interpolate(self.flow_b(feat), size=(h, w), mode='bilinear', align_corners=False)
        r = self.warp(x[:, 0:1], flow_r)
        g = x[:, 1:2]
        b = self.warp(x[:, 2:3], flow_b)
        corrected = torch.cat([r, g, b], dim=1)
        return corrected, flow_r, flow_b


class HolographicDiffractionNetwork(nn.Module):

    def __init__(self, in_channels: int = 64, holo_dim: int = 32, num_layers: int = 4, use_sn: bool = False):
        super().__init__()
        self.encoder = nn.Sequential(
            default_conv(in_channels, holo_dim, 3, use_sn=use_sn),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.holo_layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2**i
            self.holo_layers.append(
                nn.Sequential(
                    default_conv(holo_dim, holo_dim, 3, dilation=dilation, use_sn=use_sn),
                    nn.LeakyReLU(0.2, inplace=True),
                    default_conv(holo_dim, holo_dim, 3, dilation=dilation, use_sn=use_sn),
                ))
        self.fusion = nn.Sequential(
            default_conv(holo_dim * num_layers, holo_dim, 1, use_sn=use_sn),
            nn.LeakyReLU(0.2, inplace=True),
            default_conv(holo_dim, in_channels, 3, use_sn=use_sn),
        )
        self.overlay_head = nn.Sequential(
            default_conv(in_channels, in_channels // 2, 3, use_sn=use_sn),
            nn.LeakyReLU(0.2, inplace=True),
            default_conv(in_channels // 2, 3, 3),
            nn.Tanh(),
        )
        self.intensity = nn.Sequential(
            default_conv(in_channels, 16, 3, use_sn=use_sn),
            nn.LeakyReLU(0.2, inplace=True),
            default_conv(16, 1, 3),
            nn.Sigmoid(),
        )

    def forward(self, features: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        encoded = self.encoder(features)
        holo_feats = []
        x = encoded
        for layer in self.holo_layers:
            x = x + layer(x)
            holo_feats.append(x)
        fused = self.fusion(torch.cat(holo_feats, dim=1))
        intensity = self.intensity(features)
        overlay = self.overlay_head(fused)
        return fused * intensity, overlay, intensity


class ScratchDetectionModule(nn.Module):

    def __init__(self, channels: int = 16):
        super().__init__()
        angles = [0, 30, 60, 90, 120, 150]
        self.oriented_filters = nn.ModuleList()
        for angle in angles:
            kernel = self._create_line_kernel(angle, length=7)
            conv = nn.Conv2d(1, 1, 7, padding=3, bias=False)
            conv.weight.data.copy_(kernel)
            self.oriented_filters.append(conv)
        self.fusion = nn.Sequential(
            default_conv(len(angles), channels, 1),
            nn.LeakyReLU(0.2, inplace=True),
            default_conv(channels, channels, 3),
            nn.LeakyReLU(0.2, inplace=True),
            default_conv(channels, 1, 3),
            nn.Sigmoid(),
        )

    def _create_line_kernel(self, angle_deg: float, length: int = 7) -> Tensor:
        angle = math.radians(angle_deg)
        kernel = torch.zeros(1, 1, length, length)
        center = length // 2
        for i in range(length):
            dx = int(round((i - center) * math.cos(angle)))
            dy = int(round((i - center) * math.sin(angle)))
            x = center + dx
            y = center + dy
            if 0 <= x < length and 0 <= y < length:
                kernel[0, 0, y, x] = 1.0
        return kernel / kernel.sum().clamp_min(1.0)

    def forward(self, x: Tensor) -> Tensor:
        luminance = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        responses = [filt(luminance) for filt in self.oriented_filters]
        return self.fusion(torch.cat(responses, dim=1))


class ScratchInpaintingModule(nn.Module):

    def __init__(self, channels: int = 64, growth_channels: int = 32, use_sn: bool = False):
        super().__init__()
        self.encoder = nn.Sequential(
            default_conv(4, channels, 3, use_sn=use_sn),
            nn.LeakyReLU(0.2, inplace=True),
            default_conv(channels, channels, 3, use_sn=use_sn),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.body = nn.Sequential(
            RRDB(channels, growth_channels=growth_channels, use_sn=use_sn),
            RRDB(channels, growth_channels=growth_channels, use_sn=use_sn),
        )
        self.decoder = nn.Sequential(
            default_conv(channels, channels // 2, 3, use_sn=use_sn),
            nn.LeakyReLU(0.2, inplace=True),
            default_conv(channels // 2, 3, 3),
            nn.Tanh(),
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        feat = self.encoder(torch.cat([x, mask], dim=1))
        feat = self.body(feat)
        out = self.decoder(feat)
        return x * (1 - mask) + out * mask


class EdgeWhiteningCorrection(nn.Module):

    def __init__(self, channels: int = 16, use_sn: bool = False):
        super().__init__()
        self.edge_detect = nn.Sequential(
            default_conv(3, channels, 3, use_sn=use_sn),
            nn.LeakyReLU(0.2, inplace=True),
            default_conv(channels, channels, 3, use_sn=use_sn),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.distance = nn.Sequential(
            default_conv(channels, channels // 2, 3, use_sn=use_sn),
            nn.LeakyReLU(0.2, inplace=True),
            default_conv(channels // 2, 1, 3),
            nn.Sigmoid(),
        )
        self.color_correction = nn.Sequential(
            default_conv(channels + 1, channels, 3, use_sn=use_sn),
            nn.LeakyReLU(0.2, inplace=True),
            default_conv(channels, 3, 3),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        edges = self.edge_detect(x)
        dist = self.distance(edges)
        correction = self.color_correction(torch.cat([edges, dist], dim=1))
        weight = torch.exp(-dist * 5.0)
        corrected = torch.clamp(x + correction * weight, -1, 1)
        return corrected, 1 - dist


@ARCH_REGISTRY.register()
class ChromeRestorationNetwork(nn.Module):

    def __init__(self, config: Optional[ChromeRestorationConfig] = None, **kwargs):
        super().__init__()
        if config is None:
            config = ChromeRestorationConfig(**kwargs)
        self.config = config
        cfg = self.config

        self.ca_correction = ChromaticAberrationCorrection(channels=cfg.base_channels)
        self.scratch_detect = ScratchDetectionModule(channels=cfg.scratch_channels)
        self.scratch_inpaint = ScratchInpaintingModule(
            channels=cfg.base_channels,
            growth_channels=cfg.growth_channels,
            use_sn=cfg.use_spectral_norm)
        self.edge_correct = EdgeWhiteningCorrection(channels=cfg.edge_channels, use_sn=cfg.use_spectral_norm)

        self.conv_first = default_conv(cfg.in_channels, cfg.base_channels, 3, use_sn=cfg.use_spectral_norm)
        self.rrdb_blocks = nn.Sequential(*[
            RRDB(cfg.base_channels, growth_channels=cfg.growth_channels, use_sn=cfg.use_spectral_norm)
            for _ in range(cfg.num_rrdb_blocks)
        ])
        self.conv_after_rrdb = default_conv(cfg.base_channels, cfg.base_channels, 3, use_sn=cfg.use_spectral_norm)

        self.holo_net = HolographicDiffractionNetwork(
            in_channels=cfg.base_channels,
            holo_dim=cfg.holographic_dim,
            num_layers=cfg.num_holo_layers,
            use_sn=cfg.use_spectral_norm)

        if cfg.upscale_factor > 1:
            self.upsampler = nn.Sequential(
                default_conv(
                    cfg.base_channels, cfg.base_channels * (cfg.upscale_factor**2), 3, use_sn=cfg.use_spectral_norm),
                nn.PixelShuffle(cfg.upscale_factor),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            self.upsampler = nn.Identity()

        self.conv_hr = default_conv(cfg.base_channels, cfg.base_channels, 3, use_sn=cfg.use_spectral_norm)
        self.conv_last = default_conv(cfg.base_channels, cfg.out_channels, 3)
        self.bg_head = nn.Sequential(
            default_conv(cfg.base_channels, 32, 3, use_sn=cfg.use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            default_conv(32, 1, 3),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor, return_intermediates: bool = False) -> Dict[str, Tensor]:
        intermediates = {}

        ca_fixed, flow_r, flow_b = self.ca_correction(x)
        intermediates['ca_corrected'] = ca_fixed
        intermediates['flow_r'] = flow_r
        intermediates['flow_b'] = flow_b

        scratch_mask = self.scratch_detect(ca_fixed)
        intermediates['scratch_mask'] = scratch_mask

        scratch_fixed = self.scratch_inpaint(ca_fixed, scratch_mask)
        intermediates['scratch_fixed'] = scratch_fixed

        edge_fixed, edge_mask = self.edge_correct(scratch_fixed)
        intermediates['edge_corrected'] = edge_fixed
        intermediates['edge_mask'] = edge_mask

        feat = self.conv_first(edge_fixed)
        rrdb_out = self.conv_after_rrdb(self.rrdb_blocks(feat))
        feat = feat + rrdb_out

        holo_features, holo_overlay, holo_intensity = self.holo_net(feat)
        intermediates['holo_overlay'] = holo_overlay
        intermediates['holo_intensity'] = holo_intensity

        feat = feat + holo_features
        feat_hr = self.upsampler(feat)
        feat_hr = self.conv_hr(feat_hr)
        output = torch.tanh(self.conv_last(feat_hr))
        intermediates['output'] = output

        card_mask = self.bg_head(feat_hr)
        intermediates['card_mask'] = card_mask

        white_bg = torch.ones_like(output)
        ebay_ready = torch.clamp(output * card_mask + white_bg * (1 - card_mask), -1, 1)
        intermediates['ebay_ready'] = ebay_ready

        if return_intermediates:
            return intermediates
        return {'output': output, 'ebay_ready': ebay_ready}


class _PerceptualBackbone(nn.Module):

    def __init__(self, pretrained: bool = False):
        super().__init__()
        from torchvision.models import VGG19_Weights, vgg19
        weights = VGG19_Weights.IMAGENET1K_V1 if pretrained else None
        features = vgg19(weights=weights).features
        self.blocks = nn.ModuleList([features[:4], features[4:9], features[9:18]]).eval()
        for param in self.blocks.parameters():
            param.requires_grad = False

    def forward(self, x: Tensor):
        outputs = []
        for block in self.blocks:
            x = block(x)
            outputs.append(x)
        return outputs


class ChromeRestorationLoss(nn.Module):

    def __init__(self, config: Optional[ChromeRestorationConfig] = None):
        super().__init__()
        self.config = config or ChromeRestorationConfig()
        self.perceptual = None
        if self.config.use_perceptual_loss:
            self.perceptual = _PerceptualBackbone(pretrained=self.config.perceptual_pretrained)

    def vgg_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        if self.perceptual is None:
            return pred.new_zeros(())
        pred_norm = (pred + 1) / 2
        target_norm = (target + 1) / 2
        loss = pred.new_zeros(())
        pred_feats = self.perceptual(pred_norm)
        target_feats = self.perceptual(target_norm)
        for pred_feat, target_feat in zip(pred_feats, target_feats):
            loss = loss + F.l1_loss(pred_feat, target_feat)
        return loss

    def forward(self, pred: Dict[str, Tensor], target: Tensor) -> Dict[str, Tensor]:
        output = pred['output']
        pixel_loss = F.l1_loss(output, target)
        perceptual_loss = self.vgg_loss(output, target)

        holo_tv = output.new_zeros(())
        if 'holo_overlay' in pred:
            holo = pred['holo_overlay']
            holo_tv = torch.mean(torch.abs(holo[:, :, :, :-1] - holo[:, :, :, 1:])) + torch.mean(
                torch.abs(holo[:, :, :-1, :] - holo[:, :, 1:, :]))

        edge_loss = output.new_zeros(())
        if 'edge_mask' in pred:
            edge_mask = pred['edge_mask']
            if edge_mask.shape[-2:] != output.shape[-2:]:
                edge_mask = F.interpolate(edge_mask, size=output.shape[-2:], mode='bilinear', align_corners=False)
            pred_edges = self._sobel_edges(output)
            target_edges = self._sobel_edges(target)
            edge_loss = F.l1_loss(pred_edges * edge_mask, target_edges * edge_mask)

        total = pixel_loss + 0.1 * perceptual_loss + self.config.holo_loss_weight * holo_tv + self.config.edge_loss_weight * edge_loss
        return {
            'total': total,
            'pixel': pixel_loss,
            'perceptual': perceptual_loss,
            'holo_tv': holo_tv,
            'edge': edge_loss,
        }

    def _sobel_edges(self, x: Tensor) -> Tensor:
        sobel_x = x.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
        sobel_y = x.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
        edges = []
        for c in range(x.size(1)):
            gx = F.conv2d(x[:, c:c + 1], sobel_x, padding=1)
            gy = F.conv2d(x[:, c:c + 1], sobel_y, padding=1)
            edges.append(torch.sqrt(gx**2 + gy**2 + 1e-6))
        return torch.cat(edges, dim=1)


@ARCH_REGISTRY.register()
class ChromeDiscriminator(nn.Module):

    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()

        def disc_block(in_c, out_c, stride=2):
            return nn.Sequential(
                spectral_norm(nn.Conv2d(in_c, out_c, 4, stride, 1)),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.layers = nn.Sequential(
            disc_block(in_channels, base_channels),
            disc_block(base_channels, base_channels * 2),
            disc_block(base_channels * 2, base_channels * 4),
            disc_block(base_channels * 4, base_channels * 8, stride=1),
            spectral_norm(nn.Conv2d(base_channels * 8, 1, 4, 1, 1)),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)
