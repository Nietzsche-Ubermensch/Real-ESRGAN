import torch

from realesrgan.archs.chrome_restoration_arch import (ChromeDiscriminator, ChromeRestorationConfig,
                                                      ChromeRestorationLoss, ChromeRestorationNetwork)


def test_chrome_restoration_network():
    config = ChromeRestorationConfig(
        base_channels=16,
        growth_channels=8,
        num_rrdb_blocks=2,
        num_holo_layers=2,
        holographic_dim=8,
        scratch_channels=8,
        edge_channels=8,
        upscale_factor=2,
        use_perceptual_loss=False)
    net = ChromeRestorationNetwork(config)
    img = torch.rand((1, 3, 16, 16), dtype=torch.float32) * 2 - 1

    outputs = net(img, return_intermediates=True)

    assert outputs['ca_corrected'].shape == (1, 3, 16, 16)
    assert outputs['flow_r'].shape == (1, 2, 16, 16)
    assert outputs['flow_b'].shape == (1, 2, 16, 16)
    assert outputs['scratch_mask'].shape == (1, 1, 16, 16)
    assert outputs['edge_mask'].shape == (1, 1, 16, 16)
    assert outputs['holo_overlay'].shape == (1, 3, 16, 16)
    assert outputs['holo_intensity'].shape == (1, 1, 16, 16)
    assert outputs['output'].shape == (1, 3, 32, 32)
    assert outputs['card_mask'].shape == (1, 1, 32, 32)
    assert outputs['ebay_ready'].shape == (1, 3, 32, 32)
    assert torch.max(outputs['output']) <= 1
    assert torch.min(outputs['output']) >= -1


def test_chrome_restoration_loss_and_discriminator():
    config = ChromeRestorationConfig(
        base_channels=16,
        growth_channels=8,
        num_rrdb_blocks=1,
        num_holo_layers=2,
        holographic_dim=8,
        use_perceptual_loss=False)
    net = ChromeRestorationNetwork(config)
    loss_fn = ChromeRestorationLoss(config)
    disc = ChromeDiscriminator(in_channels=3, base_channels=16)

    img = torch.rand((1, 3, 16, 16), dtype=torch.float32) * 2 - 1
    target = torch.rand((1, 3, 32, 32), dtype=torch.float32) * 2 - 1

    outputs = net(img, return_intermediates=True)
    losses = loss_fn(outputs, target)
    disc_out = disc(outputs['output'])

    assert set(['total', 'pixel', 'perceptual', 'holo_tv', 'edge']).issubset(losses.keys())
    assert losses['total'].ndim == 0
    assert disc_out.shape[0] == 1
    assert disc_out.shape[1] == 1
