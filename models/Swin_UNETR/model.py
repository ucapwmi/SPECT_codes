from monai.networks.nets import SwinUNETR

def get_swin_unetr(device, roi_size=(128, 128, 128)):
    """
    Build SwinUNETR model.
    
    Args:
        device (torch.device): torch device
        roi_size (tuple): input image size (D, H, W)

    Returns:
        torch.nn.Module: SwinUNETR model
    """
    model = SwinUNETR(
        img_size=roi_size,
        in_channels=1,
        out_channels=1,
        feature_size=48,
        use_checkpoint=True,
        use_v2=True
    ).to(device)
    return model