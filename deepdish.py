import torch
import torch.nn as nn


class DeepDish(nn.Module):
    """
        Our DeepDish model.
    """

    def __init__(self, out_features, config):
        super().__init__()

        # Load in backbone
        self.backbone_str = config["backbone"]
        if self.backbone_str == "dinov2":
            self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        else:
            raise "No other backbone besides dinov2 supported, as of now"

        # Pass pseudo-input through backbone to get output feature dimensions of backbone
        with torch.no_grad():
            pseudo_output = self.backbone_forward(torch.rand(1, 3, 224, 224))
            _, in_features = pseudo_output.shape

        # Create regression head
        self.head = nn.Sequential(
            nn.Linear(in_features, config["head_hidden_size"]),
            nn.GELU(),
            nn.Dropout(config["head_dropout_p"]),
            nn.Linear(config["head_hidden_size"], out_features)
        )

        self.config = config

        # Set the requires grad properly
        self.set_requires_grad()

    def backbone_forward(self, img):
        if self.backbone_str == "dinov2":

            # Prepare the input
            x = self.backbone.prepare_tokens_with_masks(img, None)

            # Pass through each backbone block (12 for dino)
            for blk in self.backbone.blocks:
                x = blk(x)
                x = self.backbone.norm(x) # Apply norm at end

            # Remove CLS token and registered tokens for dinov2, then return
            # DINOv2 consists of 1 CLS token, 4 registered tokens, the rest are patches
            # return x[:, 1 + self.backbone.num_register_tokens:]
            return x[:, 0, :]

    def forward(self, img):
        feat = self.backbone_forward(img)
        return self.head(feat)

    def set_requires_grad(self):
        """
            Sets requires_grad for each component. The regression head is always trainable, the backbone is frozen
            by default, but the config setting unfreeze_backbone_block_after_n determines which n last blocks to
            unfreeze of the model.
        """
        
        for param_name, param in self.backbone.named_parameters():
            if ('blocks' in param_name):
                block_id = int(param_name.split('.')[1])
                if block_id >= self.config["unfreeze_backbone_block_after_n"]:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = False

        # Make the head trainable at all times
        for param in self.head.parameters():
            param.requires_grad = True