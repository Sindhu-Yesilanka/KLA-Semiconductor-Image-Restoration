import os
import sys
import time
import numpy as np
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================
# Import model definitions
# ==========================



class ResidualBlock(nn.Module):

    def __init__(
        self,
        channels
    ):

        super().__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            3,
            padding=1
        )

        self.act = nn.LeakyReLU(
            0.1,
            inplace=True
        )


    def forward(
        self,
        x
    ):

        residual = x

        x = self.conv1(
            x
        )

        x = self.act(
            x
        )

        x = self.conv2(
            x
        )

        return (
            residual
            +
            0.1 * x
        )
class EncoderBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        self.project = nn.Conv2d(
            in_channels,
            out_channels,
            3,
            padding=1
        )

        self.res1 = ResidualBlock(
            out_channels
        )

        self.res2 = ResidualBlock(
            out_channels
        )

        self.down = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            stride=2,
            padding=1
        )


    def forward(
        self,
        x
    ):

        x = self.project(
            x
        )

        x = self.res1(
            x
        )

        x = self.res2(
            x
        )

        skip = x

        x = self.down(
            x
        )

        return (
            x,
            skip
        )
class DecoderBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels
    ):

        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

        self.fuse = nn.Conv2d(
            out_channels
            +
            skip_channels,
            out_channels,
            3,
            padding=1
        )

        self.res1 = ResidualBlock(
            out_channels
        )

        self.res2 = ResidualBlock(
            out_channels
        )


    def forward(
        self,
        x,
        skip
    ):

        x = self.up(
            x
        )

        x = torch.cat(
            [
                x,
                skip
            ],
            dim=1
        )

        x = self.fuse(
            x
        )

        x = self.res1(
            x
        )

        x = self.res2(
            x
        )

        return x
class DegradationEstimator(nn.Module):

    def __init__(
        self,
        embedding_dim=16
    ):

        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(
                1,
                8,
                3,
                padding=1
            ),

            nn.LeakyReLU(
                0.1,
                inplace=True
            ),

            nn.Conv2d(
                8,
                16,
                3,
                stride=2,
                padding=1
            ),

            nn.LeakyReLU(
                0.1,
                inplace=True
            ),

            nn.Conv2d(
                16,
                32,
                3,
                stride=2,
                padding=1
            ),

            nn.LeakyReLU(
                0.1,
                inplace=True
            ),

            nn.AdaptiveAvgPool2d(
                1
            ),

            nn.Flatten(),

            nn.Linear(
                32,
                embedding_dim
            )
        )


    def forward(
        self,
        x
    ):

        return self.net(
            x
        )
class DegradationModulation(nn.Module):

    def __init__(
        self,
        channels,
        embedding_dim=16
    ):

        super().__init__()

        self.fc = nn.Linear(
            embedding_dim,
            channels * 2
        )

        # Exact identity at initialization
        nn.init.zeros_(
            self.fc.weight
        )

        nn.init.zeros_(
            self.fc.bias
        )


    def forward(
        self,
        x,
        embedding
    ):

        params = self.fc(
            embedding
        )

        scale, shift = (
            params.chunk(
                2,
                dim=1
            )
        )

        scale = (
            scale[
                :,
                :,
                None,
                None
            ]
        )

        shift = (
            shift[
                :,
                :,
                None,
                None
            ]
        )

        return (
            x
            *
            (
                1.0
                +
                scale
            )
            +
            shift
        )
class DAResidualUNetSR(nn.Module):

    def __init__(
        self,
        base_channels=48,
        scale=2,
        embedding_dim=16
    ):

        super().__init__()


        # ==================================
        # Degradation estimator
        # ==================================

        self.degradation_estimator = (
            DegradationEstimator(
                embedding_dim
            )
        )


        # ==================================
        # SAME MODEL-10 BACKBONE
        # ==================================

        self.head = nn.Conv2d(
            1,
            base_channels,
            3,
            padding=1
        )


        self.enc1 = EncoderBlock(
            base_channels,
            base_channels
        )


        self.enc2 = EncoderBlock(
            base_channels,
            base_channels * 2
        )


        self.bottleneck_in = nn.Conv2d(
            base_channels * 2,
            base_channels * 3,
            3,
            padding=1
        )


        self.bottleneck = nn.Sequential(

            ResidualBlock(
                base_channels * 3
            ),

            ResidualBlock(
                base_channels * 3
            ),

            ResidualBlock(
                base_channels * 3
            ),

            ResidualBlock(
                base_channels * 3
            )
        )


        self.dec2 = DecoderBlock(
            base_channels * 3,
            base_channels * 2,
            base_channels * 2
        )


        self.dec1 = DecoderBlock(
            base_channels * 2,
            base_channels,
            base_channels
        )


        self.refine = nn.Sequential(

            ResidualBlock(
                base_channels
            ),

            ResidualBlock(
                base_channels
            )
        )


        self.up = nn.Sequential(

            nn.Conv2d(
                base_channels,
                base_channels
                *
                scale
                *
                scale,
                3,
                padding=1
            ),

            nn.PixelShuffle(
                scale
            ),

            nn.Conv2d(
                base_channels,
                1,
                3,
                padding=1
            )
        )


        # ==================================
        # ONLY NEW COMPONENTS
        # ==================================

        self.mod_skip1 = (
            DegradationModulation(
                base_channels,
                embedding_dim
            )
        )


        self.mod_skip2 = (
            DegradationModulation(
                base_channels * 2,
                embedding_dim
            )
        )


        self.mod_bottleneck = (
            DegradationModulation(
                base_channels * 3,
                embedding_dim
            )
        )


    def forward(
        self,
        x
    ):

        # Degradation embedding
        embedding = (
            self.degradation_estimator(
                x
            )
        )


        # Shallow features
        features = self.head(
            x
        )

        shallow = features


        # Encoder 1
        features, skip1 = (
            self.enc1(
                features
            )
        )

        skip1 = self.mod_skip1(
            skip1,
            embedding
        )


        # Encoder 2
        features, skip2 = (
            self.enc2(
                features
            )
        )

        skip2 = self.mod_skip2(
            skip2,
            embedding
        )


        # Bottleneck
        features = (
            self.bottleneck_in(
                features
            )
        )

        features = (
            self.bottleneck(
                features
            )
        )

        features = (
            self.mod_bottleneck(
                features,
                embedding
            )
        )


        # Decoder
        features = self.dec2(
            features,
            skip2
        )

        features = self.dec1(
            features,
            skip1
        )


        # Refinement
        features = self.refine(
            features
        )


        # Same long residual as Model 10
        features = (
            features
            +
            shallow
        )


        # x2 reconstruction
        output = self.up(
            features
        )

        return output
# ==========================
# Load model
# ==========================

MODEL_PATH = (
    "models/da_resunet_sr_v2/"
    "da_resunet_sr_v2_best.pth"
)


device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)


model = DAResidualUNetSR(
    base_channels=checkpoint["config"]["base_channels"],
    scale=checkpoint["config"]["scale"],
    embedding_dim=checkpoint["config"]["embedding_dim"]
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


model.to(device)
model.eval()


print("Model loaded")


# ==========================
# Restoration function
# ==========================

def restore(image):

    if image.ndim == 2:
        image = image[None,None,:,:]

    elif image.ndim == 3:
        image = image[:,:,0]
        image = image[None,None,:,:]


    image = torch.tensor(
        image,
        dtype=torch.float32
    ).to(device)


    with torch.no_grad():

        output = model(image)


    output = (
        output
        .squeeze()
        .cpu()
        .numpy()
    )


    output = np.nan_to_num(
        output,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )


    output = np.clip(
        output,
        0,
        1
    )


    return output.astype(
        np.float32
    )


# ==========================
# Main execution
# ==========================

if __name__ == "__main__":


    if len(sys.argv) != 3:
        print(
            "Usage: python run.py <input-dir> <output-dir>"
        )
        sys.exit(1)


    input_dir = sys.argv[1]
    output_dir = sys.argv[2]


    os.makedirs(
        output_dir,
        exist_ok=True
    )


    files = sorted(
        [
            f for f in os.listdir(input_dir)
            if f.endswith(".npy")
        ]
    )


    start = time.time()


    for f in files:

        input_path = os.path.join(
            input_dir,
            f
        )

        output_path = os.path.join(
            output_dir,
            f
        )


        img = np.load(
            input_path
        )


        restored = restore(
            img
        )


        np.save(
            output_path,
            restored
        )


        print(
            f"Processed {f}"
        )


    end = time.time()


    print(
        "Total images:",
        len(files)
    )

    print(
        "Time:",
        end-start,
        "seconds"
    )