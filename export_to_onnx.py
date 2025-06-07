import torch
from unisal.model import UNISAL # Assuming UNISAL is directly importable
from collections import OrderedDict
import os # Added for path manipulation if needed, though not strictly in this version

def main():
    # 1. Define export parameters
    export_source = "DHF1K"
    # Assuming the script is run from the repository root
    weights_path = "training_runs/pretrained_unisal/weights_best.pth"
    onnx_model_path = "unisal.onnx"

    # For static export, time dimension is 1. Input shape (batch, time, channel, H, W)
    dummy_input_shape = (1, 1, 3, 480, 640) # batch_size=1, time=1, C=3, H=480, W=640

    # 2. Instantiate the refactored UNISAL model for export
    print(f"Instantiating UNISAL model with export_mode=True and export_source='{export_source}'...")
    # Pass all expected __init__ args, or ensure defaults are appropriate
    # The UNISAL constructor has many parameters with defaults.
    # If the defaults are fine for export (especially regarding sources list for BN momentum mapping),
    # then this is okay. The 'sources' list in UNISAL.__init__ is used to map
    # export_source to its specific bn_momentum or static_bn_momentum.
    # Default sources: ("DHF1K", "Hollywood", "UCFSports", "SALICON")
    # export_source="DHF1K" is in the default list.
    model = UNISAL(export_mode=True, export_source=export_source)
    model.eval() # Set model to evaluation mode

    # 3. Load pre-trained weights
    print(f"Loading weights from {weights_path}...")
    if not os.path.exists(weights_path):
        print(f"Error: Weights file not found at {weights_path}. Please ensure the path is correct.")
        print(f"Current working directory: {os.getcwd()}")
        # Attempt to list contents of training_runs to help debug
        if os.path.exists("training_runs"):
            print(f"Contents of training_runs: {os.listdir('training_runs')}")
            if os.path.exists("training_runs/pretrained_unisal"):
                 print(f"Contents of training_runs/pretrained_unisal: {os.listdir('training_runs/pretrained_unisal')}")
        return

    state_dict = torch.load(weights_path, map_location="cpu")

    # Handle checkpoint structure and DataParallel prefix if present
    new_state_dict = OrderedDict()
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        # This handles checkpoints from the Trainer class used in the repo
        state_dict = state_dict["model_state_dict"]

    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v  # remove `module.`
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    print("Weights loaded successfully.")

    # 4. Create a dummy input tensor
    dummy_input = torch.randn(dummy_input_shape, device="cpu")
    print(f"Created dummy input with shape: {dummy_input.shape}")

    # 5. Export the model to ONNX
    print(f"Exporting model to ONNX format at {onnx_model_path}...")

    # The forward pass of the refactored model will use static=True and source=export_source
    # due to export_mode=True being set.
    # UNISAL.forward() parameters like 'source' and 'static' will be overridden by
    # self.export_mode logic within the forward method.

    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        opset_version=11,
        verbose=True, # Set to False if output is too noisy
        export_params=True,
    )
    print(f"Model successfully exported to {onnx_model_path}")

if __name__ == "__main__":
    main()
