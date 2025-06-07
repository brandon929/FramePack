import os
import torch
from safetensors.torch import load_file
from tqdm import tqdm


def merge_lora_to_state_dict(
    state_dict: dict[str, torch.Tensor], lora_file: str, multiplier: float, device: torch.device
) -> dict[str, torch.Tensor]:
    """
    Merge LoRA weights into the state dict of a model.
    """
    lora_sd = load_file(lora_file)

    # Check the format of the LoRA file
    keys = list(lora_sd.keys())
    if keys[0].startswith("lora_unet_"):
        print(f"Musubi Tuner LoRA detected")
        return merge_musubi_tuner(lora_sd, state_dict, multiplier, device)

    transformer_prefixes = ["diffusion_model", "transformer"]  # to ignore Text Encoder modules
    lora_suffix = None
    prefix = None
    for key in keys:
        if lora_suffix is None and "lora_A" in key:
            lora_suffix = "lora_A"
        if prefix is None:
            pfx = key.split(".")[0]
            if pfx in transformer_prefixes:
                prefix = pfx
        if lora_suffix is not None and prefix is not None:
            break

    if lora_suffix == "lora_A" and prefix is not None:
        print(f"Diffusion-pipe (?) LoRA detected")
        return merge_diffusion_pipe_or_something(lora_sd, state_dict, "lora_unet_", multiplier, device)

    print(f"LoRA file format not recognized: {os.path.basename(lora_file)}")
    return state_dict


def merge_diffusion_pipe_or_something(
    lora_sd: dict[str, torch.Tensor], state_dict: dict[str, torch.Tensor], prefix: str, multiplier: float, device: torch.device
) -> dict[str, torch.Tensor]:
    """
    Convert LoRA weights to the format used by the diffusion pipeline to Musubi Tuner.
    Copy from Musubi Tuner repo.
    """
    # convert from diffusers(?) to default LoRA
    # Diffusers format: {"diffusion_model.module.name.lora_A.weight": weight, "diffusion_model.module.name.lora_B.weight": weight, ...}
    # default LoRA format: {"prefix_module_name.lora_down.weight": weight, "prefix_module_name.lora_up.weight": weight, ...}

    # note: Diffusers has no alpha, so alpha is set to rank
    new_weights_sd = {}
    lora_dims = {}
    for key, weight in lora_sd.items():
        diffusers_prefix, key_body = key.split(".", 1)
        if diffusers_prefix != "diffusion_model" and diffusers_prefix != "transformer":
            print(f"unexpected key: {key} in diffusers format")
            continue

        new_key = f"{prefix}{key_body}".replace(".", "_").replace("_lora_A_", ".lora_down.").replace("_lora_B_", ".lora_up.")
        new_weights_sd[new_key] = weight

        lora_name = new_key.split(".")[0]  # before first dot
        if lora_name not in lora_dims and "lora_down" in new_key:
            lora_dims[lora_name] = weight.shape[0]

    # add alpha with rank
    for lora_name, dim in lora_dims.items():
        new_weights_sd[f"{lora_name}.alpha"] = torch.tensor(dim)

    return merge_musubi_tuner(new_weights_sd, state_dict, multiplier, device)


def merge_musubi_tuner(
    lora_sd: dict[str, torch.Tensor], state_dict: dict[str, torch.Tensor], multiplier: float, device: torch.device
) -> dict[str, torch.Tensor]:
    """
    Merge LoRA weights into the state dict of a model.
    """
    # Check LoRA is for FramePack or for HunyuanVideo
    is_hunyuan = False
    for key in lora_sd.keys():
        if "double_blocks" in key or "single_blocks" in key:
            is_hunyuan = True
            break
    if is_hunyuan:
        print("HunyuanVideo LoRA detected, converting to FramePack format")
        lora_sd = convert_hunyuan_to_framepack(lora_sd)

    # Merge LoRA weights into the state dict
    print(f"Merging LoRA weights into state dict. multiplier: {multiplier}")

    # Create module map
    name_to_original_key = {}
    for key in state_dict.keys():
        if key.endswith(".weight"):
            lora_name = key.rsplit(".", 1)[0]  # remove trailing ".weight"
            lora_name = "lora_unet_" + lora_name.replace(".", "_")
            if lora_name not in name_to_original_key:
                name_to_original_key[lora_name] = key

    # Merge LoRA weights
    keys = list([k for k in lora_sd.keys() if "lora_down" in k])
    for key in tqdm(keys, desc="Merging LoRA weights"):
        up_key = key.replace("lora_down", "lora_up")
        alpha_key = key[: key.index("lora_down")] + "alpha"

        # find original key for this lora
        module_name = ".".join(key.split(".")[:-2])  # remove trailing ".lora_down.weight"
        if module_name not in name_to_original_key:
            print(f"No module found for LoRA weight: {key}")
            continue

        original_key = name_to_original_key[module_name]

        down_weight = lora_sd[key]
        up_weight = lora_sd[up_key]

        dim = down_weight.size()[0]
        alpha = lora_sd.get(alpha_key, dim)
        scale = alpha / dim

        weight = state_dict[original_key]
        original_device = weight.device
        if original_device != device:
            weight = weight.to(device)  # to make calculation faster

        down_weight = down_weight.to(device)
        up_weight = up_weight.to(device)

        # W <- W + U * D
        if len(weight.size()) == 2:
            # linear
            if len(up_weight.size()) == 4:  # use linear projection mismatch
                up_weight = up_weight.squeeze(3).squeeze(2)
                down_weight = down_weight.squeeze(3).squeeze(2)
            weight = weight + multiplier * (up_weight @ down_weight) * scale
        elif down_weight.size()[2:4] == (1, 1):
            # conv2d 1x1
            weight = (
                weight
                + multiplier
                * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                * scale
            )
        else:
            # conv2d 3x3
            conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
            # logger.info(conved.size(), weight.size(), module.stride, module.padding)
            weight = weight + multiplier * conved * scale

        weight = weight.to(original_device)  # move back to original device
        state_dict[original_key] = weight

    return state_dict


def convert_hunyuan_to_framepack(lora_sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Convert HunyuanVideo LoRA weights to FramePack format.
    """
    new_lora_sd = {}
    for key, weight in lora_sd.items():
        if "double_blocks" in key:
            key = key.replace("double_blocks", "transformer_blocks")
            key = key.replace("img_mod_linear", "norm1_linear")
            key = key.replace("img_attn_qkv", "attn_to_QKV")  # split later
            key = key.replace("img_attn_proj", "attn_to_out_0")
            key = key.replace("img_mlp_fc1", "ff_net_0_proj")
            key = key.replace("img_mlp_fc2", "ff_net_2")
            key = key.replace("txt_mod_linear", "norm1_context_linear")
            key = key.replace("txt_attn_qkv", "attn_add_QKV_proj")  # split later
            key = key.replace("txt_attn_proj", "attn_to_add_out")
            key = key.replace("txt_mlp_fc1", "ff_context_net_0_proj")
            key = key.replace("txt_mlp_fc2", "ff_context_net_2")
        elif "single_blocks" in key:
            key = key.replace("single_blocks", "single_transformer_blocks")
            key = key.replace("linear1", "attn_to_QKVM")  # split later
            key = key.replace("linear2", "proj_out")
            key = key.replace("modulation_linear", "norm_linear")
        else:
            print(f"Unsupported module name: {key}, only double_blocks and single_blocks are supported")
            continue

        if "QKVM" in key:
            # split QKVM into Q, K, V, M
            key_q = key.replace("QKVM", "q")
            key_k = key.replace("QKVM", "k")
            key_v = key.replace("QKVM", "v")
            key_m = key.replace("attn_to_QKVM", "proj_mlp")
            if "_down" in key or "alpha" in key:
                # copy QKVM weight or alpha to Q, K, V, M
                assert "alpha" in key or weight.size(1) == 3072, f"QKVM weight size mismatch: {key}. {weight.size()}"
                new_lora_sd[key_q] = weight
                new_lora_sd[key_k] = weight
                new_lora_sd[key_v] = weight
                new_lora_sd[key_m] = weight
            elif "_up" in key:
                # split QKVM weight into Q, K, V, M
                assert weight.size(0) == 21504, f"QKVM weight size mismatch: {key}. {weight.size()}"
                new_lora_sd[key_q] = weight[:3072]
                new_lora_sd[key_k] = weight[3072 : 3072 * 2]
                new_lora_sd[key_v] = weight[3072 * 2 : 3072 * 3]
                new_lora_sd[key_m] = weight[3072 * 3 :]  # 21504 - 3072 * 3 = 12288
            else:
                print(f"Unsupported module name: {key}")
                continue
        elif "QKV" in key:
            # split QKV into Q, K, V
            key_q = key.replace("QKV", "q")
            key_k = key.replace("QKV", "k")
            key_v = key.replace("QKV", "v")
            if "_down" in key or "alpha" in key:
                # copy QKV weight or alpha to Q, K, V
                assert "alpha" in key or weight.size(1) == 3072, f"QKV weight size mismatch: {key}. {weight.size()}"
                new_lora_sd[key_q] = weight
                new_lora_sd[key_k] = weight
                new_lora_sd[key_v] = weight
            elif "_up" in key:
                # split QKV weight into Q, K, V
                assert weight.size(0) == 3072 * 3, f"QKV weight size mismatch: {key}. {weight.size()}"
                new_lora_sd[key_q] = weight[:3072]
                new_lora_sd[key_k] = weight[3072 : 3072 * 2]
                new_lora_sd[key_v] = weight[3072 * 2 :]
            else:
                print(f"Unsupported module name: {key}")
                continue
        else:
            # no split needed
            new_lora_sd[key] = weight

    return new_lora_sd
