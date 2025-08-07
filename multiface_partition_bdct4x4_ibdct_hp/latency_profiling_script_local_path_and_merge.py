# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
from tqdm import tqdm
import torch
from models import DeepAppearanceVAE_IBDCT
performance_test = False

# Try to import thop for FLOPs calculation
try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False

# Try to import torchsummary for model summary
try:
    from torchsummary import summary
    TORCHSUMMARY_AVAILABLE = True
    TORCHINFO_AVAILABLE = False
except ImportError:
    try:
        from torchinfo import summary
        TORCHSUMMARY_AVAILABLE = True
        TORCHINFO_AVAILABLE = True
    except ImportError:
        TORCHSUMMARY_AVAILABLE = False
        TORCHINFO_AVAILABLE = False

num_freq_comp_outsourced_list = [2,4,6,8,10,12,14]
latency_list = [] if performance_test else None
flops_data = {}  # Dictionary to store FLOPs data for each configuration
for num_freq_comp_outsourced in num_freq_comp_outsourced_list:
    batch_size = 1
    device = torch.device("cuda", 0)

    model = DeepAppearanceVAE_IBDCT(
        1024, 21918, n_latent=256, n_cams=38, num_freq_comp_outsourced=num_freq_comp_outsourced, result_path="/tmp/tmp", save_latent_code=False
    ).to(device)
    z_local = torch.zeros(batch_size, 256).to(device)
    z_outsource = torch.zeros(batch_size, 256).to(device)
    v = torch.zeros(batch_size, 3).to(device)

    # Generate sample texture_outsource for tracing
    with torch.no_grad():
        view_code = model.dec.relu(model.dec.view_fc(v))
        z_code_outsource = model.dec.relu(model.dec.z_fc_outsource(z_outsource))
        feat_outsource = torch.cat((view_code, z_code_outsource), 1)
        texture_code_outsource = model.dec.relu(model.dec.texture_fc_outsource(feat_outsource))
        texture_outsource = model.dec.texture_decoder_outsource(texture_code_outsource)

    # Create a traced model for faster inference
    class TracedDecoder(torch.nn.Module):
        def __init__(self, model_dec):
            super().__init__()
            self.model_dec = model_dec
            
        def forward(self, v, z_local, texture_outsource):            
            view_code = self.model_dec.relu(self.model_dec.view_fc(v))
            z_code = self.model_dec.relu(self.model_dec.z_fc(z_local))
            mesh = self.model_dec.mesh_fc(z_code)
            texture_code = self.model_dec.relu(self.model_dec.texture_fc(torch.cat((view_code, z_code), 1)))
            texture_local = self.model_dec.texture_decoder_local(texture_code)
            texture_merge = torch.cat((texture_local, texture_outsource), dim=1)
            return (mesh, texture_merge)
    
    # Create and trace the model
    traced_decoder = TracedDecoder(model.dec)
    traced_decoder.eval()

    # Create sample inputs for tracing with correct dimensions
    sample_v = torch.zeros(batch_size, 3).to(device)
    sample_z_local = torch.zeros(batch_size, 256).to(device)
    sample_texture_outsource = texture_outsource  # Use the actual output from above

    # Trace the model
    with torch.no_grad():
        traced_model = torch.jit.trace(traced_decoder, (sample_v, sample_z_local, sample_texture_outsource))

    # Calculate FLOPs for the traced model
    print(f"\n=== FLOPs Analysis for num_freq_comp_outsourced={num_freq_comp_outsourced} ===")
    
    # Manual FLOPs calculation since thop doesn't work well with custom models
    def calculate_flops_manual(model, input_shapes):
        """Manual FLOPs calculation for custom model"""
        total_flops = 0
        
        # Calculate FLOPs for each layer
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Linear layer: input_features * output_features * batch_size
                in_features = module.in_features
                out_features = module.out_features
                batch_size = input_shapes[0][0] if input_shapes else 1
                layer_flops = in_features * out_features * batch_size
                total_flops += layer_flops
                
            elif isinstance(module, torch.nn.Conv2d):
                # Conv2d: H_out * W_out * (C_in * K_h * K_w * C_out)
                if hasattr(module, 'kernel_size'):
                    kernel_size = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
                    
                    # Estimate output size (this is approximate for deconv)
                    H_out = 256  # Approximate output size
                    W_out = 256
                    C_in = module.in_channels
                    C_out = module.out_channels
                    K_h, K_w = kernel_size
                    
                    layer_flops = H_out * W_out * C_in * K_h * K_w * C_out
                    total_flops += layer_flops
        
        return total_flops
    
    # Calculate parameters first
    total_params = sum(p.numel() for p in traced_decoder.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Manual FLOPs calculation
    input_shapes = [sample_v.shape, sample_z_local.shape, sample_texture_outsource.shape]
    estimated_flops = calculate_flops_manual(traced_decoder, input_shapes)
    
    # If thop is available, try it but don't rely on it
    if THOP_AVAILABLE:
        try:
            flops, params = profile(traced_decoder, inputs=(sample_v, sample_z_local, sample_texture_outsource), verbose=False)
            if flops > 0:
                print(f"thop FLOPs: {flops:,}")
                print(f"thop Parameters: {params:,}")
                print(f"thop FLOPs (G): {flops/1e9:.2f}G")
            else:
                print("thop returned 0 FLOPs (likely due to custom layers)")
        except Exception as e:
            print(f"thop failed: {e}")
    
    # Use our manual calculation
    print(f"Manual FLOPs estimation: {estimated_flops:,}")
    print(f"Manual FLOPs (G): {estimated_flops/1e9:.2f}G")
    
    # Also try torchprofile if available
    torchprofile_flops = None
    try:
        from torchprofile import profile_macs
        macs = profile_macs(traced_decoder, (sample_v, sample_z_local, sample_texture_outsource))
        if macs > 0:
            print(f"torchprofile MACs: {macs:,}")
            print(f"torchprofile MACs (G): {macs/1e9:.2f}G")
            # Approximate FLOPs (usually 2x MACs for most operations)
            torchprofile_flops = macs * 2
            print(f"torchprofile Estimated FLOPs: {torchprofile_flops:,}")
            print(f"torchprofile Estimated FLOPs (G): {torchprofile_flops/1e9:.2f}G")
        else:
            print("torchprofile returned 0 MACs (likely due to custom layers)")
    except ImportError:
        print("torchprofile not available")
    except Exception as e:
        print(f"torchprofile failed: {e}")
    
    # Store FLOPs data for this configuration
    flops_data[num_freq_comp_outsourced] = {
        'parameters': total_params,
        'manual_flops': estimated_flops,
        'manual_flops_g': estimated_flops/1e9,
        'torchprofile_flops': torchprofile_flops,
        'torchprofile_flops_g': torchprofile_flops/1e9 if torchprofile_flops else None
    }

    # Print model summary if available
    if TORCHSUMMARY_AVAILABLE:
        print(f"\n=== Model Summary for num_freq_comp_outsourced={num_freq_comp_outsourced} ===")
        try:
            if TORCHINFO_AVAILABLE:
                # Use torchinfo for better multi-input support
                summary_str = summary(
                    traced_decoder, 
                    input_data=[sample_v, sample_z_local, sample_texture_outsource],
                    verbose=0,
                    col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
                    depth=4
                )
                print(summary_str)
            else:
                # Fallback to basic model info
                # print("Model structure:")
                # print(traced_decoder)
                print(f"\nInput shapes:")
                print(f"  v: {sample_v.shape}")
                print(f"  z_local: {sample_z_local.shape}")
                print(f"  texture_outsource: {sample_texture_outsource.shape}")
        except Exception as e:
            print(f"Could not print model summary: {e}")

    # Warm up the traced model
    with torch.no_grad():
        for _ in range(10):
            _ = traced_model(sample_v, sample_z_local, sample_texture_outsource)

    if performance_test:
        start_time = time.time()
        total_inference = 10000

        for _ in tqdm(range(total_inference)):
            # Run the traced model for faster inference
            with torch.no_grad():
                _ = traced_model(v, z_local, texture_outsource)

        end_time = time.time()

        print(f"Under Batchsize = {batch_size}, inference latency on GPU 3090 = {(end_time - start_time) / total_inference}")
        latency_list.append((end_time - start_time) / total_inference)

if performance_test:
    print(latency_list)

# Print comprehensive FLOPs summary
# Export data to CSV for further analysis
import csv
csv_filename = "flops_analysis_results.csv"
with open(csv_filename, 'w', newline='') as csvfile:
    fieldnames = ['config', 'parameters', 'manual_flops', 'manual_flops_g', 'torchprofile_flops', 'torchprofile_flops_g']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for config in sorted(flops_data.keys()):
        data = flops_data[config]
        writer.writerow({
            'config': config,
            'parameters': data['parameters'],
            'manual_flops': data['manual_flops'],
            'manual_flops_g': data['manual_flops_g'],
            'torchprofile_flops': data['torchprofile_flops'] or 0,
            'torchprofile_flops_g': data['torchprofile_flops_g'] or 0
        })

print(f"\nData exported to {csv_filename}")

print("\n" + "="*80)
print("COMPREHENSIVE FLOPs ANALYSIS SUMMARY")
print("="*80)

# Create a formatted table
print(f"{'Config':<8} {'Parameters':<12} {'Manual FLOPs':<15} {'Manual (G)':<12} {'TP FLOPs':<15} {'TP (G)':<10}")
print("-" * 80)

for config in sorted(flops_data.keys()):
    data = flops_data[config]
    params = data['parameters']
    manual_flops = data['manual_flops']
    manual_g = data['manual_flops_g']
    tp_flops = data['torchprofile_flops']
    tp_g = data['torchprofile_flops_g']
    
    # Format the values
    params_str = f"{params:,}"
    manual_flops_str = f"{manual_flops:,}"
    manual_g_str = f"{manual_g:.2f}G"
    tp_flops_str = f"{tp_flops:,}" if tp_flops else "N/A"
    tp_g_str = f"{tp_g:.2f}G" if tp_g else "N/A"
    
    print(f"{config:<8} {params_str:<12} {manual_flops_str:<15} {manual_g_str:<12} {tp_flops_str:<15} {tp_g_str:<10}")

print("-" * 80)

# Calculate and print statistics
print("\n" + "="*50)
print("STATISTICS")
print("="*50)

# Find min/max FLOPs
tp_flops_list = [data['torchprofile_flops'] for data in flops_data.values() if data['torchprofile_flops'] is not None]
if tp_flops_list:
    min_tp_flops = min(tp_flops_list)
    max_tp_flops = max(tp_flops_list)
    min_config = [k for k, v in flops_data.items() if v['torchprofile_flops'] == min_tp_flops][0]
    max_config = [k for k, v in flops_data.items() if v['torchprofile_flops'] == max_tp_flops][0]
    
    print(f"Lowest FLOPs: {min_config} ({min_tp_flops/1e9:.2f}G)")
    print(f"Highest FLOPs: {max_config} ({max_tp_flops/1e9:.2f}G)")
    print(f"FLOPs reduction: {((max_tp_flops - min_tp_flops) / max_tp_flops * 100):.1f}%")

# Parameter statistics
params_list = [data['parameters'] for data in flops_data.values()]
min_params = min(params_list)
max_params = max(params_list)
min_param_config = [k for k, v in flops_data.items() if v['parameters'] == min_params][0]
max_param_config = [k for k, v in flops_data.items() if v['parameters'] == max_params][0]

print(f"\nLowest parameters: {min_param_config} ({min_params:,})")
print(f"Highest parameters: {max_param_config} ({max_params:,})")
print(f"Parameter reduction: {((max_params - min_params) / max_params * 100):.1f}%")

# Efficiency analysis
print("\n" + "="*50)
print("EFFICIENCY ANALYSIS")
print("="*50)

for config in sorted(flops_data.keys()):
    data = flops_data[config]
    if data['torchprofile_flops']:
        efficiency = data['torchprofile_flops'] / data['parameters']  # FLOPs per parameter
        print(f"Config {config}: {efficiency:.0f} FLOPs/parameter ({data['torchprofile_flops_g']:.2f}G FLOPs, {data['parameters']:,} params)")

print("\n" + "="*80)