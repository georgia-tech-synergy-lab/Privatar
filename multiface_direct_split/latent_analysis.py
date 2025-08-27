#!/usr/bin/env python3
"""
Latent Code Analysis Script
Analyzes the distribution and correlation between two suites of latent codes from VAE models.
"""

import os
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('default')
sns.set_palette("husl")

class LatentCodeAnalyzer:
    def __init__(self, path1, path2):
        """
        Initialize the analyzer with two paths containing latent codes.
        
        Args:
            path1 (str): Path to first suite of latent codes
            path2 (str): Path to second suite of latent codes
        """
        self.path1 = Path(path1)
        self.path2 = Path(path2)
        self.data1 = None
        self.data2 = None
        self.correlation_matrix = None
        
    def load_latent_codes(self):
        """Load all latent codes from both paths."""
        print("Loading latent codes from both paths...")
        
        # Load from path 1
        files1 = sorted(glob.glob(str(self.path1 / "z_outsource_*.pth")))
        print(f"Found {len(files1)} files in path 1")
        
        data1_list = []
        for file_path in files1:
            try:
                tensor = torch.load(file_path, map_location='cpu')
                if tensor.dim() == 1:  # [256]
                    data1_list.append(tensor.numpy())
                elif tensor.dim() == 2:  # [1, 256]
                    data1_list.append(tensor.squeeze().numpy())
                else:
                    print(f"Warning: Unexpected tensor shape {tensor.shape} in {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        # Load from path 2
        files2 = sorted(glob.glob(str(self.path2 / "z_outsource_*.pth")))
        print(f"Found {len(files2)} files in path 2")
        
        data2_list = []
        for file_path in files2:
            try:
                tensor = torch.load(file_path, map_location='cpu')
                if tensor.dim() == 1:  # [256]
                    data2_list.append(tensor.numpy())
                elif tensor.dim() == 2:  # [1, 256]
                    data2_list.append(tensor.squeeze().numpy())
                else:
                    print(f"Warning: Unexpected tensor shape {tensor.shape} in {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        # Convert to numpy arrays
        self.data1 = np.array(data1_list)  # Shape: [N1, 256]
        self.data2 = np.array(data2_list)  # Shape: [N2, 256]
        
        print(f"Loaded data shapes:")
        print(f"  Path 1: {self.data1.shape}")
        print(f"  Path 2: {self.data2.shape}")
        
        return self.data1, self.data2
    
    def analyze_distributions(self):
        """Analyze the statistical distributions of both datasets."""
        print("\n=== Distribution Analysis ===")
        
        # Basic statistics
        print(f"Path 1 Statistics:")
        print(f"  Mean: {np.mean(self.data1):.6f}")
        print(f"  Std:  {np.std(self.data1):.6f}")
        print(f"  Min:  {np.min(self.data1):.6f}")
        print(f"  Max:  {np.max(self.data1):.6f}")
        print(f"  Range: {np.max(self.data1) - np.min(self.data1):.6f}")
        
        print(f"\nPath 2 Statistics:")
        print(f"  Mean: {np.mean(self.data2):.6f}")
        print(f"  Std:  {np.std(self.data2):.6f}")
        print(f"  Min:  {np.min(self.data2):.6f}")
        print(f"  Max:  {np.max(self.data2):.6f}")
        print(f"  Range: {np.max(self.data2) - np.min(self.data2):.6f}")
        
        # Per-dimension statistics
        print(f"\nPer-dimension statistics (first 10 dimensions):")
        for i in range(min(10, self.data1.shape[1])):
            mean1, std1 = np.mean(self.data1[:, i]), np.std(self.data1[:, i])
            mean2, std2 = np.mean(self.data2[:, i]), np.std(self.data2[:, i])
            print(f"  Dim {i:3d}: Path1({mean1:8.4f}, {std1:8.4f}) | Path2({mean2:8.4f}, {std2:8.4f})")
    
    def calculate_correlations(self):
        """Calculate correlations between the two datasets."""
        print("\n=== Correlation Analysis ===")
        
        # Calculate correlation matrix between all dimensions
        # We'll use the mean of each dataset across samples for each dimension
        mean1 = np.mean(self.data1, axis=0)  # [256]
        mean2 = np.mean(self.data2, axis=0)  # [256]
        
        # Pearson correlation
        pearson_corr, pearson_p = pearsonr(mean1, mean2)
        print(f"Pearson correlation: {pearson_corr:.6f} (p-value: {pearson_p:.6f})")
        
        # Spearman correlation
        spearman_corr, spearman_p = spearmanr(mean1, mean2)
        print(f"Spearman correlation: {spearman_corr:.6f} (p-value: {spearman_p:.6f})")
        
        # Calculate cross-correlation matrix between dimensions
        # Since the datasets have different numbers of samples, we'll calculate
        # correlations between the mean values across samples for each dimension
        n_dims = self.data1.shape[1]
        
        # Calculate cross-correlation matrix
        cross_corr = np.zeros((n_dims, n_dims))
        for i in range(n_dims):
            for j in range(n_dims):
                # Calculate correlation between dimension i from dataset 1 and dimension j from dataset 2
                # We'll use the correlation between the sample values for these dimensions
                corr, _ = pearsonr(self.data1[:, i], self.data2[:min(len(self.data1), len(self.data2)), j])
                cross_corr[i, j] = corr
        
        # Create a full correlation matrix for visualization purposes
        # This will be a block matrix with internal correlations and cross-correlations
        total_dims = n_dims * 2
        self.correlation_matrix = np.zeros((total_dims, total_dims))
        
        # Fill in the blocks
        # Top-left: Dataset 1 internal correlations
        self.correlation_matrix[:n_dims, :n_dims] = np.corrcoef(self.data1.T)
        # Bottom-right: Dataset 2 internal correlations  
        self.correlation_matrix[n_dims:, n_dims:] = np.corrcoef(self.data2.T)
        # Top-right: Cross-correlations
        self.correlation_matrix[:n_dims, n_dims:] = cross_corr
        # Bottom-left: Transpose of cross-correlations
        self.correlation_matrix[n_dims:, :n_dims] = cross_corr.T
        
        print(f"Cross-correlation matrix shape: {cross_corr.shape}")
        print(f"Average cross-correlation: {np.mean(cross_corr):.6f}")
        print(f"Max cross-correlation: {np.max(cross_corr):.6f}")
        print(f"Min cross-correlation: {np.min(cross_corr):.6f}")
        
        return cross_corr
    
    def create_visualizations(self, output_dir="latent_analysis_results"):
        """Create comprehensive visualizations."""
        print(f"\n=== Creating Visualizations ===")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 1. Distribution comparison plots
        self._plot_distributions(output_path)
        
        # 2. Correlation heatmap
        self._plot_correlation_heatmap(output_path)
        
        # 3. Per-dimension comparison
        self._plot_dimension_comparison(output_path)
        
        # 4. Statistical summary plots
        self._plot_statistical_summary(output_path)
        
        print(f"Visualizations saved to: {output_path}")
    
    def _plot_distributions(self, output_path):
        """Plot distribution comparisons."""
        # Create standalone Overall Distribution Comparison plot
        self._plot_overall_distribution_standalone(output_path)
        
        # Create the original 2x2 subplot figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Latent Code Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Overall distribution
        axes[0, 0].hist(self.data1.flatten(), bins=50, alpha=0.7, label='Randomly 2 seconds', density=True)
        axes[0, 0].hist(self.data2.flatten(), bins=50, alpha=0.7, label='Historical distribution', density=True)
        axes[0, 0].set_xlabel('Latent Code Values')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Overall Distribution Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Box plot comparison
        axes[0, 1].boxplot([self.data1.flatten(), self.data2.flatten()], 
                           labels=['Randomly 2 seconds', 'Historical distribution'])
        axes[0, 1].set_ylabel('Latent Code Values')
        axes[0, 1].set_title('Box Plot Comparison')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Per-dimension mean comparison
        mean1 = np.mean(self.data1, axis=0)
        mean2 = np.mean(self.data2, axis=0)
        axes[1, 0].plot(mean1, label='Randomly 2 seconds', alpha=0.8)
        axes[1, 0].plot(mean2, label='Historical distribution', alpha=0.8)
        axes[1, 0].set_xlabel('Dimension')
        axes[1, 0].set_ylabel('Mean Value')
        axes[1, 0].set_title('Per-Dimension Mean Comparison')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Per-dimension std comparison
        std1 = np.std(self.data1, axis=0)
        std2 = np.std(self.data2, axis=0)
        axes[1, 1].plot(std1, label='Randomly 2 seconds', alpha=0.8)
        axes[1, 1].plot(std2, label='Historical distribution', alpha=0.8)
        axes[1, 1].set_xlabel('Dimension')
        axes[1, 1].set_ylabel('Standard Deviation')
        axes[1, 1].set_title('Per-Dimension Std Comparison')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_overall_distribution_standalone(self, output_path):
        """Create standalone Overall Distribution Comparison plot in PDF format."""
        # Set font size to 24
        plt.rcParams.update({'font.size': 28})
        
        # Create figure with 16:9 aspect ratio
        fig, ax = plt.subplots(figsize=(16, 9))
        
        # Plot histograms with updated labels
        ax.hist(self.data1.flatten(), bins=50, alpha=0.7, label='2s Video Clip', density=True)
        ax.hist(self.data2.flatten(), bins=50, alpha=0.7, label='Historical Distribution', density=True)
        
        # Set labels and title with larger font
        ax.set_xlabel('Latent Code Values', fontsize=28)
        ax.set_ylabel('Density', fontsize=28)
        # ax.set_title('Overall Distribution Comparison', fontsize=28, fontweight='bold')
        
        # Customize legend
        ax.legend(fontsize=28)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Adjust layout and save as PDF
        plt.tight_layout()
        plt.savefig(output_path / 'overall_distribution_comparison.pdf', format='pdf', bbox_inches='tight', dpi=300)
        plt.close()
        
        # Reset font size to default
        plt.rcParams.update({'font.size': 10})
    
    def _plot_correlation_heatmap(self, output_path):
        """Plot correlation heatmap."""
        if self.correlation_matrix is None:
            self.calculate_correlations()
        
        # Create the full correlation matrix visualization
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create labels for the heatmap
        n_dims = self.data1.shape[1]
        labels = [f'P1_{i}' for i in range(n_dims)] + [f'P2_{i}' for i in range(n_dims)]
        
        # Create mask for better visualization
        mask = np.zeros_like(self.correlation_matrix, dtype=bool)
        mask[n_dims:, :n_dims] = True  # Mask the bottom-left quadrant
        
        # Plot heatmap
        sns.heatmap(self.correlation_matrix, 
                   mask=mask,
                   annot=False,  # Too many annotations for 256x256
                   cmap='RdBu_r', 
                   center=0,
                   square=True,
                   cbar_kws={'label': 'Correlation Coefficient'},
                   ax=ax)
        
        ax.set_title('Latent Code Correlation Matrix\n(Randomly 2 seconds vs Historical distribution)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Dimensions')
        ax.set_ylabel('Dimensions')
        
        # Add text annotations for the quadrants
        ax.text(0.25, 0.25, 'Randomly 2 seconds\nInternal\nCorrelations', 
                transform=ax.transAxes, ha='center', va='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        ax.text(0.75, 0.75, 'Historical distribution\nInternal\nCorrelations', 
                transform=ax.transAxes, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        ax.text(0.75, 0.25, 'Cross\nCorrelations\n(Randomly 2 seconds vs Historical distribution)', 
                transform=ax.transAxes, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(output_path / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a focused cross-correlation heatmap
        cross_corr = self.correlation_matrix[:n_dims, n_dims:]
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(cross_corr, 
                   annot=False,
                   cmap='RdBu_r', 
                   center=0,
                   square=True,
                   cbar_kws={'label': 'Cross-Correlation Coefficient'},
                   ax=ax)
        
        ax.set_title('Cross-Correlation Heatmap\n(Randomly 2 seconds Dimensions vs Historical distribution Dimensions)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Historical distribution Dimensions')
        ax.set_ylabel('Randomly 2 seconds Dimensions')
        
        plt.tight_layout()
        plt.savefig(output_path / 'cross_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_dimension_comparison(self, output_path):
        """Plot detailed dimension-by-dimension comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Detailed Dimension Analysis', fontsize=16, fontweight='bold')
        
        # Scatter plot of means
        mean1 = np.mean(self.data1, axis=0)
        mean2 = np.mean(self.data2, axis=0)
        axes[0, 0].scatter(mean1, mean2, alpha=0.6)
        axes[0, 0].plot([mean1.min(), mean1.max()], [mean2.min(), mean2.max()], 'r--', alpha=0.8)
        axes[0, 0].set_xlabel('Randomly 2 seconds Mean Values')
        axes[0, 0].set_ylabel('Historical distribution Mean Values')
        axes[0, 0].set_title('Mean Values Correlation')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Scatter plot of standard deviations
        std1 = np.std(self.data1, axis=0)
        std2 = np.std(self.data2, axis=0)
        axes[0, 1].scatter(std1, std2, alpha=0.6)
        axes[0, 1].plot([std1.min(), std1.max()], [std2.min(), std2.max()], 'r--', alpha=0.8)
        axes[0, 1].set_xlabel('Randomly 2 seconds Standard Deviations')
        axes[0, 1].set_ylabel('Historical distribution Standard Deviations')
        axes[0, 1].set_title('Standard Deviation Correlation')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Difference plot
        diff_means = mean1 - mean2
        axes[1, 0].plot(diff_means, alpha=0.8)
        axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.8)
        axes[1, 0].set_xlabel('Dimension')
        axes[1, 0].set_ylabel('Difference (Randomly 2 seconds - Historical distribution)')
        axes[1, 0].set_title('Mean Value Differences')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Ratio plot
        ratio_means = mean1 / (mean2 + 1e-8)  # Avoid division by zero
        axes[1, 1].plot(ratio_means, alpha=0.8)
        axes[1, 1].axhline(y=1, color='r', linestyle='--', alpha=0.8)
        axes[1, 1].set_xlabel('Dimension')
        axes[1, 1].set_ylabel('Ratio (Randomly 2 seconds / Historical distribution)')
        axes[1, 1].set_title('Mean Value Ratios')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'dimension_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_statistical_summary(self, output_path):
        """Plot statistical summary information."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Statistical Summary', fontsize=16, fontweight='bold')
        
        # Histogram of correlation coefficients
        if self.correlation_matrix is None:
            self.calculate_correlations()
        
        n_dims = self.data1.shape[1]
        cross_corr = self.correlation_matrix[:n_dims, n_dims:]
        
        axes[0, 0].hist(cross_corr.flatten(), bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Cross-Correlation Coefficient')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Cross-Correlations')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Cumulative distribution of correlations
        sorted_corr = np.sort(cross_corr.flatten())
        cumulative = np.arange(1, len(sorted_corr) + 1) / len(sorted_corr)
        axes[0, 1].plot(sorted_corr, cumulative, linewidth=2)
        axes[0, 1].set_xlabel('Cross-Correlation Coefficient')
        axes[0, 1].set_ylabel('Cumulative Probability')
        axes[0, 1].set_title('Cumulative Distribution of Correlations')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Top and bottom correlations
        top_indices = np.unravel_index(np.argsort(cross_corr.flatten())[-20:], cross_corr.shape)
        bottom_indices = np.unravel_index(np.argsort(cross_corr.flatten())[:20], cross_corr.shape)
        
        axes[1, 0].bar(range(20), sorted_corr[-20:], alpha=0.7, color='green')
        axes[1, 0].set_xlabel('Rank')
        axes[1, 0].set_ylabel('Correlation Coefficient')
        axes[1, 0].set_title('Top 20 Correlations')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].bar(range(20), sorted_corr[:20], alpha=0.7, color='red')
        axes[1, 1].set_xlabel('Rank')
        axes[1, 1].set_ylabel('Correlation Coefficient')
        axes[1, 1].set_title('Bottom 20 Correlations')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'statistical_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, output_dir="latent_analysis_results"):
        """Save all analysis results to files."""
        print(f"\n=== Saving Results ===")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save numerical results
        results = {
            'path1_shape': self.data1.shape,
            'path2_shape': self.data2.shape,
            'path1_mean': np.mean(self.data1),
            'path1_std': np.std(self.data1),
            'path1_min': np.min(self.data1),
            'path1_max': np.max(self.data1),
            'path2_mean': np.mean(self.data2),
            'path2_std': np.std(self.data2),
            'path2_min': np.min(self.data2),
            'path2_max': np.max(self.data2),
        }
        
        # Calculate correlations if not already done
        if self.correlation_matrix is None:
            self.calculate_correlations()
        
        n_dims = self.data1.shape[1]
        cross_corr = self.correlation_matrix[:n_dims, n_dims:]
        
        results.update({
            'cross_correlation_mean': np.mean(cross_corr),
            'cross_correlation_std': np.std(cross_corr),
            'cross_correlation_min': np.min(cross_corr),
            'cross_correlation_max': np.max(cross_corr),
        })
        
        # Save results to text file
        with open(output_path / 'analysis_results.txt', 'w') as f:
            f.write("Latent Code Analysis Results\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Data Summary:\n")
            f.write(f"  Path 1: {results['path1_shape']}\n")
            f.write(f"  Path 2: {results['path2_shape']}\n\n")
            
            f.write("Path 1 Statistics:\n")
            f.write(f"  Mean: {results['path1_mean']:.6f}\n")
            f.write(f"  Std:  {results['path1_std']:.6f}\n")
            f.write(f"  Min:  {results['path1_min']:.6f}\n")
            f.write(f"  Max:  {results['path1_max']:.6f}\n\n")
            
            f.write("Path 2 Statistics:\n")
            f.write(f"  Mean: {results['path2_mean']:.6f}\n")
            f.write(f"  Std:  {results['path2_std']:.6f}\n")
            f.write(f"  Min:  {results['path2_min']:.6f}\n")
            f.write(f"  Max:  {results['path2_max']:.6f}\n\n")
            
            f.write("Cross-Correlation Statistics:\n")
            f.write(f"  Mean: {results['cross_correlation_mean']:.6f}\n")
            f.write(f"  Std:  {results['cross_correlation_std']:.6f}\n")
            f.write(f"  Min:  {results['cross_correlation_min']:.6f}\n")
            f.write(f"  Max:  {results['cross_correlation_max']:.6f}\n")
        
        # Save correlation matrix
        np.save(output_path / 'correlation_matrix.npy', self.correlation_matrix)
        np.save(output_path / 'cross_correlation_matrix.npy', cross_corr)
        
        # Save per-dimension statistics
        mean1 = np.mean(self.data1, axis=0)
        std1 = np.std(self.data1, axis=0)
        mean2 = np.mean(self.data2, axis=0)
        std2 = np.std(self.data2, axis=0)
        
        dim_stats = np.column_stack([mean1, std1, mean2, std2])
        np.save(output_path / 'dimension_statistics.npy', dim_stats)
        
        print(f"Results saved to: {output_path}")
        print(f"  - analysis_results.txt: Text summary")
        print(f"  - correlation_matrix.npy: Full correlation matrix")
        print(f"  - cross_correlation_matrix.npy: Cross-correlation matrix")
        print(f"  - dimension_statistics.npy: Per-dimension statistics")
    
    def run_full_analysis(self, output_dir="latent_analysis_results"):
        """Run the complete analysis pipeline."""
        print("Starting Latent Code Analysis...")
        print("=" * 50)
        
        # Load data
        self.load_latent_codes()
        
        # Analyze distributions
        self.analyze_distributions()
        
        # Calculate correlations
        self.calculate_correlations()
        
        # Create visualizations
        self.create_visualizations(output_dir)
        
        # Save results
        self.save_results(output_dir)
        
        print("\nAnalysis complete!")
        print("=" * 50)


def main():
    """Main function to run the analysis."""
    # Define paths
    path1 = "/scratch2/jianming/work/Privatar_prj/render_results/original_vae_direct_split_exp/latent_code"
    path2 = "/scratch2/jianming/work/Privatar_prj/render_results/original_vae_direct_split/latent_code"
    
    # Create analyzer
    analyzer = LatentCodeAnalyzer(path1, path2)
    
    # Run full analysis
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
