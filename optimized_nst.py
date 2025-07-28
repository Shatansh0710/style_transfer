import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time
import os
import warnings
warnings.filterwarnings('ignore')


class VGGFeatureExtractor(nn.Module):
    """
    Enhanced VGG19 model for extracting content and style features.
    Uses proper layer naming and feature extraction.
    """
    
    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        
        # Load pretrained VGG19 - Fixed deprecation warning
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.eval()
        
        # Define layer mappings (VGG19 layer indices)
        self.layer_mapping = {
            '0': 'conv1_1',   # First conv layer
            '2': 'conv1_2',
            '5': 'conv2_1',
            '7': 'conv2_2',
            '10': 'conv3_1',
            '12': 'conv3_2',
            '14': 'conv3_3',
            '16': 'conv3_4',
            '19': 'conv4_1',
            '21': 'conv4_2',  # Content layer
            '23': 'conv4_3',
            '25': 'conv4_4',
            '28': 'conv5_1',
            '30': 'conv5_2',
            '32': 'conv5_3',
            '34': 'conv5_4'
        }
        
        # Content and style layers
        self.content_layers = ['conv4_2']  # Deep layer for content
        self.style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']  # Multiple levels
        
        # Store VGG layers
        self.features = vgg
        
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad_(False)
    
    def forward(self, x):
        """Extract features from specified layers"""
        features = {}
        
        for name, layer in self.features._modules.items():
            if layer is not None:  # Check if layer exists
                x = layer(x)
                if name in self.layer_mapping:
                    layer_name = self.layer_mapping[name]
                    if layer_name in self.content_layers or layer_name in self.style_layers:
                        features[layer_name] = x
        
        return features


def gram_matrix(tensor):
    """
    Compute normalized Gram matrix for style representation.
    """
    batch_size, channels, height, width = tensor.size()
    features = tensor.view(batch_size, channels, height * width)
    gram = torch.bmm(features, features.transpose(1, 2))
    # Normalize by number of elements
    return gram.div(channels * height * width)


def total_variation_loss(tensor):
    """
    Compute total variation loss for image smoothness.
    """
    batch_size, channels, height, width = tensor.size()
    
    # Horizontal variation
    h_var = torch.sum(torch.abs(tensor[:, :, 1:, :] - tensor[:, :, :-1, :]))
    # Vertical variation  
    w_var = torch.sum(torch.abs(tensor[:, :, :, 1:] - tensor[:, :, :, :-1]))
    
    return (h_var + w_var) / (batch_size * channels * height * width)


def load_and_preprocess_image(image_path: str, image_size: int = 512, device: torch.device = torch.device('cpu')):
    """Load and preprocess image with proper error handling"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # ImageNet normalization
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    try:
        pil_image = Image.open(image_path).convert('RGB')
        tensor = transform(pil_image)
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.from_numpy(np.array(tensor))
        return tensor.unsqueeze(0).to(device, dtype=torch.float32)
    except Exception as e:
        raise ValueError(f"Error processing image {image_path}: {e}")


def tensor_to_image(tensor):
    """Convert tensor back to displayable image with proper denormalization"""
    # Denormalize using ImageNet statistics
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    tensor = tensor.cpu().clone().squeeze(0)
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy and transpose
    image = tensor.permute(1, 2, 0).detach().numpy()
    return image


def safe_item_conversion(value):
    """Safely convert tensor or float to Python float - FIXES .item() ERRORS"""
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:  # Single element tensor
            return value.item()
        else:
            return float(value.mean())  # For multi-element tensors
    elif isinstance(value, (int, float, np.number)):
        return float(value)
    else:
        return 0.0  # Fallback


class OptimizedStyleTransfer:
    """
    Optimized Neural Style Transfer with proper loss computation and optimization
    """
    
    def __init__(self, content_weight: float = 1.0, style_weight: float = 1e6, tv_weight: float = 1e-6, 
                 learning_rate: float = 1.0, device: torch.device = torch.device('cpu')):
        self.device = device
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight
        self.learning_rate = learning_rate
        
        # Initialize feature extractor
        self.vgg = VGGFeatureExtractor().to(device)
        
        # Loss tracking
        self.losses = {
            'content': [],
            'style': [],
            'tv': [],
            'total': []
        }
        
        print(f"Initialized StyleTransfer on {device}")
        print(f"Weights - Content: {content_weight}, Style: {style_weight}, TV: {tv_weight}")
    
    def compute_content_loss(self, content_features, target_features):
        """Compute normalized content loss"""
        loss = 0.0
        for layer in self.vgg.content_layers:
            if layer in content_features and layer in target_features:
                # Normalized MSE loss
                target_feat = target_features[layer]
                content_feat = content_features[layer]
                loss += nn.functional.mse_loss(target_feat, content_feat)
        return loss
    
    def compute_style_loss(self, style_features, target_features):
        """Compute normalized style loss using Gram matrices"""
        loss = 0.0
        num_layers = len(self.vgg.style_layers)
        
        for layer in self.vgg.style_layers:
            if layer in style_features and layer in target_features:
                # Compute Gram matrices
                target_gram = gram_matrix(target_features[layer])
                style_gram = gram_matrix(style_features[layer])
                
                # MSE loss between Gram matrices
                layer_loss = nn.functional.mse_loss(target_gram, style_gram)
                loss += layer_loss / num_layers  # Normalize by number of layers
        
        return loss
    
    def transfer_style(self, content_image, style_image, num_epochs=500, save_interval=100, 
                      output_dir='outputs', use_lbfgs=True):
        """
        Enhanced style transfer with better optimization - FIXES OPTIMIZER ERRORS
        """
        print(f"Starting optimized style transfer...")
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract reference features
        with torch.no_grad():
            content_features = self.vgg(content_image)
            style_features = self.vgg(style_image)
        
        # Initialize target image (start with content + noise for better convergence)
        target_image = content_image.clone()
        if not use_lbfgs:
            # Add small amount of noise for Adam optimizer
            noise = torch.randn_like(target_image) * 0.01
            target_image = target_image + noise
        
        target_image.requires_grad_(True)
        
        # Choose optimizer - FIXED INITIALIZATION
        if use_lbfgs:
            optimizer = optim.LBFGS([target_image], lr=self.learning_rate, max_iter=20)
        else:
            optimizer = optim.Adam([target_image], lr=0.01)
        
        start_time = time.time()
        epoch = 0
        
        # FIXED CLOSURE FUNCTION
        def closure():
            nonlocal epoch
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass
            target_features = self.vgg(target_image)
            
            # Compute losses
            content_loss = self.compute_content_loss(content_features, target_features)
            style_loss = self.compute_style_loss(style_features, target_features)
            tv_loss = total_variation_loss(target_image)
            
            # Total loss
            total_loss = (self.content_weight * content_loss + 
                         self.style_weight * style_loss + 
                         self.tv_weight * tv_loss)
            
            # Backward pass
            total_loss.backward()
            
            # Store losses - FIXED .item() CALLS
            self.losses['content'].append(safe_item_conversion(content_loss))
            self.losses['style'].append(safe_item_conversion(style_loss))
            self.losses['tv'].append(safe_item_conversion(tv_loss))
            self.losses['total'].append(safe_item_conversion(total_loss))
            
            # Progress reporting
            if epoch % 50 == 0:
                elapsed = time.time() - start_time
                content_val = safe_item_conversion(content_loss)
                style_val = safe_item_conversion(style_loss)
                tv_val = safe_item_conversion(tv_loss)
                total_val = safe_item_conversion(total_loss)
                
                print(f"Epoch {epoch:3d} | Content: {content_val:.4f} | "
                      f"Style: {style_val:.4f} | TV: {tv_val:.6f} | "
                      f"Total: {total_val:.4f} | Time: {elapsed:.1f}s")
                
                # Save intermediate results
                if epoch > 0 and epoch % save_interval == 0:
                    with torch.no_grad():
                        img = tensor_to_image(target_image)
                        plt.imsave(f"{output_dir}/epoch_{epoch:04d}.png", img)
            
            epoch += 1
            return total_loss
        
        # FIXED OPTIMIZATION LOOP
        if use_lbfgs:
            for i in range(num_epochs):
                # LBFGS requires closure function - FIXED
                def step_closure():
                    return closure()
                
                loss_value = optimizer.step(step_closure)
                
                # Clamp values
                with torch.no_grad():
                    target_image.data.clamp_(-3, 3)
                
                if epoch >= num_epochs:
                    break
        else:
            # Adam optimization loop
            for i in range(num_epochs):
                # Clear gradients first
                optimizer.zero_grad()
                
                # Forward pass
                target_features = self.vgg(target_image)
                
                # Compute losses
                content_loss = self.compute_content_loss(content_features, target_features)
                style_loss = self.compute_style_loss(style_features, target_features)
                tv_loss = total_variation_loss(target_image)
                
                # Total loss
                total_loss = (self.content_weight * content_loss + 
                             self.style_weight * style_loss + 
                             self.tv_weight * tv_loss)
                
                # Backward pass
                total_loss.backward()
                
                # Store losses
                self.losses['content'].append(safe_item_conversion(content_loss))
                self.losses['style'].append(safe_item_conversion(style_loss))
                self.losses['tv'].append(safe_item_conversion(tv_loss))
                self.losses['total'].append(safe_item_conversion(total_loss))
                
                # Progress reporting
                if epoch % 50 == 0:
                    elapsed = time.time() - start_time
                    content_val = safe_item_conversion(content_loss)
                    style_val = safe_item_conversion(style_loss)
                    tv_val = safe_item_conversion(tv_loss)
                    total_val = safe_item_conversion(total_loss)
                    
                    print(f"Epoch {epoch:3d} | Content: {content_val:.4f} | "
                          f"Style: {style_val:.4f} | TV: {tv_val:.6f} | "
                          f"Total: {total_val:.4f} | Time: {elapsed:.1f}s")
                    
                    # Save intermediate results
                    if epoch > 0 and epoch % save_interval == 0:
                        with torch.no_grad():
                            img = tensor_to_image(target_image)
                            plt.imsave(f"{output_dir}/epoch_{epoch:04d}.png", img)
                
                # Optimizer step - handle different optimizer types
                if isinstance(optimizer, optim.LBFGS):
                    # LBFGS needs closure
                    def step_closure():
                        return total_loss
                    optimizer.step(step_closure)
                else:
                    # Adam and other optimizers don't need closure
                    optimizer.step()
                
                # Clamp values
                with torch.no_grad():
                    target_image.data.clamp_(-3, 3)
                
                epoch += 1
        
        total_time = time.time() - start_time
        print(f"\nOptimization completed in {total_time:.2f} seconds")
        print(f"Total iterations: {len(self.losses['total'])}")
        
        return target_image.detach()
    
    def plot_losses(self, save_path='outputs/loss_curves.png'):
        """Plot comprehensive loss curves"""
        if not any(self.losses.values()):
            print("No loss data to plot")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Content loss
        if self.losses['content']:
            axes[0, 0].plot(self.losses['content'])
            axes[0, 0].set_title('Content Loss')
            axes[0, 0].set_xlabel('Iteration')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True)
            axes[0, 0].set_yscale('log')
        
        # Style loss
        if self.losses['style']:
            axes[0, 1].plot(self.losses['style'], color='orange')
            axes[0, 1].set_title('Style Loss')
            axes[0, 1].set_xlabel('Iteration')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True)
            axes[0, 1].set_yscale('log')
        
        # Total Variation loss
        if self.losses['tv']:
            axes[1, 0].plot(self.losses['tv'], color='green')
            axes[1, 0].set_title('Total Variation Loss')
            axes[1, 0].set_xlabel('Iteration')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True)
            axes[1, 0].set_yscale('log')
        
        # Total loss
        if self.losses['total']:
            axes[1, 1].plot(self.losses['total'], color='red')
            axes[1, 1].set_title('Total Loss')
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].grid(True)
            axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Loss curves saved to {save_path}")
    
    def create_comparison_grid(self, content_img, style_img, result_img, save_path='outputs/comparison.png'):
        """Create enhanced comparison visualization"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Top row - original images
        axes[0, 0].imshow(tensor_to_image(content_img))
        axes[0, 0].set_title('Content Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(tensor_to_image(style_img))
        axes[0, 1].set_title('Style Image', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(tensor_to_image(result_img))
        axes[0, 2].set_title('Stylized Result', fontsize=14, fontweight='bold')
        axes[0, 2].axis('off')
        
        # Bottom row - analysis
        axes[1, 0].imshow(tensor_to_image(content_img))
        axes[1, 0].set_title('Original Content', fontsize=12)
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(tensor_to_image(style_img))
        axes[1, 1].set_title('Style Reference', fontsize=12)
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(tensor_to_image(result_img))
        final_loss = self.losses['total'][-1] if self.losses['total'] else 0
        convergence = ((self.losses['total'][0] - self.losses['total'][-1]) / 
                      self.losses['total'][0] * 100) if len(self.losses['total']) > 1 else 0
        axes[1, 2].set_title(f'Final Result\nLoss: {final_loss:.2f}, Conv: {convergence:.1f}%', 
                            fontsize=10)
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Enhanced comparison saved to {save_path}")


def main():
    """Enhanced main function with better argument handling"""
    parser = argparse.ArgumentParser(description='Optimized Neural Style Transfer')
    parser.add_argument('--content', type=str, required=True, help='Content image path')
    parser.add_argument('--style', type=str, required=True, help='Style image path')
    parser.add_argument('--output', type=str, default='outputs/stylized_result.png', help='Output path')
    parser.add_argument('--size', type=int, default=512, help='Image size (default: 512)')
    parser.add_argument('--epochs', type=int, default=300, help='Optimization iterations (default: 300)')
    parser.add_argument('--content_weight', type=float, default=1.0, help='Content loss weight (default: 1.0)')
    parser.add_argument('--style_weight', type=float, default=1e6, help='Style loss weight (default: 1e6)')
    parser.add_argument('--tv_weight', type=float, default=1e-6, help='Total variation weight (default: 1e-6)')
    parser.add_argument('--lr', type=float, default=1.0, help='Learning rate (default: 1.0 for LBFGS)')
    parser.add_argument('--optimizer', type=str, default='lbfgs', choices=['lbfgs', 'adam'], 
                       help='Optimizer choice (default: lbfgs)')
    
    args = parser.parse_args()
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    try:
        # Load and validate images
        print("Loading images...")
        content_image = load_and_preprocess_image(args.content, args.size, device)
        style_image = load_and_preprocess_image(args.style, args.size, device)
        
        # Initialize model
        model = OptimizedStyleTransfer(
            content_weight=args.content_weight,
            style_weight=args.style_weight,
            tv_weight=args.tv_weight,
            learning_rate=args.lr,
            device=device
        )
        
        # Perform style transfer
        use_lbfgs = args.optimizer == 'lbfgs'
        result = model.transfer_style(
            content_image, style_image, 
            num_epochs=args.epochs,
            use_lbfgs=use_lbfgs
        )
        
        # Save results
        os.makedirs(os.path.dirname(args.output) or 'outputs', exist_ok=True)
        final_image = tensor_to_image(result)
        plt.imsave(args.output, final_image)
        print(f"Final result saved to {args.output}")
        
        # Generate analysis
        model.plot_losses()
        model.create_comparison_grid(content_image, style_image, result)
        
        # Print comprehensive evaluation - FIXED LOSS ACCESS
        print("\n" + "="*60)
        print("OPTIMIZATION RESULTS")
        print("="*60)
        
        if model.losses['total']:
            final_content = model.losses['content'][-1] if model.losses['content'] else 0
            final_style = model.losses['style'][-1] if model.losses['style'] else 0
            final_tv = model.losses['tv'][-1] if model.losses['tv'] else 0
            final_total = model.losses['total'][-1] if model.losses['total'] else 0
            convergence = ((model.losses['total'][0] - final_total) / 
                          model.losses['total'][0] * 100) if len(model.losses['total']) > 1 else 0
            
            print(f"Final Content Loss:    {final_content:.6f}")
            print(f"Final Style Loss:      {final_style:.6f}")
            print(f"Final TV Loss:         {final_tv:.6f}")
            print(f"Final Total Loss:      {final_total:.6f}")
            print(f"Loss Convergence:      {convergence:.2f}%")
            print(f"Total Iterations:      {len(model.losses['total'])}")
            print(f"Optimizer Used:        {args.optimizer.upper()}")
            
            # Quality assessment
            if final_total < 1000:
                quality = "Excellent"
            elif final_total < 5000:
                quality = "Good"
            elif final_total < 15000:
                quality = "Fair"
            else:
                quality = "Poor - Consider adjusting weights"
                
            print(f"Result Quality:        {quality}")
        
        print("="*60)
        
    except Exception as e:
        print(f"Error during style transfer: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        print("Optimized Neural Style Transfer")
        print("=" * 40)
        print("Usage:")
        print("python optimized_nst.py --content content.jpg --style style.jpg")
        print("\nOptional parameters:")
        print("  --size 512              # Image size")
        print("  --epochs 300            # Optimization iterations") 
        print("  --content_weight 1.0    # Content preservation")
        print("  --style_weight 1e6      # Style transfer strength")
        print("  --tv_weight 1e-6        # Smoothness regularization")
        print("  --optimizer lbfgs       # lbfgs or adam")
        print("\nExample:")
        print("python optimized_nst.py --content photo.jpg --style painting.jpg --epochs 500")
    else:
        exit_code = main()
        sys.exit(exit_code)
