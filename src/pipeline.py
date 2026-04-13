"""
Restoration Pipeline Module (Core + Novelty 2: Multi-Step)
==========================================================

Wraps the base InstructIR `process_img` and extends it with:
- Multi-step restoration via parsed compound instructions
- Region-aware restoration (Phase 3)
- Confidence-aware restoration (Phase 4)

Usage:
    pipeline = RestorationPipeline(model, language_model, lm_head, device)
    result = pipeline.restore("Remove noise and then sharpen edges", image)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any

from .instruction_parser import InstructionParser


class RestorationPipeline:
    """
    Unified restoration pipeline wrapping InstructIR with novelty extensions.
    """
    
    def __init__(self, model, language_model, lm_head, device):
        """
        Args:
            model: The InstructIR image restoration model.
            language_model: The LanguageModel for text encoding.
            lm_head: The LMHead for degradation prediction.
            device: torch device (cuda or cpu).
        """
        self.model = model
        self.language_model = language_model
        self.lm_head = lm_head
        self.device = device
        self.parser = InstructionParser()
    
    def process_single(self, image: np.ndarray, prompt: str) -> np.ndarray:
        """
        Single-step restoration (base InstructIR behavior).
        
        Args:
            image: RGB image as numpy array normalized to [0,1], shape (H, W, 3).
            prompt: Plain text instruction string.
            
        Returns:
            Restored image as numpy array, shape (H, W, 3), clipped to [0,1].
        """
        # Convert image → tensor → move to device
        y = torch.Tensor(image).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # Text embedding
        lm_embd = self.language_model(prompt)
        lm_embd = lm_embd.to(self.device)
        text_embd, deg_pred = self.lm_head(lm_embd)
        
        # Forward pass
        with torch.no_grad():
            x_hat = self.model(y, text_embd)
        
        # Convert output → numpy
        restored_img = x_hat[0].permute(1, 2, 0).cpu().detach().numpy()
        restored_img = np.clip(restored_img, 0., 1.)
        
        return restored_img
    
    def process_single_with_embeddings(
        self, image: np.ndarray, prompt: str
    ) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """
        Single-step restoration returning embeddings for downstream use.
        
        Returns:
            (restored_image, text_embedding, degradation_prediction)
        """
        y = torch.Tensor(image).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        lm_embd = self.language_model(prompt)
        lm_embd = lm_embd.to(self.device)
        text_embd, deg_pred = self.lm_head(lm_embd)
        
        with torch.no_grad():
            x_hat = self.model(y, text_embd)
        
        restored_img = x_hat[0].permute(1, 2, 0).cpu().detach().numpy()
        restored_img = np.clip(restored_img, 0., 1.)
        
        return restored_img, text_embd.detach(), deg_pred.detach()
    
    # ─── Novelty 2: Multi-Step Restoration ────────────────────────────────
    
    def restore_multistep(
        self, image: np.ndarray, compound_prompt: str
    ) -> Dict[str, Any]:
        """
        Multi-step restoration: parse compound prompt and apply sequentially.
        
        Args:
            image: Input degraded image (H, W, 3), normalized to [0,1].
            compound_prompt: Compound instruction (may contain multiple steps).
            
        Returns:
            Dictionary with:
                'final': final restored image
                'steps': list of dicts, each with 'label', 'instruction', 'output'
                'original': the input image
                'compound_prompt': the original compound prompt
        """
        steps = self.parser.parse(compound_prompt)
        
        results = {
            'original': image.copy(),
            'compound_prompt': compound_prompt,
            'steps': [],
            'final': None,
        }
        
        current = image.copy()
        
        for label, sub_instruction in steps:
            restored = self.process_single(current, sub_instruction)
            results['steps'].append({
                'label': label,
                'instruction': sub_instruction,
                'input': current.copy(),
                'output': restored.copy(),
            })
            current = restored
        
        results['final'] = current
        return results
    
    # ─── Novelty 3: Region-Aware Restoration ──────────────────────────────
    
    def restore_region(
        self, image: np.ndarray, prompt: str, 
        mask: np.ndarray, feather_radius: int = 5
    ) -> Dict[str, Any]:
        """
        Region-aware restoration: apply restoration only to masked regions.
        
        Args:
            image: Input image (H, W, 3), normalized to [0,1].
            prompt: Restoration instruction.
            mask: Binary mask (H, W) or (H, W, 1). 1 = restore, 0 = keep original.
            feather_radius: Gaussian blur radius for soft mask edges.
            
        Returns:
            Dictionary with 'final', 'full_restored', 'mask', 'blended'.
        """
        from .region_selector import RegionSelector
        
        # Process full image with the model
        full_restored = self.process_single(image, prompt)
        
        # Apply mask blending
        selector = RegionSelector()
        blended = selector.apply_mask_blend(image, full_restored, mask, feather_radius)
        
        return {
            'original': image.copy(),
            'full_restored': full_restored,
            'mask': mask,
            'blended': blended,
            'final': blended,
            'prompt': prompt,
        }
    
    # ─── Novelty 4: Confidence-Aware Restoration ──────────────────────────
    
    def restore_with_confidence(
        self, image: np.ndarray, prompt: str
    ) -> Dict[str, Any]:
        """
        Confidence-aware restoration: includes confidence scores and maps.
        
        Returns:
            Dictionary with 'final', 'confidence_score', 'confidence_map',
            'instruction_clarity', 'degradation_prediction'.
        """
        from .confidence_estimator import ConfidenceEstimator
        
        # Get restoration + embeddings
        restored, text_embd, deg_pred = self.process_single_with_embeddings(image, prompt)
        
        # Compute confidence
        estimator = ConfidenceEstimator()
        confidence_score = estimator.instruction_confidence(prompt, deg_pred)
        confidence_map = estimator.pixel_confidence_map(image, restored)
        clarity = estimator.instruction_clarity(prompt)
        
        return {
            'original': image.copy(),
            'final': restored,
            'confidence_score': confidence_score,
            'confidence_map': confidence_map,
            'instruction_clarity': clarity,
            'degradation_prediction': deg_pred,
            'prompt': prompt,
        }
    
    # ─── Visualization Helpers ────────────────────────────────────────────
    
    @staticmethod
    def plot_multistep_results(results: Dict[str, Any], figsize: Tuple = None):
        """Visualize multi-step restoration progression."""
        steps = results['steps']
        n = len(steps) + 1  # original + each step output
        
        if figsize is None:
            figsize = (5 * n, 5)
        
        fig, axes = plt.subplots(1, n, figsize=figsize)
        if n == 1:
            axes = [axes]
        
        # Plot original
        axes[0].imshow(results['original'])
        axes[0].set_title("Original (Degraded)", fontsize=10)
        axes[0].axis('off')
        
        # Plot each step
        for i, step in enumerate(steps):
            axes[i + 1].imshow(step['output'])
            axes[i + 1].set_title(
                f"{step['label']}: {step['instruction'][:40]}...", 
                fontsize=9
            )
            axes[i + 1].axis('off')
        
        plt.suptitle(
            f"Multi-Step Restoration\n\"{results['compound_prompt'][:80]}\"",
            fontsize=12, fontweight='bold'
        )
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_comparison(images: List[np.ndarray], titles: List[str], 
                        figsize: Tuple = None):
        """Generic side-by-side comparison plot."""
        n = len(images)
        if figsize is None:
            figsize = (5 * n, 5)
        
        fig, axes = plt.subplots(1, n, figsize=figsize)
        if n == 1:
            axes = [axes]
        
        for ax, img, title in zip(axes, images, titles):
            ax.imshow(img)
            ax.set_title(title, fontsize=10)
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()
