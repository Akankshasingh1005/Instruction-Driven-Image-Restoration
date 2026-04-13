"""
Region Selector Module (Novelty 3: Region-Aware Restoration)
=============================================================

Enables instructions to be applied only to selected image regions.

Supports:
- Manual binary masks (from file or array)
- Auto-generated region masks from instruction keywords 
  ("the sky", "the background", "top half", "left side")
- Soft mask blending with Gaussian feathering

Usage:
    selector = RegionSelector()
    mask = selector.create_mask_from_instruction(image, "the sky region")
    blended = selector.apply_mask_blend(original, restored, mask)
"""

import numpy as np
from typing import Optional, Tuple
import re


class RegionSelector:
    """
    Creates and applies region masks for selective restoration.
    """
    
    # Keyword patterns for auto-region detection
    REGION_PATTERNS = {
        'top': r'\b(top|upper)\b',
        'bottom': r'\b(bottom|lower)\b',
        'left': r'\b(left)\b',
        'right': r'\b(right)\b',
        'center': r'\b(center|centre|middle)\b',
        'sky': r'\b(sky|skies)\b',
        'background': r'\b(background|bg)\b',
        'foreground': r'\b(foreground|fg|subject)\b',
    }
    
    def create_mask_from_instruction(
        self, image: np.ndarray, instruction: str
    ) -> np.ndarray:
        """
        Auto-generate a region mask from instruction keywords.
        
        Args:
            image: Input image (H, W, 3).
            instruction: Text instruction containing region hints.
            
        Returns:
            Binary mask (H, W) with 1 = region to restore.
        """
        h, w = image.shape[:2]
        instruction_lower = instruction.lower()
        
        # Check for spatial region keywords
        for region, pattern in self.REGION_PATTERNS.items():
            if re.search(pattern, instruction_lower):
                return self._spatial_mask(h, w, region, image)
        
        # Default: full image mask
        return np.ones((h, w), dtype=np.float32)
    
    def _spatial_mask(self, h: int, w: int, region: str, 
                      image: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate spatial region masks."""
        mask = np.zeros((h, w), dtype=np.float32)
        
        if region == 'top':
            mask[:h // 2, :] = 1.0
        elif region == 'bottom':
            mask[h // 2:, :] = 1.0
        elif region == 'left':
            mask[:, :w // 2] = 1.0
        elif region == 'right':
            mask[:, w // 2:] = 1.0
        elif region == 'center':
            margin_h, margin_w = h // 4, w // 4
            mask[margin_h:h - margin_h, margin_w:w - margin_w] = 1.0
        elif region == 'sky':
            # Heuristic: sky is usually in the upper portion
            # Use brightness-based segmentation 
            if image is not None:
                mask = self._sky_mask(image)
            else:
                mask[:h // 3, :] = 1.0  # fallback: top third
        elif region == 'background':
            # Heuristic: background = non-center region
            margin_h, margin_w = h // 4, w // 4
            mask[:, :] = 1.0
            mask[margin_h:h - margin_h, margin_w:w - margin_w] = 0.0
        elif region == 'foreground':
            margin_h, margin_w = h // 4, w // 4
            mask[margin_h:h - margin_h, margin_w:w - margin_w] = 1.0
        
        return mask
    
    def _sky_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Simple sky detection based on color and brightness.
        Sky pixels tend to be bright and bluish.
        """
        h, w = image.shape[:2]
        
        # Convert to grayscale
        gray = np.mean(image, axis=2)
        
        # Blue channel dominance (R-channel ratio)
        # Sky tends to: high blue, lower red/green
        if image.shape[2] >= 3:
            blue_ratio = image[:, :, 2] / (np.mean(image, axis=2) + 1e-6)
            brightness = gray
            
            # Sky mask: bright + blue-dominant + upper region bias
            height_prior = np.linspace(1, 0, h).reshape(-1, 1) * np.ones((1, w))
            
            sky_score = (
                0.3 * (brightness > 0.5).astype(float) +
                0.3 * (blue_ratio > 1.1).astype(float) +
                0.4 * height_prior
            )
            mask = (sky_score > 0.5).astype(np.float32)
        else:
            # Fallback: upper third
            mask = np.zeros((h, w), dtype=np.float32)
            mask[:h // 3, :] = 1.0
        
        return mask
    
    def create_rectangular_mask(
        self, h: int, w: int, 
        top: int, left: int, bottom: int, right: int
    ) -> np.ndarray:
        """Create a rectangular region mask."""
        mask = np.zeros((h, w), dtype=np.float32)
        mask[top:bottom, left:right] = 1.0
        return mask
    
    def create_circular_mask(
        self, h: int, w: int, 
        center_y: int, center_x: int, radius: int
    ) -> np.ndarray:
        """Create a circular region mask."""
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((Y - center_y) ** 2 + (X - center_x) ** 2)
        mask = (dist <= radius).astype(np.float32)
        return mask
    
    def feather_mask(self, mask: np.ndarray, radius: int = 5) -> np.ndarray:
        """
        Apply Gaussian feathering to mask edges for smooth blending.
        
        Args:
            mask: Binary mask (H, W).
            radius: Gaussian blur radius.
            
        Returns:
            Soft mask with feathered edges.
        """
        from scipy.ndimage import gaussian_filter
        if radius <= 0:
            return mask
        soft_mask = gaussian_filter(mask.astype(np.float64), sigma=radius)
        return soft_mask.astype(np.float32)
    
    def apply_mask_blend(
        self, original: np.ndarray, restored: np.ndarray,
        mask: np.ndarray, feather_radius: int = 5
    ) -> np.ndarray:
        """
        Blend original and restored images using the mask.
        
        output = mask * restored + (1 - mask) * original
        
        Args:
            original: Original image (H, W, 3).
            restored: Restored image (H, W, 3).
            mask: Region mask (H, W). 1 = use restored, 0 = keep original.
            feather_radius: Gaussian blur radius for soft edges.
            
        Returns:
            Blended image (H, W, 3).
        """
        # Feather the mask for smooth transitions
        soft_mask = self.feather_mask(mask, feather_radius)
        
        # Expand mask to 3 channels
        if soft_mask.ndim == 2:
            soft_mask = soft_mask[:, :, np.newaxis]
        
        # Blend
        blended = soft_mask * restored + (1.0 - soft_mask) * original
        return np.clip(blended, 0., 1.)
    
    def visualize_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Create a visualization overlay of the mask on the image.
        
        Returns:
            Image with red-tinted mask overlay.
        """
        overlay = image.copy()
        
        if mask.ndim == 3:
            mask_2d = mask[:, :, 0]
        else:
            mask_2d = mask
        
        # Red tint on masked region
        overlay[:, :, 0] = np.clip(overlay[:, :, 0] + mask_2d * 0.3, 0, 1)
        overlay[:, :, 1] = overlay[:, :, 1] * (1 - mask_2d * 0.2)
        overlay[:, :, 2] = overlay[:, :, 2] * (1 - mask_2d * 0.2)
        
        return overlay
    
    def is_region_instruction(self, instruction: str) -> bool:
        """Check if instruction contains region-related keywords."""
        instruction_lower = instruction.lower()
        for pattern in self.REGION_PATTERNS.values():
            if re.search(pattern, instruction_lower):
                return True
        return False
