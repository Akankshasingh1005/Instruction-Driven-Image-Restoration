"""
Instruction Faithfulness Metric (Novelty 1)
============================================

Evaluates how well a restored image follows the given text instruction.

Composite Instruction Faithfulness Score (IFS) with three sub-metrics:
1. CLIP-Based Alignment Score - text-image semantic alignment
2. Degradation-Aware Improvement Score - task-specific quality improvement
3. Perceptual Consistency Score - content preservation check

Usage:
    ifs = InstructionFaithfulnessScore(device)
    score = ifs.compute(degraded_image, restored_image, instruction)
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import re


class InstructionFaithfulnessScore:
    """
    Composite metric to evaluate instruction-following quality of restoration.
    """
    
    # Weights for combining sub-metrics
    DEFAULT_WEIGHTS = {
        'clip_alignment': 0.35,
        'degradation_improvement': 0.40,
        'perceptual_consistency': 0.25,
    }
    
    # Keywords mapping instructions to degradation types
    DEGRADATION_KEYWORDS = {
        'noise': ['noise', 'noisy', 'grain', 'grainy', 'dots', 'speckle', 'denoise', 'clean'],
        'blur': ['blur', 'blurry', 'sharp', 'sharpen', 'focus', 'stabilize', 'deblur', 'fuzzy'],
        'rain': ['rain', 'rainy', 'raindrop', 'streak', 'derain', 'wet'],
        'haze': ['haze', 'hazy', 'fog', 'foggy', 'smog', 'dehaze', 'mist', 'misty'],
        'low_light': ['dark', 'dim', 'bright', 'brighten', 'light', 'exposure', 'illuminate',
                      'low-light', 'underexposed'],
        'enhancement': ['enhance', 'improve', 'better', 'stunning', 'professional', 'beautiful',
                       'retouch', 'color', 'colour', 'vibrant', 'vivid', 'contrast', 'saturate'],
    }
    
    def __init__(self, device='cpu', weights: Dict[str, float] = None):
        self.device = device
        self.weights = weights or self.DEFAULT_WEIGHTS
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_tokenize = None
    
    def _load_clip(self):
        """Lazy-load CLIP model (heavy, only load when needed)."""
        if self._clip_model is not None:
            return
        
        try:
            import clip
            self._clip_model, self._clip_preprocess = clip.load("ViT-B/32", device=self.device)
            self._clip_tokenize = clip.tokenize
            self._clip_model.eval()
        except ImportError:
            print("Warning: 'clip' package not installed. Install with: pip install git+https://github.com/openai/CLIP.git")
            print("CLIP-based alignment score will be unavailable.")
            self._clip_model = None
    
    def compute(
        self, 
        degraded: np.ndarray, 
        restored: np.ndarray, 
        instruction: str,
        ground_truth: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute the composite Instruction Faithfulness Score.
        
        Args:
            degraded: Degraded input image (H, W, 3), float [0,1].
            restored: Restored output image (H, W, 3), float [0,1].
            instruction: Text instruction used for restoration.
            ground_truth: Optional clean ground truth image.
            
        Returns:
            Dictionary with individual scores and composite IFS.
        """
        scores = {}
        
        # 1. CLIP-Based Alignment Score
        scores['clip_alignment'] = self.clip_alignment_score(restored, instruction)
        
        # 2. Degradation-Aware Improvement Score
        scores['degradation_improvement'] = self.degradation_improvement_score(
            degraded, restored, instruction, ground_truth
        )
        
        # 3. Perceptual Consistency Score
        scores['perceptual_consistency'] = self.perceptual_consistency_score(
            degraded, restored
        )
        
        # Composite IFS
        ifs = sum(self.weights[k] * scores[k] for k in self.weights)
        scores['IFS'] = ifs
        
        return scores
    
    # ─── Sub-metric 1: CLIP Alignment ─────────────────────────────────────
    
    def clip_alignment_score(self, image: np.ndarray, instruction: str) -> float:
        """
        Compute CLIP cosine similarity between image and instruction text.
        
        Returns:
            Score in [0, 1]. Higher = better alignment.
        """
        self._load_clip()
        
        if self._clip_model is None:
            # Fallback: use keyword-based heuristic instead
            return self._keyword_alignment_fallback(image, instruction)
        
        from PIL import Image
        
        # Prepare image
        if isinstance(image, np.ndarray):
            img_pil = Image.fromarray((image * 255).astype(np.uint8))
        else:
            img_pil = image
        
        img_tensor = self._clip_preprocess(img_pil).unsqueeze(0).to(self.device)
        
        # Prepare text - prefix for better alignment
        text_prompt = f"A photo that has been processed to: {instruction}"
        text_tokens = self._clip_tokenize([text_prompt]).to(self.device)
        
        with torch.no_grad():
            img_features = self._clip_model.encode_image(img_tensor)
            txt_features = self._clip_model.encode_text(text_tokens)
            
            # Normalize
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            txt_features = txt_features / txt_features.norm(dim=-1, keepdim=True)
            
            # Cosine similarity
            similarity = (img_features @ txt_features.T).item()
        
        # Map from [-1, 1] to [0, 1]
        score = (similarity + 1.0) / 2.0
        return float(np.clip(score, 0, 1))
    
    def _keyword_alignment_fallback(self, image: np.ndarray, instruction: str) -> float:
        """Fallback alignment score when CLIP is not available."""
        # Simple heuristic based on image statistics
        detected_type = self._detect_degradation_type(instruction)
        
        if detected_type == 'noise':
            # Check if image looks clean (low variance in smooth regions)
            gray = np.mean(image, axis=2)
            local_var = self._local_variance(gray, window=5)
            cleanliness = 1.0 - np.clip(np.mean(local_var) * 10, 0, 1)
            return cleanliness
        elif detected_type == 'blur':
            # Check if image is sharp (high-frequency content)
            sharpness = self._compute_sharpness(image)
            return np.clip(sharpness / 100, 0, 1)
        else:
            return 0.5  # Neutral score for unknown types
    
    # ─── Sub-metric 2: Degradation-Aware Improvement ──────────────────────
    
    def degradation_improvement_score(
        self, degraded: np.ndarray, restored: np.ndarray,
        instruction: str, ground_truth: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute improvement score based on the degradation type.
        
        Uses ground truth if available, otherwise uses no-reference metrics.
        """
        deg_type = self._detect_degradation_type(instruction)
        
        if ground_truth is not None:
            return self._reference_improvement(degraded, restored, ground_truth, deg_type)
        else:
            return self._no_reference_improvement(degraded, restored, deg_type)
    
    def _detect_degradation_type(self, instruction: str) -> str:
        """Detect degradation type from instruction keywords."""
        instruction_lower = instruction.lower()
        
        best_type = 'enhancement'  # default
        best_count = 0
        
        for deg_type, keywords in self.DEGRADATION_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in instruction_lower)
            if count > best_count:
                best_count = count
                best_type = deg_type
        
        return best_type
    
    def _reference_improvement(
        self, degraded: np.ndarray, restored: np.ndarray, 
        ground_truth: np.ndarray, deg_type: str
    ) -> float:
        """Improvement score when ground truth is available."""
        # PSNR improvement
        psnr_degraded = psnr(ground_truth, degraded, data_range=1.0)
        psnr_restored = psnr(ground_truth, restored, data_range=1.0)
        
        # SSIM improvement
        ssim_degraded = ssim(ground_truth, degraded, data_range=1.0, channel_axis=2)
        ssim_restored = ssim(ground_truth, restored, data_range=1.0, channel_axis=2)
        
        # Normalize improvements
        psnr_improvement = np.clip((psnr_restored - psnr_degraded) / 10.0, 0, 1)  # 10dB = perfect
        ssim_improvement = np.clip((ssim_restored - ssim_degraded) / 0.3, 0, 1)   # 0.3 = perfect
        
        return 0.5 * psnr_improvement + 0.5 * ssim_improvement
    
    def _no_reference_improvement(
        self, degraded: np.ndarray, restored: np.ndarray, deg_type: str
    ) -> float:
        """No-reference improvement using degradation-specific metrics."""
        
        if deg_type == 'noise':
            # Compare noise levels (local variance)
            noise_before = self._estimate_noise_level(degraded)
            noise_after = self._estimate_noise_level(restored)
            if noise_before > 0:
                reduction = (noise_before - noise_after) / noise_before
                return float(np.clip(reduction, 0, 1))
            return 0.5
        
        elif deg_type == 'blur':
            # Compare sharpness
            sharp_before = self._compute_sharpness(degraded)
            sharp_after = self._compute_sharpness(restored)
            if sharp_before > 0:
                improvement = (sharp_after - sharp_before) / max(sharp_before, 1e-6)
                return float(np.clip(improvement / 2.0, 0, 1))  # 200% = perfect
            return 0.5
        
        elif deg_type == 'haze':
            # Compare contrast
            contrast_before = self._compute_contrast(degraded)
            contrast_after = self._compute_contrast(restored)
            if contrast_before > 0:
                improvement = (contrast_after - contrast_before) / max(contrast_before, 1e-6)
                return float(np.clip(improvement, 0, 1))
            return 0.5
        
        elif deg_type == 'rain':
            # Use general quality improvement (SSIM of diff)
            diff_before = np.std(degraded - np.mean(degraded, axis=2, keepdims=True))
            diff_after = np.std(restored - np.mean(restored, axis=2, keepdims=True))
            # Rain adds high-frequency artifacts, restoration should reduce them
            reduction = (diff_before - diff_after) / max(diff_before, 1e-6)
            return float(np.clip(reduction + 0.5, 0, 1))
        
        elif deg_type == 'low_light':
            # Compare brightness
            bright_before = np.mean(degraded)
            bright_after = np.mean(restored)
            improvement = (bright_after - bright_before) / max(1 - bright_before, 1e-6)
            return float(np.clip(improvement, 0, 1))
        
        else:  # enhancement
            # General quality: SSIM between degraded/restored (moderate = good)
            s = ssim(degraded, restored, data_range=1.0, channel_axis=2)
            # We want some change but not too much: peak around 0.7-0.9 SSIM
            if s > 0.95:  # too little change
                return 0.3
            elif s > 0.7:  # good range
                return 0.8 + 0.2 * (s - 0.7) / 0.25
            elif s > 0.5:  # moderate change
                return 0.5 + 0.3 * (s - 0.5) / 0.2
            else:  # too much change
                return max(0.1, s)
    
    # ─── Sub-metric 3: Perceptual Consistency ─────────────────────────────
    
    def perceptual_consistency_score(
        self, degraded: np.ndarray, restored: np.ndarray
    ) -> float:
        """
        Ensure restoration preserves image content (no hallucinations/artifacts).
        
        Uses SSIM to measure structural preservation.
        Higher score = better content preservation.
        """
        # Structural similarity (want high: content preserved)
        s = ssim(degraded, restored, data_range=1.0, channel_axis=2)
        
        # Check for artifacts: extreme pixel changes
        diff = np.abs(restored.astype(float) - degraded.astype(float))
        extreme_changes = np.mean(diff > 0.5)  # fraction of pixels changed by > 50%
        
        # Penalize extreme changes
        artifact_penalty = 1.0 - np.clip(extreme_changes * 5, 0, 0.5)  # max 0.5 penalty
        
        # Combine
        score = 0.7 * s + 0.3 * artifact_penalty
        return float(np.clip(score, 0, 1))
    
    # ─── Helper Functions ─────────────────────────────────────────────────
    
    @staticmethod
    def _estimate_noise_level(image: np.ndarray) -> float:
        """Estimate noise level using median absolute deviation."""
        gray = np.mean(image, axis=2)
        # Laplacian-based noise estimation
        from scipy.ndimage import laplace
        lap = laplace(gray)
        sigma = np.median(np.abs(lap)) / 0.6745
        return float(sigma)
    
    @staticmethod
    def _compute_sharpness(image: np.ndarray) -> float:
        """Compute image sharpness using Laplacian variance."""
        gray = np.mean(image, axis=2)
        from scipy.ndimage import laplace
        lap = laplace(gray)
        return float(np.var(lap))
    
    @staticmethod
    def _compute_contrast(image: np.ndarray) -> float:
        """Compute image contrast as standard deviation of intensity."""
        gray = np.mean(image, axis=2)
        return float(np.std(gray))
    
    @staticmethod
    def _local_variance(gray: np.ndarray, window: int = 5) -> np.ndarray:
        """Compute local variance map."""
        from scipy.ndimage import uniform_filter
        mean = uniform_filter(gray, size=window)
        mean_sq = uniform_filter(gray ** 2, size=window)
        var = mean_sq - mean ** 2
        return np.clip(var, 0, None)
    
    # ─── Reporting ────────────────────────────────────────────────────────
    
    @staticmethod
    def format_scores(scores: Dict[str, float]) -> str:
        """Pretty-format IFS scores for display."""
        lines = ["┌─── Instruction Faithfulness Score (IFS) ───┐"]
        for key, val in scores.items():
            bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
            label = key.replace('_', ' ').title()
            if key == 'IFS':
                lines.append(f"├────────────────────────────────────────────┤")
                lines.append(f"│ {'★ ' + label + ':':30s} {val:.4f} {bar} │")
            else:
                lines.append(f"│ {label + ':':30s} {val:.4f} {bar} │")
        lines.append("└────────────────────────────────────────────┘")
        return "\n".join(lines)
