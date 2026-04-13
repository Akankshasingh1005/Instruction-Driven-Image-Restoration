"""
Confidence Estimator Module (Novelty 4: Confidence-Aware Restoration)
=====================================================================

Handles vague/uncertain instructions by producing confidence-aware outputs.

Two levels of confidence:
1. Instruction Confidence Score - how clear/specific the instruction is
2. Pixel Confidence Map - per-pixel confidence of the restoration

Usage:
    estimator = ConfidenceEstimator()
    clarity = estimator.instruction_clarity("Remove the noise")  # 0.9
    clarity = estimator.instruction_clarity("Make it look nice")  # 0.3
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import re


class ConfidenceEstimator:
    """
    Estimates confidence of instruction-guided restoration.
    """
    
    # Clear, specific restoration keywords → high confidence
    CLEAR_KEYWORDS = {
        'denoise': 0.95, 'deblur': 0.95, 'dehaze': 0.95, 'derain': 0.95,
        'sharpen': 0.90, 'brighten': 0.85, 'enhance contrast': 0.85,
        'remove noise': 0.95, 'remove blur': 0.95, 'remove rain': 0.95,
        'remove haze': 0.95, 'remove fog': 0.90, 'remove grain': 0.90,
        'reduce noise': 0.90, 'fix blur': 0.85, 'fix blurry': 0.85,
        'noise removal': 0.90, 'stabilize': 0.85, 'de-noise': 0.90,
    }
    
    # Vague, ambiguous phrases → low confidence
    VAGUE_PATTERNS = [
        (r'\bmake\s+it\s+(look\s+)?(better|nice|good|great)\b', 0.25),
        (r'\bfix\s+it\b', 0.20),
        (r'\bimprove\s+it\b', 0.30),
        (r'\bclean\s+it\s+up\b', 0.35),
        (r'\bdo\s+something\b', 0.10),
        (r'\benhance\b', 0.40),
        (r'\bretouche?\b', 0.45),
        (r'\bprofessional\b', 0.35),
        (r'\bstunning\b', 0.30),
        (r'\bbeautiful\b', 0.25),
        (r'\bpretty\b', 0.25),
        (r'\bpleasant\b', 0.35),
    ]
    
    def instruction_clarity(self, instruction: str) -> float:
        """
        Score the clarity/specificity of an instruction.
        
        Args:
            instruction: Text instruction string.
            
        Returns:
            Clarity score in [0, 1]. Higher = more specific/clear.
        """
        instruction_lower = instruction.lower().strip()
        
        if not instruction_lower:
            return 0.0
        
        # Check for clear keywords
        max_clear_score = 0.0
        for keyword, score in self.CLEAR_KEYWORDS.items():
            if keyword in instruction_lower:
                max_clear_score = max(max_clear_score, score)
        
        # Check for vague patterns
        max_vague_score = 0.0
        for pattern, score in self.VAGUE_PATTERNS:
            if re.search(pattern, instruction_lower):
                max_vague_score = max(max_vague_score, score)
        
        # If clear keywords found, use their score (boosted slightly)
        if max_clear_score > 0 and max_vague_score > 0:
            # Mix: clear keyword + vague phrasing
            return max_clear_score * 0.7 + max_vague_score * 0.3
        elif max_clear_score > 0:
            return max_clear_score
        elif max_vague_score > 0:
            return max_vague_score
        else:
            # Unknown instruction pattern: moderate confidence
            # Longer, more descriptive instructions get slightly higher score
            word_count = len(instruction_lower.split())
            return min(0.5, 0.15 + word_count * 0.05)
    
    def instruction_confidence(
        self, instruction: str, deg_pred: Optional[torch.Tensor] = None
    ) -> float:
        """
        Combined confidence score using instruction clarity and model prediction.
        
        Args:
            instruction: Text instruction.
            deg_pred: Degradation prediction logits from LMHead (optional).
            
        Returns:
            Confidence score in [0, 1].
        """
        clarity = self.instruction_clarity(instruction)
        
        if deg_pred is not None:
            # Use entropy of degradation prediction as model confidence
            model_conf = self._prediction_confidence(deg_pred)
            # Combine: 60% instruction clarity + 40% model confidence
            return 0.6 * clarity + 0.4 * model_conf
        
        return clarity
    
    def _prediction_confidence(self, deg_pred: torch.Tensor) -> float:
        """
        Compute confidence from degradation prediction logits.
        
        High confidence = peaked softmax (low entropy).
        Low confidence = flat softmax (high entropy).
        """
        # Softmax
        probs = F.softmax(deg_pred.float().flatten(), dim=0)
        
        # Entropy
        log_probs = torch.log(probs + 1e-8)
        entropy = -(probs * log_probs).sum().item()
        
        # Max entropy for uniform distribution
        n_classes = len(probs)
        max_entropy = np.log(n_classes) if n_classes > 1 else 1.0
        
        # Normalize: 0 entropy = perfect confidence, max entropy = no confidence
        normalized_entropy = entropy / max(max_entropy, 1e-8)
        confidence = 1.0 - normalized_entropy
        
        return float(np.clip(confidence, 0, 1))
    
    def pixel_confidence_map(
        self, degraded: np.ndarray, restored: np.ndarray,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate per-pixel confidence map based on restoration changes.
        
        Regions with moderate, consistent changes → higher confidence.
        Regions with extreme changes or no change → lower confidence.
        
        Args:
            degraded: Input degraded image (H, W, 3).
            restored: Restored image (H, W, 3).
            normalize: If True, normalize to [0, 1].
            
        Returns:
            Confidence map (H, W) in [0, 1].
        """
        # Per-pixel absolute difference
        diff = np.abs(restored.astype(np.float64) - degraded.astype(np.float64))
        diff_magnitude = np.mean(diff, axis=2)  # average across channels
        
        # Confidence is highest for moderate changes
        # Too little change = uncertain (model didn't act)
        # Too much change = uncertain (potential artifacts)
        
        # Gaussian-shaped confidence centered around moderate change
        optimal_change = np.median(diff_magnitude[diff_magnitude > 0.01]) if np.any(diff_magnitude > 0.01) else 0.05
        optimal_change = max(optimal_change, 0.02)
        
        # Distance from optimal
        distance = np.abs(diff_magnitude - optimal_change) / max(optimal_change, 1e-6)
        
        confidence = np.exp(-0.5 * distance ** 2)  # Gaussian falloff
        
        # Boost confidence where restoration clearly acted
        active_mask = (diff_magnitude > 0.005).astype(float)
        confidence = confidence * (0.3 + 0.7 * active_mask)
        
        if normalize:
            conf_min = confidence.min()
            conf_max = confidence.max()
            if conf_max > conf_min:
                confidence = (confidence - conf_min) / (conf_max - conf_min)
        
        return confidence.astype(np.float32)
    
    def generate_variants(
        self, image: np.ndarray, prompt: str, pipeline, n_variants: int = 3
    ) -> list:
        """
        For low-confidence instructions, generate multiple restoration variants
        by slightly perturbing the approach.
        
        Args:
            image: Input image.
            prompt: Instruction text.
            pipeline: RestorationPipeline instance.
            n_variants: Number of variants to generate.
            
        Returns:
            List of (variant_image, variant_prompt) tuples.
        """
        variants = []
        
        # Variant 1: Original prompt
        result = pipeline.process_single(image, prompt)
        variants.append((result, prompt))
        
        # Variant 2: More specific interpretation
        specific_prompt = self._make_specific(prompt)
        if specific_prompt != prompt:
            result = pipeline.process_single(image, specific_prompt)
            variants.append((result, specific_prompt))
        
        # Variant 3: Alternative interpretation
        alt_prompt = self._make_alternative(prompt)
        if alt_prompt != prompt and alt_prompt != specific_prompt:
            result = pipeline.process_single(image, alt_prompt)
            variants.append((result, alt_prompt))
        
        return variants[:n_variants]
    
    def _make_specific(self, prompt: str) -> str:
        """Attempt to make a vague prompt more specific."""
        replacements = {
            'make it look better': 'Remove noise and enhance the image quality',
            'fix it': 'Remove degradation and restore the image',
            'improve it': 'Enhance the image sharpness and colors',
            'clean it up': 'Remove noise and artifacts from the image',
            'enhance': 'Enhance the image colors and contrast',
        }
        result = prompt.lower()
        for vague, specific in replacements.items():
            if vague in result:
                return specific
        return prompt
    
    def _make_alternative(self, prompt: str) -> str:
        """Generate an alternative interpretation of the prompt."""
        alternatives = {
            'make it look better': 'Sharpen the image and improve contrast',
            'fix it': 'Denoise and deblur the image',
            'improve it': 'Remove noise and brighten the image',
            'clean it up': 'Remove grain and smooth the image',
            'enhance': 'Boost saturation and sharpen edges',
        }
        result = prompt.lower()
        for vague, alt in alternatives.items():
            if vague in result:
                return alt
        return prompt
    
    @staticmethod
    def classify_confidence(score: float) -> str:
        """Classify confidence score into human-readable category."""
        if score >= 0.8:
            return "High Confidence ✓"
        elif score >= 0.5:
            return "Moderate Confidence ~"
        elif score >= 0.3:
            return "Low Confidence !"
        else:
            return "Very Low Confidence ✗"
