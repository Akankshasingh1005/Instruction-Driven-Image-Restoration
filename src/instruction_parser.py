"""
Instruction Parser Module (Novelty 2: Multi-Step Instructions)
==============================================================

Parses compound natural language instructions into ordered sub-instructions.
Supports various delimiters: 'then', 'and then', 'followed by', 'after that',
semicolons, numbered lists, etc.

Example:
    "Remove the noise and then sharpen the edges"
    → [("Step 1", "Remove the noise"), ("Step 2", "sharpen the edges")]
"""

import re
from typing import List, Tuple


class InstructionParser:
    """
    Parses a single compound instruction into ordered sub-instructions.
    
    Handles multiple patterns:
    - Sequential keywords: "then", "and then", "followed by", "after that", "next"
    - Semicolons and periods as delimiters  
    - Numbered lists: "1. ... 2. ... 3. ..."
    - Bullet lists: "- ... - ..."
    """
    
    # Ordered list of delimiter patterns (checked in priority order)
    SEQUENTIAL_PATTERNS = [
        r'\.\s*(?:and\s+)?then\s+',          # ". then" or ". and then"
        r',?\s*(?:and\s+)?then\s+',           # ", then" or ", and then"  
        r',?\s*followed\s+by\s+',             # ", followed by"
        r',?\s*after\s+that\s*,?\s+',         # ", after that,"
        r',?\s*(?:and\s+)?next\s*,?\s+',      # ", next" or ", and next"
        r',?\s*(?:and\s+)?finally\s*,?\s+',   # ", finally" or ", and finally"
        r',?\s*afterwards?\s+',               # ", afterward(s)"
        r',?\s*subsequently\s+',              # ", subsequently"
    ]
    
    # Numbered list pattern: "1. ...", "2) ...", etc. (works mid-line too)
    NUMBERED_LIST_PATTERN = r'(?:^|\s)\d+[\.\)]\s+'
    
    # Semicolon delimiter
    SEMICOLON_PATTERN = r'\s*;\s*'
    
    def __init__(self):
        # Compile all patterns for efficiency
        self._sequential_regex = re.compile(
            '|'.join(f'({p})' for p in self.SEQUENTIAL_PATTERNS),
            re.IGNORECASE
        )
        self._numbered_regex = re.compile(self.NUMBERED_LIST_PATTERN, re.IGNORECASE)
        self._semicolon_regex = re.compile(self.SEMICOLON_PATTERN)
    
    def parse(self, instruction: str) -> List[Tuple[str, str]]:
        """
        Parse a compound instruction into ordered sub-instructions.
        
        Args:
            instruction: A natural language instruction string.
            
        Returns:
            List of (step_label, sub_instruction) tuples.
            If no compound structure detected, returns single-element list.
        """
        instruction = instruction.strip()
        if not instruction:
            return [("Step 1", "")]
        
        # Try each parsing strategy in order of specificity
        
        # 1. Try numbered list first (most explicit)
        steps = self._parse_numbered_list(instruction)
        if len(steps) > 1:
            return self._format_steps(steps)
        
        # 2. Try sequential keywords
        steps = self._parse_sequential_keywords(instruction)
        if len(steps) > 1:
            return self._format_steps(steps)
        
        # 3. Try semicolons
        steps = self._parse_semicolons(instruction)
        if len(steps) > 1:
            return self._format_steps(steps)
        
        # 4. Fallback: treat as single instruction
        return [("Step 1", instruction)]
    
    def _parse_numbered_list(self, instruction: str) -> List[str]:
        """Parse numbered lists like '1. denoise 2. sharpen 3. enhance'."""
        parts = self._numbered_regex.split(instruction)
        # Filter out empty strings
        parts = [p.strip().rstrip('.') for p in parts if p.strip()]
        return parts
    
    def _parse_sequential_keywords(self, instruction: str) -> List[str]:
        """Parse instructions connected by sequential keywords."""
        parts = self._sequential_regex.split(instruction)
        # regex split includes the groups, filter to non-None actual text
        result = []
        for part in parts:
            if part is not None:
                cleaned = part.strip()
                # Skip if this part is just a delimiter match
                is_delimiter = False
                for pattern in self.SEQUENTIAL_PATTERNS:
                    if re.fullmatch(pattern, part, re.IGNORECASE):
                        is_delimiter = True
                        break
                if not is_delimiter and cleaned:
                    result.append(cleaned.rstrip('.'))
        return result
    
    def _parse_semicolons(self, instruction: str) -> List[str]:
        """Parse instructions separated by semicolons."""
        parts = self._semicolon_regex.split(instruction)
        parts = [p.strip().rstrip('.') for p in parts if p.strip()]
        return parts
    
    def _format_steps(self, steps: List[str]) -> List[Tuple[str, str]]:
        """Format a list of step strings into labeled tuples."""
        return [(f"Step {i+1}", step) for i, step in enumerate(steps)]
    
    def is_multi_step(self, instruction: str) -> bool:
        """Check if an instruction contains multiple steps."""
        return len(self.parse(instruction)) > 1
    
    def get_step_count(self, instruction: str) -> int:
        """Return the number of steps in an instruction."""
        return len(self.parse(instruction))


# --- Convenience functions ---

def parse_instruction(instruction: str) -> List[Tuple[str, str]]:
    """Shortcut: parse a compound instruction into steps."""
    parser = InstructionParser()
    return parser.parse(instruction)


def print_parsed_steps(instruction: str):
    """Pretty-print parsed instruction steps."""
    steps = parse_instruction(instruction)
    print(f"\nInstruction: \"{instruction}\"")
    print(f"  → {len(steps)} step(s) detected:")
    for label, step in steps:
        print(f"    [{label}] {step}")


# --- Self-test / demo ---
if __name__ == "__main__":
    test_instructions = [
        # Single step
        "Remove noise from the image",
        
        # Sequential keywords
        "Remove the noise and then sharpen the edges",
        "Denoise the image, followed by enhancing the colors",
        "First remove blur, then enhance contrast, and finally sharpen",
        "Clean the noise, after that improve brightness",
        
        # Numbered lists
        "1. Denoise 2. Dehaze 3. Enhance colours",
        "1) Remove rain 2) Improve sharpness 3) Boost saturation",
        
        # Semicolons
        "remove noise; enhance edges; boost colors",
        
        # Complex
        "Can you remove the little dots in the image? is very unpleasant",
        "Remove rain. Then make it look stunning like a professional photo",
    ]
    
    for instruction in test_instructions:
        print_parsed_steps(instruction)
    print()
