"""LoRA multipliers utility for LTX-2 pipeline (simplified version without mmgp)."""
from typing import List


def update_loras_slists(trans, slists_dict, num_inference_steps, phase_switch_step=None, phase_switch_step2=None):
    """No-op version of update_loras_slists for when LoRA is not used."""
    # This is a simplified version that doesn't require mmgp
    # For full LoRA support, mmgp would be needed
    pass
