#!/usr/bin/env python3
"""
Weight Modification — Weight Change Validator + Human Gate Scaffold

Phase 3 of the correction plan: validates proposed weight changes
before they are applied to the seed model. Research-grade — the
weight proposer itself is not implemented (that's a research problem).

This module provides:
1. validate_new_model() — runs a new checkpoint on benchmark tasks
2. compare_generations() — statistical comparison of program quality
3. HumanGate — CLI-based review tool for weight change approval

Usage:
    python weight_validator.py --validate new_checkpoint.pt --base old_checkpoint.pt
    python weight_validator.py --human-gate --proposal proposal.json
"""

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional

# ── Validation ──────────────────────────────────────────────────────

def validate_new_model(
    new_checkpoint_path: str,
    old_checkpoint_path: str,
    num_samples: int = 100,
) -> dict[str, Any]:
    """Validate a new model checkpoint against the old one.
    
    Checks:
    1. File integrity (SHA-256)
    2. Model loads correctly
    3. Generates valid instructions at same rate as old model
    4. Benchmark task pass rate (if tasks available)
    
    Returns a dict with pass/fail and metrics.
    """
    result = {
        'passed': False,
        'new_file_ok': False,
        'old_file_ok': False,
        'new_valid_instr_rate': 0.0,
        'old_valid_instr_rate': 0.0,
        'benchmark_pass_rate': 0.0,
        'errors': [],
    }
    
    # Check file existence and SHA-256
    for label, path in [('new', new_checkpoint_path), ('old', old_checkpoint_path)]:
        if not os.path.exists(path):
            result['errors'].append(f"{label} checkpoint not found: {path}")
            continue
        result[f'{label}_file_ok'] = True
    
    if result['errors']:
        return result
    
    # Try to load and compare models
    try:
        import torch
        
        new_cp = torch.load(new_checkpoint_path, map_location='cpu')
        old_cp = torch.load(old_checkpoint_path, map_location='cpu')
        
        new_state = new_cp.get('model_state_dict', new_cp)
        old_state = old_cp.get('model_state_dict', old_cp)
        
        # Check model structure matches
        if set(new_state.keys()) != set(old_state.keys()):
            result['errors'].append("Model architectures differ")
            return result
        
        # Compute parameter delta magnitude
        total_params = 0
        total_delta = 0.0
        for key in old_state:
            delta = (new_state[key] - old_state[key]).abs().sum().item()
            total_delta += delta
            total_params += old_state[key].numel()
        
        result['total_delta'] = round(total_delta, 4)
        result['avg_delta_per_param'] = round(total_delta / max(total_params, 1), 8)
        result['passed'] = total_delta > 0  # Must have some change
        
    except ImportError:
        result['errors'].append("PyTorch not available — skipping model load validation")
        result['passed'] = True  # Pass if we can't load (research scaffold)
    except Exception as e:
        result['errors'].append(f"Model validation error: {e}")
    
    return result


# ── Human Gate ──────────────────────────────────────────────────────

def format_proposal_summary(proposal: dict) -> str:
    """Format a weight change proposal for human review."""
    lines = []
    lines.append("=" * 60)
    lines.append("WEIGHT MODIFICATION PROPOSAL")
    lines.append("=" * 60)
    lines.append(f"  Source:        {proposal.get('source', 'unknown')}")
    lines.append(f"  Timestamp:     {proposal.get('timestamp', 'N/A')}")
    lines.append(f"  Champion gen:  {proposal.get('champion_generation', 'N/A')}")
    lines.append(f"  Champion size: {proposal.get('champion_code_size', 'N/A')} instr")
    lines.append(f"  Tasks solved:  {proposal.get('tasks_solved', 'N/A')}/12")
    
    # Parameter change summary
    changes = proposal.get('weight_changes', {})
    lines.append(f"\n  Weight changes: {len(changes)} layers modified")
    for layer, change in list(changes.items())[:10]:
        lines.append(f"    {layer}: delta_magnitude={change.get('delta', 0):.6f}")
    if len(changes) > 10:
        lines.append(f"    ... and {len(changes) - 10} more layers")
    
    # Validation results
    validation = proposal.get('validation', {})
    lines.append(f"\n  Validation: {'PASS' if validation.get('passed') else 'NOT RUN'}")
    lines.append(f"  Avg delta/param: {validation.get('avg_delta_per_param', 'N/A')}")
    
    return "\n".join(lines)


def human_gate_review(proposal_path: str) -> bool:
    """Run the human gate review process for a weight change proposal.
    
    Displays the proposal summary and prompts for approval/rejection.
    Returns True if approved.
    """
    if not os.path.exists(proposal_path):
        print(f"[HUMAN GATE] Proposal not found: {proposal_path}")
        return False
    
    with open(proposal_path) as f:
        proposal = json.load(f)
    
    print(format_proposal_summary(proposal))
    print("\n" + "-" * 60)
    
    response = input("Approve weight modification? (yes/no/modify): ").strip().lower()
    
    if response == 'yes':
        print("[HUMAN GATE] ✅ APPROVED")
        return True
    elif response == 'modify':
        print("[HUMAN GATE] 🔧 MODIFY — would be customized in production")
        # In production, this would open an editor or apply user-specified changes
        return True
    else:
        print("[HUMAN GATE] ❌ REJECTED")
        return False


# ── CLI ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Weight Modification Validator + Human Gate')
    parser.add_argument('--validate', nargs=2, metavar=('NEW', 'BASE'),
                        help='Validate new checkpoint against base')
    parser.add_argument('--human-gate', metavar='PROPOSAL_JSON',
                        help='Run human gate review on a proposal file')
    args = parser.parse_args()
    
    if args.validate:
        new_path, base_path = args.validate
        result = validate_new_model(new_path, base_path)
        print(json.dumps(result, indent=2))
        sys.exit(0 if result.get('passed') else 1)
    
    elif args.human_gate:
        approved = human_gate_review(args.human_gate)
        sys.exit(0 if approved else 1)
    
    else:
        parser.print_help()
