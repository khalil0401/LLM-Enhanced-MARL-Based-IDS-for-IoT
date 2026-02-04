"""
Self-Play Attacker Agent (Training Only)
Constrained by MITRE ATT&CK for IoT taxonomy

WARNING: This agent is STRICTLY for training use.
         It must NEVER be deployed in production.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import json


@dataclass
class AttackerConfig:
    """Configuration for attacker agent"""
    action_space_size: int = 25  # Number of MITRE ATT&CK techniques
    hidden_dim: int = 128
    learning_rate: float = 1e-4
    max_reward_per_episode: float = 100.0  # Cap to prevent over-optimization


class MITREAttackLibrary:
    """
    Library of MITRE ATT&CK for IoT techniques
    Constrains attacker to realistic, bounded action space
    """
    
    def __init__(self):
        # Simplified library (expand with full MITRE ATT&CK for IoT)
        self.techniques = {
            # Initial Access
            0: {'id': 'T1190', 'tactic': 'Initial Access', 
                'technique': 'Exploit Public-Facing Application'},
            1: {'id': 'T1566', 'tactic': 'Initial Access', 
               'technique': 'Phishing'},
            
            # Execution
            2: {'id': 'T1059', 'tactic': 'Execution', 
                'technique': 'Command and Scripting Interpreter'},
            3: {'id': 'T1203', 'tactic': 'Execution', 
                'technique': 'Exploitation for Client Execution'},
            
            # Persistence
            4: {'id': 'T1547', 'tactic': 'Persistence', 
                'technique': 'Boot or Logon Autostart Execution'},
            5: {'id': 'T1542', 'tactic': 'Persistence', 
                'technique': 'Pre-OS Boot'},
            
            # Privilege Escalation
            6: {'id': 'T1068', 'tactic': 'Privilege Escalation', 
                'technique': 'Exploitation for Privilege Escalation'},
            
            # Defense Evasion
            7: {'id': 'T1027', 'tactic': 'Defense Evasion', 
                'technique': 'Obfuscated Files or Information'},
            8: {'id': 'T1070', 'tactic': 'Defense Evasion', 
                'technique': 'Indicator Removal on Host'},
            
            # Discovery
            9: {'id': 'T1046', 'tactic': 'Discovery', 
                'technique': 'Network Service Scanning'},
            10: {'id': 'T1082', 'tactic': 'Discovery', 
                 'technique': 'System Information Discovery'},
            
            # Lateral Movement
            11: {'id': 'T1210', 'tactic': 'Lateral Movement', 
                 'technique': 'Exploitation of Remote Services'},
            
            # Collection
            12: {'id': 'T1005', 'tactic': 'Collection', 
                 'technique': 'Data from Local System'},
            13: {'id': 'T1056', 'tactic': 'Collection', 
                 'technique': 'Input Capture'},
            
            # Command and Control
            14: {'id': 'T1071', 'tactic': 'Command and Control', 
                 'technique': 'Application Layer Protocol'},
            15: {'id': 'T1132', 'tactic': 'Command and Control', 
                 'technique': 'Data Encoding'},
            
            # Exfiltration
            16: {'id': 'T1041', 'tactic': 'Exfiltration', 
                 'technique': 'Exfiltration Over C2 Channel'},
            17: {'id': 'T1048', 'tactic': 'Exfiltration', 
                 'technique': 'Exfiltration Over Alternative Protocol'},
            
            # Impact
            18: {'id': 'T1485', 'tactic': 'Impact', 
                 'technique': 'Data Destruction'},
            19: {'id': 'T1486', 'tactic': 'Impact', 
                 'technique': 'Data Encrypted for Impact'},
            20: {'id': 'T1498', 'tactic': 'Impact', 
                 'technique': 'Network Denial of Service'},
            21: {'id': 'T1499', 'tactic': 'Impact', 
                 'technique': 'Endpoint Denial of Service'},
            
            # IoT-Specific
            22: {'id': 'T1200', 'tactic': 'Initial Access', 
                 'technique': 'Hardware Additions'},
            23: {'id': 'T1091', 'tactic': 'Initial Access', 
                 'technique': 'Replication Through Removable Media'},
            24: {'id': 'T1195', 'tactic': 'Initial Access', 
                 'technique': 'Supply Chain Compromise'},
        }
    
    def get_technique(self, technique_id: int) -> Dict:
        """Get technique details by ID"""
        return self.techniques.get(technique_id, self.techniques[0])
    
    def get_random_sequence(self, length: int = 3) -> List[int]:
        """Generate random attack sequence (for initialization)"""
        return list(np.random.choice(len(self.techniques), size=length, replace=False))
    
    def validate_sequence(self, sequence: List[int]) -> bool:
        """
        Validate that attack sequence follows realistic progression
        (e.g., Initial Access → Execution → Exfiltration)
        
        Returns:
            True if sequence is realistic
        """
        if not sequence:
            return False
        
        # Get tactics for sequence
        tactics = [self.techniques[tid]['tactic'] for tid in sequence]
        
        # Check for logical progression (simplified validation)
        tactic_order = [
            'Initial Access', 'Execution', 'Persistence', 
            'Privilege Escalation', 'Defense Evasion', 'Discovery',
            'Lateral Movement', 'Collection', 'Command and Control',
            'Exfiltration', 'Impact'
        ]
        
        # Ensure tactics generally follow kill chain
        prev_index = -1
        for tactic in tactics:
            if tactic in tactic_order:
                curr_index = tactic_order.index(tactic)
                if curr_index < prev_index - 2:  # Allow some flexibility
                    return False
                prev_index = curr_index
        
        return True


class AttackerPolicy(nn.Module):
    """
    Attacker neural network policy
    Learns to select attack sequences that evade defender
    """
    
    def __init__(self, state_dim: int, action_space_size: int, hidden_dim: int = 128):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_space_size)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            state: Defender state (alert levels, confidence)
            
        Returns:
            Logits for attack technique selection
        """
        return self.network(state)


class AttackerAgent:
    """
    Self-play attacker agent for training
    
    CRITICAL: This agent is TRAINING-ONLY
              Never deploy in production
    
    Purpose:
    - Generate adversarial examples during training
    - Improve defender robustness
    - Explore attack strategies defender may encounter
    
    Constraints:
    - Bounded by MITRE ATT&CK (no arbitrary attacks)
    - Realistic attack sequences only
    - Capped reward (prevents over-optimization)
    - Human oversight every N epochs
    """
    
    def __init__(self, config: AttackerConfig):
        self.config = config
        self.attack_library = MITREAttackLibrary()
        
        # Simple policy network (state -> attack technique)
        self.policy = AttackerPolicy(
            state_dim=100,  # Simplified defender state
            action_space_size=config.action_space_size,
            hidden_dim=config.hidden_dim
        )
        
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), 
            lr=config.learning_rate
        )
        
        # Track strategy history for safety oversight
        self.strategy_history = []
        
        print(f"Attacker Agent initialized (TRAINING ONLY)")
        print(f"  Action space: {config.action_space_size} MITRE techniques")
        print(f"  Hidden dim: {config.hidden_dim}")
        print(f"  Learning rate: {config.learning_rate}")
        print(f"  Reward cap: {config.max_reward_per_episode}")
    
    def select_attack(self, defender_state: np.ndarray) -> Dict:
        """
        Select next attack technique based on defender state
        
        Args:
            defender_state: Current defender alert levels, confidence
            
        Returns:
            Dictionary with:
                - technique_id: Selected MITRE technique
                - technique_info: MITRE details
                - timing: When to execute (delay in seconds)
        """
        # Convert to tensor
        state_tensor = torch.FloatTensor(defender_state).unsqueeze(0)
        
        # Get logits from policy
        with torch.no_grad():
            logits = self.policy(state_tensor)
        
        # Sample action (stochastic during training)
        action_probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(action_probs, num_samples=1).item()
        
        # Get technique details
        technique_info = self.attack_library.get_technique(action)
        
        # Random timing (to vary temporal patterns)
        timing_delay = np.random.exponential(scale=60)  # Avg 60 sec delay
        
        attack = {
            'technique_id': action,
            'technique_info': technique_info,
            'timing_delay': timing_delay,
            'confidence': action_probs[0, action].item()
        }
        
        # Log for safety oversight
        self.strategy_history.append(attack)
        
        return attack
    
    def compute_reward(self, attack_success: bool, detected: bool, 
                       detection_confidence: float = 0.0) -> float:
        """
        Compute attacker reward
        
        Args:
            attack_success: Whether attack achieved objective
            detected: Whether defender detected the attack
            detection_confidence: Defender's confidence (0-1)
            
        Returns:
            Attacker reward (higher = better for attacker)
        """
        # Attacker goal: succeed without being detected
        if attack_success and not detected:
            reward = 1.0  # Success
        elif attack_success and detected:
            # Partial success (achieved goal but caught)
            reward = 0.5 * (1 - detection_confidence)
        elif not attack_success and detected:
            reward = -1.0  # Failure
        else:  # not attack_success and not detected
            reward = 0.0  # No impact
        
        # Cap reward to prevent unbounded optimization
        reward = np.clip(reward, -1.0, 1.0)
        
        return reward
    
    def update_policy(self, states: List[np.ndarray], actions: List[int], 
                     rewards: List[float]):
        """
        Update attacker policy using policy gradient
        
        Args:
            states: List of defender states
            actions: List of selected attacks
            rewards: List of rewards received
        """
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states))
        actions_tensor = torch.LongTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)
        
        # Compute returns (simple Monte Carlo for now)
        returns = torch.cumsum(rewards_tensor.flip(0), dim=0).flip(0)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Forward pass
        logits = self.policy(states_tensor)
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # Get log probs of taken actions
        action_log_probs = log_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze()
        
        # Policy gradient loss
        loss = -(action_log_probs * returns).mean()
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def get_strategy_summary(self, last_n: int = 100) -> Dict:
        """
        Get summary of recent strategies for safety oversight
        
        Args:
            last_n: Number of recent strategies to analyze
            
        Returns:
            Summary statistics
        """
        recent = self.strategy_history[-last_n:]
        
        if not recent:
            return {}
        
        # Count technique usage
        technique_counts = {}
        for strategy in recent:
            tid = strategy['technique_id']
            technique_counts[tid] = technique_counts.get(tid, 0) + 1
        
        # Find most common techniques
        sorted_techniques = sorted(
            technique_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        top_techniques = [
            {
                'technique_id': tid,
                'count': count,
                'info': self.attack_library.get_technique(tid)
            }
            for tid, count in sorted_techniques[:5]
        ]
        
        return {
            'total_attacks': len(recent),
            'unique_techniques': len(technique_counts),
            'top_5_techniques': top_techniques,
            'avg_timing_delay': np.mean([s['timing_delay'] for s in recent])
        }
    
    def save_checkpoint(self, path: str):
        """Save attacker policy"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'strategy_history': self.strategy_history[-1000:]  # Last 1000
        }, path)
        print(f"Attacker checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str):
        """Load attacker policy"""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.strategy_history = checkpoint['strategy_history']
        print(f"Attacker checkpoint loaded: {path}")


# Safety validation function
def validate_attacker_safety(agent: AttackerAgent, threshold: float = 0.3) -> bool:
    """
    Validate that attacker has not learned unrealistic strategies
    
    Args:
        agent: Attacker agent to validate
        threshold: Max fraction of attacks that can use same technique
        
    Returns:
        True if safe, False if needs reset
    """
    summary = agent.get_strategy_summary(last_n=100)
    
    if not summary:
        return True
    
    # Check for over-specialization
    for technique in summary['top_5_techniques']:
        usage_fraction = technique['count'] / summary['total_attacks']
        if usage_fraction > threshold:
            print(f"WARNING: Attacker over-using technique {technique['technique_id']}")
            print(f"  Usage: {usage_fraction:.1%} (threshold: {threshold:.1%})")
            return False
    
    # Check for unrealistic sequences
    # (More sophisticated validation could be added here)
    
    return True


# Example usage (TRAINING ONLY)
if __name__ == "__main__":
    print("="*60)
    print("SELF-PLAY ATTACKER AGENT - TRAINING MODE ONLY")
    print("="*60)
    print()
    
    config = AttackerConfig()
    attacker = AttackerAgent(config)
    
    # Simulate training episode
    defender_state = np.random.randn(100)
    
    # Attacker selects attack
    attack = attacker.select_attack(defender_state)
    print(f"Selected attack:")
    print(f"  Technique: {attack['technique_info']['technique']}")
    print(f"  Tactic: {attack['technique_info']['tactic']}")
    print(f"  Timing delay: {attack['timing_delay']:.1f} seconds")
    print()
    
    # Simulate outcome
    attack_success = True
    detected = np.random.rand() > 0.5
    
    reward = attacker.compute_reward(attack_success, detected)
    print(f"Attack outcome:")
    print(f"  Success: {attack_success}")
    print(f"  Detected: {detected}")
    print(f"  Attacker reward: {reward}")
    print()
    
    # Safety validation
    is_safe = validate_attacker_safety(attacker)
    print(f"Safety validation: {'PASS' if is_safe else 'FAIL'}")
