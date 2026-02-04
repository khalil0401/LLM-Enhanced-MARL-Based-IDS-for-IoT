"""
MARL Environment for IoT IDS Training
Implements Multi-Agent Dec-POMDP for intrusion detection
"""

import gym
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json


@dataclass
class MARLConfig:
    """Configuration for MARL environment"""
    num_agents: int = 10
    observation_dim: int = 5000
    max_episode_steps: int = 1000
    reward_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.reward_weights is None:
            self.reward_weights = {
                'detect': 1.0,
                'fp': -0.5,
                'latency': -0.2,
                'resource': -0.1
            }


class IoTIDSEnvironment(gym.Env):
    """
    Multi-Agent IoT IDS Environment
    
    Based on Dec-POMDP formulation:
    - N agents (one per IoT device/subnet)
    - Decentralized execution (local observations only)
    - Centralized training (global reward)
    
    Compatible with RLlib MultiAgentEnv
    """
    
    def __init__(self, config: Dict):
        """
        Initialize environment
        
        Args:
            config: Dictionary with:
                - num_agents: Number of MARL agents
                - observation_dim: Dimension of state space
                - dataset_path: Path to preprocessed dataset
                - self_play: Whether to include attacker agent
        """
        self.config = MARLConfig(**config)
        self.num_agents = self.config.num_agents
        
        # Define observation and action spaces
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.config.observation_dim,),
            dtype=np.float32
        )
        
        # Action space: (alert_level, attack_type, confidence)
        self.action_space = gym.spaces.Dict({
            'alert_level': gym.spaces.Discrete(5),  # 0-4
            'attack_type': gym.spaces.Discrete(7),  # None, DDoS, Malware, etc.
            'confidence': gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        })
        
        # Load dataset
        self.dataset = self._load_dataset(config.get('dataset_path'))
        self.current_step = 0
        self.episode_step = 0
        
        # State tracking
        self.agent_observations = {}
        self.ground_truth_labels = None
        
        print(f"MARL Environment initialized:")
        print(f"  - Agents: {self.num_agents}")
        print(f"  - Observation dim: {self.config.observation_dim}")
        print(f"  - Dataset samples: {len(self.dataset)}")
        
    def reset(self) -> Dict[str, np.ndarray]:
        """
        Reset environment to initial state
        
        Returns:
            Dictionary mapping agent_id -> observation
        """
        self.current_step = 0
        self.episode_step = 0
        
        # Sample initial batch from dataset
        batch = self._sample_batch()
        
        # Construct observations for each agent
        self.agent_observations = self._construct_observations(batch)
        self.ground_truth_labels = batch['labels']
        
        return self.agent_observations
    
    def step(self, action_dict: Dict[str, Dict]) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        Execute one environment step
        
        Args:
            action_dict: {agent_id: {'alert_level': x, 'attack_type': y, 'confidence': z}}
            
        Returns:
            observations: {agent_id: obs}
            rewards: {agent_id: reward}
            dones: {agent_id: done, '__all__': all_done}
            infos: {agent_id: info}
        """
        # Compute rewards based on actions and ground truth
        rewards = self._compute_rewards(action_dict, self.ground_truth_labels)
        
        # Advance to next timestep
        self.episode_step += 1
        self.current_step += 1
        
        # Check if episode is done
        done = self.episode_step >= self.config.max_episode_steps
        dones = {agent_id: done for agent_id in self.agent_observations.keys()}
        dones['__all__'] = done
        
        # Get next observations
        if not done:
            batch = self._sample_batch()
            self.agent_observations = self._construct_observations(batch)
            self.ground_truth_labels = batch['labels']
        
        # Info dict
        infos = {
            agent_id: {
                'episode_step': self.episode_step,
                'ground_truth': self.ground_truth_labels.get(agent_id, 0)
            }
            for agent_id in self.agent_observations.keys()
        }
        
        return self.agent_observations, rewards, dones, infos
    
    def _compute_rewards(self, action_dict: Dict[str, Dict], 
                        labels: Dict[str, int]) -> Dict[str, float]:
        """
        Compute multi-objective reward function
        
        R = w_detect * R_detect - w_fp * R_fp - w_latency * R_latency - w_resource * R_resource
        
        Args:
            action_dict: Agent actions
            labels: Ground truth labels {agent_id: 0 (benign) or 1 (attack)}
            
        Returns:
            {agent_id: reward}
        """
        rewards = {}
        w = self.config.reward_weights
        
        for agent_id, action in action_dict.items():
            alert_level = action['alert_level']  # 0-4
            ground_truth = labels.get(agent_id, 0)  # 0 or 1
            
            # Convert alert level to binary prediction (>= 2 is alert)
            predicted = 1 if alert_level >= 2 else 0
            
            # Detection reward component
            if predicted == 1 and ground_truth == 1:
                r_detect = 10.0  # True Positive
            elif predicted == 0 and ground_truth == 0:
                r_detect = 1.0   # True Negative
            elif predicted == 0 and ground_truth == 1:
                r_detect = -20.0  # False Negative (worst!)
            else:  # predicted == 1 and ground_truth == 0
                r_detect = -5.0   # False Positive
            
            # False positive penalty (if FP rate would exceed threshold)
            r_fp = 0.0  # Simplified for now (would track FP rate over episode)
            
            # Latency penalty (encourage fast detection)
            # In real system, would measure actual latency
            r_latency = -0.1 * self.episode_step  # Penalize late detection
            
            # Resource penalty (encourage low alert_level when appropriate)
            # Higher alert levels -> more analyst attention -> higher cost
            r_resource = -0.05 * alert_level
            
            # Combined reward
            reward = (w['detect'] * r_detect + 
                     w['fp'] * r_fp + 
                     w['latency'] * r_latency + 
                     w['resource'] * r_resource)
            
            rewards[agent_id] = reward
        
        return rewards
    
    def _construct_observations(self, batch: Dict) -> Dict[str, np.ndarray]:
        """
        Construct decentralized observations for each agent
        
        Args:
            batch: Data batch with features for each agent
            
        Returns:
            {agent_id: observation_vector}
        """
        observations = {}
        
        for agent_id in range(self.num_agents):
            # Local features for this agent
            local_features = batch['features'][agent_id]
            
            # Neighbor features (simplified: average of all other agents)
            neighbor_features = np.mean(
                [batch['features'][i] for i in range(self.num_agents) if i != agent_id],
                axis=0
            )
            
            # Device context (simulated)
            device_context = np.random.randn(20)  # Placeholder
            
            # Temporal history (last 10 timesteps, simplified as zeros for now)
            temporal_history = np.zeros(10 * (50 + 384))
            
            # Concatenate all observation components
            observation = np.concatenate([
                local_features,
                neighbor_features,
                device_context,
                temporal_history
            ])
            
            # Pad or truncate to observation_dim
            if len(observation) < self.config.observation_dim:
                observation = np.pad(
                    observation, 
                    (0, self.config.observation_dim - len(observation))
                )
            else:
                observation = observation[:self.config.observation_dim]
            
            observations[f'agent_{agent_id}'] = observation.astype(np.float32)
        
        return observations
    
    def _sample_batch(self) -> Dict:
        """
        Sample a batch of data from dataset
        
        Returns:
            Dictionary with:
                - features: List of feature vectors (one per agent)
                - labels: {agent_id: label}
        """
        # Sample random indices
        if self.current_step >= len(self.dataset):
            self.current_step = 0  # Loop back
            
        indices = np.random.choice(
            len(self.dataset), 
            size=self.num_agents, 
            replace=False
        )
        
        features = [self.dataset[i]['features'] for i in indices]
        labels = {f'agent_{i}': self.dataset[idx]['label'] 
                 for i, idx in enumerate(indices)}
        
        return {'features': features, 'labels': labels}
    
    def _load_dataset(self, dataset_path: Optional[str]) -> List[Dict]:
        """
        Load preprocessed dataset
        
        Returns:
            List of samples, each with 'features' and 'label'
        """
        if dataset_path is None:
            # Generate synthetic data for testing
            print("No dataset path provided, using synthetic data")
            num_samples = 10000
            return [
                {
                    'features': np.random.randn(50 + 384),  # Statistical + semantic
                    'label': np.random.choice([0, 1], p=[0.8, 0.2])  # 20% attacks
                }
                for _ in range(num_samples)
            ]
        
        # Load from HDF5 or JSON
        # TODO: Implement actual loading
        raise NotImplementedError("Dataset loading not yet implemented")


# Example usage with RLlib
if __name__ == "__main__":
    # Test environment
    config = {
        'num_agents': 5,
        'observation_dim': 5000,
        'dataset_path': None  # Use synthetic data
    }
    
    env = IoTIDSEnvironment(config)
    
    print("\n=== Testing Environment ===")
    obs = env.reset()
    print(f"Initial observations: {len(obs)} agents")
    print(f"Observation shape: {obs['agent_0'].shape}")
    
    # Simulate one step
    actions = {
        agent_id: {
            'alert_level': np.random.randint(0, 5),
            'attack_type': np.random.randint(0, 7),
            'confidence': np.random.rand(1)
        }
        for agent_id in obs.keys()
    }
    
    next_obs, rewards, dones, infos = env.step(actions)
    
    print(f"\nStep results:")
    print(f"  Rewards: {list(rewards.values())[:3]}...")
    print(f"  Dones: {dones}")
    print(f"  Next obs shape: {next_obs['agent_0'].shape}")
