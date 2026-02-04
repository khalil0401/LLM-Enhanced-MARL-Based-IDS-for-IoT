"""
MAPPO Training Script for IoT IDS
Multi-Agent Proximal Policy Optimization with parameter sharing
"""

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import argparse
import yaml
from pathlib import Path

from marl_environment import IoTIDSEnvironment


def get_training_config(config_path: str = None) -> dict:
    """Load training configuration from YAML file"""
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    # Default configuration
    return {
        'env_config': {
            'num_agents': 10,
            'observation_dim': 5000,
            'max_episode_steps': 1000,
            'dataset_path': 'data/iot23_processed.h5',
            'self_play': False  # Set to True for adversarial training
        },
        'training': {
            'lr': 3e-4,
            'gamma': 0.99,
            'lambda': 0.95,
            'clip_param': 0.2,
            'train_batch_size': 4096,
            'sgd_minibatch_size': 128,
            'num_sgd_iter': 10,
            'num_workers': 8,
            'num_gpus': 4,
            'framework': 'torch'
        },
        'experiment': {
            'total_iterations': 1000,
            'checkpoint_freq': 10,
            'evaluation_interval': 10,
            'checkpoint_dir': 'checkpoints/mappo'
        }
    }


def train_mappo(config: dict, experiment_name: str = 'mappo_iot_ids'):
    """
    Train MAPPO agents for IoT IDS
    
    Args:
        config: Training configuration dictionary
        experiment_name: Name for experiment tracking
    """
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    env_config = config['env_config']
    training_config = config['training']
    experiment_config = config['experiment']
    
    # Create policy mapping (all agents share one policy)
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        return "shared_policy"
    
    # Configure MAPPO
    algo_config = (
        PPOConfig()
        .environment(
            IoTIDSEnvironment,
            env_config=env_config
        )
        .framework(training_config['framework'])
        .training(
            lr=training_config['lr'],
            gamma=training_config['gamma'],
            lambda_=training_config['lambda'],
            clip_param=training_config['clip_param'],
            train_batch_size=training_config['train_batch_size'],
            sgd_minibatch_size=training_config['sgd_minibatch_size'],
            num_sgd_iter=training_config['num_sgd_iter'],
            vf_clip_param=10.0,
            entropy_coeff=0.01,
        )
        .multi_agent(
            policies={
                "shared_policy": PolicySpec(
                    observation_space=None,  # Inferred from env
                    action_space=None,      # Inferred from env
                    config={}
                )
            },
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["shared_policy"]
        )
        .rollouts(
            num_rollout_workers=training_config['num_workers'],
            num_envs_per_worker=1,
        )
        .resources(
            num_gpus=training_config['num_gpus'],
            num_cpus_per_worker=1,
        )
        .evaluation(
            evaluation_interval=experiment_config['evaluation_interval'],
            evaluation_duration=10,
            evaluation_config={
                "explore": False,
            }
        )
        .reporting(
            min_time_s_per_iteration=0,
            min_sample_timesteps_per_iteration=1000,
        )
    )
    
    # Build algorithm
    algo = algo_config.build()
    
    print("\n" + "="*60)
    print(f"MAPPO Training: {experiment_name}")
    print("="*60)
    print(f"Configuration:")
    print(f"  - Agents: {env_config['num_agents']}")
    print(f"  - Learning rate: {training_config['lr']}")
    print(f"  - Batch size: {training_config['train_batch_size']}")
    print(f"  - Workers: {training_config['num_workers']}")
    print(f"  - GPUs: {training_config['num_gpus']}")
    print(f"  - Total iterations: {experiment_config['total_iterations']}")
    print("="*60 + "\n")
    
    # Training loop
    best_reward = float('-inf')
    
    for i in range(experiment_config['total_iterations']):
        result = algo.train()
        
        # Extract metrics
        episode_reward_mean = result.get('episode_reward_mean', 0)
        episode_len_mean = result.get('episode_len_mean', 0)
        
        # Print progress
        if i % 10 == 0:
            print(f"Iteration {i:4d}:")
            print(f"  Reward: {episode_reward_mean:.2f}")
            print(f"  Episode Length: {episode_len_mean:.1f}")
            print(f"  Timesteps: {result['timesteps_total']}")
            
        # Save checkpoint
        if i % experiment_config['checkpoint_freq'] == 0:
            checkpoint_dir = algo.save(experiment_config['checkpoint_dir'])
            print(f"  Checkpoint saved: {checkpoint_dir}")
            
        # Track best model
        if episode_reward_mean > best_reward:
            best_reward = episode_reward_mean
            best_checkpoint = algo.save(f"{experiment_config['checkpoint_dir']}/best")
            print(f"  NEW BEST! Reward: {best_reward:.2f}")
    
    # Final checkpoint
    final_checkpoint = algo.save(f"{experiment_config['checkpoint_dir']}/final")
    print(f"\nTraining complete!")
    print(f"  Final checkpoint: {final_checkpoint}")
    print(f"  Best reward: {best_reward:.2f}")
    
    algo.stop()
    ray.shutdown()
    
    return final_checkpoint


def evaluate_model(checkpoint_path: str, num_episodes: int = 10):
    """
    Evaluate trained MAPPO model
    
    Args:
        checkpoint_path: Path to model checkpoint
        num_episodes: Number of evaluation episodes
    """
    from ray.rllib.algorithms.algorithm import Algorithm
    
    ray.init(ignore_reinit_error=True)
    
    # Load algorithm from checkpoint
    algo = Algorithm.from_checkpoint(checkpoint_path)
    
    print(f"\nEvaluating model: {checkpoint_path}")
    print(f"Episodes: {num_episodes}\n")
    
    env_config = algo.config['env_config']
    env = IoTIDSEnvironment(env_config)
    
    episode_rewards = []
    detection_accuracy = []
    
    for ep in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        correct_detections = 0
        total_samples = 0
        done = False
        
        while not done:
            # Get actions from all agents
            actions = {}
            for agent_id in obs.keys():
                action = algo.compute_single_action(
                    obs[agent_id],
                    policy_id="shared_policy",
                    explore=False
                )
                actions[agent_id] = action
            
            # Environment step
            obs, rewards, dones, infos = env.step(actions)
            
            # Track metrics
            episode_reward += sum(rewards.values())
            
            for agent_id, info in infos.items():
                if agent_id == '__all__':
                    continue
                ground_truth = info['ground_truth']
                predicted = 1 if actions[agent_id]['alert_level'] >= 2 else 0
                if predicted == ground_truth:
                    correct_detections += 1
                total_samples += 1
            
            done = dones['__all__']
        
        accuracy = correct_detections / total_samples if total_samples > 0 else 0
        episode_rewards.append(episode_reward)
        detection_accuracy.append(accuracy)
        
        print(f"Episode {ep+1}: Reward={episode_reward:.2f}, Accuracy={accuracy:.3f}")
    
    print(f"\nEvaluation Results:")
    print(f"  Avg Reward: {sum(episode_rewards)/len(episode_rewards):.2f}")
    print(f"  Avg Accuracy: {sum(detection_accuracy)/len(detection_accuracy):.3f}")
    print(f"  Std Reward: {np.std(episode_rewards):.2f}")
    
    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MAPPO for IoT IDS')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config YAML file')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'evaluate'],
                       help='Train or evaluate mode')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Checkpoint path for evaluation')
    parser.add_argument('--name', type=str, default='mappo_iot_ids',
                       help='Experiment name')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        config = get_training_config(args.config)
        checkpoint = train_mappo(config, args.name)
        print(f"\nTo evaluate: python train_mappo.py --mode evaluate --checkpoint {checkpoint}")
        
    elif args.mode == 'evaluate':
        if args.checkpoint is None:
            print("Error: --checkpoint required for evaluation mode")
        else:
            import numpy as np
            evaluate_model(args.checkpoint, num_episodes=10)
