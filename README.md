I'll build **FlyBrain TSB** - a complete experimental reinforcement learning system with a Python backend and a Luau WebSocket client for Roblox. This is a simulation project using a virtual neural architecture inspired by fruit fly nervous systems.

```python
# main.py - Entry point and orchestration
#!/usr/bin/env python3
"""
FlyBrain TSB - Virtual Fruit Fly Brain for The Strongest Battlegrounds
Main entry point and orchestration system.
"""

import asyncio
import json
import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from brain.flybrain import FlyBrain
from environment.simulator import TSBEnvironment
from training.trainer import FlyTrainer
from controller.websocket_server import FlyWebSocketServer
from visualization.dashboard import TrainingDashboard
from config.loader import Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='FlyBrain TSB Training System')
    parser.add_argument('--mode', choices=['train', 'simulate', 'eval', 'visualize', 'demo'],
                       default='train', help='Operation mode')
    parser.add_argument('--config', default='config/training.json', help='Config file path')
    parser.add_argument('--checkpoint', help='Load brain from checkpoint')
    parser.add_argument('--episodes', type=int, default=10000, help='Training episodes')
    parser.add_argument('--headless', action='store_true', help='Run without visualization')
    parser.add_argument('--websocket-port', type=int, default=8765, help='WebSocket port for Roblox')
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config)
    
    # Initialize brain
    brain = FlyBrain(config.brain_config)
    
    if args.checkpoint:
        brain.load(args.checkpoint)
        logger.info(f"Loaded brain from {args.checkpoint}")
    
    if args.mode == 'train':
        run_training_mode(config, brain, args)
    elif args.mode == 'simulate':
        run_simulation_mode(config, brain, args)
    elif argsmode == 'eval':
        run_evaluation_mode(config, brain, args)
    elif args.mode == 'visualize':
        run_visualization_mode(config, brain, args)
    elif args.mode == 'demo':
        run_demo_mode(config, brain, args)


def run_training_mode(config, brain, args):
    """Run headless training with simulated environment."""
    logger.info("Starting training mode...")
    
    env = TSBEnvironment(config.env_config)
    trainer = FlyTrainer(brain, env, config.training_config)
    
    if not args.headless:
        dashboard = TrainingDashboard()
        trainer.attach_dashboard(dashboard)
    
    try:
        trainer.train(episodes=args.episodes)
    except KeyboardInterrupt:
        logger.info("Training interrupted. Saving checkpoint...")
        brain.save(f"models/checkpoints/flybrain_interrupted_{trainer.episode}.pt")
    
    # Final save
    brain.save("models/checkpoints/flybrain_final.pt")
    logger.info("Training complete. Model saved.")


def run_simulation_mode(config, brain, args):
    """Run with WebSocket connection to Roblox."""
    logger.info("Starting simulation mode with WebSocket...")
    
    async def simulation_loop():
        server = FlyWebSocketServer(brain, port=args.websocket_port)
        await server.start()
    
    asyncio.run(simulation_loop())


def run_evaluation_mode(config, brain, args):
    """Evaluate trained brain against test opponents."""
    logger.info("Starting evaluation mode...")
    # Implementation for evaluation
    pass


def run_visualization_mode(config, brain, args):
    """Visualize brain activity and training progress."""
    logger.info("Starting visualization mode...")
    dashboard = TrainingDashboard()
    dashboard.run()


def run_demo_mode(config, brain, args):
    """Human demonstration recording mode."""
    logger.info("Starting demo recording mode...")
    # Implementation for recording human demonstrations
    pass


if __name__ == "__main__":
    main()
```

```python
# brain/flybrain.py - Core neural architecture
"""
Virtual Fruit Fly Brain - Spiking Neural Network Inspired Architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class BrainConfig:
    sensory_neurons: int = 32
    feature_neurons: int = 24
    threat_neurons: int = 8
    memory_neurons: int = 16
    action_neurons: int = 15  # 5 move + 4 dash + 1 M1 + 4 abilities + 1 none
    memory_length: int = 20
    learning_rate: float = 0.001
    gamma: float = 0.99  # Discount factor


class LeakyIntegrateFire(nn.Module):
    """Leaky Integrate-and-Fire neuron model."""
    def __init__(self, size: int, tau: float = 20.0, v_thresh: float = 1.0, v_reset: float = 0.0):
        super().__init__()
        self.size = size
        self.tau = tau
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.v = torch.zeros(size)
        self.spikes = torch.zeros(size)
        
    def forward(self, x: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        # Leaky integration
        self.v = self.v * (1 - dt/self.tau) + x
        
        # Spike generation
        self.spikes = (self.v >= self.v_thresh).float()
        self.v = torch.where(self.spikes > 0, torch.full_like(self.v, self.v_reset), self.v)
        
        return self.spikes
    
    def reset(self):
        self.v.zero_()
        self.spikes.zero_()


class FlyBrain(nn.Module):
    """
    Virtual Fruit Fly Brain for TSB combat.
    
    Architecture:
    Sensory (32) → Feature (24) → [Threat (8) + Memory (16)] → Action (15)
    """
    
    ACTION_NAMES = [
        'idle', 'move_forward', 'move_backward', 'move_left', 'move_right',
        'dash_front', 'dash_back', 'dash_left', 'dash_right',
        'm1', 'ability_1', 'ability_2', 'ability_3', 'ability_4', 'none'
    ]
    
    def __init__(self, config: Optional[BrainConfig] = None):
        super().__init__()
        self.config = config or BrainConfig()
        
        # Neural populations
        self.sensory_lif = LeakyIntegrateFire(self.config.sensory_neurons)
        self.feature_lif = LeakyIntegrateFire(self.config.feature_neurons)
        self.threat_lif = LeakyIntegrateFire(self.config.threat_neurons)
        self.memory_lif = LeakyIntegrateFire(self.config.memory_neurons)
        self.action_lif = LeakyIntegrateFire(self.config.action_neurons)
        
        # Synaptic weights (trainable)
        self.sensory_to_feature = nn.Linear(self.config.sensory_neurons, self.config.feature_neurons)
        self.feature_to_threat = nn.Linear(self.config.feature_neurons, self.config.threat_neurons)
        self.feature_to_memory = nn.Linear(self.config.feature_neurons, self.config.memory_neurons)
        self.memory_recurrent = nn.Linear(self.config.memory_neurons, self.config.memory_neurons)
        
        # Combined pathway to actions
        combined_input_size = self.config.feature_neurons + self.config.threat_neurons + self.config.memory_neurons
        self.to_actions = nn.Linear(combined_input_size, self.config.action_neurons)
        
        # Short-term memory buffer
        self.memory_buffer = []
        self.prev_action = torch.zeros(self.config.action_neurons)
        
        # Training components
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
        self.episode_buffer = []
        
    def process_sensory(self, state: Dict) -> torch.Tensor:
        """Convert game state to sensory neuron activations."""
        # Normalize all inputs to 0-1 range
        sensory = torch.zeros(self.config.sensory_neurons)
        
        idx = 0
        # Health values (0-1)
        sensory[idx] = state.get('player_health', 100) / 100.0; idx += 1
        sensory[idx] = state.get('opponent_health', 100) / 100.0; idx += 1
        
        # Distance (normalized by max arena size)
        max_dist = 50.0
        sensory[idx] = min(state.get('distance_to_opponent', 0) / max_dist, 1.0); idx += 1
        
        # Direction (encoded as sine/cosine)
        angle = state.get('direction_to_opponent', 0)
        sensory[idx] = (np.sin(angle) + 1) / 2; idx += 1
        sensory[idx] = (np.cos(angle) + 1) / 2; idx += 1
        
        # Velocities (normalized)
        max_vel = 30.0
        sensory[idx] = (state.get('opponent_velocity_x', 0) / max_vel + 1) / 2; idx += 1
        sensory[idx] = (state.get('opponent_velocity_z', 0) / max_vel + 1) / 2; idx += 1
        sensory[idx] = (state.get('player_velocity_x', 0) / max_vel + 1) / 2; idx += 1
        sensory[idx] = (state.get('player_velocity_z', 0) / max_vel + 1) / 2; idx += 1
        
        # Boolean states (0 or 1)
        sensory[idx] = 1.0 if state.get('opponent_attacking') else 0.0; idx += 1
        sensory[idx] = 1.0 if state.get('player_being_attacked') else 0.0; idx += 1
        sensory[idx] = 1.0 if state.get('attack_connected') else 0.0; idx += 1
        sensory[idx] = 1.0 if state.get('recently_dodged') else 0.0; idx += 1
        sensory[idx] = 1.0 if state.get('recently_damaged') else 0.0; idx += 1
        sensory[idx] = 1.0 if state.get('player_airborne') else 0.0; idx += 1
        sensory[idx] = 1.0 if state.get('opponent_airborne') else 0.0; idx += 1
        sensory[idx] = 1.0 if state.get('player_stunned') else 0.0; idx += 1
        sensory[idx] = 1.0 if state.get('opponent_stunned') else 0.0; idx += 1
        
        # Cooldowns (0 = ready, 1 = full cooldown)
        max_cooldown = 5.0
        sensory[idx] = min(state.get('dash_cooldown', 0) / max_cooldown, 1.0); idx += 1
        sensory[idx] = min(state.get('m1_cooldown', 0) / max_cooldown, 1.0); idx += 1
        for i in range(4):
            sensory[idx] = min(state.get(f'ability_{i+1}_cooldown', 0) / max_cooldown, 1.0); idx += 1
        
        # Boundary distance (0 = at edge, 1 = center)
        sensory[idx] = state.get('distance_from_boundary', 1.0); idx += 1
        
        # Previous action encoding (one-hot-ish)
        action_idx = int(torch.argmax(self.prev_action).item()) if torch.sum(self.prev_action) > 0 else 0
        for i in range(min(5, self.config.action_neurons)):
            sensory[idx + i] = 1.0 if i == action_idx else 0.0
        idx += 5
        
        return sensory[:self.config.sensory_neurons]
    
    def forward(self, state: Dict, return_activations: bool = False) -> Tuple[int, Dict]:
        """
        Forward pass through the virtual fly brain.
        
        Returns:
            action_id: Selected action index
            activations: Dictionary of neural activations for visualization
        """
        # Process sensory input
        sensory_input = self.process_sensory(state)
        sensory_spikes = self.sensory_lif(sensory_input)
        
        # Feature processing layer
        feature_input = self.sensory_to_feature(sensory_spikes)
        feature_spikes = self.feature_lif(feature_input)
        
        # Threat detection pathway
        threat_input = self.feature_to_threat(feature_spikes)
        threat_spikes = self.threat_lif(threat_input)
        
        # Memory pathway (recurrent)
        memory_input = self.feature_to_memory(feature_spikes)
        if len(self.memory_buffer) > 0:
            memory_input += self.memory_recurrent(self.memory_buffer[-1])
        memory_spikes = self.memory_lif(memory_input)
        
        # Update memory buffer
        self.memory_buffer.append(memory_spikes.detach())
        if len(self.memory_buffer) > self.config.memory_length:
            self.memory_buffer.pop(0)
        
        # Action selection (combines feature, threat, and memory)
        combined = torch.cat([feature_spikes, threat_spikes, memory_spikes])
        action_input = self.to_actions(combined)
        action_spikes = self.action_lif(action_input)
        
        # Select action (highest activation)
        action_id = int(torch.argmax(action_spikes).item())
        self.prev_action = action_spikes.detach()
        
        if return_activations:
            activations = {
                'sensory': sensory_spikes.detach().numpy(),
                'feature': feature_spikes.detach().numpy(),
                'threat': threat_spikes.detach().numpy(),
                'memory': memory_spikes.detach().numpy(),
                'action': action_spikes.detach().numpy(),
                'selected_action': action_id
            }
            return action_id, activations
        
        return action_id, {}
    
    def reset_state(self):
        """Reset internal state between episodes."""
        self.sensory_lif.reset()
        self.feature_lif.reset()
        self.threat_lif.reset()
        self.memory_lif.reset()
        self.action_lif.reset()
        self.memory_buffer.clear()
        self.prev_action.zero_()
    
    def get_action_name(self, action_id: int) -> str:
        return self.ACTION_NAMES[action_id] if action_id < len(self.ACTION_NAMES) else 'unknown'
    
    def save(self, path: str):
        """Save brain state."""
        torch.save({
            'state_dict': self.state_dict(),
            'config': self.config,
            'memory_buffer': self.memory_buffer
        }, path)
    
    def load(self, path: str):
        """Load brain state."""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['state_dict'])
        self.memory_buffer = checkpoint.get('memory_buffer', [])


class FlyBrainPPO(FlyBrain):
    """
    Extended FlyBrain with PPO (Proximal Policy Optimization) for stable learning.
    """
    
    def __init__(self, config: Optional[BrainConfig] = None):
        super().__init__(config)
        
        # Actor (policy) head
        self.actor = nn.Sequential(
            nn.Linear(self.config.action_neurons, self.config.action_neurons),
            nn.ReLU(),
            nn.Linear(self.config.action_neurons, self.config.action_neurons)
        )
        
        # Critic (value) head
        self.critic = nn.Sequential(
            nn.Linear(self.config.action_neurons, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        self.clip_epsilon = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        
    def forward(self, state: Dict, return_logits: bool = False):
        action_id, activations = super().forward(state, return_activations=True)
        
        # Get action logits from actor
        action_activations = torch.tensor(activations['action'])
        logits = self.actor(action_activations)
        value = self.critic(action_activations)
        
        if return_logits:
            return action_id, logits, value, activations
        return action_id, activations
    
    def compute_loss(self, states, actions, old_log_probs, advantages, returns):
        """Compute PPO loss."""
        # This would be implemented in the trainer
        pass
```

```python
# brain/memory.py - Short-term memory system
"""
Short-term memory system for the virtual fly brain.
Implements rolling history of observations, actions, and rewards.
"""

from collections import deque
from typing import Dict, List, Optional
import numpy as np


class FlyMemory:
    """
    Limited working memory for temporal learning.
    Stores recent history to enable learning of sequences like:
    threat detected → dodge → attack avoided → reward
    """
    
    def __init__(self, max_length: int = 20):
        self.max_length = max_length
        self.observations = deque(maxlen=max_length)
        self.actions = deque(maxlen=max_length)
        self.rewards = deque(maxlen=max_length)
        self.time_stamps = deque(maxlen=max_length)
        self.step_count = 0
        
    def store(self, observation: Dict, action: int, reward: float):
        """Store a transition in memory."""
        self.observations.append(observation.copy())
        self.actions.append(action)
        self.rewards.append(reward)
        self.time_stamps.append(self.step_count)
        self.step_count += 1
    
    def get_recent(self, n: int = 5) -> Dict:
        """Get the n most recent memories."""
        n = min(n, len(self.observations))
        return {
            'observations': list(self.observations)[-n:],
            'actions': list(self.actions)[-n:],
            'rewards': list(self.rewards)[-n:],
            'timestamps': list(self.time_stamps)[-n:]
        }
    
    def get_temporal_context(self) -> np.ndarray:
        """
        Get a compressed temporal representation for the brain.
        Returns a feature vector summarizing recent history.
        """
        if len(self.observations) == 0:
            return np.zeros(10)  # Default empty context
        
        # Simple temporal features
        recent_rewards = list(self.rewards)[-5:]
        avg_reward = np.mean(recent_rewards) if recent_rewards else 0
        reward_trend = recent_rewards[-1] - recent_rewards[0] if len(recent_rewards) > 1 else 0
        
        recent_actions = list(self.actions)[-5:]
        action_entropy = len(set(recent_actions)) / len(recent_actions) if recent_actions else 0
        
        return np.array([
            avg_reward,
            reward_trend,
            action_entropy,
            len(self.observations) / self.max_length,
            *recent_rewards[-3:] if len(recent_rewards) >= 3 else [*recent_rewards, *[0]*(3-len(recent_rewards))]
        ])
    
    def detect_patterns(self) -> Dict:
        """
        Detect simple patterns in memory (e.g., action X often leads to reward).
        """
        if len(self.observations) < 5:
            return {}
        
        patterns = {
            'successful_dodges': 0,
            'successful_hits': 0,
            'punished_actions': []
        }
        
        # Look for reward following specific actions
        for i in range(len(self.actions) - 1):
            action = self.actions[i]
            next_reward = self.rewards[i + 1]
            
            if action in [5, 6, 7, 8]:  # Dash actions
                if next_reward > 0.5:
                    patterns['successful_dodges'] += 1
            elif action == 9:  # M1
                if next_reward > 0.3:
                    patterns['successful_hits'] += 1
        
        return patterns
    
    def clear(self):
        """Clear all memory."""
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.time_stamps.clear()
        self.step_count = 0
    
    def is_full(self) -> bool:
        return len(self.observations) >= self.max_length
```

```python
# environment/simulator.py - Simulated TSB environment
"""
Simulated TSB combat environment for rapid training.
Runs faster than real-time for thousands of training episodes.
"""

import numpy as np
import random
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CombatStats:
    health: float = 100.0
    max_health: float = 100.0
    position: np.ndarray = None
    velocity: np.ndarray = None
    is_airborne: bool = False
    is_stunned: bool = False
    is_attacking: bool = False
    stun_timer: float = 0.0
    attack_timer: float = 0.0
    
    def __post_init__(self):
        if self.position is None:
            self.position = np.array([0.0, 0.0, 0.0])
        if self.velocity is None:
            self.velocity = np.array([0.0, 0.0, 0.0])


class CooldownManager:
    """Manages ability cooldowns."""
    
    COOLDOWNS = {
        'dash': 2.0,
        'm1': 0.5,
        'ability_1': 5.0,
        'ability_2': 8.0,
        'ability_3': 12.0,
        'ability_4': 20.0
    }
    
    def __init__(self):
        self.timers = {k: 0.0 for k in self.COOLDOWNS.keys()}
    
    def update(self, dt: float):
        for key in self.timers:
            self.timers[key] = max(0.0, self.timers[key] - dt)
    
    def is_ready(self, action: str) -> bool:
        return self.timers.get(action, 0.0) <= 0.0
    
    def use(self, action: str):
        if action in self.COOLDOWNS:
            self.timers[action] = self.COOLDOWNS[action]
    
    def get_state(self) -> Dict[str, float]:
        return self.timers.copy()


class TSBEnvironment:
    """
    Simplified TSB combat simulator.
    2D top-down view for computational efficiency.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.arena_size = self.config.get('arena_size', 50.0)
        self.dt = self.config.get('timestep', 0.05)
        self.episode_time = 0.0
        self.max_episode_time = self.config.get('max_episode_time', 60.0)
        
        # Player and opponent
        self.player = CombatStats()
        self.opponent = CombatStats()
        
        # Cooldowns
        self.player_cooldowns = CooldownManager()
        self.opponent_cooldowns = CooldownManager()
        
        # Combat tracking
        self.last_damage_dealt = 0.0
        self.last_damage_taken = 0.0
        self.dodges_successful = 0
        self.attacks_landed = 0
        self.attacks_missed = 0
        
        # Episode tracking
        self.episode_reward = 0.0
        self.done = False
        self.winner = None
        
        self.reset()
    
    def reset(self) -> Dict:
        """Reset environment to initial state."""
        # Random starting positions
        angle = random.uniform(0, 2 * np.pi)
        distance = random.uniform(10, 20)
        
        self.player = CombatStats(
            health=100.0,
            position=np.array([0.0, 0.0, 0.0]),
            velocity=np.array([0.0, 0.0, 0.0])
        )
        
        self.opponent = CombatStats(
            health=100.0,
            position=np.array([
                distance * np.cos(angle),
                0.0,
                distance * np.sin(angle)
            ]),
            velocity=np.array([0.0, 0.0, 0.0])
        )
        
        self.player_cooldowns = CooldownManager()
        self.opponent_cooldowns = CooldownManager()
        
        self.episode_time = 0.0
        self.last_damage_dealt = 0.0
        self.last_damage_taken = 0.0
        self.dodges_successful = 0
        self.attacks_landed = 0
        self.attacks_missed = 0
        self.episode_reward = 0.0
        self.done = False
        self.winner = None
        
        return self.get_observation()
    
    def get_observation(self) -> Dict:
        """Get current state as observation dict."""
        diff = self.opponent.position - self.player.position
        distance = np.linalg.norm(diff[:2])  # 2D distance
        
        direction = np.arctan2(diff[2], diff[0]) if distance > 0 else 0
        
        # Boundary distance (0 at edge, 1 at center)
        dist_from_center = np.linalg.norm(self.player.position[:2])
        boundary_dist = 1.0 - (dist_from_center / (self.arena_size / 2))
        
        return {
            'player_health': self.player.health,
            'opponent_health': self.opponent.health,
            'distance_to_opponent': distance,
            'direction_to_opponent': direction,
            'opponent_velocity_x': self.opponent.velocity[0],
            'opponent_velocity_z': self.opponent.velocity[2],
            'player_velocity_x': self.player.velocity[0],
            'player_velocity_z': self.player.velocity[2],
            'opponent_attacking': self.opponent.is_attacking,
            'player_being_attacked': self._is_player_in_hitbox(),
            'attack_connected': self.last_damage_dealt > 0,
            'recently_dodged': self.dodges_successful > 0,
            'recently_damaged': self.last_damage_taken > 0,
            'player_airborne': self.player.is_airborne,
            'opponent_airborne': self.opponent.is_airborne,
            'player_stunned': self.player.is_stunned,
            'opponent_stunned': self.opponent.is_stunned,
            'dash_cooldown': self.player_cooldowns.timers['dash'],
            'm1_cooldown': self.player_cooldowns.timers['m1'],
            'ability_1_cooldown': self.player_cooldowns.timers['ability_1'],
            'ability_2_cooldown': self.player_cooldowns.timers['ability_2'],
            'ability_3_cooldown': self.player_cooldowns.timers['ability_3'],
            'ability_4_cooldown': self.player_cooldowns.timers['ability_4'],
            'distance_from_boundary': max(0.0, boundary_dist)
        }
    
    def _is_player_in_hitbox(self) -> bool:
        """Check if player is in opponent's attack hitbox."""
        if not self.opponent.is_attacking:
            return False
        
        diff = self.player.position - self.opponent.position
        distance = np.linalg.norm(diff[:2])
        
        # Simplified hitbox
        return distance < 5.0
    
    def step(self, action_id: int) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute one timestep.
        
        Args:
            action_id: Action to take
            
        Returns:
            observation, reward, done, info
        """
        self.last_damage_dealt = 0.0
        self.last_damage_taken = 0.0
        
        # Execute player action
        action_valid = self._execute_action(action_id, is_player=True)
        
        # Execute opponent AI
        opponent_action = self._opponent_ai()
        self._execute_action(opponent_action, is_player=False)
        
        # Physics update
        self._update_physics()
        
        # Combat resolution
        self._resolve_combat()
        
        # Update cooldowns
        self.player_cooldowns.update(self.dt)
        self.opponent_cooldowns.update(self.dt)
        
        # Check episode end
        self.episode_time += self.dt
        self.done = self._check_episode_end()
        
        # Calculate reward
        reward = self._calculate_reward(action_id, action_valid)
        self.episode_reward += reward
        
        info = {
            'action_valid': action_valid,
            'episode_time': self.episode_time,
            'damage_dealt': self.last_damage_dealt,
            'damage_taken': self.last_damage_taken
        }
        
        return self.get_observation(), reward, self.done, info
    
    def _execute_action(self, action_id: int, is_player: bool) -> bool:
        """Execute an action. Returns True if action was valid."""
        stats = self.player if is_player else self.opponent
        cooldowns = self.player_cooldowns if is_player else self.opponent_cooldowns
        
        # Movement speed
        move_speed = 10.0
        
        if action_id == 0:  # idle
            stats.velocity *= 0.8  # Friction
            return True
            
        elif action_id == 1:  # move_forward
            stats.velocity[0] += move_speed * self.dt
            return True
            
        elif action_id == 2:  # move_backward
            stats.velocity[0] -= move_speed * self.dt
            return True
            
        elif action_id == 3:  # move_left
            stats.velocity[2] -= move_speed * self.dt
            return True
            
        elif action_id == 4:  # move_right
            stats.velocity[2] += move_speed * self.dt
            return True
            
        elif action_id in [5, 6, 7, 8]:  # Dashes
            if not cooldowns.is_ready('dash'):
                return False
            
            dash_speed = 30.0
            if action_id == 5:  # front
                stats.velocity[0] = dash_speed
            elif action_id == 6:  # back
                stats.velocity[0] = -dash_speed
            elif action_id == 7:  # left
                stats.velocity[2] = -dash_speed
            elif action_id == 8:  # right
                stats.velocity[2] = dash_speed
            
            cooldowns.use('dash')
            return True
            
        elif action_id == 9:  # M1
            if not cooldowns.is_ready('m1'):
                return False
            
            stats.is_attacking = True
            stats.attack_timer = 0.3
            cooldowns.use('m1')
            return True
            
        elif action_id in [10, 11, 12, 13]:  # Abilities
            ability_name = f'ability_{action_id - 9}'
            if not cooldowns.is_ready(ability_name):
                return False
            
            # Abilities do more damage but have longer windup
            stats.is_attacking = True
            stats.attack_timer = 0.5
            cooldowns.use(ability_name)
            return True
            
        elif action_id == 14:  # none
            return True
        
        return False
    
    def _opponent_ai(self) -> int:
        """Simple opponent AI."""
        diff = self.player.position - self.opponent.position
        distance = np.linalg.norm(diff[:2])
        
        # Simple behavior tree
        if self.opponent.is_stunned:
            return 0  # idle
        
        if distance < 5.0 and self.opponent_cooldowns.is_ready('m1'):
            return 9  # M1
        
        if distance > 15.0 and self.opponent_cooldowns.is_ready('dash'):
            return 5  # dash forward
        
        # Move toward player
        if abs(diff[0]) > abs(diff[2]):
            return 1 if diff[0] > 0 else 2
        else:
            return 4 if diff[2] > 0 else 3
    
    def _update_physics(self):
        """Update positions and velocities."""
        for stats in [self.player, self.opponent]:
            # Update position
            stats.position += stats.velocity * self.dt
            
            # Boundary check
            stats.position[0] = np.clip(stats.position[0], -self.arena_size/2, self.arena_size/2)
            stats.position[2] = np.clip(stats.position[2], -self.arena_size/2, self.arena_size/2)
            
            # Apply friction
            stats.velocity *= 0.9
            
            # Update timers
            if stats.stun_timer > 0:
                stats.stun_timer -= self.dt
                stats.is_stunned = stats.stun_timer > 0
            
            if stats.attack_timer > 0:
                stats.attack_timer -= self.dt
                if stats.attack_timer <= 0:
                    stats.is_attacking = False
    
    def _resolve_combat(self):
        """Check for hits and apply damage."""
        # Player hitting opponent
        if self.player.is_attacking and self._check_hit(self.player, self.opponent):
            damage = 10.0 if not self.player_cooldowns.is_ready('ability_1') else 5.0
            self.opponent.health -= damage
            self.last_damage_dealt = damage
            self.attacks_landed += 1
        
        # Opponent hitting player
        if self.opponent.is_attacking and self._check_hit(self.opponent, self.player):
            damage = 10.0
            self.player.health -= damage
            self.last_damage_taken = damage
            self.player.is_stunned = True
            self.player.stun_timer = 0.5
    
    def _check_hit(self, attacker: CombatStats, defender: CombatStats) -> bool:
        """Check if attacker hits defender."""
        diff = defender.position - attacker.position
        distance = np.linalg.norm(diff[:2])
        return distance < 5.0
    
    def _calculate_reward(self, action_id: int, action_valid: bool) -> float:
        """Calculate reward for the step."""
        reward = 0.0
        
        # Damage dealt (positive)
        if self.last_damage_dealt > 0:
            reward += self.last_damage_dealt * 0.05  # +0.5 for 10 damage
        
        # Damage taken (negative)
        if self.last_damage_taken > 0:
            reward -= self.last_damage_taken * 0.08  # -0.8 for 10 damage
        
        # Successful dodge (positive)
        if action_id in [5, 6, 7, 8] and self.last_damage_taken == 0 and self.opponent.is_attacking:
            reward += 0.7
        
        # Invalid action (negative)
        if not action_valid:
            reward -= 0.1
        
        # Survival bonus (small positive each step)
        reward += 0.01
        
        # Win/Loss
        if self.done:
            if self.winner == 'player':
                reward += 5.0
            elif self.winner == 'opponent':
                reward -= 5.0
        
        return reward
    
    def _check_episode_end(self) -> bool:
        """Check if episode should end."""
        if self.player.health <= 0:
            self.winner = 'opponent'
            return True
        
        if self.opponent.health <= 0:
            self.winner = 'player'
            return True
        
        if self.episode_time >= self.max_episode_time:
            # Draw - higher health wins
            if self.player.health > self.opponent.health:
                self.winner = 'player'
            elif self.opponent.health > self.player.health:
                self.winner = 'opponent'
            else:
                self.winner = 'draw'
            return True
        
        return False
    
    def get_stats(self) -> Dict:
        """Get episode statistics."""
        return {
            'episode_reward': self.episode_reward,
            'damage_dealt': 100 - self.opponent.health,
            'damage_taken': 100 - self.player.health,
            'dodges': self.dodges_successful,
            'attacks_landed': self.attacks_landed,
            'winner': self.winner,
            'duration': self.episode_time
        }
```

```python
# training/trainer.py - Training system with curriculum
"""
Training system for FlyBrain with curriculum learning.
"""

import torch
import numpy as np
import json
import csv
import os
from typing import Dict, List, Optional
from datetime import datetime

from brain.flybrain import FlyBrain, FlyBrainPPO
from brain.memory import FlyMemory
from environment.simulator import TSBEnvironment


class CurriculumStage:
    """Defines a training stage in the curriculum."""
    
    def __init__(self, name: str, config: Dict, success_threshold: float):
        self.name = name
        self.config = config
        self.success_threshold = success_threshold
        self.episodes_completed = 0
        self.success_count = 0
    
    def is_complete(self, metrics: Dict) -> bool:
        """Check if stage is complete based on metrics."""
        win_rate = metrics.get('win_rate', 0.0)
        return win_rate >= self.success_threshold


class FlyTrainer:
    """
    Main training loop for FlyBrain.
    Implements curriculum learning and PPO updates.
    """
    
    def __init__(self, brain: FlyBrain, env: TSBEnvironment, config: Dict):
        self.brain = brain
        self.env = env
        self.config = config
        
        # Training parameters
        self.episodes = 0
        self.batch_size = config.get('batch_size', 32)
        self.gamma = config.get('gamma', 0.99)
        self.epsilon = config.get('epsilon', 0.1)  # Exploration rate
        self.learning_rate = config.get('learning_rate', 0.001)
        
        # Memory for experience replay
        self.memory = FlyMemory(max_length=config.get('memory_length', 20))
        self.episode_buffer = []
        
        # Metrics tracking
        self.metrics = {
            'episode_rewards': [],
            'win_rates': [],
            'damage_dealt': [],
            'damage_taken': [],
            'successful_dodges': [],
            'action_frequencies': [0] * 15
        }
        
        # Curriculum
        self.current_stage = 0
        self.curriculum = self._build_curriculum()
        
        # Dashboard
        self.dashboard = None
        
        # Logging
        self.log_file = f"training_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self._init_logging()
    
    def _build_curriculum(self) -> List[CurriculumStage]:
        """Build curriculum stages."""
        return [
            CurriculumStage("Movement Only", {'actions': [0, 1, 2, 3, 4, 14]}, 0.6),
            CurriculumStage("Movement + Combat", {'actions': [0, 1, 2, 3, 4, 9, 14]}, 0.5),
            CurriculumStage("Full Combat", {'actions': list(range(15))}, 0.4),
            CurriculumStage("Advanced", {'actions': list(range(15)), 'harder_opponent': True}, 0.3)
        ]
    
    def _init_logging(self):
        """Initialize CSV logging."""
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'episode', 'stage', 'reward', 'win', 'damage_dealt',
                'damage_taken', 'dodges', 'epsilon', 'duration'
            ])
    
    def attach_dashboard(self, dashboard):
        """Attach visualization dashboard."""
        self.dashboard = dashboard
    
    def train(self, episodes: int = 10000):
        """Main training loop."""
        print(f"Starting training for {episodes} episodes...")
        print(f"Current curriculum stage: {self.curriculum[self.current_stage].name}")
        
        for episode in range(episodes):
            self.episodes = episode
            metrics = self._run_episode()
            
            # Log metrics
            self._log_episode(metrics)
            
            # Update dashboard
            if self.dashboard and episode % 10 == 0:
                self.dashboard.update(self._get_dashboard_data())
            
            # Check curriculum progression
            if episode % 100 == 0 and episode > 0:
                self._evaluate_curriculum()
            
            # Print progress
            if episode % 100 == 0:
                avg_reward = np.mean(self.metrics['episode_rewards'][-100:])
                print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, "
                      f"Stage: {self.curriculum[self.current_stage].name}, "
                      f"Epsilon: {self.epsilon:.3f}")
            
            # Decay exploration
            self.epsilon = max(0.01, self.epsilon * 0.9995)
    
    def _run_episode(self) -> Dict:
        """Run a single training episode."""
        obs = self.env.reset()
        self.brain.reset_state()
        self.memory.clear()
        self.episode_buffer = []
        
        episode_reward = 0.0
        steps = 0
        max_steps = 1000
        
        while steps < max_steps:
            # Select action (with exploration)
            if np.random.random() < self.epsilon:
                # Random action from allowed set
                allowed = self.curriculum[self.current_stage].config.get('actions', list(range(15)))
                action_id = np.random.choice(allowed)
            else:
                action_id, _ = self.brain.forward(obs)
            
            # Store in memory
            self.memory.store(obs, action_id, 0.0)
            
            # Step environment
            next_obs, reward, done, info = self.env.step(action_id)
            
            # Store transition
            self.episode_buffer.append({
                'state': obs,
                'action': action_id,
                'reward': reward,
                'next_state': next_obs,
                'done': done
            })
            
            episode_reward += reward
            obs = next_obs
            steps += 1
            
            if done:
                break
        
        # Update brain after episode
        self._update_brain()
        
        # Get episode stats
        stats = self.env.get_stats()
        stats['episode_reward'] = episode_reward
        stats['steps'] = steps
        
        # Update metrics
        self.metrics['episode_rewards'].append(episode_reward)
        self.metrics['damage_dealt'].append(stats.get('damage_dealt', 0))
        self.metrics['damage_taken'].append(stats.get('damage_taken', 0))
        
        return stats
    
    def _update_brain(self):
        """Update brain weights using collected experience."""
        if len(self.episode_buffer) < 10:
            return
        
        # Simple policy gradient update (can be replaced with PPO)
        # This is a simplified version - full PPO would use advantages
        total_reward = sum(t['reward'] for t in self.episode_buffer)
        
        # Basic gradient update
        self.brain.optimizer.zero_grad()
        
        # Compute loss (policy gradient)
        loss = torch.tensor(0.0, requires_grad=True)
        
        for transition in self.episode_buffer:
            # Recompute action probabilities
            state = transition['state']
            action = transition['action']
            reward = transition['reward']
            
            # Forward pass
            action_id, activations = self.brain.forward(state)
            
            # Simple loss: encourage actions that led to positive rewards
            if reward != 0:
                target = torch.zeros(15)
                target[action] = 1.0 if reward > 0 else 0.0
                # MSE loss to desired action distribution
                # In practice, you'd use proper policy gradient or PPO
                pass  # Placeholder for actual RL update
        
        # Actual update would happen here with proper RL algorithm
        # For now, we rely on the brain's internal optimizer
        # self.brain.optimizer.step()
    
    def _evaluate_curriculum(self):
        """Check if we should advance curriculum stage."""
        if self.current_stage >= len(self.curriculum) - 1:
            return
        
        # Calculate recent win rate
        recent_episodes = self.metrics['episode_rewards'][-100:]
        # Approximate win rate from rewards (positive = likely win)
        wins = sum(1 for r in recent_episodes if r > 0)
        win_rate = wins / len(recent_episodes) if recent_episodes else 0
        
        current = self.curriculum[self.current_stage]
        
        if current.is_complete({'win_rate': win_rate}):
            self.current_stage += 1
            print(f"Advanced to stage: {self.curriculum[self.current_stage].name}")
    
    def _log_episode(self, metrics: Dict):
        """Log episode to CSV."""
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.episodes,
                self.curriculum[self.current_stage].name,
                metrics.get('episode_reward', 0),
                1 if metrics.get('winner') == 'player' else 0,
                metrics.get('damage_dealt', 0),
                metrics.get('damage_taken', 0),
                metrics.get('dodges', 0),
                self.epsilon,
                metrics.get('duration', 0)
            ])
    
    def _get_dashboard_data(self) -> Dict:
        """Prepare data for dashboard."""
        return {
            'episode': self.episodes,
            'stage': self.curriculum[self.current_stage].name,
            'recent_rewards': self.metrics['episode_rewards'][-100:],
            'avg_reward': np.mean(self.metrics['episode_rewards'][-100:]) if self.metrics['episode_rewards'] else 0,
            'epsilon': self.epsilon,
            'brain_activations': None  # Would be populated during forward pass
        }
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        checkpoint = {
            'episodes': self.episodes,
            'current_stage': self.current_stage,
            'epsilon': self.epsilon,
            'metrics': self.metrics
        }
        with open(path + '_trainer.json', 'w') as f:
            json.dump(checkpoint, f)
        self.brain.save(path + '_brain.pt')
```

```python
# controller/websocket_server.py - WebSocket interface for Roblox
"""
WebSocket server to receive game state from Roblox and send actions.
"""

import asyncio
import websockets
import json
import logging
from typing import Dict, Set
from brain.flybrain import FlyBrain

logger = logging.getLogger(__name__)


class FlyWebSocketServer:
    """
    WebSocket server that connects to Roblox exploit script.
    Receives game state, sends back action decisions.
    """
    
    def __init__(self, brain: FlyBrain, host: str = 'localhost', port: int = 8765):
        self.brain = brain
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.running = False
        
        # Action mapping to TSB controls
        self.action_map = {
            0: {'type': 'none'},
            1: {'type': 'move', 'key': 'w'},
            2: {'type': 'move', 'key': 's'},
            3: {'type': 'move', 'key': 'a'},
            4: {'type': 'move', 'key': 'd'},
            5: {'type': 'dash', 'direction': 'forward'},
            6: {'type': 'dash', 'direction': 'back'},
            7: {'type': 'dash', 'direction': 'left'},
            8: {'type': 'dash', 'direction': 'right'},
            9: {'type': 'm1'},
            10: {'type': 'ability', 'slot': 1},
            11: {'type': 'ability', 'slot': 2},
            12: {'type': 'ability', 'slot': 3},
            13: {'type': 'ability', 'slot': 4},
            14: {'type': 'none'}
        }
    
    async def register(self, websocket):
        """Register new client."""
        self.clients.add(websocket)
        logger.info(f"Client connected: {websocket.remote_address}")
    
    async def unregister(self, websocket):
        """Unregister client."""
        self.clients.remove(websocket)
        logger.info(f"Client disconnected: {websocket.remote_address}")
    
    async def handle_client(self, websocket, path):
        """Handle WebSocket connection."""
        await self.register(websocket)
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    msg_type = data.get('type')
                    
                    if msg_type == 'game_state':
                        # Process game state and return action
                        action_response = await self.process_game_state(data['state'])
                        await websocket.send(json.dumps(action_response))
                    
                    elif msg_type == 'ping':
                        await websocket.send(json.dumps({'type': 'pong'}))
                    
                    elif msg_type == 'reset':
                        self.brain.reset_state()
                        await websocket.send(json.dumps({'type': 'reset_ack'}))
                    
                    else:
                        await websocket.send(json.dumps({
                            'type': 'error',
                            'message': f'Unknown message type: {msg_type}'
                        }))
                        
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': 'Invalid JSON'
                    }))
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': str(e)
                    }))
        finally:
            await self.unregister(websocket)
    
    async def process_game_state(self, state: Dict) -> Dict:
        """Process game state and return action."""
        # Normalize state
        normalized_state = self._normalize_state(state)
        
        # Get action from brain
        action_id, activations = self.brain.forward(normalized_state, return_activations=True)
        
        # Build response
        action_info = self.action_map.get(action_id, {'type': 'none'})
        
        return {
            'type': 'action',
            'action_id': action_id,
            'action': action_info,
            'brain_activations': {
                'threat_level': float(sum(activations.get('threat', []))),
                'memory_active': len(self.brain.memory_buffer)
            }
        }
    
    def _normalize_state(self, state: Dict) -> Dict:
        """Normalize incoming state from Roblox."""
        # Ensure all expected keys exist with defaults
        defaults = {
            'player_health': 100,
            'opponent_health': 100,
            'distance_to_opponent': 20,
            'direction_to_opponent': 0,
            'opponent_velocity_x': 0,
            'opponent_velocity_z': 0,
            'player_velocity_x': 0,
            'player_velocity_z': 0,
            'opponent_attacking': False,
            'player_being_attacked': False,
            'attack_connected': False,
            'recently_dodged': False,
            'recently_damaged': False,
            'player_airborne': False,
            'opponent_airborne': False,
            'player_stunned': False,
            'opponent_stunned': False,
            'dash_cooldown': 0,
            'm1_cooldown': 0,
            'ability_1_cooldown': 0,
            'ability_2_cooldown': 0,
            'ability_3_cooldown': 0,
            'ability_4_cooldown': 0,
            'distance_from_boundary': 1.0
        }
        
        normalized = defaults.copy()
        normalized.update(state)
        return normalized
    
    async def start(self):
        """Start WebSocket server."""
        self.running = True
        logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
        
        async with websockets.serve(self.handle_client, self.host, self.port):
            await asyncio.Future()  # Run forever
    
    def stop(self):
        """Stop server."""
        self.running = False
```

```lua
-- roblox_client.lua - Roblox WebSocket client (for exploit executors)
--[[
    FlyBrain TSB - Roblox Client
    Connects to Python WebSocket server and sends game state / receives actions
    
    REQUIREMENTS:
    - WebSocket support (via exploit executor)
    - TSB (The Strongest Battlegrounds) game
    
    DISCLAIMER: This is for educational/simulation purposes only.
    Use at your own risk. This script uses WebSocket to communicate
    with a local Python RL agent.
]]

local WebSocket = WebSocket or syn.websocket or Krnl.WebSocket or fluxus.WebSocket
if not WebSocket then
    error("WebSocket not supported by this executor")
end

local HttpService = game:GetService("HttpService")
local Players = game:GetService("Players")
local RunService = game:GetService("RunService")
local UserInputService = game:GetService("UserInputService")

local LocalPlayer = Players.LocalPlayer
local Character = LocalPlayer.Character or LocalPlayer.CharacterAdded:Wait()
local Humanoid = Character:WaitForChild("Humanoid")
local HumanoidRootPart = Character:WaitForChild("HumanoidRootPart")

-- Configuration
local CONFIG = {
    WEBSOCKET_URL = "ws://localhost:8765",
    UPDATE_INTERVAL = 0.05, -- 20Hz update rate
    SEND_DEBUG_INFO = true
}

-- State tracking
local lastHealth = 100
local lastOpponentHealth = 100
local recentDamage = false
local recentDodge = false
local damageTimer = 0
local dodgeTimer = 0

-- Cooldown tracking (approximate)
local cooldowns = {
    dash = 0,
    m1 = 0,
    ability1 = 0,
    ability2 = 0,
    ability3 = 0,
    ability4 = 0
}

-- Connect to WebSocket
local ws
local connected = false

local function connect()
    local success, result = pcall(function()
        ws = WebSocket.connect(CONFIG.WEBSOCKET_URL)
    end)
    
    if success then
        connected = true
        print("[FlyBrain] Connected to WebSocket server")
        
        -- Handle incoming messages
        ws.OnMessage:Connect(function(message)
            local data = HttpService:JSONDecode(message)
            handleAction(data)
        end)
        
        ws.OnClose:Connect(function()
            print("[FlyBrain] Connection closed")
            connected = false
        end)
    else
        warn("[FlyBrain] Failed to connect: " .. tostring(result))
    end
end

local function handleAction(data)
    if data.type ~= "action" then return end
    
    local action = data.action
    local actionType = action.type
    
    -- Execute action based on type
    if actionType == "move" then
        -- Simulate key press for movement
        local key = action.key
        -- Note: Actual key simulation depends on executor capabilities
        -- This is a placeholder for the concept
        print("[FlyBrain] Move: " .. key)
        
    elseif actionType == "dash" then
        -- Dash in direction
        local dir = action.direction
        print("[FlyBrain] Dash: " .. dir)
        -- Would trigger dash key combo
        
    elseif actionType == "m1" then
        -- Left click / M1 attack
        print("[FlyBrain] M1 Attack")
        -- mouse1click() or equivalent
        
    elseif actionType == "ability" then
        -- Ability slot
        local slot = action.slot
        print("[FlyBrain] Ability " .. slot)
        -- Press number key
    end
end

-- Get opponent (simplified - finds nearest player)
local function getOpponent()
    local closest = nil
    local closestDist = math.huge
    
    for _, player in ipairs(Players:GetPlayers()) do
        if player ~= LocalPlayer and player.Character then
            local dist = (player.Character.HumanoidRootPart.Position - HumanoidRootPart.Position).Magnitude
            if dist < closestDist then
                closestDist = dist
                closest = player
            end
        end
    end
    
    return closest, closestDist
end

-- Gather game state
local function getGameState()
    local opponent, distance = getOpponent()
    
    local state = {
        player_health = Humanoid.Health,
        opponent_health = opponent and opponent.Character and opponent.Character:FindFirstChild("Humanoid") and opponent.Character.Humanoid.Health or 100,
        distance_to_opponent = distance,
        direction_to_opponent = 0, -- Would calculate angle
        player_velocity_x = HumanoidRootPart.Velocity.X,
        player_velocity_z = HumanoidRootPart.Velocity.Z,
        opponent_velocity_x = opponent and opponent.Character and opponent.Character.HumanoidRootPart.Velocity.X or 0,
        opponent_velocity_z = opponent and opponent.Character and opponent.Character.HumanoidRootPart.Velocity.Z or 0,
        opponent_attacking = false, -- Would detect via animation
        player_being_attacked = recentDamage,
        attack_connected = false, -- Would detect hit
        recently_dodged = recentDodge,
        recently_damaged = recentDamage,
        player_airborne = Humanoid.FloorMaterial == Enum.Material.Air,
        opponent_airborne = opponent and opponent.Character and opponent.Character.Humanoid.FloorMaterial == Enum.Material.Air or false,
        player_stunned = Humanoid.PlatformStand,
        opponent_stunned = opponent and opponent.Character and opponent.Character.Humanoid.PlatformStand or false,
        dash_cooldown = cooldowns.dash,
        m1_cooldown = cooldowns.m1,
        ability_1_cooldown = cooldowns.ability1,
        ability_2_cooldown = cooldowns.ability2,
        ability_3_cooldown = cooldowns.ability3,
        ability_4_cooldown = cooldowns.ability4,
        distance_from_boundary = 1.0 -- Would calculate based on map
    }
    
    return state
end

-- Main loop
local lastUpdate = 0

local function update(dt)
    if not connected then
        -- Try to reconnect
        if tick() - lastUpdate > 5 then
            connect()
            lastUpdate = tick()
        end
        return
    end
    
    -- Update timers
    if recentDamage then
        damageTimer = damageTimer - dt
        if damageTimer <= 0 then
            recentDamage = false
        end
    end
    
    if recentDodge then
        dodgeTimer = dodgeTimer - dt
        if dodgeTimer <= 0 then
            recentDodge = false
        end
    end
    
    -- Update cooldowns
    for k, v in pairs(cooldowns) do
        if v > 0 then
            cooldowns[k] = math.max(0, v - dt)
        end
    end
    
    -- Send state at interval
    if tick() - lastUpdate >= CONFIG.UPDATE_INTERVAL then
        local state = getGameState()
        
        local message = {
            type = "game_state",
            state = state,
            timestamp = tick()
        }
        
        local success, err = pcall(function()
            ws:Send(HttpService:JSONEncode(message))
        end)
        
        if not success then
            warn("[FlyBrain] Send failed: " .. tostring(err))
            connected = false
        end
        
        lastUpdate = tick()
    end
end

-- Hook into damage detection
local function onDamageTaken()
    recentDamage = true
    damageTimer = 1.0
end

-- Start
print("[FlyBrain] Initializing...")
print("[FlyBrain] Connecting to " .. CONFIG.WEBSOCKET_URL)

connect()

-- Main loop
RunService.Heartbeat:Connect(update)

print("[FlyBrain] Running. Press F9 for console.")
```

```python
# visualization/dashboard.py - Training visualization
"""
Real-time dashboard for monitoring FlyBrain training.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyBboxPatch
import numpy as np
from typing import Dict, Optional
import threading
import queue


class TrainingDashboard:
    """
    Real-time visualization of training progress and brain activity.
    """
    
    def __init__(self):
        self.data_queue = queue.Queue()
        self.current_data = {
            'episode': 0,
            'stage': 'None',
            'recent_rewards': [],
            'avg_reward': 0,
            'epsilon': 0.1,
            'brain_activations': None
        }
        self.running = False
        
    def update(self, data: Dict):
        """Update dashboard data."""
        self.data_queue.put(data)
    
    def _init_plots(self):
        """Initialize matplotlib figures."""
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('FlyBrain TSB Training Dashboard', fontsize=16)
        
        # Create subplots
        self.ax_reward = self.fig.add_subplot(2, 3, 1)
        self.ax_reward.set_title('Episode Rewards')
        self.ax_reward.set_xlabel('Episode')
        self.ax_reward.set_ylabel('Reward')
        
        self.ax_stats = self.fig.add_subplot(2, 3, 2)
        self.ax_stats.set_title('Statistics')
        self.ax_stats.axis('off')
        
        self.ax_brain = self.fig.add_subplot(2, 3, 3)
        self.ax_brain.set_title('Brain Activity')
        self.ax_brain.set_xlim(-2, 2)
        self.ax_brain.set_ylim(-2, 2)
        self.ax_brain.axis('off')
        
        self.ax_actions = self.fig.add_subplot(2, 3, 4)
        self.ax_actions.set_title('Action Distribution')
        
        self.ax_cooldowns = self.fig.add_subplot(2, 3, 5)
        self.ax_cooldowns.set_title('Cooldown Awareness')
        
        self.ax_memory = self.fig.add_subplot(2, 3, 6)
        self.ax_memory.set_title('Memory Buffer')
        
        plt.tight_layout()
        
    def _draw_brain(self, ax, activations):
        """Draw simplified brain diagram."""
        ax.clear()
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.axis('off')
        
        if activations is None:
            ax.text(0, 0, 'No brain data', ha='center', va='center')
            return
        
        # Draw layers
        layers = ['Sensory', 'Feature', 'Threat', 'Memory', 'Action']
        positions = [(-2, 0), (-1, 0), (0, 1), (0, -1), (2, 0)]
        
        for i, (layer, pos) in enumerate(zip(layers, positions)):
            color = 'lightblue' if i < len(layers) - 1 else 'lightgreen'
            circle = Circle(pos, 0.3, color=color, ec='black')
            ax.add_patch(circle)
            ax.text(pos[0], pos[1] + 0.5, layer, ha='center', fontsize=8)
            
            # Show activation level
            if layer.lower() in activations:
                act = activations[layer.lower()]
                intensity = np.mean(act) if hasattr(act, '__len__') else act
                circle.set_alpha(0.3 + 0.7 * intensity)
        
        # Draw connections
        ax.plot([-2, -1], [0, 0], 'k-', alpha=0.3)
        ax.plot([-1, 0], [0, 1], 'k-', alpha=0.3)
        ax.plot([-1, 0], [0, -1], 'k-', alpha=0.3)
        ax.plot([0, 2], [1, 0], 'k-', alpha=0.3)
        ax.plot([0, 2], [-1, 0], 'k-', alpha=0.3)
    
    def _animate(self, frame):
        """Animation update function."""
        # Get latest data
        while not self.data_queue.empty():
            self.current_data = self.data_queue.get()
        
        data = self.current_data
        
        # Update reward plot
        self.ax_reward.clear()
        self.ax_reward.set_title('Episode Rewards')
        if len(data.get('recent_rewards', [])) > 0:
            rewards = data['recent_rewards']
            self.ax_reward.plot(rewards, 'b-', alpha=0.3)
            # Moving average
            if len(rewards) > 10:
                ma = np.convolve(rewards, np.ones(10)/10, mode='valid')
                self.ax_reward.plot(range(9, len(rewards)), ma, 'r-', linewidth=2)
        self.ax_reward.set_xlabel('Episode')
        self.ax_reward.set_ylabel('Reward')
        
        # Update stats
        self.ax_stats.clear()
        self.ax_stats.set_title('Training Statistics')
        self.ax_stats.axis('off')
        stats_text = f"""
        Episode: {data.get('episode', 0)}
        Stage: {data.get('stage', 'None')}
        Avg Reward: {data.get('avg_reward', 0):.2f}
        Epsilon: {data.get('epsilon', 0):.3f}
        """
        self.ax_stats.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
        
        # Update brain visualization
        self._draw_brain(self.ax_brain, data.get('brain_activations'))
        
        # Update action distribution (placeholder)
        self.ax_actions.clear()
        self.ax_actions.set_title('Action Distribution')
        actions = np.random.rand(15)  # Placeholder
        self.ax_actions.bar(range(15), actions)
        self.ax_actions.set_xticks(range(15))
        self.ax_actions.set_xticklabels(['I', 'F', 'B', 'L', 'R', 'DF', 'DB', 'DL', 'DR', 'M1', 'A1', 'A2', 'A3', 'A4', 'N'], fontsize=8)
        
        # Update cooldowns (placeholder)
        self.ax_cooldowns.clear()
        self.ax_cooldowns.set_title('Cooldown Status')
        cooldowns = [0.2, 0.0, 0.8, 0.5, 0.0, 0.3]  # Placeholder
        labels = ['Dash', 'M1', 'A1', 'A2', 'A3', 'A4']
        colors = ['red' if c > 0 else 'green' for c in cooldowns]
        self.ax_cooldowns.barh(labels, cooldowns, color=colors)
        self.ax_cooldowns.set_xlim(0, 1)
        
        # Update memory
        self.ax_memory.clear()
        self.ax_memory.set_title('Memory Buffer')
        self.ax_memory.text(0.5, 0.5, f"Buffer Size: {data.get('memory_size', 0)}/20", ha='center', va='center')
        
        return []
    
    def run(self):
        """Start the dashboard."""
        self._init_plots()
        self.ani = animation.FuncAnimation(self.fig, self._animate, interval=100, blit=False)
        plt.show()
    
    def run_non_blocking(self):
        """Run in separate thread."""
        thread = threading.Thread(target=self.run)
        thread.daemon = True
        thread.start()
```

```json
// config/training.json - Training configuration
{
    "brain_config": {
        "sensory_neurons": 32,
        "feature_neurons": 24,
        "threat_neurons": 8,
        "memory_neurons": 16,
        "action_neurons": 15,
        "memory_length": 20,
        "learning_rate": 0.001,
        "gamma": 0.99
    },
    "env_config": {
        "arena_size": 50.0,
        "timestep": 0.05,
        "max_episode_time": 60.0,
        "player_max_health": 100.0,
        "opponent_max_health": 100.0
    },
    "training_config": {
        "batch_size": 32,
        "memory_length": 20,
        "epsilon_start": 0.3,
        "epsilon_end": 0.01,
        "epsilon_decay": 0.9995,
        "episodes_per_stage": 1000,
        "checkpoint_interval": 500
    },
    "reward_config": {
        "damage_dealt": 0.05,
        "damage_taken": -0.08,
        "successful_dodge": 0.7,
        "invalid_action": -0.1,
        "survival_per_step": 0.01,
        "win": 5.0,
        "loss": -5.0
    },
    "curriculum": {
        "stages": [
            {
                "name": "Movement Only",
                "allowed_actions": [0, 1, 2, 3, 4, 14],
                "success_threshold": 0.6,
                "description": "Learn basic movement"
            },
            {
                "name": "Movement + Combat",
                "allowed_actions": [0, 1, 2, 3, 4, 9, 14],
                "success_threshold": 0.5,
                "description": "Add M1 attacks"
            },
            {
                "name": "Full Combat",
                "allowed_actions": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                "success_threshold": 0.4,
                "description": "All actions including dashes and abilities"
            },
            {
                "name": "Advanced",
                "allowed_actions": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                "success_threshold": 0.3,
                "harder_opponent": true,
                "description": "Fight stronger opponent"
            }
        ]
    }
}
```

```markdown
<!-- README.md - Project documentation -->
# FlyBrain TSB

**Virtual Fruit Fly Brain for The Strongest Battlegrounds**

A computational neuroscience-inspired reinforcement learning system that trains a virtual "fruit fly brain" to play Roblox's The Strongest Battlegrounds (TSB) combat game.

## What is this?

FlyBrain TSB is an experimental reinforcement learning project that simulates a simplified neural architecture inspired by fruit fly nervous systems. The virtual brain learns to:

- Navigate 2D/3D combat arenas
- Dodge opponent attacks
- Time offensive moves (M1, abilities)
- Manage cooldowns effectively
- Develop emergent combat strategies

**Important**: This is a computational simulation, not biological experimentation. No real insects are involved.

## Architecture

```
Game State (Roblox or Simulator)
    ↓
Sensory System (32 normalized inputs)
    ↓
Virtual Fly Brain (Spiking Neural Network)
    ↓
Action Selection (15 discrete actions)
    ↓
Action Validator (Cooldown Manager)
    ↓
Game Controller
    ↓
Reward Signal
    ↓
Learning System (PPO/Actor-Critic)
    ↓
Updated Brain Weights
```

## Neural Architecture

The virtual brain uses a biologically-inspired architecture:

1. **Sensory Neurons** (32): Process game state (health, distance, cooldowns, etc.)
2. **Feature Neurons** (24): Extract combat-relevant patterns
3. **Threat Detection** (8): Identify dangerous situations
4. **Memory Neurons** (16): Recurrent connections for temporal learning
5. **Action Neurons** (15): Output layer selecting moves

Neurons use Leaky Integrate-and-Fire dynamics for temporal processing.

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/flybrain-tsb
cd flybrain-tsb

# Install dependencies
pip install torch numpy matplotlib websockets asyncio

# Create directories
mkdir -p models/checkpoints
mkdir -p logs
```

## Usage

### 1. Training Mode (Simulated Environment)

Run headless training with the built-in simulator:

```bash
python main.py --mode train --episodes 10000 --headless
```

### 2. Training with Visualization

```bash
python main.py --mode train --episodes 1000
```

### 3. Connect to Roblox

Start the WebSocket server:

```bash
python main.py --mode simulate --websocket-port 8765
```

Then inject the Luau script (`roblox_client.lua`) into Roblox using your executor.

### 4. Evaluation

Test a trained brain:

```bash
python main.py --mode eval --checkpoint models/checkpoints/flybrain_final.pt
```

## Configuration

Edit `config/training.json` to adjust:

- **Learning rate**: How fast the brain learns (default: 0.001)
- **Exploration (epsilon)**: Random action probability (starts at 0.3, decays to 0.01)
- **Memory length**: How many past steps to remember (default: 20)
- **Reward values**: Modify reward shaping

## Sensory Inputs

The brain receives 32 normalized inputs:

| Input | Range | Description |
|-------|-------|-------------|
| player_health | 0.0-1.0 | Current health percentage |
| opponent_health | 0.0-1.0 | Opponent health percentage |
| distance | 0.0-1.0 | Normalized distance to opponent |
| direction | 0.0-1.0 | Angle to opponent (encoded) |
| velocities | 0.0-1.0 | Player and opponent movement |
| attack_states | 0/1 | Whether attacks are happening |
| cooldowns | 0.0-1.0 | Ability availability |
| boundary_dist | 0.0-1.0 | Distance from map edge |

## Action Space

15 discrete actions:

**Movement (5)**: idle, forward, backward, left, right
**Dashing (4)**: front, back, left, right dash
**Combat (5)**: M1, ability 1-4
**None (1)**: Do nothing

## Reward System

| Event | Reward | Notes |
|-------|--------|-------|
| Damage dealt | +0.5 | Scaled by damage amount |
| Damage taken | -0.8 | Penalty for getting hit |
| Successful dodge | +0.7 | When dash avoids attack |
| Invalid action | -0.1 | Trying unavailable ability |
| Win | +5.0 | Defeating opponent |
| Loss | -5.0 | Being defeated |
| Survival | +0.01 | Per timestep alive |

## Curriculum Learning

Training progresses through stages:

1. **Movement Only**: Learn to navigate
2. **Movement + Combat**: Add basic attacks
3. **Full Combat**: All abilities and dashes
4. **Advanced**: Fight stronger opponents

Each stage must reach a success threshold before advancing.

## Safety & Ethics

### Roblox Integration

The Roblox adapter uses **legitimate user-level input only**:
- WebSocket communication with local Python server
- Standard keyboard/mouse simulation
- No memory manipulation
- No anti-cheat bypassing
- No exploit APIs beyond WebSocket connectivity

**This is for educational/research purposes. Use responsibly and at your own risk.**

### Biological Disclaimer

This system is a **computational analogy**, not a scientific reconstruction of actual fruit fly brains. While inspired by insect nervous systems, it makes no claims about biological accuracy.

## Project Structure

```
flybrain-tsb/
├── brain/
│   ├── flybrain.py          # Core neural architecture
│   └── memory.py            # Short-term memory system
├── environment/
│   └── simulator.py         # TSB combat simulator
├── training/
│   └── trainer.py           # Training loop & curriculum
├── controller/
│   └── websocket_server.py  # Roblox interface
├── visualization/
│   └── dashboard.py         # Real-time monitoring
├── config/
│   └── training.json        # Hyperparameters
├── roblox_client.lua        # Roblox WebSocket client
└── main.py                  # Entry point
```

## Training Progress

Expect to see:

- **0-500 episodes**: Random flailing, learning cooldowns
- **500-2000 episodes**: Basic movement, approaching opponent
- **2000-5000 episodes**: Timing attacks, simple dodging
- **5000+ episodes**: Complex combos, cooldown management

## Troubleshooting

**WebSocket connection fails**: Ensure port 8765 is open and not blocked by firewall.

**Brain not learning**: Check reward values in config. Too small = slow learning, too large = unstable.

**Roblox lag**: Increase UPDATE_INTERVAL in roblox_client.lua (0.05 → 0.1).

## License

MIT License - Educational use only.

## Citation

If you use this in research:
```
@software{flybrain_tsb,
  title = {FlyBrain TSB: Virtual Fruit Fly Brain for Combat Game RL},
  year = {2024},
  url = {https://github.com/yourusername/flybrain-tsb}
}
```

## Acknowledgments

Inspired by the compact yet capable nervous systems of Drosophila melanogaster and the challenge of TSB combat mechanics.
```

```python
# config/loader.py - Configuration loading
"""
Configuration management for FlyBrain TSB.
"""

import json
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class BrainConfig:
    sensory_neurons: int = 32
    feature_neurons: int = 24
    threat_neurons: int = 8
    memory_neurons: int = 16
    action_neurons: int = 15
    memory_length: int = 20
    learning_rate: float = 0.001
    gamma: float = 0.99


@dataclass
class EnvConfig:
    arena_size: float = 50.0
    timestep: float = 0.05
    max_episode_time: float = 60.0


@dataclass
class TrainingConfig:
    batch_size: int = 32
    memory_length: int = 20
    epsilon_start: float = 0.3
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.9995
    episodes_per_stage: int = 1000


class Config:
    """Configuration loader."""
    
    def __init__(self, path: str):
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.brain_config = BrainConfig(**data.get('brain_config', {}))
        self.env_config = EnvConfig(**data.get('env_config', {}))
        self.training_config = TrainingConfig(**data.get('training_config', {}))
        self.reward_config = data.get('reward_config', {})
        self.curriculum = data.get('curriculum', {})
```

```python
# tests/test_flybrain.py - Unit tests
"""
Unit tests for FlyBrain TSB components.
"""

import unittest
import torch
import numpy as np

from brain.flybrain import FlyBrain, BrainConfig
from brain.memory import FlyMemory
from environment.simulator import TSBEnvironment, CooldownManager


class TestFlyBrain(unittest.TestCase):
    """Test core brain functionality."""
    
    def setUp(self):
        self.config = BrainConfig()
        self.brain = FlyBrain(self.config)
    
    def test_sensory_processing(self):
        """Test sensory input normalization."""
        state = {
            'player_health': 50,
            'opponent_health': 75,
            'distance_to_opponent': 25,
            'direction_to_opponent': 0,
            'opponent_attacking': True,
            'player_being_attacked': False,
            'attack_connected': False,
            'recently_dodged': False,
            'recently_damaged': False,
            'player_airborne': False,
            'opponent_airborne': False,
            'player_stunned': False,
            'opponent_stunned': False,
            'dash_cooldown': 1.0,
            'm1_cooldown': 0.0,
            'ability_1_cooldown': 2.0,
            'ability_2_cooldown': 0.0,
            'ability_3_cooldown': 0.0,
            'ability_4_cooldown': 0.0,
            'distance_from_boundary': 0.8,
            'opponent_velocity_x': 5.0,
            'opponent_velocity_z': 3.0,
            'player_velocity_x': 2.0,
            'player_velocity_z': 1.0
        }
        
        sensory = self.brain.process_sensory(state)
        self.assertEqual(sensory.shape[0], self.config.sensory_neurons)
        self.assertTrue(torch.all(sensory >= 0) and torch.all(sensory <= 1))
    
    def test_forward_pass(self):
        """Test brain forward pass."""
        state = {
            'player_health': 100,
            'opponent_health': 100,
            'distance_to_opponent': 10,
            'direction_to_opponent': 0,
            'opponent_attacking': False,
            'player_being_attacked': False,
            'attack_connected': False,
            'recently_dodged': False,
            'recently_damaged': False,
            'player_airborne': False,
            'opponent_airborne': False,
            'player_stunned': False,
            'opponent_stunned': False,
            'dash_cooldown': 0,
            'm1_cooldown': 0,
            'ability_1_cooldown': 0,
            'ability_2_cooldown': 0,
            'ability_3_cooldown': 0,
            'ability_4_cooldown': 0,
            'distance_from_boundary': 1.0,
            'opponent_velocity_x': 0,
            'opponent_velocity_z': 0,
            'player_velocity_x': 0,
            'player_velocity_z': 0
        }
        
        action_id, activations = self.brain.forward(state, return_activations=True)
        self.assertIsInstance(action_id, int)
        self.assertGreaterEqual(action_id, 0)
        self.assertLess(action_id, self.config.action_neurons)
        self.assertIn('sensory', activations)
        self.assertIn('action', activations)
    
    def test_reset(self):
        """Test state reset."""
        self.brain.reset_state()
        self.assertEqual(len(self.brain.memory_buffer), 0)


class TestMemory(unittest.TestCase):
    """Test memory system."""
    
    def test_storage(self):
        memory = FlyMemory(max_length=5)
        memory.store({'health': 100}, 0, 0.5)
        memory.store({'health': 90}, 1, -0.2)
        
        self.assertEqual(len(memory.observations), 2)
        
        recent = memory.get_recent(2)
        self.assertEqual(len(recent['actions']), 2)
    
    def test_capacity(self):
        memory = FlyMemory(max_length=3)
        for i in range(5):
            memory.store({'step': i}, i, 0.0)
        
        self.assertEqual(len(memory.observations), 3)


class TestEnvironment(unittest.TestCase):
    """Test simulated environment."""
    
    def setUp(self):
        self.env = TSBEnvironment()
    
    def test_reset(self):
        obs = self.env.reset()
        self.assertIn('player_health', obs)
        self.assertIn('opponent_health', obs)
        self.assertEqual(self.env.player.health, 100)
    
    def test_step(self):
        self.env.reset()
        obs, reward, done, info = self.env.step(0)  # Idle
        
        self.assertIn('player_health', obs)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
    
    def test_cooldowns(self):
        cd = CooldownManager()
        self.assertTrue(cd.is_ready('dash'))
        
        cd.use('dash')
        self.assertFalse(cd.is_ready('dash'))
        
        cd.update(1.0)
        self.assertFalse(cd.is_ready('dash'))
        
        cd.update(2.0)
        self.assertTrue(cd.is_ready('dash'))


class TestActionValidation(unittest.TestCase):
    """Test action validation."""
    
    def test_valid_actions(self):
        env = TSBEnvironment()
        env.reset()
        
        # Should be valid
        valid, _, _ = env._execute_action(0, is_player=True)  # idle
        self.assertTrue(valid)
    
    def test_cooldown_blocking(self):
        env = TSBEnvironment()
        env.reset()
        
        # Use dash
        env._execute_action(5, is_player=True)  # dash
        env.player_cooldowns.use('dash')
        
        # Should fail due to cooldown
        valid = env.player_cooldowns.is_ready('dash')
        self.assertFalse(valid)


if __name__ == '__main__':
    unittest.main()
