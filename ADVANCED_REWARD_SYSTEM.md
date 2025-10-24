# Advanced Rocket League Reward System

This document explains the comprehensive reward system implemented for training advanced Rocket League AI agents.

## 🎯 **System Overview**

The new reward system implements **4 major categories** of rewards that train the AI to play like a professional Rocket League player:

1. **Offensive Play Rewards** - Ball control, attacking, and scoring
2. **Defensive Play Rewards** - Goal defense, clearance, and positioning  
3. **Team Coordination Rewards** - Teamwork, spacing, and rotation
4. **Positional Awareness Rewards** - Field coverage, time management, and strategy

## 🚀 **Offensive Play Rewards**

### **Ball Control and Attacking**

| Scenario | Reward | Description |
|----------|--------|-------------|
| **Successfully reach ball** | +100 points | Base reward for touching the ball |
| **High-quality approach** (<30° angle) | +50 points | Excellent approach angle |
| **Decent approach** (<60° angle) | +25 points | Good approach angle |
| **Poor approach** (>60° angle) | +10 points | Basic success |
| **Optimal approach speed** (3-8 units) | +5x distance improvement | Smart speed control |
| **Standard approach** | +2x distance improvement | Normal movement |
| **Good angle to ball** (<45°) | +2 points | Maintaining good angle |
| **Decent angle** (<90°) | +1 point | Acceptable angle |
| **Poor positioning** (close + bad angle) | -5 points | Penalty for bad positioning |

### **Ball Chasing vs. Positioning**

| Scenario | Reward | Description |
|----------|--------|-------------|
| **Over-chasing** (too fast to ball) | -2 points | Penalty for excessive ball chasing |
| **Good spacing** (30-100 units from ball) | +2 points | Reward for proper positioning |

## 🛡️ **Defensive Play Rewards**

### **Goal Defense (Blocking Shots)**

| Scenario | Reward | Description |
|----------|--------|-------------|
| **Ball near goal** (<100 units) + **Good defensive position** (<50 units from goal) | +10 points | Basic defensive positioning |
| **Between ball and goal** | +15 points | Excellent defensive positioning |
| **Out of position** when ball near goal | -10 points | Penalty for poor defense |

### **Clearance and Positioning for Defense**

| Scenario | Reward | Description |
|----------|--------|-------------|
| **Successful clearance** (ball to offensive half) | +20 points | Excellent defensive play |
| **Partial clearance** (ball still in defensive half) | +5 points | Basic defensive play |
| **Good defensive position** (<80 units from goal) | +5 points | Reward for proper positioning |

### **Back Positioning (Returning to Net)**

| Scenario | Reward | Description |
|----------|--------|-------------|
| **Far from goal** (>100 units) + **Ball near goal** (<150 units) | -10 points | Penalty for being out of position |
| **Good defensive position** (<80 units from goal) | +5 points | Reward for proper positioning |

## 🤝 **Team Coordination Rewards**

### **Teammate Proximity and Support**

| Scenario | Reward | Description |
|----------|--------|-------------|
| **Good midfield position** (50-150 units from center) | +3 points | Reward for supportive positioning |
| **Over-chasing** (very close + fast movement) | -2 points | Penalty for excessive ball chasing |
| **Good spacing** (30-100 units from ball) | +2 points | Reward for proper team spacing |

### **Rotation and Positioning**

| Scenario | Reward | Description |
|----------|--------|-------------|
| **Good distance from ball** (30-100 units) | +2 points | Reward for proper rotation |
| **Too close to ball** (<20 units) | Risk of over-chasing | Potential penalty |

## ⚽ **Positional Awareness Rewards**

### **Midfield Play**

| Scenario | Reward | Description |
|----------|--------|-------------|
| **Good midfield position** (50-120 units from center) | +2 points | Reward for midfield presence |
| **Too far from center** (>200 units) | -3 points | Penalty for poor positioning |

### **Time Management (Clock Awareness)**

| Scenario | Reward | Description |
|----------|--------|-------------|
| **Late episode** (>80% complete) + **Close to ball** (<50 units) | +5 points | Urgency bonus |
| **Late episode** + **Far from ball** (>50 units) | -2 points | Penalty for being out of action |

### **Field Coverage**

| Scenario | Reward | Description |
|----------|--------|-------------|
| **Good field coverage** (50-250 units from edges) | +1 point | Reward for useful positioning |

## 📊 **Reward Scale Examples**

### **Typical Episode Rewards:**

| Scenario | Expected Reward Range |
|----------|----------------------|
| **Poor Performance** | -50 to +50 points |
| **Average Performance** | +50 to +200 points |
| **Good Performance** | +200 to +500 points |
| **Excellent Performance** | +500 to +1000+ points |

### **High-Reward Scenarios:**

| Scenario | Total Reward |
|----------|--------------|
| **Perfect approach + success** | +150+ points |
| **Excellent defensive play** | +35+ points |
| **Good team positioning** | +10+ points |
| **Efficient completion** | +100+ points |

## 🎮 **Training Implications**

### **What the AI Learns:**

1. **Offensive Skills:**
   - Approach the ball at optimal angles
   - Control approach speed
   - Maintain good positioning
   - Avoid over-chasing

2. **Defensive Skills:**
   - Position between ball and goal
   - Clear ball to offensive half
   - Return to defensive positions
   - Block incoming shots

3. **Team Skills:**
   - Maintain good spacing
   - Support teammates
   - Avoid ball chasing
   - Rotate properly

4. **Strategic Skills:**
   - Maintain midfield presence
   - Manage time effectively
   - Cover field efficiently
   - Adapt to game state

## 🔧 **Customization Options**

### **Reward Scale Multiplier:**
```python
env = RocketLeagueSB3Env(reward_scale=2.0)  # Double all rewards
```

### **Individual Component Tuning:**
You can modify individual reward values in the environment file:
- Offensive rewards: Lines 298-330
- Defensive rewards: Lines 332-379  
- Team coordination: Lines 381-412
- Positional awareness: Lines 414-443

## 🚀 **Usage Examples**

### **Basic Training:**
```python
from rktl_autonomy.training_env.envs.rocket_league_sb3 import RocketLeagueSB3Env

env = RocketLeagueSB3Env(
    field_width=304.8,
    field_height=426.72,
    max_episode_steps=1000,
    reward_scale=1.0  # Standard rewards
)
```

### **High-Reward Training:**
```python
env = RocketLeagueSB3Env(
    reward_scale=5.0,  # 5x rewards for faster learning
    max_episode_steps=500
)
```

### **Testing Rewards:**
```bash
python test_advanced_rewards.py
```

## 📈 **Expected Training Results**

With this reward system, you should see:

1. **Faster Learning:** More nuanced rewards guide better behavior
2. **Strategic Play:** AI learns offensive and defensive strategies
3. **Efficient Movement:** AI learns optimal positioning and movement
4. **Team Awareness:** AI develops spatial and temporal awareness
5. **Professional Behavior:** AI plays more like a skilled human player

## 🎯 **Next Steps**

1. **Test the system:** Run `python test_advanced_rewards.py`
2. **Start training:** Run `python examples/quick_start_training.py`
3. **Monitor progress:** Use TensorBoard to track reward components
4. **Tune parameters:** Adjust reward values based on training results
5. **Scale up:** Train with longer episodes and more complex scenarios

The advanced reward system transforms simple ball-chasing into sophisticated Rocket League gameplay! 🚀
