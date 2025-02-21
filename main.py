import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import json
import os
import matplotlib.pyplot as plt

# Initialize Pygame and Matplotlib
pygame.init()
plt.ion()  # Enable interactive mode for live plotting

# Constants
WIDTH, HEIGHT = 600, 400
GRID_SIZE = 20
GRID_WIDTH = WIDTH // GRID_SIZE
GRID_HEIGHT = HEIGHT // GRID_SIZE
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)

# Snake Class
class Snake:
    def __init__(self, algorithm, hyperparams):
        """Initialize a snake with a specific ML algorithm and hyperparameters."""
        self.algorithm = algorithm
        self.hp = hyperparams
        self.color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        
        # Algorithm-specific initialization
        if algorithm == 'q_learning':
            self.q_table = {}
        elif algorithm == 'genetic':
            self.population = [self._create_genetic_individual() for _ in range(self.hp['population_size'])]
            self.current_individual = 0
            self.weights = self.population[self.current_individual]['weights']
            self.genetic_generation = 0
        elif algorithm == 'dqn':
            self.model = self._create_dqn_model()
            self.target_model = self._create_dqn_model()
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.hp['learning_rate'])
            self.memory = deque(maxlen=self.hp['memory_size'])
            self.steps = 0
        
        self.reset()
        self.load_model(f"{algorithm}_model")  # Load saved model if exists

    def _create_dqn_model(self):
        """Create a neural network for DQN."""
        return nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 4)
        )

    def _create_genetic_individual(self):
        """Create a neural network individual for Genetic Algorithm."""
        return {
            "weights": [np.random.randn(8, 16) * 0.1, np.random.randn(16, 4) * 0.1],
            "fitness": 0
        }

    def reset(self):
        """Reset snake state."""
        self.body = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
        self.direction = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
        self.score = 0
        self.alive = True
        self.steps = 0
        self.food_eaten = 0

    def get_state(self, food, other_snakes, obstacles):
        """Generate an 8-element binary state vector."""
        head = self.body[0]
        danger_up = int(head[1] - 1 < 0 or (head[0], head[1] - 1) in obstacles or (head[0], head[1] - 1) in self.body or any((head[0], head[1] - 1) in s.body for s in other_snakes))
        danger_down = int(head[1] + 1 >= GRID_HEIGHT or (head[0], head[1] + 1) in obstacles or (head[0], head[1] + 1) in self.body or any((head[0], head[1] + 1) in s.body for s in other_snakes))
        danger_left = int(head[0] - 1 < 0 or (head[0] - 1, head[1]) in obstacles or (head[0] - 1, head[1]) in self.body or any((head[0] - 1, head[1]) in s.body for s in other_snakes))
        danger_right = int(head[0] + 1 >= GRID_WIDTH or (head[0] + 1, head[1]) in obstacles or (head[0] + 1, head[1]) in self.body or any((head[0] + 1, head[1]) in s.body for s in other_snakes))
        food_up = int(food[1] < head[1])
        food_down = int(food[1] > head[1])
        food_left = int(food[0] < head[0])
        food_right = int(food[0] > head[0])
        state = [danger_up, danger_down, danger_left, danger_right, food_up, food_down, food_left, food_right]
        return tuple(state) if self.algorithm == 'q_learning' else np.array(state, dtype=np.float32)

    def move(self, food, other_snakes, obstacles):
        """Move the snake based on the chosen action."""
        if not self.alive:
            return False
        state = self.get_state(food, other_snakes, obstacles)
        action = self.get_action(state)
        self.steps += 1
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        self.direction = directions[action]
        new_head = (self.body[0][0] + self.direction[0], self.body[0][1] + self.direction[1])
        if (new_head[0] < 0 or new_head[0] >= GRID_WIDTH or new_head[1] < 0 or new_head[1] >= GRID_HEIGHT or 
            new_head in obstacles or new_head in self.body or any(new_head in s.body for s in other_snakes)):
            self.alive = False
            return False
        self.body.insert(0, new_head)
        if new_head == food:
            self.score += 1
            self.food_eaten += 1
            return True
        self.body.pop()
        return False

    def get_action(self, state):
        """Choose an action based on the algorithm."""
        if self.algorithm == 'q_learning':
            if random.random() < self.hp['epsilon']:
                return random.randint(0, 3)
            if state not in self.q_table:
                self.q_table[state] = [0, 0, 0, 0]
            return np.argmax(self.q_table[state])
        elif self.algorithm == 'genetic':
            x = state
            for w in self.weights:
                x = np.tanh(np.dot(x, w))
            return np.argmax(x)
        else:  # DQN
            if random.random() < self.hp['epsilon']:
                return random.randint(0, 3)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(state_tensor)
            return q_values.argmax().item()

    def train(self, old_state, action, reward, new_state):
        """Train the snake's model based on experience."""
        if self.algorithm == 'q_learning':
            if old_state not in self.q_table:
                self.q_table[old_state] = [0, 0, 0, 0]
            if new_state not in self.q_table:
                self.q_table[new_state] = [0, 0, 0, 0]
            old_value = self.q_table[old_state][action]
            next_max = max(self.q_table[new_state])
            new_value = (1 - self.hp['learning_rate']) * old_value + self.hp['learning_rate'] * (reward + self.hp['discount'] * next_max)
            self.q_table[old_state][action] = new_value
            self.hp['epsilon'] = max(self.hp['min_epsilon'], self.hp['epsilon'] * self.hp['epsilon_decay'])
        elif self.algorithm == 'dqn':
            self.memory.append((old_state, action, reward, new_state))
            if len(self.memory) >= self.hp['batch_size']:
                batch = random.sample(self.memory, self.hp['batch_size'])
                states, actions, rewards, next_states = zip(*batch)
                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor(next_states)
                q_values = self.model(states).gather(1, actions.unsqueeze(1))
                next_q_values = self.target_model(next_states).max(1)[0].detach()
                targets = rewards + self.hp['gamma'] * next_q_values
                loss = nn.MSELoss()(q_values.squeeze(), targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.hp['epsilon'] = max(self.hp['min_epsilon'], self.hp['epsilon'] * self.hp['epsilon_decay'])
                if self.steps % self.hp['target_update'] == 0:
                    self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self, filename):
        """Save the trained model to a file."""
        try:
            if self.algorithm == 'q_learning':
                with open(f"{filename}.json", 'w') as f:
                    json.dump({str(k): v for k, v in self.q_table.items()}, f)
            elif self.algorithm == 'genetic':
                np.save(f"{filename}.npy", self.population)
            else:  # DQN
                torch.save(self.model.state_dict(), f"{filename}.pth")
        except Exception as e:
            print(f"Error saving model for {self.algorithm}: {e}")

    def load_model(self, filename):
        """Load a trained model from a file."""
        if self.algorithm == 'q_learning' and os.path.exists(f"{filename}.json"):
            try:
                with open(f"{filename}.json", 'r') as f:
                    data = json.load(f)
                    self.q_table = {eval(k): v for k, v in data.items()}
            except Exception as e:
                print(f"Error loading Q-learning model: {e}")
        elif self.algorithm == 'genetic' and os.path.exists(f"{filename}.npy"):
            try:
                self.population = np.load(f"{filename}.npy", allow_pickle=True).tolist()
                self.weights = self.population[self.current_individual]['weights']
            except Exception as e:
                print(f"Error loading Genetic model: {e}")
        elif self.algorithm == 'dqn' and os.path.exists(f"{filename}.pth"):
            try:
                self.model.load_state_dict(torch.load(f"{filename}.pth"))
                self.target_model.load_state_dict(self.model.state_dict())
            except Exception as e:
                print(f"Error loading DQN model: {e}")

# Game Class
class Game:
    def __init__(self):
        """Initialize the game environment."""
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("ML Snake Game")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.mode = 'training'
        self.training_snake_index = 0
        self.difficulty_level = 0
        self.max_difficulty = 5
        self.score_history = {i: [] for i in range(3)}
        self.game_count = {i: 0 for i in range(3)}
        self.snakes = [
            Snake('q_learning', {'learning_rate': 0.1, 'discount': 0.95, 'epsilon': 0.1, 'epsilon_decay': 0.995, 'min_epsilon': 0.01}),
            Snake('genetic', {'population_size': 50, 'elite_size': 10, 'mutation_rate': 0.1}),
            Snake('dqn', {'learning_rate': 0.001, 'gamma': 0.95, 'epsilon': 0.1, 'epsilon_decay': 0.995, 'min_epsilon': 0.01, 'batch_size': 64, 'memory_size': 10000, 'target_update': 100})
        ]
        self.reset()

    def reset(self):
        """Reset the game state."""
        self.obstacles = []
        num_obstacles = self.difficulty_level * 2
        while len(self.obstacles) < num_obstacles:
            pos = (random.randint(0, GRID_WIDTH-1), random.randint(0, GRID_HEIGHT-1))
            if all(pos not in snake.body for snake in self.snakes) and pos not in self.obstacles:
                self.obstacles.append(pos)
        self.food = self.new_food()
        for snake in self.snakes:
            snake.reset()
            if self.mode == 'training':
                snake.alive = (snake == self.snakes[self.training_snake_index])
                if snake.algorithm == 'genetic':
                    snake.current_individual = 0
                    snake.weights = snake.population[snake.current_individual]['weights']
            else:
                snake.alive = True

    def new_food(self):
        """Generate a new food position."""
        while True:
            food = (random.randint(0, GRID_WIDTH-1), random.randint(0, GRID_HEIGHT-1))
            if food not in self.obstacles and all(food not in snake.body for snake in self.snakes):
                return food

    def evolve_genetic(self, snake):
        """Evolve the genetic population."""
        snake.population.sort(key=lambda x: x['fitness'], reverse=True)
        new_population = snake.population[:snake.hp['elite_size']]
        while len(new_population) < snake.hp['population_size']:
            parent1, parent2 = random.sample(snake.population[:snake.hp['elite_size']], 2)
            child = {'weights': [], 'fitness': 0}
            for w1, w2 in zip(parent1['weights'], parent2['weights']):
                crossover_point = random.randint(0, w1.shape[1])
                child_weight = np.concatenate([w1[:, :crossover_point], w2[:, crossover_point:]], axis=1)
                if random.random() < snake.hp['mutation_rate']:
                    child_weight += np.random.randn(*child_weight.shape) * 0.1
                child['weights'].append(child_weight)
            new_population.append(child)
        snake.population = new_population
        snake.genetic_generation += 1
        print(f"{snake.algorithm} - Gen {snake.genetic_generation} - Best Fitness: {snake.population[0]['fitness']}")

    def plot_progress(self):
        """Plot the learning progress of all snakes."""
        plt.clf()
        for i, snake in enumerate(self.snakes):
            if self.score_history[i]:
                plt.plot(self.score_history[i], label=f"{snake.algorithm} (Games: {self.game_count[i]})")
        plt.legend()
        plt.title("Snake Learning Progress")
        plt.xlabel("Game")
        plt.ylabel("Score")
        plt.pause(0.1)

    def run(self):
        """Main game loop."""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            if self.mode == 'training':
                training_snake = self.snakes[self.training_snake_index]
                if training_snake.alive:
                    old_state = training_snake.get_state(self.food, [], self.obstacles)
                    ate = training_snake.move(self.food, [], self.obstacles)
                    reward = -0.01
                    if not training_snake.alive:
                        reward = -1
                    elif ate:
                        reward = 1
                        self.food = self.new_food()
                    new_state = training_snake.get_state(self.food, [], self.obstacles)
                    training_snake.train(old_state, training_snake.get_action(old_state), reward, new_state)
                else:
                    self.game_count[self.training_snake_index] += 1
                    self.score_history[self.training_snake_index].append(training_snake.score)
                    if training_snake.algorithm == 'genetic':
                        training_snake.population[training_snake.current_individual]['fitness'] = training_snake.score
                        training_snake.current_individual += 1
                        if training_snake.current_individual >= training_snake.hp['population_size']:
                            self.evolve_genetic(training_snake)
                            best_fitness = max(ind['fitness'] for ind in training_snake.population)
                            threshold = max(1, 5 - self.difficulty_level)
                            if best_fitness >= threshold:
                                self.difficulty_level += 1
                                if self.difficulty_level > self.max_difficulty:
                                    self.training_snake_index += 1
                                    self.difficulty_level = 0
                                    if self.training_snake_index >= len(self.snakes):
                                        self.mode = 'competition'
                                        self.difficulty_level = self.max_difficulty
                            training_snake.current_individual = 0
                        training_snake.weights = training_snake.population[training_snake.current_individual]['weights']
                    else:
                        if len(self.score_history[self.training_snake_index]) >= 10:
                            avg_score = np.mean(self.score_history[self.training_snake_index][-10:])
                            threshold = max(1, 5 - self.difficulty_level)
                            if avg_score >= threshold:
                                self.difficulty_level += 1
                                self.score_history[self.training_snake_index] = []
                                if self.difficulty_level > self.max_difficulty:
                                    self.training_snake_index += 1
                                    self.difficulty_level = 0
                                    if self.training_snake_index >= len(self.snakes):
                                        self.mode = 'competition'
                                        self.difficulty_level = self.max_difficulty
                    self.reset()
                    if self.game_count[self.training_snake_index] % 10 == 0:
                        self.plot_progress()
            else:  # Competition mode
                all_dead = True
                for snake in self.snakes:
                    if snake.alive:
                        all_dead = False
                        other_snakes = [s for s in self.snakes if s != snake and s.alive]
                        old_state = snake.get_state(self.food, other_snakes, self.obstacles)
                        ate = snake.move(self.food, other_snakes, self.obstacles)
                        reward = -0.01
                        if not snake.alive:
                            reward = -1
                        elif ate:
                            reward = 1
                            self.food = self.new_food()
                        new_state = snake.get_state(self.food, other_snakes, self.obstacles)
                        snake.train(old_state, snake.get_action(old_state), reward, new_state)
                if all_dead:
                    for i in range(len(self.snakes)):
                        self.game_count[i] += 1
                        self.score_history[i].append(self.snakes[i].score)
                    self.reset()
                    if sum(self.game_count.values()) % 10 == 0:
                        self.plot_progress()

            # Rendering
            self.screen.fill(BLACK)
            for obstacle in self.obstacles:
                pygame.draw.rect(self.screen, YELLOW, (obstacle[0]*GRID_SIZE, obstacle[1]*GRID_SIZE, GRID_SIZE, GRID_SIZE))
            for snake in self.snakes:
                if snake.alive or self.mode == 'training':
                    for i, segment in enumerate(snake.body):
                        color = snake.color if i == 0 else tuple(c//2 for c in snake.color)
                        pygame.draw.rect(self.screen, color, (segment[0]*GRID_SIZE, segment[1]*GRID_SIZE, GRID_SIZE, GRID_SIZE))
            pygame.draw.rect(self.screen, WHITE, (self.food[0]*GRID_SIZE, self.food[1]*GRID_SIZE, GRID_SIZE, GRID_SIZE))
            mode_text = f"Training {self.snakes[self.training_snake_index].algorithm} - Diff {self.difficulty_level}" if self.mode == 'training' else "Competition Mode"
            text = self.font.render(mode_text, True, YELLOW)
            self.screen.blit(text, (10, 10))
            pygame.display.flip()
            self.clock.tick(FPS)

        # Save models on exit
        for snake in self.snakes:
            snake.save_model(f"{snake.algorithm}_model")
        pygame.quit()
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    game = Game()
    game.run()
