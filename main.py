import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import copy
import matplotlib.pyplot as plt
import json
import os

pygame.init()

# Constants
WIDTH, HEIGHT = 600, 400
GRID_SIZE = 20
GRID_WIDTH = WIDTH // GRID_SIZE
GRID_HEIGHT = HEIGHT // GRID_SIZE
FPS = 60

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)

class Snake:
    def __init__(self, color, algorithm):
        self.color = color
        self.algorithm = algorithm
        self.reset()
        
        # Enhanced hyperparameters
        self.hyperparams = {
            "q_learning": {"lr": 0.1, "discount": 0.95, "epsilon": 0.1, "epsilon_decay": 0.995, "min_epsilon": 0.01},
            "genetic": {"mutation_rate": 0.1, "population_size": 5, "elite_size": 2},
            "dqn": {"lr": 0.001, "gamma": 0.95, "epsilon": 0.1, "epsilon_decay": 0.995, 
                    "min_epsilon": 0.01, "batch_size": 64, "memory_size": 10000}
        }
        
        # ML specific initializations
        if algorithm == "q_learning":
            self.q_table = {}
            self.hp = self.hyperparams["q_learning"]
            
        elif algorithm == "genetic":
            self.population = [self._create_genetic_individual() for _ in range(self.hyperparams["genetic"]["population_size"])]
            self.current_individual = 0
            self.weights = self.population[self.current_individual]["weights"]
            self.fitness = 0
            self.hp = self.hyperparams["genetic"]
            
        elif algorithm == "dqn":
            self.model = self._create_dqn_model()
            self.target_model = self._create_dqn_model()
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.hyperparams["dqn"]["lr"])
            self.memory = deque(maxlen=self.hyperparams["dqn"]["memory_size"])
            self.hp = self.hyperparams["dqn"]

    def _create_dqn_model(self):
        return nn.Sequential(
            nn.Linear(12, 64),  # Increased input size and network capacity
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

    def _create_genetic_individual(self):
        return {
            "weights": [np.random.randn(12, 8) * 0.1, 
                       np.random.randn(8, 4) * 0.1],
            "fitness": 0
        }

    def reset(self):
        self.body = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
        self.direction = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
        self.score = 0
        self.alive = True
        self.steps = 0
        self.food_eaten = 0

    def get_state(self, food, other_snakes):
        head_x, head_y = self.body[0]
        fx, fy = food
        
        # More sophisticated state representation
        state = [
            fx - head_x,  # x distance to food
            fy - head_y,  # y distance to food
            head_x,       # distance to left wall
            GRID_WIDTH - head_x,  # distance to right wall
            head_y,       # distance to top wall
            GRID_HEIGHT - head_y,  # distance to bottom wall
            int((head_x + 1, head_y) in self.body),  # right collision
            int((head_x - 1, head_y) in self.body),  # left collision
            int(any((head_x + 1, head_y) in s.body for s in other_snakes)),  # other snakes right
            int(any((head_x - 1, head_y) in s.body for s in other_snakes)),  # other snakes left
            self.direction[0],  # current x direction
            self.direction[1]   # current y direction
        ]
        return tuple(state) if self.algorithm == "q_learning" else np.array(state, dtype=np.float32)

    def move(self, food, other_snakes):
        if not self.alive:
            return False

        state = self.get_state(food, other_snakes)
        action = self.get_action(state)
        self.steps += 1
        
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        self.direction = directions[action]
        
        new_head = (self.body[0][0] + self.direction[0], 
                   self.body[0][1] + self.direction[1])
        
        # Enhanced collision detection
        if (new_head[0] < 0 or new_head[0] >= GRID_WIDTH or 
            new_head[1] < 0 or new_head[1] >= GRID_HEIGHT or 
            new_head in self.body or 
            any(new_head in s.body for s in other_snakes)):
            self.alive = False
            return False
        
        self.body.insert(0, new_head)
        
        if new_head == food:
            self.score += 1
            self.food_eaten += 1
            return True
        else:
            self.body.pop()
            return False

    def get_action(self, state):
        if self.algorithm == "q_learning":
            return self._q_learning_action(state)
        elif self.algorithm == "genetic":
            return self._genetic_action(state)
        else:
            return self._dqn_action(state)

    def _q_learning_action(self, state):
        if random.random() < self.hp["epsilon"]:
            return random.randint(0, 3)
        if state not in self.q_table:
            self.q_table[state] = [0, 0, 0, 0]
        return np.argmax(self.q_table[state])

    def _genetic_action(self, state):
        x = state
        for w in self.weights:
            x = np.tanh(np.dot(x, w))
        return np.argmax(x)

    def _dqn_action(self, state):
        if random.random() < self.hp["epsilon"]:
            return random.randint(0, 3)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return q_values.argmax().item()

    def train(self, old_state, action, reward, new_state):
        if self.algorithm == "q_learning":
            if old_state not in self.q_table:
                self.q_table[old_state] = [0, 0, 0, 0]
            if new_state not in self.q_table:
                self.q_table[new_state] = [0, 0, 0, 0]
            
            old_value = self.q_table[old_state][action]
            next_max = max(self.q_table[new_state])
            new_value = (1 - self.hp["lr"]) * old_value + \
                       self.hp["lr"] * (reward + self.hp["discount"] * next_max)
            self.q_table[old_state][action] = new_value
            self.hp["epsilon"] = max(self.hp["min_epsilon"], self.hp["epsilon"] * self.hp["epsilon_decay"])
            
        elif self.algorithm == "dqn" and len(self.memory) >= self.hp["batch_size"]:
            self.memory.append((old_state, action, reward, new_state))
            batch = random.sample(self.memory, self.hp["batch_size"])
            states, actions, rewards, next_states = zip(*batch)
            
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            
            q_values = self.model(states).gather(1, actions.unsqueeze(1))
            next_q_values = self.target_model(next_states).max(1)[0].detach()
            targets = rewards + self.hp["gamma"] * next_q_values
            
            loss = nn.MSELoss()(q_values.squeeze(), targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.hp["epsilon"] = max(self.hp["min_epsilon"], self.hp["epsilon"] * self.hp["epsilon_decay"])
            if self.steps % 100 == 0:  # Update target network
                self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self, filename):
        if self.algorithm == "q_learning":
            with open(filename, 'w') as f:
                json.dump({str(k): v for k, v in self.q_table.items()}, f)
        elif self.algorithm == "genetic":
            np.save(filename, self.population)
        else:  # dqn
            torch.save(self.model.state_dict(), filename)

    def load_model(self, filename):
        if not os.path.exists(filename):
            return
        if self.algorithm == "q_learning":
            with open(filename, 'r') as f:
                data = json.load(f)
                self.q_table = {eval(k): v for k, v in data.items()}
        elif self.algorithm == "genetic":
            self.population = np.load(filename, allow_pickle=True).tolist()
            self.weights = self.population[self.current_individual]["weights"]
        else:  # dqn
            self.model.load_state_dict(torch.load(filename))
            self.target_model.load_state_dict(torch.load(filename))

class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Enhanced ML Snake Game")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.reset()
        self.scores_history = {"q_learning": [], "genetic": [], "dqn": []}
        self.plot_window = deque(maxlen=50)  # Last 50 games for plotting

    def reset(self):
        self.snakes = [
            Snake(GREEN, "q_learning"),
            Snake(RED, "genetic"),
            Snake(BLUE, "dqn")
        ]
        for i, snake in enumerate(self.snakes):
            snake.load_model(f"snake_{i}_model")
        self.food = self.new_food()
        self.generation = 0
        self.best_genetic = None

    def new_food(self):
        while True:
            food = (random.randint(0, GRID_WIDTH-1), 
                   random.randint(0, GRID_HEIGHT-1))
            if all(food not in snake.body for snake in self.snakes):
                return food

    def evolve_genetic(self):
        genetic_snake = next(s for s in self.snakes if s.algorithm == "genetic")
        genetic_snake.population[genetic_snake.current_individual]["fitness"] = genetic_snake.fitness
        
        # Sort by fitness and select elite
        genetic_snake.population.sort(key=lambda x: x["fitness"], reverse=True)
        new_population = genetic_snake.population[:genetic_snake.hp["elite_size"]]
        
        # Create new individuals through crossover and mutation
        while len(new_population) < genetic_snake.hp["population_size"]:
            parent1, parent2 = random.sample(genetic_snake.population[:genetic_snake.hp["elite_size"]], 2)
            child = {"weights": [], "fitness": 0}
            for w1, w2 in zip(parent1["weights"], parent2["weights"]):
                crossover_point = random.randint(0, w1.shape[1])
                child_weight = np.concatenate([w1[:, :crossover_point], w2[:, crossover_point:]], axis=1)
                if random.random() < genetic_snake.hp["mutation_rate"]:
                    child_weight += np.random.randn(*child_weight.shape) * 0.1
                child["weights"].append(child_weight)
            new_population.append(child)
        
        genetic_snake.population = new_population
        genetic_snake.current_individual = (genetic_snake.current_individual + 1) % genetic_snake.hp["population_size"]
        genetic_snake.weights = genetic_snake.population[genetic_snake.current_individual]["weights"]
        self.generation += 1
        print(f"Generation {self.generation} - Best Fitness: {genetic_snake.population[0]['fitness']}")

    def plot_progress(self):
        plt.clf()
        for algo in self.scores_history:
            plt.plot(self.scores_history[algo], label=algo)
        plt.legend()
        plt.title("Snake Learning Progress")
        plt.xlabel("Game")
        plt.ylabel("Score")
        plt.pause(0.1)

    def run(self):
        running = True
        game_count = 0
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            all_dead = True
            for snake in self.snakes:
                if snake.alive:
                    all_dead = False
                    old_state = snake.get_state(self.food, [s for s in self.snakes if s != snake])
                    ate = snake.move(self.food, [s for s in self.snakes if s != snake])
                    
                    # Better reward shaping
                    reward = -0.01  # Small penalty for each step
                    if not snake.alive:
                        reward = -1 - (snake.food_eaten * 0.1)  # Penalty scaled by performance
                    elif ate:
                        reward = 1 + (1 / (snake.steps - snake.food_eaten))  # Reward efficiency
                        self.food = self.new_food()
                    elif np.sqrt((snake.body[0][0] - self.food[0])**2 + 
                               (snake.body[0][1] - self.food[1])**2) < 5:
                        reward += 0.1  # Bonus for getting close to food
                    
                    new_state = snake.get_state(self.food, [s for s in self.snakes if s != snake])
                    snake.train(old_state, snake.get_action(old_state), reward, new_state)

            if all_dead:
                for snake in self.snakes:
                    self.scores_history[snake.algorithm].append(snake.score)
                    snake.save_model(f"snake_{self.snakes.index(snake)}_model")
                self.evolve_genetic()
                self.reset()
                game_count += 1
                if game_count % 10 == 0:  # Plot every 10 games
                    self.plot_progress()

            # Render
            self.screen.fill(BLACK)
            for snake in self.snakes:
                for i, segment in enumerate(snake.body):
                    color = snake.color if i == 0 else tuple(c//2 for c in snake.color)  # Head brighter
                    pygame.draw.rect(self.screen, color, 
                                   (segment[0]*GRID_SIZE, segment[1]*GRID_SIZE, 
                                    GRID_SIZE, GRID_SIZE))
            
            pygame.draw.rect(self.screen, WHITE, 
                           (self.food[0]*GRID_SIZE, self.food[1]*GRID_SIZE, 
                            GRID_SIZE, GRID_SIZE))
            
            # Display scores
            for i, snake in enumerate(self.snakes):
                score_text = self.font.render(f"{snake.algorithm}: {snake.score}", True, YELLOW)
                self.screen.blit(score_text, (10, 10 + i * 30))
            
            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()
        plt.close()

if __name__ == "__main__":
    game = Game()
    game.run()
