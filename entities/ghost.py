import pygame
import math
import constants
import random
from utils.helpers import is_in_ghost_house, is_ghost_house_door

class Ghost:
    def __init__(self, x, y, color=(255, 0, 0), scatter_target=(0, 0)):
        """Initialize ghost with position, colour and behaviour parameters."""
        self.x = x
        self.y = y
        self.radius = constants.TILE_SIZE // 2 - 2
        self.direction = pygame.Vector2(0, 0)
        self.color = color  
        self.exited = False
        self.scatter_target = scatter_target
        self.chasing = False

        # Set ghost speed based on maze complexity
        if constants.CURRENT_MAZE_TYPE == "SIMPLE":
            self.speed = 2  # Slower in simple maze
            self.chase_threshold = constants.TILE_SIZE * 3
        else:
            self.speed = 3  # Original speed in complex maze
            self.chase_threshold = constants.CHASE_THRESHOLD
        
    def update(self, maze, pacman):
        """Update ghost position and behaviour."""
        current_tile_x = int(self.x // constants.TILE_SIZE)
        current_tile_y = int(self.y // constants.TILE_SIZE)
        center_x = current_tile_x * constants.TILE_SIZE + constants.TILE_SIZE // 2
        center_y = current_tile_y * constants.TILE_SIZE + constants.TILE_SIZE // 2

        if abs(self.x - center_x) < self.speed and abs(self.y - center_y) < self.speed:
            self.x = center_x
            self.y = center_y

            if not is_in_ghost_house(current_tile_x, current_tile_y):
                self.exited = True

            valid_directions = []
            for d in [pygame.Vector2(1, 0), pygame.Vector2(-1, 0), pygame.Vector2(0, 1), pygame.Vector2(0, -1)]:
                nx = current_tile_x + int(d.x)
                ny = current_tile_y + int(d.y)
                if 0 <= nx < constants.COLS and 0 <= ny < constants.ROWS and maze[ny][nx] == 0:
                    valid_directions.append(d)
            
            if is_in_ghost_house(current_tile_x, current_tile_y) and pygame.Vector2(0, -1) in valid_directions:
                self.direction = pygame.Vector2(0, -1)
            else:
                if self.exited:
                    valid_directions = [
                        d for d in valid_directions
                        if not is_in_ghost_house(current_tile_x + int(d.x), current_tile_y + int(d.y))
                    ]
                if len(valid_directions) > 1:
                    reverse = pygame.Vector2(-self.direction.x, -self.direction.y)
                    for d in valid_directions[:]:
                        if d.x == reverse.x and d.y == reverse.y:
                            valid_directions.remove(d)
                            break
                            
                if valid_directions:
                    if math.hypot(center_x - pacman.x, center_y - pacman.y) < self.chase_threshold:
                        target = (pacman.x, pacman.y)
                        self.chasing = True
                    else:
                        target = self.scatter_target
                        self.chasing = False

                    best_direction = None
                    best_distance = float('inf')
                    for d in valid_directions:
                        next_tile_x = current_tile_x + int(d.x)
                        next_tile_y = current_tile_y + int(d.y)
                        next_tile_center = (
                            next_tile_x * constants.TILE_SIZE + constants.TILE_SIZE // 2,
                            next_tile_y * constants.TILE_SIZE + constants.TILE_SIZE // 2
                        )
                        distance = math.hypot(
                            next_tile_center[0] - target[0],
                            next_tile_center[1] - target[1]
                        )
                        if distance < best_distance:
                            best_distance = distance
                            best_direction = d

                    current_target_distance = math.hypot(center_x - target[0], center_y - target[1])
                    if current_target_distance < constants.TILE_SIZE * 2 or random.random() < 0.3:
                        self.direction = random.choice(valid_directions)
                    elif best_direction is not None:
                        self.direction = best_direction
        
        self.x += self.direction.x * self.speed
        self.y += self.direction.y * self.speed
        
        if self.x < 0:
            self.x = constants.WIDTH
        elif self.x > constants.WIDTH:
            self.x = 0

    def draw(self, screen):
        """Draw ghost with optional chase mode indicator."""
        if self.chasing:
            pygame.draw.circle(screen, (255, 255, 255), (int(self.x), int(self.y)), self.radius + 2)
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)