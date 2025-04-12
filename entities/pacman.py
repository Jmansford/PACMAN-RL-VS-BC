import pygame
import math
import numpy as np
from utils.helpers import grid_to_pixel, pixel_to_grid, collides_with_wall
from utils.a_star import a_star
import constants
import random

class Pacman:
    def __init__(self, x, y):
        """Initialise Pac-Man with position and movement parameters."""
        self.x = x
        self.y = y
        self.radius = constants.TILE_SIZE // 2 - 2
        self.speed = 3
        self.direction = pygame.Vector2(0, 0)
        self.desired_direction = pygame.Vector2(0, 0)
        
        # A* pathfinding attributes
        self.path = []
        self.target_pellet = None
        self.path_finding_cooldown = 0
        
        # DQN attributes
        self.last_action = None
        self.dqn_model = None
        self.action_history = []
        self.last_positions = []

    def update(self, maze, active_ghosts=None, pellet_grid=None):
        """Update Pac-Man's position and state."""
        prev_x, prev_y = self.x, self.y

        if constants.GAME_MODE == "DQN" and self.dqn_model:
            current_cell = pixel_to_grid(self.x, self.y)
            model_obs_dim = self.dqn_model.policy.observation_space.shape[0]
            observation = self._create_observation(active_ghosts, pellet_grid, maze)
            
            if len(observation) > model_obs_dim:
                observation = observation[:model_obs_dim]
            
            action, _ = self.dqn_model.predict(observation, deterministic=True)
            action = int(action.item()) if hasattr(action, 'item') else int(action)
            self.last_action = action
            
            direction_map = {
                0: pygame.Vector2(1, 0),   # Right
                1: pygame.Vector2(-1, 0),  # Left
                2: pygame.Vector2(0, 1),   # Down
                3: pygame.Vector2(0, -1)   # Up
            }
            self.desired_direction = direction_map[action]
            
            self.action_history.append(action)
            if len(self.action_history) > 10:
                self.action_history.pop(0)
            
            current_pos = pixel_to_grid(self.x, self.y)
            self.last_positions.append(current_pos)
            if len(self.last_positions) > 5:
                self.last_positions.pop(0)
            
        elif constants.GAME_MODE == "A_STAR":
            current_cell = pixel_to_grid(self.x, self.y)
            
            if self.path_finding_cooldown > 0:
                self.path_finding_cooldown -= 1
            
            if (not self.path or len(self.path) <= 1 or self.target_pellet is None or 
                not pellet_grid[self.target_pellet[1]][self.target_pellet[0]]) and self.path_finding_cooldown == 0:
                
                best_goal = None
                best_distance = float('inf')
                
                for r in range(constants.ROWS):
                    for c in range(constants.COLS):
                        if pellet_grid[r][c]:
                            d = abs(current_cell[0] - c) + abs(current_cell[1] - r)
                            if d < best_distance:
                                best_distance = d
                                best_goal = (c, r)
                
                if best_goal is not None:
                    self.target_pellet = best_goal
                    self.path = a_star(current_cell, best_goal, maze)
                    
                    if not self.path:
                        self.path_finding_cooldown = 10
                        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                        random.shuffle(directions)
                        for dx, dy in directions:
                            test_x = self.x + dx * self.speed
                            test_y = self.y + dy * self.speed
                            if not collides_with_wall(test_x, test_y, self.radius, maze):
                                self.desired_direction = pygame.Vector2(dx, dy)
                                break
            
            if self.path and len(self.path) > 1:
                next_cell = self.path[1]
                next_pixel_x, next_pixel_y = grid_to_pixel(next_cell[0], next_cell[1])
                dx = next_pixel_x - self.x
                dy = next_pixel_y - self.y
                
                length = math.sqrt(dx*dx + dy*dy)
                if length > 0:
                    dx /= length
                    dy /= length
                
                if abs(dx) > abs(dy):
                    self.desired_direction = pygame.Vector2(1 if dx > 0 else -1, 0)
                else:
                    self.desired_direction = pygame.Vector2(0, 1 if dy > 0 else -1)
                
                if length < self.speed * 2:
                    self.path.pop(0)

        door_rect = None
        if constants.CURRENT_MAZE_TYPE == "COMPLEX":
            ghost_info = constants.get_ghost_house_info()
            door_rect = pygame.Rect(
                ghost_info["door_col_start"] * constants.TILE_SIZE, 
                ghost_info["door_row"] * constants.TILE_SIZE, 
                (ghost_info["door_col_end"] - ghost_info["door_col_start"] + 1) * constants.TILE_SIZE, 
                constants.TILE_SIZE
            )
        
        if self.desired_direction.length_squared() > 0:
            test_x = self.x + self.desired_direction.x * self.speed
            test_y = self.y + self.desired_direction.y * self.speed
            test_rect = pygame.Rect(test_x - self.radius, test_y - self.radius, 2 * self.radius, 2 * self.radius)
            
            if not (constants.CURRENT_MAZE_TYPE == "COMPLEX" and door_rect and test_rect.colliderect(door_rect)):
                if not collides_with_wall(test_x, test_y, self.radius, maze):
                    self.direction = self.desired_direction

        if self.direction.length_squared() > 0:
            new_x = self.x + self.direction.x * self.speed
            new_y = self.y + self.direction.y * self.speed
            new_rect = pygame.Rect(new_x - self.radius, new_y - self.radius, 2 * self.radius, 2 * self.radius)
            
            if not (constants.CURRENT_MAZE_TYPE == "COMPLEX" and door_rect and new_rect.colliderect(door_rect)):
                if not collides_with_wall(new_x, new_y, self.radius, maze):
                    self.x = new_x
                    self.y = new_y

        if self.x < 0:
            self.x = constants.WIDTH
        elif self.x > constants.WIDTH:
            self.x = 0
        
        if abs(self.x - prev_x) < 0.1 and abs(self.y - prev_y) < 0.1 and self.direction.length_squared() > 0:
            self.wall_collision_count = getattr(self, 'wall_collision_count', 0) + 1
        else:
            self.wall_collision_count = 0
            
    def _get_valid_actions(self, maze):
        """Get list of valid actions from current position."""
        valid_actions = []
        directions = [
            (0, pygame.Vector2(1, 0)),    # Right
            (1, pygame.Vector2(-1, 0)),   # Left
            (2, pygame.Vector2(0, 1)),    # Down
            (3, pygame.Vector2(0, -1))    # Up
        ]
        
        door_rect = None
        if constants.CURRENT_MAZE_TYPE == "COMPLEX":
            ghost_info = constants.get_ghost_house_info()
            door_rect = pygame.Rect(
                ghost_info["door_col_start"] * constants.TILE_SIZE, 
                ghost_info["door_row"] * constants.TILE_SIZE, 
                (ghost_info["door_col_end"] - ghost_info["door_col_start"] + 1) * constants.TILE_SIZE, 
                constants.TILE_SIZE
            )
        
        for action, direction in directions:
            test_x = self.x + direction.x * self.speed
            test_y = self.y + direction.y * self.speed
            test_rect = pygame.Rect(test_x - self.radius, test_y - self.radius, 2 * self.radius, 2 * self.radius)
            
            if not (constants.CURRENT_MAZE_TYPE == "COMPLEX" and door_rect and test_rect.colliderect(door_rect)):
                if not collides_with_wall(test_x, test_y, self.radius, maze):
                    valid_actions.append(action)
                
        return valid_actions
            
    def _create_observation(self, ghosts, pellet_grid=None, maze=None):
        """Create observation vector for DQN agent."""
        pacman_x, pacman_y = pixel_to_grid(self.x, self.y)
        norm_x = pacman_x / constants.COLS
        norm_y = pacman_y / constants.ROWS
        
        direction_vec = [0, 0, 0, 0]  # right, left, down, up
        if self.direction.x > 0:
            direction_vec[0] = 1
        elif self.direction.x < 0:
            direction_vec[1] = 1
        elif self.direction.y > 0:
            direction_vec[2] = 1
        elif self.direction.y < 0:
            direction_vec[3] = 1
        
        # Define directions for sensors
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # right, left, down, up
        
        # --- WALL SENSORS ---
        # Distance measurements to walls in each direction
        wall_sensors = [0, 0, 0, 0]  # right, left, down, up
        
        if maze is not None:
            for i, (dx, dy) in enumerate(directions):
                # Search for walls in this direction
                for dist in range(1, 8):  # Look up to 7 cells away
                    check_x = pacman_x + (dx * dist)
                    check_y = pacman_y + (dy * dist)
                    
                    # Check if we're out of bounds or hit a wall
                    if (check_x < 0 or check_x >= constants.COLS or 
                        check_y < 0 or check_y >= constants.ROWS or 
                        (maze[check_y][check_x] == 1)):
                        # Wall found - encode its distance (closer = higher value)
                        wall_sensors[i] = 1.0 - ((dist - 1) / 7.0)
                        break
        
        # --- JUNCTION INFORMATION ---
        # Explicitly mark if current position is a junction (3+ valid moves)
        is_junction = 0.0
        valid_moves = [0, 0, 0, 0]  # Valid moves in each direction
        
        if maze is not None:
            valid_move_count = 0
            for i, (dx, dy) in enumerate(directions):
                next_x = pacman_x + dx
                next_y = pacman_y + dy
                
                if (0 <= next_y < constants.ROWS and 
                    0 <= next_x < constants.COLS and 
                    maze[next_y][next_x] == 0):
                    valid_move_count += 1
                    valid_moves[i] = 1.0  # Mark this as a valid move
            
            is_junction = 1.0 if valid_move_count >= 3 else 0.0
        
        # --- PELLET DETECTION ---
        # Improved pellet detection with layered sensors
        pellet_sensors = [0, 0, 0, 0]  # right, left, down, up
        
        if pellet_grid is not None and maze is not None:
            for i, (dx, dy) in enumerate(directions):
                # Search for pellets at different distances with diminishing importance
                for dist in range(1, 10):
                    check_x = pacman_x + (dx * dist)
                    check_y = pacman_y + (dy * dist)
                    
                    # Check if position is valid
                    if (0 <= check_y < constants.ROWS and 
                        0 <= check_x < constants.COLS):
                        # Stop searching if we hit a wall
                        if maze[check_y][check_x] == 1:
                            break
                        
                        # Check if there's a pellet here
                        if pellet_grid[check_y][check_x]:
                            # Scale the sensor value based on distance
                            if dist <= 3:
                                pellet_sensors[i] = 1.0  # Very close pellet
                            elif dist <= 6:
                                pellet_sensors[i] = 0.7  # Medium distance pellet
                            else:
                                pellet_sensors[i] = 0.4  # Far pellet
                            break
        
        # --- AREA EVALUATION ---
        # Analyze pellet density in each quadrant
        quadrant_pellets = [0, 0, 0, 0]  # top-right, top-left, bottom-right, bottom-left
        
        if pellet_grid is not None:
            # Count pellets in each quadrant
            quadrant_counts = [0, 0, 0, 0]
            quadrant_totals = [0, 0, 0, 0]
            
            # Define quadrant boundaries
            mid_col = constants.COLS // 2
            mid_row = constants.ROWS // 2
            
            for r in range(constants.ROWS):
                for c in range(constants.COLS):
                    # Skip walls (can't have pellets)
                    if maze is not None and maze[r][c] == 1:
                        continue
                    
                    # Determine quadrant
                    quadrant = 0
                    if c < mid_col:
                        quadrant += 1  # Left side
                    if r >= mid_row:
                        quadrant += 2  # Bottom half
                    
                    # Count this position
                    quadrant_totals[quadrant] += 1
                    
                    # Check if there's a pellet
                    if pellet_grid[r][c]:
                        quadrant_counts[quadrant] += 1
            
            # Calculate normalized pellet density for each quadrant
            for i in range(4):
                if quadrant_totals[i] > 0:
                    quadrant_pellets[i] = quadrant_counts[i] / quadrant_totals[i]
        
        # --- GHOST SENSORS ---
        # Enhanced ghost detection with improved directional awareness and path detection
        ghost_sensors = [0, 0, 0, 0]  # right, left, down, up
        closest_ghost_dist = 1.0  # Normalized distance to nearest ghost
        ghost_direction_vec = [0, 0, 0, 0]  # Direction closest ghost is moving (one-hot)
        ghost_in_path = [0, 0, 0, 0]  # Indicates if a ghost is in pacman's path in each direction
        
        if ghosts:
            # Calculate distances to all ghosts
            ghost_distances = []
            for ghost in ghosts:
                ghost_x, ghost_y = pixel_to_grid(ghost.x, ghost.y)
                # Manhattan distance
                dist = abs(ghost_x - pacman_x) + abs(ghost_y - pacman_y)
                # Store as tuple (ghost, distance, position)
                ghost_distances.append((ghost, dist, (ghost_x, ghost_y)))
            
            # Sort by distance (closest first)
            ghost_distances.sort(key=lambda x: x[1])
            
            # Process closest ghost
            if ghost_distances:
                closest_ghost, distance, ghost_pos = ghost_distances[0]
                ghost_x, ghost_y = ghost_pos
                
                # Update closest ghost distance (normalize to 0-1)
                closest_ghost_dist = min(1.0, distance / 15.0)
                
                # Get ghost's direction of movement
                if closest_ghost.direction.x > 0:
                    ghost_direction_vec[0] = 1  # Right
                elif closest_ghost.direction.x < 0:
                    ghost_direction_vec[1] = 1  # Left
                elif closest_ghost.direction.y > 0:
                    ghost_direction_vec[2] = 1  # Down
                elif closest_ghost.direction.y < 0:
                    ghost_direction_vec[3] = 1  # Up
                
                # Check if ghosts are in Pacman's path
                # For each direction, check if there's a ghost along that line
                for i, (dx, dy) in enumerate(directions):
                    # Check several steps in this direction
                    for steps in range(1, 8):
                        check_x = pacman_x + (dx * steps)
                        check_y = pacman_y + (dy * steps)
                        
                        # Stop at walls
                        if (check_x < 0 or check_x >= constants.COLS or 
                            check_y < 0 or check_y >= constants.ROWS or 
                            (maze and maze[check_y][check_x] == 1)):
                            break
                        
                        # Check if any ghost is at or near this position
                        for _, _, (gx, gy) in ghost_distances:
                            if abs(gx - check_x) <= 1 and abs(gy - check_y) <= 1:
                                ghost_in_path[i] = 1.0
                                break
                    
                # Update directional ghost sensors for nearby ghosts
                for ghost, dist, (gx, gy) in ghost_distances:
                    if dist > 10:  # Ignore ghosts that are too far
                        continue
                        
                    dx = gx - pacman_x
                    dy = gy - pacman_y
                    
                    # Determine dominant direction
                    if abs(dx) > abs(dy):
                        idx = 0 if dx > 0 else 1  # right or left
                    else:
                        idx = 2 if dy > 0 else 3  # down or up
                    
                    # Scale intensity by distance (closer = stronger signal)
                    intensity = max(0, 1.0 - (dist / 10.0))
                    ghost_sensors[idx] = max(ghost_sensors[idx], intensity)
        
        # --- MOVEMENT HISTORY ---
        # Information about recent moves to help detect patterns
        movement_history = [0, 0, 0, 0]  # How often moved in each direction recently
        
        if hasattr(self, 'action_history') and self.action_history:
            # Count occurrences of each action
            action_counts = [0, 0, 0, 0]
            for action in self.action_history[-8:]:  # Last 8 actions
                if action is not None and 0 <= action < 4:
                    action_counts[action] += 1
            
            # Normalize to 0-1 range
            total = sum(action_counts)
            if total > 0:
                movement_history = [count / total for count in action_counts]
        
        # --- STUCK DETECTION ---
        # Whether Pacman has been in the same position for several frames
        is_stuck = 0.0
        
        if hasattr(self, 'last_positions') and len(self.last_positions) >= 3:
            current_pos = pixel_to_grid(self.x, self.y)
            if all(pos == current_pos for pos in self.last_positions[-3:]):
                is_stuck = 1.0
        
        # --- OSCILLATION DETECTION ---
        # Check for back-and-forth movement pattern
        is_oscillating = 0.0
        
        if hasattr(self, 'last_positions') and len(self.last_positions) >= 4:
            pos_history = self.last_positions[-4:]
            # Check if positions are alternating between two locations
            if pos_history[0] == pos_history[2] and pos_history[1] == pos_history[3]:
                is_oscillating = 1.0
        
        # --- ASSEMBLE OBSERVATION ---
        # Base observation components (18 features)
        base_observation = np.array([
            norm_x, norm_y,  # Position (2)
            *direction_vec,  # Current direction (4)
            *wall_sensors,   # Wall distances (4)
            *pellet_sensors, # Pellet detection (4)
            *ghost_sensors   # Ghost directions (4)
        ], dtype=np.float32)
        
        # Enhanced features (19 additional features)
        enhanced_features = np.array([
            *quadrant_pellets,     # Area evaluation (4)
            is_junction,           # Junction detection (1)
            closest_ghost_dist,    # Distance to nearest ghost (1)
            is_stuck,              # Stuck detection (1)
            is_oscillating,        # Oscillation detection (1)
            *valid_moves,          # Valid moves (4)
            *movement_history,     # Movement history (4)
            *ghost_direction_vec,  # Direction of closest ghost (4)
            *ghost_in_path         # Ghost in path detection (4)
        ], dtype=np.float32)
        
        # Combine everything for a total of 37 features
        full_observation = np.concatenate([base_observation, enhanced_features])
        
        return full_observation

    def draw(self, screen, score_font):
        # Determine the facing angle using the direction vector
        if self.direction.length_squared() > 0:
            angle = math.atan2(self.direction.y, self.direction.x)
        else:
            angle = 0  # Default facing right

        # Set the mouth opening angle
        mouth_angle_deg = 30  # Total opening angle
        half_mouth_rad = math.radians(mouth_angle_deg / 2)

        # Draw Pac-Man's body (a yellow circle)
        pygame.draw.circle(screen, (255, 255, 0), (int(self.x), int(self.y)), self.radius)

        # Calculate the points for the mouth wedge
        point1 = (self.x + self.radius * math.cos(angle + half_mouth_rad),
                 self.y + self.radius * math.sin(angle + half_mouth_rad))
        point2 = (self.x + self.radius * math.cos(angle - half_mouth_rad),
                 self.y + self.radius * math.sin(angle - half_mouth_rad))

        # Draw the mouth (a black triangle)
        pygame.draw.polygon(screen, (0, 0, 0), [(self.x, self.y), point1, point2])

        # Debug: Draw the path if in A_STAR mode
        if constants.GAME_MODE == "A_STAR" and self.path:
            for i in range(len(self.path)-1):
                start_x, start_y = grid_to_pixel(self.path[i][0], self.path[i][1])
                end_x, end_y = grid_to_pixel(self.path[i+1][0], self.path[i+1][1])
                pygame.draw.line(screen, (255, 0, 255), (start_x, start_y), (end_x, end_y), 2)
                
        # Debug: Show target pellet if exists in A_STAR mode
        if constants.GAME_MODE == "A_STAR" and self.target_pellet:
            target_x, target_y = grid_to_pixel(self.target_pellet[0], self.target_pellet[1])
            pygame.draw.circle(screen, (255, 255, 0), (target_x, target_y), 8, 2)
            
        # Debug: Show DQN action if DQN mode
        if constants.GAME_MODE == "DQN" and self.last_action is not None:
            action_names = ["→", "←", "↓", "↑"]
            text = score_font.render(action_names[self.last_action], True, (0, 255, 0))
            screen.blit(text, (self.x - 10, self.y - 30))

    def _get_nearby_ghosts(self, active_ghosts):
        """Get a list of nearby ghosts, sorted by distance"""
        if not active_ghosts:
            return []
            
        ghost_distances = []
        for ghost in active_ghosts:
            dist = math.hypot(ghost.x - self.x, ghost.y - self.y) / constants.TILE_SIZE
            ghost_distances.append((ghost, dist))
        
        # Sort by distance
        ghost_distances.sort(key=lambda x: x[1])
        
        # Return the ghosts only (not distances)
        return [g for g, _ in ghost_distances]