import pygame as pg
import numpy as np
import sys
import math
import random
import time
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.arrays import vbo
from PIL import Image

class Vector3:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z
    
    def __add__(self, v):
        if isinstance(v, Vector3):
            return Vector3(self.x + v.x, self.y + v.y, self.z + v.z)
        return Vector3(self.x + v, self.y + v, self.z + v)
    
    def __sub__(self, v):
        if isinstance(v, Vector3):
            return Vector3(self.x - v.x, self.y - v.y, self.z - v.z)
        return Vector3(self.x - v, self.y - v, self.z - v)
    
    def __mul__(self, v):
        if isinstance(v, Vector3):
            return Vector3(self.x * v.x, self.y * v.y, self.z * v.z)
        return Vector3(self.x * v, self.y * v, self.z * v)
    
    def __truediv__(self, v):
        if isinstance(v, Vector3):
            return Vector3(self.x / v.x, self.y / v.y, self.z / v.z)
        return Vector3(self.x / v, self.y / v, self.z / v)
    
    def dot(self, v):
        return self.x * v.x + self.y * v.y + self.z * v.z
    
    def cross(self, v):
        return Vector3(
            self.y * v.z - self.z * v.y,
            self.z * v.x - self.x * v.z,
            self.x * v.y - self.y * v.x
        )
    
    def magnitude(self):
        return math.sqrt(self.dot(self))
    
    def normalize(self):
        mag = self.magnitude()
        if mag == 0:
            return Vector3(0, 0, 0)
        return Vector3(self.x / mag, self.y / mag, self.z / mag)
    
    def as_tuple(self):
        return (self.x, self.y, self.z)
    
    def __str__(self):
        return f"Vector3({self.x}, {self.y}, {self.z})"


class Camera:
    def __init__(self, position=Vector3(0, 0, 0), target=Vector3(0, 0, -1), up=Vector3(0, 1, 0)):
        self.position = position
        self.target = target
        self.up = up
        self.yaw = -90
        self.pitch = 0
        self.speed = 0.1
        self.sensitivity = 0.1
        self.first_mouse = True
        self.last_x = 0
        self.last_y = 0
        self.fov = 70
        self.update_vectors()

    def update_vectors(self):
        front_x = math.cos(math.radians(self.yaw)) * math.cos(math.radians(self.pitch))
        front_y = math.sin(math.radians(self.pitch))
        front_z = math.sin(math.radians(self.yaw)) * math.cos(math.radians(self.pitch))
        self.front = Vector3(front_x, front_y, front_z).normalize()
        self.right = self.front.cross(Vector3(0, 1, 0)).normalize()
        self.up = self.right.cross(self.front).normalize()

    def process_mouse(self, x_offset, y_offset):
        x_offset *= self.sensitivity
        y_offset *= self.sensitivity
        
        self.yaw += x_offset
        self.pitch -= y_offset
        
        if self.pitch > 89:
            self.pitch = 89
        if self.pitch < -89:
            self.pitch = -89
            
        self.update_vectors()
        
    def process_keyboard(self, direction, delta_time):
        velocity = self.speed * delta_time
        if direction == "FORWARD":
            self.position = self.position + self.front * velocity
        if direction == "BACKWARD":
            self.position = self.position - self.front * velocity
        if direction == "LEFT":
            self.position = self.position - self.right * velocity
        if direction == "RIGHT":
            self.position = self.position + self.right * velocity
        if direction == "UP":
            self.position = self.position + Vector3(0, 1, 0) * velocity
        if direction == "DOWN":
            self.position = self.position - Vector3(0, 1, 0) * velocity
    
    def get_view_matrix(self):
        return look_at(self.position, self.position + self.front, self.up)


def look_at(position, target, up):
    z_axis = (position - target).normalize()
    x_axis = up.cross(z_axis).normalize()
    y_axis = z_axis.cross(x_axis).normalize()
    
    translation = Vector3(
        -x_axis.dot(position),
        -y_axis.dot(position),
        -z_axis.dot(position)
    )
    
    return np.array([
        [x_axis.x, y_axis.x, z_axis.x, 0],
        [x_axis.y, y_axis.y, z_axis.y, 0],
        [x_axis.z, y_axis.z, z_axis.z, 0],
        [translation.x, translation.y, translation.z, 1]
    ], dtype=np.float32)


def load_texture(filename):
    image = Image.open(filename)
    image_data = image.convert("RGBA").tobytes()
    width, height = image.size
    
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data)
    glGenerateMipmap(GL_TEXTURE_2D)
    
    return texture_id


class Entity:
    def __init__(self, position=Vector3(0, 0, 0), rotation=Vector3(0, 0, 0), scale=Vector3(1, 1, 1)):
        self.position = position
        self.rotation = rotation
        self.scale = scale
        self.velocity = Vector3(0, 0, 0)
        self.acceleration = Vector3(0, 0, 0)
        self.mesh = None
        self.collision_radius = 1.0
        self.health = 100
        self.max_health = 100
        self.is_active = True
        self.entity_type = "generic"
        self.attack_power = 0
        self.defense = 0
        self.experience = 0
        self.level = 1
    
    def update(self, delta_time, world):
        if not self.is_active:
            return
            
        self.velocity = self.velocity + self.acceleration * delta_time
        self.position = self.position + self.velocity * delta_time
        self.acceleration = self.acceleration * 0
    
    def render(self):
        if not self.is_active or not self.mesh:
            return
            
        glPushMatrix()
        glTranslatef(self.position.x, self.position.y, self.position.z)
        glRotatef(self.rotation.x, 1, 0, 0)
        glRotatef(self.rotation.y, 0, 1, 0)
        glRotatef(self.rotation.z, 0, 0, 1)
        glScalef(self.scale.x, self.scale.y, self.scale.z)
        
        self.mesh.render()
        
        glPopMatrix()
    
    def apply_force(self, force):
        self.acceleration = self.acceleration + force
    
    def is_colliding_with(self, other):
        distance = (self.position - other.position).magnitude()
        return distance < (self.collision_radius + other.collision_radius)
    
    def take_damage(self, amount):
        actual_damage = max(0, amount - self.defense)
        self.health -= actual_damage
        if self.health <= 0:
            self.health = 0
            self.on_death()
        return actual_damage
    
    def heal(self, amount):
        self.health = min(self.max_health, self.health + amount)
    
    def on_death(self):
        self.is_active = False
    
    def attack(self, target):
        damage_dealt = target.take_damage(self.attack_power)
        return damage_dealt


class Player(Entity):
    def __init__(self, position=Vector3(0, 2, 0)):
        super().__init__(position)
        self.entity_type = "player"
        self.camera = Camera(position)
        self.speed = 5.0
        self.jump_force = 10.0
        self.is_jumping = False
        self.is_grounded = False
        self.stamina = 100
        self.max_stamina = 100
        self.stamina_regen = 5
        self.oxygen = 100
        self.max_oxygen = 100
        self.is_underwater = False
        self.inventory = Inventory(20)
        self.attack_power = 10
        self.defense = 2
        self.level = 1
        self.experience = 0
        self.experience_to_next_level = 100
        self.skill_points = 0
        self.skills = {
            "strength": 1,
            "agility": 1,
            "intelligence": 1,
            "vitality": 1
        }
        self.equipped_items = {
            "weapon": None,
            "armor": None,
            "helmet": None,
            "boots": None,
            "amulet": None
        }
        self.collision_radius = 0.5
        self.height = 1.8
        self.is_sprinting = False
        self.footstep_timer = 0
        self.footstep_interval = 0.5
        self.last_damage_time = 0
        self.immunity_time = 1.0
    
    def update(self, delta_time, world):
        super().update(delta_time, world)
        
        # Update camera position to match player
        self.camera.position = self.position + Vector3(0, self.height * 0.8, 0)
        
        # Handle gravity
        if not self.is_grounded:
            self.apply_force(Vector3(0, -9.8, 0))
        
        # Regenerate stamina when not sprinting
        if not self.is_sprinting:
            self.stamina = min(self.max_stamina, self.stamina + self.stamina_regen * delta_time)
        
        # Handle oxygen underwater
        if self.is_underwater:
            self.oxygen -= 5 * delta_time
            if self.oxygen <= 0:
                self.oxygen = 0
                self.take_damage(5 * delta_time)
        else:
            self.oxygen = min(self.max_oxygen, self.oxygen + 10 * delta_time)
        
        # Footstep sounds
        if self.velocity.magnitude() > 0.1 and self.is_grounded:
            self.footstep_timer += delta_time
            if self.footstep_timer >= self.footstep_interval:
                self.footstep_timer = 0
                # Play footstep sound here
        
        # Level up check
        if self.experience >= self.experience_to_next_level:
            self.level_up()
    
    def move(self, direction, delta_time):
        move_speed = self.speed
        
        if self.is_sprinting:
            if self.stamina > 0:
                move_speed *= 1.5
                self.stamina -= 10 * delta_time
            else:
                self.is_sprinting = False
        
        if direction == "FORWARD":
            forward = Vector3(self.camera.front.x, 0, self.camera.front.z).normalize()
            self.position = self.position + forward * move_speed * delta_time
        elif direction == "BACKWARD":
            forward = Vector3(self.camera.front.x, 0, self.camera.front.z).normalize()
            self.position = self.position - forward * move_speed * delta_time
        elif direction == "LEFT":
            self.position = self.position - self.camera.right * move_speed * delta_time
        elif direction == "RIGHT":
            self.position = self.position + self.camera.right * move_speed * delta_time
        elif direction == "JUMP" and self.is_grounded:
            self.velocity.y = self.jump_force
            self.is_jumping = True
            self.is_grounded = False
    
    def sprint(self, is_sprinting):
        self.is_sprinting = is_sprinting and self.stamina > 0
    
    def look(self, x_offset, y_offset):
        self.camera.process_mouse(x_offset, y_offset)
    
    def level_up(self):
        self.level += 1
        self.experience -= self.experience_to_next_level
        self.experience_to_next_level = int(self.experience_to_next_level * 1.5)
        self.skill_points += 3
        self.max_health += 10
        self.health = self.max_health
        self.max_stamina += 5
        self.stamina = self.max_stamina
        # Play level up sound/effect
    
    def add_experience(self, amount):
        self.experience += amount
    
    def upgrade_skill(self, skill_name, points=1):
        if skill_name in self.skills and self.skill_points >= points:
            self.skills[skill_name] += points
            self.skill_points -= points
            
            # Apply skill effects
            if skill_name == "strength":
                self.attack_power += 2 * points
            elif skill_name == "agility":
                self.speed += 0.2 * points
            elif skill_name == "intelligence":
                pass  # Affects magic if implemented
            elif skill_name == "vitality":
                max_health_increase = 5 * points
                self.max_health += max_health_increase
                self.health += max_health_increase
    
    def equip_item(self, item):
        if not item:
            return False
            
        slot = item.item_type
        if slot in self.equipped_items:
            old_item = self.equipped_items[slot]
            if old_item:
                # Remove old item stats
                self.attack_power -= old_item.attack_bonus
                self.defense -= old_item.defense_bonus
                self.inventory.add_item(old_item)
            
            # Apply new item stats
            self.attack_power += item.attack_bonus
            self.defense += item.defense_bonus
            self.equipped_items[slot] = item
            return True
        
        return False
    
    def unequip_item(self, slot):
        if slot in self.equipped_items and self.equipped_items[slot]:
            item = self.equipped_items[slot]
            if self.inventory.add_item(item):
                # Remove item stats
                self.attack_power -= item.attack_bonus
                self.defense -= item.defense_bonus
                self.equipped_items[slot] = None
                return True
        
        return False
    
    def interact(self, world):
        # Check for interaction with objects in front of the player
        interaction_distance = 2.0
        interaction_point = self.position + self.camera.front * interaction_distance
        
        for entity in world.entities:
            if entity == self or not entity.is_active:
                continue
                
            if (entity.position - interaction_point).magnitude() < entity.collision_radius:
                if hasattr(entity, "on_interact"):
                    entity.on_interact(self)
                    return True
        
        # Check for terrain interaction
        # This would involve terrain manipulation, mining, etc.
        
        return False
    
    def attack(self, world):
        attack_distance = 2.0
        attack_point = self.position + self.camera.front * attack_distance
        
        damage_dealt = 0
        
        for entity in world.entities:
            if entity == self or not entity.is_active:
                continue
                
            if (entity.position - attack_point).magnitude() < entity.collision_radius + 0.5:
                damage_dealt = super().attack(entity)
                # Play attack sound/effect
                if entity.health <= 0 and entity.entity_type == "enemy":
                    self.add_experience(entity.experience)
                
                break
        
        return damage_dealt
    
    def take_damage(self, amount):
        current_time = time.time()
        if current_time - self.last_damage_time < self.immunity_time:
            return 0
            
        damage = super().take_damage(amount)
        if damage > 0:
            self.last_damage_time = current_time
            # Play damage sound/effect
        
        return damage


class Enemy(Entity):
    def __init__(self, position, enemy_type="basic"):
        super().__init__(position)
        self.entity_type = "enemy"
        self.enemy_type = enemy_type
        self.target = None
        self.detection_radius = 10.0
        self.attack_radius = 1.5
        self.speed = 2.0
        self.attack_cooldown = 1.0
        self.last_attack_time = 0
        self.wandering = True
        self.wander_target = None
        self.wander_radius = 5.0
        self.wander_change_time = 5.0
        self.last_wander_change = time.time()
        
        # Set enemy type specific stats
        if enemy_type == "basic":
            self.health = 20
            self.max_health = 20
            self.attack_power = 5
            self.defense = 1
            self.experience = 10
            self.collision_radius = 0.6
        elif enemy_type == "strong":
            self.health = 50
            self.max_health = 50
            self.attack_power = 10
            self.defense = 3
            self.experience = 25
            self.collision_radius = 0.8
        elif enemy_type == "boss":
            self.health = 200
            self.max_health = 200
            self.attack_power = 20
            self.defense = 8
            self.experience = 100
            self.detection_radius = 20.0
            self.speed = 1.5
            self.collision_radius = 1.2
    
    def update(self, delta_time, world):
        super().update(delta_time, world)
        
        if not self.is_active:
            return
        
        # Find player
        player = world.get_player()
        if not player:
            return
            
        distance_to_player = (player.position - self.position).magnitude()
        
        # Check if player is within detection radius
        if distance_to_player < self.detection_radius:
            self.target = player
            self.wandering = False
        else:
            self.target = None
            self.wandering = True
        
        # Attack target if close enough
        if self.target and distance_to_player < self.attack_radius:
            current_time = time.time()
            if current_time - self.last_attack_time >= self.attack_cooldown:
                self.attack(self.target)
                self.last_attack_time = current_time
        
        # Move towards target or wander
        if self.target:
            direction = (self.target.position - self.position).normalize()
            self.velocity = direction * self.speed
        elif self.wandering:
            current_time = time.time()
            
            # Change wander target periodically
            if current_time - self.last_wander_change >= self.wander_change_time or not self.wander_target:
                angle = random.uniform(0, math.pi * 2)
                distance = random.uniform(3, self.wander_radius)
                self.wander_target = self.position + Vector3(
                    math.cos(angle) * distance,
                    0,
                    math.sin(angle) * distance
                )
                self.last_wander_change = current_time
            
            # Move towards wander target
            if self.wander_target:
                direction = (self.wander_target - self.position)
                if direction.magnitude() > 0.5:
                    self.velocity = direction.normalize() * (self.speed * 0.5)
                else:
                    self.velocity = Vector3(0, 0, 0)
    
    def on_death(self):
        super().on_death()
        # Drop loot
        # Play death animation


class Item:
    def __init__(self, name, item_type="misc", description=""):
        self.name = name
        self.item_type = item_type  # weapon, armor, helmet, boots, amulet, consumable, misc
        self.description = description
        self.quantity = 1
        self.stackable = item_type == "consumable" or item_type == "misc"
        self.max_stack = 99 if self.stackable else 1
        self.icon_texture = None
        self.model = None
        self.attack_bonus = 0
        self.defense_bonus = 0
        self.effects = []
        self.value = 1
        self.rarity = "common"  # common, uncommon, rare, epic, legendary
    
    def use(self, player):
        if self.item_type == "consumable":
            for effect in self.effects:
                effect.apply(player)
            return True
        elif self.item_type in ["weapon", "armor", "helmet", "boots", "amulet"]:
            return player.equip_item(self)
        return False
    
    def get_display_name(self):
        rarity_colors = {
            "common": (255, 255, 255),     # White
            "uncommon": (0, 255, 0),       # Green
            "rare": (0, 112, 221),         # Blue
            "epic": (163, 53, 238),        # Purple
            "legendary": (255, 128, 0)     # Orange
        }
        color = rarity_colors.get(self.rarity, (255, 255, 255))
        return f"{{color:{color[0]},{color[1]},{color[2]}}}{{self.name}}"


class ItemEffect:
    def __init__(self, effect_type, value, duration=0):
        self.effect_type = effect_type  # health, stamina, speed, etc.
        self.value = value
        self.duration = duration  # 0 for instant effects
    
    def apply(self, player):
        if self.effect_type == "health":
            player.heal(self.value)
        elif self.effect_type == "stamina":
            player.stamina = min(player.max_stamina, player.stamina + self.value)
        elif self.effect_type == "oxygen":
            player.oxygen = min(player.max_oxygen, player.oxygen + self.value)
        # Add other effect types as needed


class Inventory:
    def __init__(self, size):
        self.size = size
        self.items = [None] * size
        self.selected_slot = 0
    
    def add_item(self, item):
        if not item:
            return False
            
        # Check if item can be stacked with existing items
        if item.stackable:
            for i, slot_item in enumerate(self.items):
                if slot_item and slot_item.name == item.name and slot_item.quantity < slot_item.max_stack:
                    available_space = slot_item.max_stack - slot_item.quantity
                    added_amount = min(available_space, item.quantity)
                    self.items[i].quantity += added_amount
                    item.quantity -= added_amount
                    
                    if item.quantity <= 0:
                        return True
        
        # Find empty slot for item
        for i in range(self.size):
            if self.items[i] is None:
                self.items[i] = item
                return True
        
        return False
    
    def remove_item(self, index, quantity=1):
        if 0 <= index < self.size and self.items[index]:
            if self.items[index].quantity <= quantity:
                removed_item = self.items[index]
                self.items[index] = None
                return removed_item
            else:
                self.items[index].quantity -= quantity
                removed_item = Item(self.items[index].name, self.items[index].item_type)
                removed_item.quantity = quantity
                return removed_item
        
        return None
    
    def get_selected_item(self):
        if 0 <= self.selected_slot < self.size:
            return self.items[self.selected_slot]
        return None
    
    def select_slot(self, slot):
        if 0 <= slot < self.size:
            self.selected_slot = slot
    
    def use_selected_item(self, player):
        item = self.get_selected_item()
        if item:
            if item.use(player):
                if item.item_type == "consumable":
                    self.remove_item(self.selected_slot, 1)
                elif item.item_type in ["weapon", "armor", "helmet", "boots", "amulet"]:
                    self.remove_item(self.selected_slot)
                return True
        
        return False


class Mesh:
    def __init__(self, vertices, indices, uvs=None, normals=None, material=None):
        self.vertices = np.array(vertices, dtype=np.float32)
        self.indices = np.array(indices, dtype=np.uint32)
        
        if uvs is None:
            self.uvs = np.zeros((len(vertices), 2), dtype=np.float32)
        else:
            self.uvs = np.array(uvs, dtype=np.float32)
            
        if normals is None:
            self.calculate_normals()
        else:
            self.normals = np.array(normals, dtype=np.float32)
            
        self.material = material
        self.setup_vao()
    
    def calculate_normals(self):
        self.normals = np.zeros((len(self.vertices), 3), dtype=np.float32)
        
        for i in range(0, len(self.indices), 3):
            v1 = self.vertices[self.indices[i]]
            v2 = self.vertices[self.indices[i+1]]
            v3 = self.vertices[self.indices[i+2]]
            
            edge1 = v2 - v1
            edge2 = v3 - v1
            
            normal = np.cross(edge1, edge2)
            normal = normal / np.linalg.norm(normal)
            
            self.normals[self.indices[i]] += normal
            self.normals[self.indices[i+1]] += normal
            self.normals[self.indices[i+2]] += normal
        
        # Normalize all normals
        for i in range(len(self.normals)):
            norm = np.linalg.norm(self.normals[i])
            if norm > 0:
                self.normals[i] = self.normals[i] / norm
    
    def setup_vao(self):
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        
        # Position VBO
        self.position_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.position_vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        
        # Normal VBO
        self.normal_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.normal_vbo)
        glBufferData(GL_ARRAY_BUFFER, self.normals.nbytes, self.normals, GL_STATIC_DRAW)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
        
        # UV VBO
        self.uv_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.uv_vbo)
        glBufferData(GL_ARRAY_BUFFER, self.uvs.nbytes, self.uvs, GL_STATIC_DRAW)
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, None)
        
        # Index EBO
        self.ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)
        
        glBindVertexArray(0)
    
    def render(self):
        if self.material:
            self.material.use()
            
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
        
        if self.material:
            self.material.unuse()


class Material:
    def __init__(self, diffuse_texture=None, specular_texture=None, normal_texture=None, shininess=32.0):
        self.diffuse_texture = diffuse_texture
        self.specular_texture = specular_texture
        self.normal_texture = normal_texture
        self.shininess = shininess
        self.ambient_color = (0.2, 0.2, 0.2, 1.0)
        self.diffuse_color = (0.8, 0.8, 0.8, 1.0)
        self.specular_color = (1.0, 1.0, 1.0, 1.0)
    
    def use(self):
        # Set material properties
        glMaterialfv(GL_FRONT, GL_AMBIENT, self.ambient_color)
        glMaterialfv(GL_FRONT, GL_DIFFUSE, self.diffuse_color)
        glMaterialfv(GL_FRONT, GL_SPECULAR, self.specular_color)
        glMaterialf(GL_FRONT, GL_SHININESS, self.shininess)
        
        # Bind diffuse texture
        if self.diffuse_texture:
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.diffuse_texture)
            glEnable(GL_TEXTURE_2D)
        
        # Bind specular texture
        if self.specular_texture:
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, self.specular_texture)
            glEnable(GL_TEXTURE_2D)
        
        # Bind normal texture
        if self.normal_texture:
            glActiveTexture(GL_TEXTURE2)
            glBindTexture(GL_TEXTURE_2D, self.normal_texture)
            glEnable(GL_TEXTURE_2D)
    
    def unuse(self):
        glDisable(GL_TEXTURE_2D)
        glActiveTexture(GL_TEXTURE0)


class Shader:
    def __init__(self, vertex_code, fragment_code):
        self.program = glCreateProgram()
        
        # Compile vertex shader
        vertex = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vertex, vertex_code)
        glCompileShader(vertex)
        self.check_compile_errors(vertex, "VERTEX")
        
        # Compile fragment shader
        fragment = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(fragment, fragment_code)
        glCompileShader(fragment)
        self.check_compile_errors(fragment, "FRAGMENT")
        
        # Link program
        glAttachShader(self.program, vertex)
        glAttachShader(self.program, fragment)
        glLinkProgram(self.program)
        self.check_compile_errors(self.program, "PROGRAM")
        
        # Delete shaders as they're linked into the program and no longer needed
        glDeleteShader(vertex)
        glDeleteShader(fragment)
        
        # Get uniform locations
        self.uniform_locations = {}
    
    def check_compile_errors(self, shader, shader_type):
        if shader_type != "PROGRAM":
            if not glGetShaderiv(shader, GL_COMPILE_STATUS):
                info_log = glGetShaderInfoLog(shader)
                print(f"ERROR::SHADER_COMPILATION_ERROR of type: {shader_type}\n{info_log}")
        else:
            if not glGetProgramiv(shader, GL_LINK_STATUS):
                info_log = glGetProgramInfoLog(shader)
                print(f"ERROR::PROGRAM_LINKING_ERROR of type: {shader_type}\n{info_log}")
    
    def use(self):
        glUseProgram(self.program)
    
    def get_uniform_location(self, name):
        if name not in self.uniform_locations:
            self.uniform_locations[name] = glGetUniformLocation(self.program, name)
        return self.uniform_locations[name]
    
    def set_bool(self, name, value):
        glUniform1i(self.get_uniform_location(name), int(value))
    
    def set_int(self, name, value):
        glUniform1i(self.get_uniform_location(name), value)
    
    def set_float(self, name, value):
        glUniform1f(self.get_uniform_location(name), value)
    
    def set_vec2(self, name, x, y=None):
        if y is None:  # x is a container
            glUniform2fv(self.get_uniform_location(name), 1, x)
        else:
            glUniform2f(self.get_uniform_location(name), x, y)
    
    def set_vec3(self, name, x, y=None, z=None):
        if y is None:  # x is a container
            glUniform3fv(self.get_uniform_location(name), 1, x)
        else:
            glUniform3f(self.get_uniform_location(name), x, y, z)
    
    def set_vec4(self, name, x, y=None, z=None, w=None):
        if y is None:  # x is a container
            glUniform4fv(self.get_uniform_location(name), 1, x)
        else:
            glUniform4f(self.get_uniform_location(name), x, y, z, w)
    
    def set_mat2(self, name, mat):
        glUniformMatrix2fv(self.get_uniform_location(name), 1, GL_FALSE, mat)
    
    def set_mat3(self, name, mat):
        glUniformMatrix3fv(self.get_uniform_location(name), 1, GL_FALSE, mat)
    
    def set_mat4(self, name, mat):
        glUniformMatrix4fv(self.get_uniform_location(name), 1, GL_FALSE, mat)


class Terrain:
    def __init__(self, width, depth, height_scale=10.0, resolution=128):
        self.width = width
        self.depth = depth
        self.height_scale = height_scale
        self.resolution = resolution
        self.heights = np.zeros((resolution, resolution), dtype=np.float32)
        self.generate_terrain()
        self.create_mesh()
        self.textures = {
            "grass": None,
            "rock": None,
            "sand": None,
            "snow": None
        }
    
    def load_textures(self):
        # These would be loaded from files in a real application
        self.textures["grass"] = 1  # Dummy texture ID
        self.textures["rock"] = 2   # Dummy texture ID
        self.textures["sand"] = 3   # Dummy texture ID
        self.textures["snow"] = 4   # Dummy texture ID
    
    def generate_terrain(self):
        # Simple perlin noise terrain generation
        scale = 0.1
        octaves = 6
        persistence = 0.5
        lacunarity = 2.0
        
        for i in range(self.resolution):
            for j in range(self.resolution):
                x = i / self.resolution * scale
                y = j / self.resolution * scale
                amplitude = 1.0
                frequency = 1.0
                height = 0.0
                
                for _ in range(octaves):
                    noise_val = perlin(x * frequency, y * frequency) * 2 - 1
                    height += noise_val * amplitude
                    amplitude *= persistence
                    frequency *= lacunarity
                
                # Normalize height to [0, 1]
                height = (height + 1) / 2
                self.heights[i, j] = height * self.height_scale
    
    def get_height(self, world_x, world_z):
        # Convert world coordinates to terrain coordinates
        terrain_x = (world_x + self.width / 2) / self.width * (self.resolution - 1)
        terrain_z = (world_z + self.depth / 2) / self.depth * (self.resolution - 1)
        
        # Clamp coordinates to valid range
        terrain_x = max(0, min(terrain_x, self.resolution - 2))
        terrain_z = max(0, min(terrain_z, self.resolution - 2))
        
        # Get grid coordinates
        grid_x = int(terrain_x)
        grid_z = int(terrain_z)
        
        # Get fractional part for interpolation
        x_frac = terrain_x - grid_x
        z_frac = terrain_z - grid_z
        
        # Bilinear interpolation
        h1 = self.heights[grid_x, grid_z]
        h2 = self.heights[grid_x + 1, grid_z]
        h3 = self.heights[grid_x, grid_z + 1]
        h4 = self.heights[grid_x + 1, grid_z + 1]
        
        height = (1 - x_frac) * (1 - z_frac) * h1 + \
                 x_frac * (1 - z_frac) * h2 + \
                 (1 - x_frac) * z_frac * h3 + \
                 x_frac * z_frac * h4
        
        return height
    
    def get_terrain_type(self, height):
        if height > 0.8 * self.height_scale:
            return "snow"
        elif height > 0.4 * self.height_scale:
            return "rock"
        elif height > 0.2 * self.height_scale:
            return "grass"
        else:
            return "sand"
    
    def create_mesh(self):
        vertices = []
        indices = []
        uvs = []
        normals = []
        
        # Generate vertices and UVs
        for z in range(self.resolution):
            for x in range(self.resolution):
                # Calculate vertex position
                pos_x = x / (self.resolution - 1) * self.width - self.width / 2
                pos_z = z / (self.resolution - 1) * self.depth - self.depth / 2
                pos_y = self.heights[x, z]
                
                vertices.append([pos_x, pos_y, pos_z])
                
                # Calculate UVs (for texturing)
                u = x / (self.resolution - 1)
                v = z / (self.resolution - 1)
                uvs.append([u, v])
                
                # Calculate normals
                if x > 0 and z > 0 and x < self.resolution - 1 and z < self.resolution - 1:
                    height_l = self.heights[x - 1, z]
                    height_r = self.heights[x + 1, z]
                    height_d = self.heights[x, z - 1]
                    height_u = self.heights[x, z + 1]
                    
                    # Calculate tangent vectors
                    tangent_x = Vector3(2.0, height_r - height_l, 0.0).normalize()
                    tangent_z = Vector3(0.0, height_u - height_d, 2.0).normalize()
                    
                    # Cross product to get normal
                    normal = tangent_x.cross(tangent_z).normalize()
                    normals.append([normal.x, normal.y, normal.z])
                else:
                    normals.append([0.0, 1.0, 0.0])
        
        # Generate indices (triangles)
        for z in range(self.resolution - 1):
            for x in range(self.resolution - 1):
                top_left = z * self.resolution + x
                top_right = top_left + 1
                bottom_left = (z + 1) * self.resolution + x
                bottom_right = bottom_left + 1
                
                # First triangle (top-left, bottom-left, bottom-right)
                indices.extend([top_left, bottom_left, bottom_right])
                
                # Second triangle (top-left, bottom-right, top-right)
                indices.extend([top_left, bottom_right, top_right])
        
        # Create the mesh
        material = Material()
        material.diffuse_texture = self.textures["grass"]  # Default texture
        self.mesh = Mesh(vertices, indices, uvs, normals, material)
    
    def render(self):
        self.mesh.render()
    
    def is_point_above_terrain(self, position):
        if abs(position.x) > self.width / 2 or abs(position.z) > self.depth / 2:
            return True  # Point is outside terrain bounds
        
        terrain_height = self.get_height(position.x, position.z)
        return position.y > terrain_height


def perlin(x, y):
    # Simple placeholder for Perlin noise function
    # In a real application, you would use a library like noise or implement a proper Perlin noise function
    X = int(x) & 255
    Y = int(y) & 255
    x -= int(x)
    y -= int(y)
    fade_x = fade(x)
    fade_y = fade(y)
    
    A = p[X] + Y
    B = p[X + 1] + Y
    
    return lerp(
        fade_y,
        lerp(fade_x, grad(p[A], x, y), grad(p[B], x - 1, y)),
        lerp(fade_x, grad(p[A + 1], x, y - 1), grad(p[B + 1], x - 1, y - 1))
    )


def fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def lerp(t, a, b):
    return a + t * (b - a)


def grad(hash, x, y):
    h = hash & 15
    u = x if h < 8 else y
    v = y if h < 4 else (x if h == 12 or h == 14 else 0)
    return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)


# Permutation table for Perlin noise
p = [151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225,
     140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148,
     247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32,
     57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175,
     74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122,
     60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54,
     65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169,
     200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64,
     52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212,
     207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213,
     119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9,
     129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104,
     218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241,
     81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157,
     184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93,
     222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180]
p.extend(p)  # Duplicate the array for easier computation


class Water:
    def __init__(self, width, depth, height=0.0, resolution=32):
        self.width = width
        self.depth = depth
        self.height = height
        self.resolution = resolution
        self.create_mesh()
        self.time = 0.0
        self.wave_speed = 0.05
        self.wave_height = 0.2
    
    def create_mesh(self):
        vertices = []
        indices = []
        uvs = []
        normals = []
        
        # Generate a flat grid for water surface
        for z in range(self.resolution):
            for x in range(self.resolution):
                # Calculate vertex position
                pos_x = x / (self.resolution - 1) * self.width - self.width / 2
                pos_z = z / (self.resolution - 1) * self.depth - self.depth / 2
                pos_y = self.height
                
                vertices.append([pos_x, pos_y, pos_z])
                
                # Calculate UVs
                u = x / (self.resolution - 1) * 10  # Repeat texture multiple times
                v = z / (self.resolution - 1) * 10
                uvs.append([u, v])
                
                # Normal points up for flat water
                normals.append([0.0, 1.0, 0.0])
        
        # Generate indices (triangles)
        for z in range(self.resolution - 1):
            for x in range(self.resolution - 1):
                top_left = z * self.resolution + x
                top_right = top_left + 1
                bottom_left = (z + 1) * self.resolution + x
                bottom_right = bottom_left + 1
                
                # First triangle (top-left, bottom-left, bottom-right)
                indices.extend([top_left, bottom_left, bottom_right])
                
                # Second triangle (top-left, bottom-right, top-right)
                indices.extend([top_left, bottom_right, top_right])
        
        # Create the mesh with a semi-transparent blue material
        material = Material()
        material.diffuse_color = (0.0, 0.4, 0.8, 0.7)  # Semi-transparent blue
        self.mesh = Mesh(vertices, indices, uvs, normals, material)
    
    def update(self, delta_time):
        self.time += delta_time
        
        # Update vertices for wave animation
        vertices = []
        normals = []
        
        for z in range(self.resolution):
            for x in range(self.resolution):
                pos_x = x / (self.resolution - 1) * self.width - self.width / 2
                pos_z = z / (self.resolution - 1) * self.depth - self.depth / 2
                
                # Simple sine wave animation
                wave1 = math.sin(pos_x * 0.5 + self.time * self.wave_speed) * self.wave_height
                wave2 = math.sin(pos_z * 0.3 + self.time * self.wave_speed * 0.7) * self.wave_height
                pos_y = self.height + wave1 + wave2
                
                vertices.append([pos_x, pos_y, pos_z])
                
                # Calculate normal based on wave gradients
                dx = math.cos(pos_x * 0.5 + self.time * self.wave_speed) * self.wave_height * 0.5
                dz = math.cos(pos_z * 0.3 + self.time * self.wave_speed * 0.7) * self.wave_height * 0.3
                normal = Vector3(-dx, 1.0, -dz).normalize()
                normals.append([normal.x, normal.y, normal.z])
        
        # Update mesh vertices and normals
        self.mesh.vertices = np.array(vertices, dtype=np.float32)
        self.mesh.normals = np.array(normals, dtype=np.float32)
        
        # Update VBOs
        glBindBuffer(GL_ARRAY_BUFFER, self.mesh.position_vbo)
        glBufferData(GL_ARRAY_BUFFER, self.mesh.vertices.nbytes, self.mesh.vertices, GL_STATIC_DRAW)
        
        glBindBuffer(GL_ARRAY_BUFFER, self.mesh.normal_vbo)
        glBufferData(GL_ARRAY_BUFFER, self.mesh.normals.nbytes, self.mesh.normals, GL_STATIC_DRAW)
    
    def render(self):
        # Enable blending for transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        self.mesh.render()
        
        glDisable(GL_BLEND)


class Skybox:
    def __init__(self, size=1000.0):
        self.size = size
        self.texture_ids = [0] * 6  # Cube map textures
        self.create_mesh()
    
    def load_textures(self, texture_paths):
        # Load the six faces of the skybox
        for i in range(6):
            self.texture_ids[i] = load_texture(texture_paths[i])
    
    def create_mesh(self):
        # Create a cube centered at (0,0,0)
        s = self.size / 2
        vertices = [
            # Front face
            [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s],
            # Back face
            [-s, -s, -s], [-s, s, -s], [s, s, -s], [s, -s, -s],
            # Left face
            [-s, -s, -s], [-s, -s, s], [-s, s, s], [-s, s, -s],
            # Right face
            [s, -s, s], [s, -s, -s], [s, s, -s], [s, s, s],
            # Top face
            [-s, s, s], [s, s, s], [s, s, -s], [-s, s, -s],
            # Bottom face
            [-s, -s, s], [-s, -s, -s], [s, -s, -s], [s, -s, s]
        ]
        
        # Each face is a quad with 4 vertices, with UVs covering the whole texture
        uvs = [[0, 0], [1, 0], [1, 1], [0, 1]] * 6
        
        # Normals point inward for a skybox
        normals = [
            [0, 0, -1], [0, 0, -1], [0, 0, -1], [0, 0, -1],  # Front
            [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],      # Back
            [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],      # Left
            [-1, 0, 0], [-1, 0, 0], [-1, 0, 0], [-1, 0, 0],  # Right
            [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0],  # Top
            [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]       # Bottom
        ]
        
        # Create indices for each face (2 triangles per face)
        indices = []
        for i in range(0, 24, 4):
            indices.extend([i, i+1, i+2, i, i+2, i+3])
        
        # Create a mesh for the skybox
        self.mesh = Mesh(vertices, indices, uvs, normals)
    
    def render(self, camera):
        # Save current OpenGL state
        glPushAttrib(GL_ENABLE_BIT)
        glDepthMask(GL_FALSE)  # Disable depth writing
        
        # Move skybox with camera
        glPushMatrix()
        glTranslatef(camera.position.x, camera.position.y, camera.position.z)
        
        # Render each face with its texture
        for i in range(6):
            glBindTexture(GL_TEXTURE_2D, self.texture_ids[i])
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, ctypes.c_void_p(i * 6 * 4))
        
        glPopMatrix()
        
        # Restore OpenGL state
        glDepthMask(GL_TRUE)
        glPopAttrib()


class ParticleSystem:
    def __init__(self, position=Vector3(0, 0, 0), max_particles=1000):
        self.position = position
        self.max_particles = max_particles
        self.particles = []
        self.texture = None
        self.active = True
        self.emit_rate = 50  # Particles per second
        self.emit_timer = 0.0
        self.particle_life = 2.0
        self.particle_size = 0.1
        self.particle_color = (1.0, 1.0, 1.0, 1.0)
        self.velocity = Vector3(0, 1, 0)
        self.velocity_variation = Vector3(0.5, 0.5, 0.5)
        self.gravity = Vector3(0, -0.5, 0)
    
    def update(self, delta_time):
        if not self.active:
            return
        
        # Remove dead particles
        self.particles = [p for p in self.particles if p["life"] > 0]
        
        # Update existing particles
        for particle in self.particles:
            particle["life"] -= delta_time
            particle["position"] = particle["position"] + particle["velocity"] * delta_time
            particle["velocity"] = particle["velocity"] + self.gravity * delta_time
            
            # Update color based on life (fade out)
            life_ratio = particle["life"] / self.particle_life
            particle["color"] = (
                self.particle_color[0],
                self.particle_color[1],
                self.particle_color[2],
                self.particle_color[3] * life_ratio
            )
        
        # Emit new particles
        self.emit_timer += delta_time
        particles_to_emit = int(self.emit_rate * self.emit_timer)
        
        if particles_to_emit > 0:
            self.emit_timer -= particles_to_emit / self.emit_rate
            
            for _ in range(min(particles_to_emit, self.max_particles - len(self.particles))):
                # Randomize velocity
                vel_x = self.velocity.x + random.uniform(-1, 1) * self.velocity_variation.x
                vel_y = self.velocity.y + random.uniform(-1, 1) * self.velocity_variation.y
                vel_z = self.velocity.z + random.uniform(-1, 1) * self.velocity_variation.z
                
                self.particles.append({
                    "position": Vector3(self.position.x, self.position.y, self.position.z),
                    "velocity": Vector3(vel_x, vel_y, vel_z),
                    "color": self.particle_color,
                    "life": self.particle_life,
                    "size": self.particle_size * random.uniform(0.8, 1.2)
                })
    
    def render(self, camera):
        if not self.active or not self.particles:
            return
        
        glPushAttrib(GL_ENABLE_BIT)
        
        # Enable blending for transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Disable depth writing but keep depth testing
        glDepthMask(GL_FALSE)
        
        # Enable point sprites
        glEnable(GL_POINT_SPRITE)
        glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE)
        glPointSize(50.0)  # Base point size
        
        # Bind texture if available
        if self.texture:
            glBindTexture(GL_TEXTURE_2D, self.texture)
        
        # Use billboard technique to make particles face camera
        right = camera.right
        up = camera.up
        
        # Render each particle
        glBegin(GL_POINTS)
        for particle in self.particles:
            pos = particle["position"]
            size = particle["size"]
            color = particle["color"]
            
            # Set particle color
            glColor4f(color[0], color[1], color[2], color[3])
            
            # Calculate billboard vertices
            center = Vector3(pos.x, pos.y, pos.z)
            
            # Apply point size based on distance
            distance = (center - camera.position).magnitude()
            adjusted_size = size * 100.0 / distance
            glPointSize(adjusted_size)
            
            # Render point
            glVertex3f(center.x, center.y, center.z)
        
        glEnd()
        
        # Reset state
        glDepthMask(GL_TRUE)
        glDisable(GL_POINT_SPRITE)
        glPopAttrib()


class World:
    def __init__(self, width=1024, depth=1024):
        self.width = width
        self