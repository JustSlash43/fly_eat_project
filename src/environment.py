"""
A minimal 2D Pygame world: a fly agent that walks/turns based on motor
commands and tries to reach randomly placed food particles.
"""
from __future__ import annotations

import math
import os
import random

import numpy as np
import pygame

WIDTH, HEIGHT = 900, 650
FLY_RADIUS = 8
FOOD_RADIUS = 6
FOOD_COUNT = 6
MAX_TURN_RATE = 0.12       # radians per frame at turn = 1
MAX_SPEED = 3.5            # pixels per frame at speed = 1
EAT_DISTANCE = FLY_RADIUS + FOOD_RADIUS

POV_FOV_DEG = 330.0              # panoramic field of view; the remaining 30 deg behind is a blind spot
FOOD_VISUAL_RADIUS = 14.0        # px, apparent size used for angular width
MIN_FOOD_HALF_WIDTH_DEG = 2.0    # keep far food at least a few photoreceptors wide
BRIGHTNESS_FALLOFF = 150.0       # px; brightness = 1 / (1 + d / falloff)
BACKGROUND_RGB = (0, 0, 0)          # pure dark: only food excites the photoreceptors
POV_STRIP_W, POV_STRIP_H = 330, 40
FOOD_RGB = (60, 230, 90)


def advance(x: float, y: float, heading: float, turn: float, speed: float) -> tuple[float, float, float]:
    """One frame of fly kinematics, bouncing off the arena walls. Returns (x, y, heading)."""
    heading += turn * MAX_TURN_RATE
    x += math.cos(heading) * speed * MAX_SPEED
    y += math.sin(heading) * speed * MAX_SPEED
    # Bounce off the walls. Without this the fly can end up walking
    # head-on into a wall with food balanced on both sides of its view
    # (equal left/right drive -> no turn) and stay pinned there forever.
    if not FLY_RADIUS <= x <= WIDTH - FLY_RADIUS:
        heading = math.pi - heading
    if not FLY_RADIUS <= y <= HEIGHT - FLY_RADIUS:
        heading = -heading
    x = min(max(x, FLY_RADIUS), WIDTH - FLY_RADIUS)
    y = min(max(y, FLY_RADIUS), HEIGHT - FLY_RADIUS)
    return x, y, heading


def render_panorama(
    x: float, y: float, heading: float, objects: list[tuple[float, float, tuple[int, int, int]]], size: int = 128
) -> np.ndarray:
    """
    The panoramic view from (x, y, heading) of objects given as (ox, oy, rgb),
    as an (size, size, 3) uint8 array: see FlyEnvironment.render_pov_frame.
    """
    frame = np.empty((size, size, 3), dtype=np.uint8)
    frame[:] = BACKGROUND_RGB
    col_azimuth = (np.arange(size) / (size - 1) - 0.5) * POV_FOV_DEG

    bearings = []
    for ox, oy, rgb in objects:
        dx, dy = ox - x, oy - y
        angle = (math.atan2(dy, dx) - heading + math.pi) % (2 * math.pi) - math.pi
        bearings.append((math.degrees(angle), math.hypot(dx, dy), rgb))

    # far objects first so nearer ones are drawn on top
    for azimuth, distance, rgb in sorted(bearings, key=lambda b: -b[1]):
        if abs(azimuth) > POV_FOV_DEG / 2:
            continue
        half_width = max(math.degrees(math.atan2(FOOD_VISUAL_RADIUS, distance)), MIN_FOOD_HALF_WIDTH_DEG)
        brightness = 1.0 / (1.0 + distance / BRIGHTNESS_FALLOFF)
        colour = (np.asarray(rgb) * brightness).astype(np.uint8)
        frame[:, np.abs(col_azimuth - azimuth) <= half_width] = colour
    return frame


class FlyEnvironment:
    def __init__(self, seed: int | None = None, headless: bool = False):
        self.rng = random.Random(seed)
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("DOOMFLY-style demo: MaleCNS connectome controls a fly")
        self.font = pygame.font.SysFont(None, 22)
        self.clock = pygame.time.Clock()

        self.x, self.y = WIDTH / 2, HEIGHT / 2
        self.heading = self.rng.uniform(0, 2 * math.pi)
        self.food = [self._random_food() for _ in range(FOOD_COUNT)]
        self.score = 0

    def _random_food(self):
        return (self.rng.uniform(20, WIDTH - 20), self.rng.uniform(20, HEIGHT - 20))

    def nearest_food_angle_distance(self) -> tuple[float, float, float]:
        fx, fy = min(
            self.food, key=lambda f: (f[0] - self.x) ** 2 + (f[1] - self.y) ** 2
        )
        dx, dy = fx - self.x, fy - self.y
        distance = math.hypot(dx, dy)
        target_angle = math.atan2(dy, dx)
        angle = (target_angle - self.heading + math.pi) % (2 * math.pi) - math.pi
        max_distance = math.hypot(WIDTH, HEIGHT)
        return angle, distance, max_distance

    def step(self, turn: float, speed: float) -> bool:
        """Apply an action, advance physics one frame. Returns False if the window closed."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False

        self.x, self.y, self.heading = advance(self.x, self.y, self.heading, turn, speed)

        for i, (fx, fy) in enumerate(self.food):
            if math.hypot(fx - self.x, fy - self.y) < EAT_DISTANCE:
                self.food[i] = self._random_food()
                self.score += 1
        return True

    def render_pov_frame(self, size: int = 128) -> np.ndarray:
        """
        Renders the fly's panoramic first-person view as an (size, size, 3)
        uint8 RGB array. Columns are azimuth relative to the heading, from
        -FOV/2 (far left) to +FOV/2 (far right), with straight ahead in the
        middle -- matching vision_mapping.py's UV layout (left eye on the left
        half, right eye on the right half, binocular overlap in the centre).

        Each food item is a full-height vertical bar whose angular width is
        its real angular size and whose brightness falls with distance, so
        nearer food dominates what both eyes see. Food in the rear blind spot
        (|azimuth| > FOV/2) is invisible, as for a real fly.
        """
        return render_panorama(self.x, self.y, self.heading, [(fx, fy, FOOD_RGB) for fx, fy in self.food], size)

    def render(
        self, turn: float, speed: float, diagnostics: dict | None = None, pov: np.ndarray | None = None
    ) -> None:
        self.screen.fill((15, 15, 25))
        for fx, fy in self.food:
            pygame.draw.circle(self.screen, (80, 200, 80), (int(fx), int(fy)), FOOD_RADIUS)

        tip = (self.x + math.cos(self.heading) * 16, self.y + math.sin(self.heading) * 16)
        pygame.draw.circle(self.screen, (230, 230, 240), (int(self.x), int(self.y)), FLY_RADIUS)
        pygame.draw.line(self.screen, (240, 90, 90), (self.x, self.y), tip, 3)

        d = diagnostics or {}
        hud = [
            f"score: {self.score}",
            f"turn: {turn:+.2f}  speed: {speed:.2f}",
            f"R1-R6 firing: {d.get('r16_active', 0)}/{d.get('r16_total', 0)}   "
            f"brain active: {d.get('active', 0):,} neurons",
            f"visual hemispheres L={d.get('vis_l', 0):.0f} R={d.get('vis_r', 0):.0f}",
            f"DNa02 L={d.get('dna02_l', 0)} R={d.get('dna02_r', 0)}  "
            f"DNp09={d.get('dnp09', 0)}  MDN={d.get('mdn', 0)}  (diagnostic)",
            "ESC to quit",
        ]
        for i, line in enumerate(hud):
            surf = self.font.render(line, True, (200, 200, 210))
            self.screen.blit(surf, (10, 10 + 20 * i))

        if pov is not None:
            # what the retina samples: left = fly's left, centre = straight ahead
            strip = pygame.surfarray.make_surface(np.transpose(pov, (1, 0, 2)))
            strip = pygame.transform.scale(strip, (POV_STRIP_W, POV_STRIP_H))
            x0, y0 = WIDTH - POV_STRIP_W - 10, 10
            self.screen.blit(strip, (x0, y0))
            pygame.draw.rect(self.screen, (90, 90, 110), (x0, y0, POV_STRIP_W, POV_STRIP_H), 1)
            pygame.draw.line(self.screen, (240, 90, 90), (x0 + POV_STRIP_W // 2, y0), (x0 + POV_STRIP_W // 2, y0 + POV_STRIP_H))
            self.screen.blit(self.font.render("fly view (330 deg)", True, (150, 150, 165)), (x0, y0 + POV_STRIP_H + 4))

        pygame.display.flip()

    def tick(self, fps: int = 30) -> None:
        self.clock.tick(fps)

    def close(self) -> None:
        pygame.quit()
