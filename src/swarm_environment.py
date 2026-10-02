"""
Several connectome-driven flies sharing one arena: they compete for the same
food, see each other in their panoramic views, and collide.

Same rules as environment.FlyEnvironment for each fly (kinematics, wall
bounce, panoramic rendering), plus:
  - other flies appear in each fly's view as grey bars, dimmer than food
    (fly_brightness scales them; 0 makes flies invisible to each other),
  - overlapping flies are pushed apart along the line joining them, and each
    new contact between a pair counts as one collision for both,
  - each food item is eaten by whichever fly reaches it first.
"""
from __future__ import annotations

import colorsys
import math
import os
import random
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pygame

from environment import (
    EAT_DISTANCE,
    FLY_RADIUS,
    FOOD_COUNT,
    FOOD_RADIUS,
    FOOD_RGB,
    HEIGHT,
    POV_STRIP_H,
    POV_STRIP_W,
    WIDTH,
    advance,
    render_panorama,
)

FLY_RGB = (255, 255, 255)
TRAIL_LENGTH = 90


@dataclass
class Fly:
    x: float
    y: float
    heading: float
    colour: tuple[int, int, int]
    score: int = 0
    collisions: int = 0
    trail: deque = field(default_factory=lambda: deque(maxlen=TRAIL_LENGTH))


class SwarmEnvironment:
    def __init__(
        self,
        n_flies: int,
        seed: int | None = None,
        headless: bool = False,
        fly_brightness: float = 0.35,
        food_count: int | None = None,
    ):
        self.rng = random.Random(seed)
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption(f"MaleCNS connectome: {n_flies} flies")
        self.font = pygame.font.SysFont(None, 22)
        self.small_font = pygame.font.SysFont(None, 16)
        self.clock = pygame.time.Clock()

        self.fly_rgb = tuple(int(c * fly_brightness) for c in FLY_RGB)
        self.flies: list[Fly] = []
        for i in range(n_flies):
            self.flies.append(self._spawn(i, n_flies))
        self.food = [self._random_food() for _ in range(food_count or max(FOOD_COUNT, n_flies + 2))]
        self.contacts: set[tuple[int, int]] = set()
        self.total_collisions = 0
        self.selected = 0  # fly whose view and brain are shown in the HUD

    def _spawn(self, i: int, n: int) -> Fly:
        """Random non-overlapping start position (gives up spacing after many tries)."""
        hue = i / n
        colour = tuple(int(255 * c) for c in colorsys.hsv_to_rgb(hue, 0.65, 1.0))
        for _ in range(200):
            x = self.rng.uniform(4 * FLY_RADIUS, WIDTH - 4 * FLY_RADIUS)
            y = self.rng.uniform(4 * FLY_RADIUS, HEIGHT - 4 * FLY_RADIUS)
            if all(math.hypot(f.x - x, f.y - y) > 6 * FLY_RADIUS for f in self.flies):
                break
        return Fly(x, y, self.rng.uniform(0, 2 * math.pi), colour)

    def _random_food(self):
        return (self.rng.uniform(20, WIDTH - 20), self.rng.uniform(20, HEIGHT - 20))

    def render_pov_frame(self, i: int, size: int = 128) -> np.ndarray:
        """Panoramic view of fly i: food plus every other fly."""
        fly = self.flies[i]
        objects = [(fx, fy, FOOD_RGB) for fx, fy in self.food]
        if any(self.fly_rgb):
            objects += [(o.x, o.y, self.fly_rgb) for j, o in enumerate(self.flies) if j != i]
        return render_panorama(fly.x, fly.y, fly.heading, objects, size)

    def step(self, turns: list[float], speeds: list[float]) -> bool:
        """Apply one action per fly, advance physics one frame. Returns False if the window closed."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_TAB:
                    self.selected = (self.selected + 1) % len(self.flies)
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                nearest = min(range(len(self.flies)), key=lambda j: math.hypot(self.flies[j].x - mx, self.flies[j].y - my))
                if math.hypot(self.flies[nearest].x - mx, self.flies[nearest].y - my) < 4 * FLY_RADIUS:
                    self.selected = nearest

        for fly, turn, speed in zip(self.flies, turns, speeds):
            fly.x, fly.y, fly.heading = advance(fly.x, fly.y, fly.heading, turn, speed)
        self._resolve_collisions()

        # random order so no fly always wins a tie for the same food item
        for fly in self.rng.sample(self.flies, len(self.flies)):
            fly.trail.append((fly.x, fly.y))
            for k, (fx, fy) in enumerate(self.food):
                if math.hypot(fx - fly.x, fy - fly.y) < EAT_DISTANCE:
                    self.food[k] = self._random_food()
                    fly.score += 1
        return True

    def _resolve_collisions(self) -> None:
        """Push overlapping flies apart; count each new pair contact once."""
        touching = set()
        for i in range(len(self.flies)):
            for j in range(i + 1, len(self.flies)):
                a, b = self.flies[i], self.flies[j]
                dx, dy = b.x - a.x, b.y - a.y
                distance = math.hypot(dx, dy)
                overlap = 2 * FLY_RADIUS - distance
                if overlap <= 0:
                    continue
                touching.add((i, j))
                if distance < 1e-6:  # exactly on top of each other: pick any direction
                    dx, dy, distance = 1.0, 0.0, 1.0
                push = overlap / 2
                a.x -= dx / distance * push
                a.y -= dy / distance * push
                b.x += dx / distance * push
                b.y += dy / distance * push
                for f in (a, b):
                    f.x = min(max(f.x, FLY_RADIUS), WIDTH - FLY_RADIUS)
                    f.y = min(max(f.y, FLY_RADIUS), HEIGHT - FLY_RADIUS)

        for i, j in touching - self.contacts:
            self.flies[i].collisions += 1
            self.flies[j].collisions += 1
            self.total_collisions += 1
        self.contacts = touching

    def render(self, actions: list[tuple[float, float]], diagnostics: dict | None = None, pov: np.ndarray | None = None) -> None:
        self.screen.fill((15, 15, 25))
        for fx, fy in self.food:
            pygame.draw.circle(self.screen, (80, 200, 80), (int(fx), int(fy)), FOOD_RADIUS)

        for fly in self.flies:
            if len(fly.trail) > 1:
                dim = tuple(c // 3 for c in fly.colour)
                pygame.draw.lines(self.screen, dim, False, list(fly.trail), 1)
        for i, fly in enumerate(self.flies):
            tip = (fly.x + math.cos(fly.heading) * 16, fly.y + math.sin(fly.heading) * 16)
            pygame.draw.circle(self.screen, fly.colour, (int(fly.x), int(fly.y)), FLY_RADIUS)
            pygame.draw.line(self.screen, (240, 90, 90), (fly.x, fly.y), tip, 3)
            if i == self.selected:
                pygame.draw.circle(self.screen, (255, 255, 255), (int(fly.x), int(fly.y)), FLY_RADIUS + 5, 1)
            label = self.small_font.render(str(i), True, (220, 220, 230))
            self.screen.blit(label, (fly.x + FLY_RADIUS + 2, fly.y - FLY_RADIUS - 8))

        sel = self.flies[self.selected]
        turn, speed = actions[self.selected]
        d = diagnostics or {}
        hud = [
            f"{len(self.flies)} flies   food eaten: {sum(f.score for f in self.flies)}   collisions: {self.total_collisions}",
            f"fly {self.selected} (TAB / click to switch): score {sel.score}  collisions {sel.collisions}",
            f"turn: {turn:+.2f}  speed: {speed:.2f}   brain active: {d.get('active', 0):,} neurons",
            f"visual hemispheres L={d.get('vis_l', 0):.0f} R={d.get('vis_r', 0):.0f}",
            "ESC to quit",
        ]
        for i, line in enumerate(hud):
            self.screen.blit(self.font.render(line, True, (200, 200, 210)), (10, 10 + 20 * i))

        # leaderboard, bottom left
        ranking = sorted(range(len(self.flies)), key=lambda j: -self.flies[j].score)
        for row, j in enumerate(ranking[:10]):
            f = self.flies[j]
            text = self.small_font.render(f"fly {j}: {f.score} food, {f.collisions} hits", True, f.colour)
            self.screen.blit(text, (10, HEIGHT - 16 * (min(len(ranking), 10) - row) - 6))

        if pov is not None:
            strip = pygame.surfarray.make_surface(np.transpose(pov, (1, 0, 2)))
            strip = pygame.transform.scale(strip, (POV_STRIP_W, POV_STRIP_H))
            x0, y0 = WIDTH - POV_STRIP_W - 10, 10
            self.screen.blit(strip, (x0, y0))
            pygame.draw.rect(self.screen, sel.colour, (x0, y0, POV_STRIP_W, POV_STRIP_H), 1)
            pygame.draw.line(self.screen, (240, 90, 90), (x0 + POV_STRIP_W // 2, y0), (x0 + POV_STRIP_W // 2, y0 + POV_STRIP_H))
            caption = f"fly {self.selected} view (330 deg): green = food, grey = other flies"
            self.screen.blit(self.small_font.render(caption, True, (150, 150, 165)), (x0, y0 + POV_STRIP_H + 4))

        pygame.display.flip()

    def tick(self, fps: int = 30) -> None:
        self.clock.tick(fps)

    def close(self) -> None:
        pygame.quit()
