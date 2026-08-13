#!/usr/bin/env python3
"""Real-time ant colony visualization.

Usage:
    python run_sim.py [--ants 300] [--evaporation 0.015] [--seed 42]

Controls:
    SPACE        pause / resume
    UP / DOWN    simulation speed (steps per frame, 1-16)
    LEFT CLICK   drop a new food source
    X            remove the food source nearest the mouse
    A            add 20 ants at the mouse position
    HOLD W       draw walls at the mouse position
    HOLD E       erase walls at the mouse position
    RIGHT DRAG   draw obstacles      (hold SHIFT to erase)
    R            reset the world (new random layout)
    ESC / Q      quit
"""

import argparse

import numpy as np
import pygame

from antsim import SimConfig, Simulation

SCALE = 4
HUD_HEIGHT = 34

BG = (10, 14, 20)
OBSTACLE_COLOR = (86, 92, 104)
FOOD_COLOR = (120, 220, 90)
COLONY_COLOR = (245, 180, 83)
ANT_COLOR = (232, 238, 247)
ANT_CARRY_COLOR = (157, 255, 110)
HUD_TEXT = (160, 175, 195)


def render(screen, sim, font, steps_per_frame, paused):
    cfg = sim.cfg
    world = sim.world

    # Pheromone + obstacles as one RGB image, scaled up
    img = np.zeros((cfg.height, cfg.width, 3), dtype=np.uint8)
    p = np.sqrt(np.clip(world.pheromone / cfg.pheromone_max, 0.0, 1.0))
    img[..., 0] = (p * 40).astype(np.uint8) + BG[0]
    img[..., 1] = (p * 200).astype(np.uint8) + BG[1]
    img[..., 2] = (p * 140).astype(np.uint8) + BG[2]
    img[world.obstacles] = OBSTACLE_COLOR
    surf = pygame.surfarray.make_surface(img.swapaxes(0, 1))
    surf = pygame.transform.scale(surf, (cfg.width * SCALE, cfg.height * SCALE))
    screen.blit(surf, (0, 0))

    # Food sources (radius shrinks as they deplete)
    for f in world.food_sources:
        if f.amount <= 0:
            continue
        r = (2 + 6 * np.sqrt(f.amount / f.initial)) * SCALE * 0.6
        pygame.draw.circle(screen, FOOD_COLOR, (int(f.x * SCALE), int(f.y * SCALE)), int(r))

    # Colony
    cx, cy = int(world.colony[0] * SCALE), int(world.colony[1] * SCALE)
    pygame.draw.circle(screen, COLONY_COLOR, (cx, cy), int(cfg.colony_radius * SCALE), 2)

    # Ants
    pts = (sim.ants.pos * SCALE).astype(int)
    carrying = sim.ants.carrying
    for i in range(len(pts)):
        color = ANT_CARRY_COLOR if carrying[i] else ANT_COLOR
        screen.fill(color, (pts[i, 0], pts[i, 1], 2, 2))

    # HUD
    hud_y = cfg.height * SCALE
    pygame.draw.rect(screen, (16, 21, 30), (0, hud_y, cfg.width * SCALE, HUD_HEIGHT))
    status = "PAUSED" if paused else f"{steps_per_frame}x"
    text = (f"step {sim.step_count}   speed {status}   ants {sim.ants.n}   "
            f"food collected {sim.total_delivered}   food left {world.food_remaining}   "
            f"explored {world.explored_fraction * 100:.0f}%")
    screen.blit(font.render(text, True, HUD_TEXT), (10, hud_y + 9))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ants", type=int, default=300)
    parser.add_argument("--evaporation", type=float, default=0.015)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--frames", type=int, default=0,
                        help="auto-quit after N frames (for testing)")
    parser.add_argument("--screenshot", default=None,
                        help="save a PNG of the final frame before quitting")
    args = parser.parse_args()

    cfg = SimConfig(n_ants=args.ants, evaporation=args.evaporation, seed=args.seed)
    sim = Simulation(cfg)

    pygame.init()
    screen = pygame.display.set_mode(
        (cfg.width * SCALE, cfg.height * SCALE + HUD_HEIGHT))
    pygame.display.set_caption("Digital Ant Colony - pheromone simulation")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("menlo, monospace", 14)

    steps_per_frame = 1
    paused = False
    frame = 0
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_UP:
                    steps_per_frame = min(16, steps_per_frame + 1)
                elif event.key == pygame.K_DOWN:
                    steps_per_frame = max(1, steps_per_frame - 1)
                elif event.key == pygame.K_r:
                    cfg.seed = None
                    sim = Simulation(cfg)
                elif event.key == pygame.K_x:
                    mx, my = pygame.mouse.get_pos()
                    sim.world.remove_food_near(mx / SCALE, my / SCALE)
                elif event.key == pygame.K_a:
                    mx, my = pygame.mouse.get_pos()
                    if my < cfg.height * SCALE:
                        sim.ants.add_ants(mx / SCALE, my / SCALE, 20)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                if my < cfg.height * SCALE:
                    sim.world.add_food(mx / SCALE, my / SCALE)

        # Right-drag: paint obstacles (shift erases). Hold W / E does the same
        # from the keyboard (easier on a trackpad).
        keys = pygame.key.get_pressed()
        mx, my = pygame.mouse.get_pos()
        if my < cfg.height * SCALE:
            if pygame.mouse.get_pressed()[2]:
                erase = bool(pygame.key.get_mods() & pygame.KMOD_SHIFT)
                sim.world.paint_obstacle(mx / SCALE, my / SCALE, radius=4, erase=erase)
            if keys[pygame.K_w]:
                sim.world.paint_obstacle(mx / SCALE, my / SCALE, radius=4)
            if keys[pygame.K_e]:
                sim.world.paint_obstacle(mx / SCALE, my / SCALE, radius=6, erase=True)

        if not paused:
            for _ in range(steps_per_frame):
                sim.step()

        render(screen, sim, font, steps_per_frame, paused)
        pygame.display.flip()
        clock.tick(60)

        frame += 1
        if args.frames and frame >= args.frames:
            running = False

    if args.screenshot:
        pygame.image.save(screen, args.screenshot)
    pygame.quit()


if __name__ == "__main__":
    main()
