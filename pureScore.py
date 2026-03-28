import numpy as np
import pygame
import time
import json
import os
import sys
from simulator import Game, FIELD_WIDTH, FIELD_HEIGHT, GOAL_DEPTH, GOAL_HEIGHT, SIDE_WALL, CAR_SIZE, CAR_TURN

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION & PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════════════
CONFIG_FILE = "spline_config.json"

DEFAULT_PARAMS = {
    "GATE_OFFSET": 90.0,      
    "TANGENT_SCALE": 1.1,     
    "STEER_GAIN": 3.5,        
    "LOOK_AHEAD": 0.22        
}

def load_params():
    params = DEFAULT_PARAMS.copy()
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                saved = json.load(f)
                params.update(saved)
        except: pass
    return params

def save_params(params):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(params, f, indent=4)

# ═══════════════════════════════════════════════════════════════════════════════
# CAR SOCCER CONTROLLER
# ═══════════════════════════════════════════════════════════════════════════════

class CarSoccer:
    def __init__(self, render_mode="human", tune_mode=False):
        self.params = load_params()
        self.game = Game(render=(render_mode == "human"))
        self.render_mode = render_mode
        self.tune_mode = tune_mode
        self.dt = 0.1
        self.mode = "spline"
        
        self.goal_pos = np.array([FIELD_WIDTH, FIELD_HEIGHT / 2])
        self.ball_initial_pos = None 
        
        if render_mode == "human":
            pygame.font.init()
            self.font = pygame.font.SysFont("Arial", 18)

    @staticmethod
    def wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi

    def hermite(self, t, p0, v0, p1, v1):
        t2, t3 = t**2, t**3
        h00, h10 = 2*t3 - 3*t2 + 1, t3 - 2*t2 + t
        h01, h11 = -2*t3 + 3*t2, t3 - t2
        return h00*p0 + h10*v0 + h01*p1 + h11*v1

    def getAction(self):
        car = self.game.cars[0]
        ball = self.game.ball

        p_car = np.array(car.getPos()[:2])
        angle_car = np.radians(car.getAngle())
        v_car_dir = np.array([np.cos(angle_car), np.sin(angle_car)])
        p_ball = np.array(ball.getPos()[:2])
        v_ball = np.array(ball.body.velocity)

        # 1. Geometry
        shot_vec = self.goal_pos - p_ball
        shot_unit = shot_vec / (np.linalg.norm(shot_vec) + 1e-9)
        p_gate = p_ball - (shot_unit * self.params["GATE_OFFSET"])
        v_gate_dir = shot_unit 

        dist_g = np.linalg.norm(p_gate - p_car)
        if self.mode == "spline" and dist_g < 20.0: self.mode = "linear"

        # ── BALL SPLINE (DIRECT: BALL -> GOAL) ──
        # Independent of the gate; visualizes the predicted ball arc
        v_ball_start = v_ball * 500 
        v_ball_end = shot_unit * 500

        pts_ball = [self.hermite(t, p_ball, v_ball_start, self.goal_pos, v_ball_end) for t in np.linspace(0, 1, 100)]

        # 2. Car Path Logic
        to_gate_unit = (p_gate - p_car) / (dist_g + 1e-9)
        dot = np.dot(v_car_dir, to_gate_unit)
        
        adaptive_scale = self.params["TANGENT_SCALE"]
        is_tight = dot < 0.4
        if is_tight: adaptive_scale *= (1.6 - dot) 

        v0, v1 = np.zeros(2), np.zeros(2)
        if self.mode == "spline":
            v0 = v_car_dir * dist_g * adaptive_scale
            v1 = v_gate_dir * dist_g * adaptive_scale
            target_p = self.hermite(self.params["LOOK_AHEAD"], p_car, v0, p_gate, v1)
            throttle = 0.8
        else:
            target_p = p_ball
            throttle = 1.0

        # 3. Steering
        desired_angle = np.arctan2(target_p[1]-p_car[1], target_p[0]-p_car[0])
        err = self.wrap(desired_angle - angle_car)
        steer = np.clip(err * self.params["STEER_GAIN"] / np.radians(30), -1, 1)

        self._dbg = {
            "p_car": p_car, "p_gate": p_gate, "p_ball": p_ball,
            "v_ball": v_ball, "v0": v0, "v1": v1, "target": target_p, 
            "mode": self.mode, "tight": is_tight, "pts_ball": pts_ball
        }
        return [float(throttle), float(steer)]

    def render(self):
        scr = self.game.screen
        scr.fill((255, 255, 255))
        
        # Goals
        pygame.draw.rect(scr, (200, 0, 0), (FIELD_WIDTH, SIDE_WALL, GOAL_DEPTH, GOAL_HEIGHT), 2)
        pygame.draw.rect(scr, (0, 0, 200), (0, SIDE_WALL, GOAL_DEPTH, GOAL_HEIGHT), 2)
        
        self.game.gameSpace.debug_draw(self.game.draw_options)

        if hasattr(self, "_dbg"):
            d = self._dbg
            
            # 1. Static Initial Axis (Gray)
            if self.ball_initial_pos is not None:
                pygame.draw.line(scr, (220, 220, 220), self.ball_initial_pos.astype(int), self.goal_pos.astype(int), 1)
            
            # 2. Direct Ball-to-Goal Spline (Orange)
            for i in range(len(d['pts_ball'])-1):
                pygame.draw.line(scr, (255, 120, 0), d['pts_ball'][i].astype(int), d['pts_ball'][i+1].astype(int), 2)

            # 4. Ball Physics
            if np.linalg.norm(d['v_ball']) > 0.1:
                vel_end = d['p_ball'] + d['v_ball'] * 1.8 
                pygame.draw.line(scr, (0, 0, 255), d['p_ball'].astype(int), vel_end.astype(int), 2)
                pygame.draw.circle(scr, (0, 0, 255), vel_end.astype(int), 3)
            pygame.draw.circle(scr, (0, 0, 255), d['p_ball'].astype(int), 5)

            # 5. Car Pathing
            if d['mode'] == "spline":
                pts_car = [self.hermite(t, d['p_car'], d['v0'], d['p_gate'], d['v1']) for t in np.linspace(0, 1, 25)]
                path_color = (255, 69, 0) if d['tight'] else (0, 0, 0)
                path_width = 3 if d['tight'] else 1
                for i in range(len(pts_car)-1):
                    pygame.draw.line(scr, path_color, pts_car[i].astype(int), pts_car[i+1].astype(int), path_width)
            
            # Markers
            pygame.draw.circle(scr, (255, 0, 255), d['p_gate'].astype(int), 10, 3) 
            pygame.draw.circle(scr, (0, 200, 0), d['target'].astype(int), 8, 2)
            
        pygame.display.update()

    def run_simulation(self):
        self.game.addDefaultObjects()
        self.ball_initial_pos = np.array(self.game.ball.getPos()[:2])
        start_t = time.time()
        hit_t = None
        
        while True:
            action = self.getAction()
            self.game.cars[0].update(action)
            self.game.gameSpace.step(self.dt)

            car_p = np.array(self.game.cars[0].getPos()[:2])
            ball_p = np.array(self.game.ball.getPos()[:2])

            if ball_p[0] > FIELD_WIDTH and SIDE_WALL < ball_p[1] < (SIDE_WALL + GOAL_HEIGHT):
                break

            if hit_t is None and np.linalg.norm(car_p - ball_p) < 18:
                hit_t = time.time()

            if time.time() - start_t > 15.0 and self.tune_mode: break
            if hit_t and (time.time() - hit_t > 1.8) and self.tune_mode: break 

            if self.render_mode == "human":
                self.render()
                for e in pygame.event.get():
                    if e.type == pygame.QUIT: return False
                    if e.type == pygame.KEYDOWN and e.key == pygame.K_n: return True
                time.sleep(0.01)

        if self.tune_mode:
            choice = self.show_tuning_ui()
            if choice == "UNDER TURN":
                self.params["TANGENT_SCALE"] *= 1.15
                self.params["STEER_GAIN"] *= 1.05
            elif choice == "OVER TURN":
                self.params["TANGENT_SCALE"] *= 0.85
                self.params["STEER_GAIN"] *= 0.95
            elif choice == "RESTART": return True
            elif choice is None: return False
            save_params(self.params)
            return True
        return True

    def show_tuning_ui(self):
        scr = self.game.screen
        btns = [{"label": "UNDER TURN", "rect": pygame.Rect(50, 350, 250, 80), "color": (255, 200, 200)},
                {"label": "JUST RIGHT", "rect": pygame.Rect(375, 350, 250, 80), "color": (200, 255, 200)},
                {"label": "OVER TURN", "rect": pygame.Rect(700, 350, 250, 80), "color": (200, 200, 255)}]
        while True:
            scr.fill((40, 40, 40))
            m_pos = pygame.mouse.get_pos()
            for b in btns:
                c = tuple(min(x+30, 255) if b["rect"].collidepoint(m_pos) else x for x in b["color"])
                pygame.draw.rect(scr, c, b["rect"], border_radius=8)
                txt = self.font.render(b["label"], True, (0, 0, 0))
                scr.blit(txt, (b["rect"].centerx-55, b["rect"].centery-10))
            pygame.display.update()
            for e in pygame.event.get():
                if e.type == pygame.QUIT: return None
                if e.type == pygame.MOUSEBUTTONDOWN:
                    for b in btns:
                        if b["rect"].collidepoint(e.pos): return b["label"]
                if e.type == pygame.KEYDOWN and e.key == pygame.K_n: return "RESTART"

def select_mode_ui():
    pygame.init()
    scr = pygame.display.set_mode((FIELD_WIDTH + GOAL_DEPTH, FIELD_HEIGHT))
    font = pygame.font.SysFont("Arial", 32, bold=True)
    btns = [
        {"label": "TUNE MODE", "rect": pygame.Rect(250, 350, 200, 100), "color": (150, 255, 150), "val": True},
        {"label": "NORMAL MODE", "rect": pygame.Rect(550, 350, 200, 100), "color": (200, 200, 200), "val": False}
    ]
    while True:
        scr.fill((50, 50, 50))
        m_pos = pygame.mouse.get_pos()
        for b in btns:
            c = tuple(min(x+30, 255) if b["rect"].collidepoint(m_pos) else x for x in b["color"])
            pygame.draw.rect(scr, c, b["rect"], border_radius=12)
            txt = pygame.font.SysFont("Arial", 22).render(b["label"], True, (0, 0, 0))
            scr.blit(txt, (b["rect"].centerx - 60, b["rect"].centery - 10))
        pygame.display.update()
        for e in pygame.event.get():
            if e.type == pygame.QUIT: pygame.quit(); sys.exit()
            if e.type == pygame.MOUSEBUTTONDOWN:
                for b in btns:
                    if b["rect"].collidepoint(e.pos): return b["val"]

if __name__ == "__main__":
    is_tuning = select_mode_ui()
    pygame.quit() 
    while True:
        sim = CarSoccer(render_mode="human", tune_mode=is_tuning)
        restart = sim.run_simulation()
        pygame.quit()
        if not restart: break