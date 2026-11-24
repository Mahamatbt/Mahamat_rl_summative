"""Interactive 3D visualisation of the Water Point Dispatch environment (pygame + PyOpenGL).

Scene
    * Rolling savanna terrain with acacia trees, dirt roads from the depot and fog
    * Villages (thatched huts) around every water point - more huts = more people served
    * Elevated water tanks: the tank band shows the TRUE status
      (green = functional, amber = needs repair, red = non functional) with a flashing
      beacon on broken points, flowing water + villagers at working taps
    * Floating diamond above each tank = what the crew BELIEVES (last known status),
      fading to grey as that information gets older
    * Detailed pickup truck with spinning wheels and a dust trail; its recent route is drawn
    * Effects: scanning ring for inspections, sparks for repairs, fuel ring at the depot

Camera / controls (human mode)
    left-drag      orbit (turn around the scene)        scroll / Z X   zoom
    right-drag     pan            W A S D  pan            Q E  rotate      T G  tilt
    click          select a water point (details panel)  TAB  next water point
    F              follow the truck on/off               R    reset camera
    SPACE          pause / resume  + -  animation speed  H    help on/off
    ESC            quit
"""

from __future__ import annotations

import math

import numpy as np
import pygame
from OpenGL.GL import *  # noqa: F403
from OpenGL.GLU import *  # noqa: F403

from environment.custom_env import FUNCTIONAL, NEEDS_REPAIR, NON_FUNCTIONAL, STATUS_NAMES

STATUS_RGB = {
    FUNCTIONAL: (0.20, 0.78, 0.35),
    NEEDS_REPAIR: (1.00, 0.70, 0.10),
    NON_FUNCTIONAL: (0.90, 0.18, 0.15),
}
ACTION_RGB = {"inspect": (0.2, 0.85, 1.0), "minor repair": (1.0, 0.95, 0.2), "major repair": (1.0, 0.5, 0.1)}
SKY_TOP, SKY_HORIZON = (0.36, 0.58, 0.86), (0.86, 0.90, 0.93)

WIDTH, HEIGHT = 1360, 800
HUD_W = 330
WORLD = 24.0  # world units across the district


def _rng_colour(rng, base, jitter):
    return tuple(float(np.clip(b + rng.uniform(-jitter, jitter), 0, 1)) for b in base)


class WaterDispatchRenderer:
    def __init__(self, env, mode: str = "human", fps: int = 60):
        self.env = env
        self.mode = mode
        self.fps = fps
        self.t = 0.0
        self.caption_extra = ""
        self.speed = 1.0
        self.paused = False
        self.show_help = True
        self.follow = False
        self.selected: int | None = None
        self.particles: list[list] = []  # [x, y, z, vx, vy, vz, age, life, r, g, b, size]
        self.route: list[np.ndarray] = []
        self.wheel_angle = 0.0
        self._drag = None
        self._drag_moved = 0
        self.reset_camera()

        pygame.init()
        pygame.font.init()
        flags = pygame.DOUBLEBUF | pygame.OPENGL
        if mode == "rgb_array":
            flags |= pygame.HIDDEN
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)
        try:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT), flags)
        except pygame.error:
            pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 0)
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT), flags)
        pygame.display.set_caption("Water Point Maintenance Dispatch - RL agent (drag to orbit, scroll to zoom, H for help)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Helvetica,Arial", 17)
        self.font_b = pygame.font.SysFont("Helvetica,Arial", 20, bold=True)
        self.font_s = pygame.font.SysFont("Helvetica,Arial", 13)
        self.font_sb = pygame.font.SysFont("Helvetica,Arial", 14, bold=True)
        self.quadric = gluNewQuadric()
        gluQuadricNormals(self.quadric, GLU_SMOOTH)

        self.scale = WORLD / env.cfg.map_size_km
        self.site_xz = np.array([self._xz(c) for c in env.coords])  # n points + depot
        self._init_gl()
        self._build_static_scene()
        self.truck_pos = self._world(env.coords[env.n])
        self.truck_heading = 0.0
        self._matrices = None

    # ================================================================ geometry
    def _xz(self, xy):
        half = self.env.cfg.map_size_km / 2
        return np.array([(xy[0] - half) * self.scale, (xy[1] - half) * self.scale])

    def height(self, x: float, z: float) -> float:
        h = 0.55 * math.sin(0.23 * x + 0.4) * math.cos(0.19 * z - 0.3) + 0.3 * math.sin(0.41 * x - 0.57 * z + 1.1)
        h += 0.9 * math.exp(-((x + 9.0) ** 2 + (z - 9.5) ** 2) / 18.0)  # a hill in one corner
        # flatten the ground around sites so tanks and depot stand level
        d2 = np.min(np.sum((self.site_xz - np.array([x, z])) ** 2, axis=1)) if hasattr(self, "site_xz") else 99
        flat = 1.0 - math.exp(-d2 / 1.6)
        flat *= min(1.0, max(0.0, (WORLD / 2 + 4.0 - max(abs(x), abs(z))) / 3.5))  # blend into the far plains
        return 0.6 + h * flat if flat > 0 else 0.6

    def site_height(self, k: int) -> float:
        x, z = self.site_xz[k]
        return self.height(x, z)

    def _world(self, xy) -> np.ndarray:
        x, z = self._xz(xy)
        return np.array([x, self.height(x, z), z])

    # ================================================================ GL setup
    def _init_gl(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_NORMALIZE)
        glShadeModel(GL_SMOOTH)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.38, 0.38, 0.40, 1.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.92, 0.88, 0.80, 1.0))
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, (0.15, 0.15, 0.17, 1.0))
        glFogi(GL_FOG_MODE, GL_LINEAR)
        glFogfv(GL_FOG_COLOR, (*SKY_HORIZON, 1.0))
        try:
            glEnable(GL_MULTISAMPLE)
        except Exception:
            pass

    # ================================================================ primitives
    def _box(self, sx, sy, sz):
        x, y, z = sx / 2, sy / 2, sz / 2
        faces = [
            ((0, 0, 1), [(-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z)]),
            ((0, 0, -1), [(-x, -y, -z), (-x, y, -z), (x, y, -z), (x, -y, -z)]),
            ((1, 0, 0), [(x, -y, -z), (x, y, -z), (x, y, z), (x, -y, z)]),
            ((-1, 0, 0), [(-x, -y, -z), (-x, -y, z), (-x, y, z), (-x, y, -z)]),
            ((0, 1, 0), [(-x, y, -z), (-x, y, z), (x, y, z), (x, y, -z)]),
            ((0, -1, 0), [(-x, -y, -z), (x, -y, -z), (x, -y, z), (-x, -y, z)]),
        ]
        glBegin(GL_QUADS)
        for n, vs in faces:
            glNormal3f(*n)
            for v in vs:
                glVertex3f(*v)
        glEnd()

    def _cyl(self, r, h, top=None, slices=20, cap=True):
        glPushMatrix()
        glRotatef(-90, 1, 0, 0)
        gluCylinder(self.quadric, r, r if top is None else top, h, slices, 1)
        if cap:
            glTranslatef(0, 0, h)
            gluDisk(self.quadric, 0, r if top is None else top, slices, 1)
        glPopMatrix()

    def _cone(self, r, h, slices=20):
        self._cyl(r, h, top=0.0, slices=slices, cap=False)

    def _sphere(self, r, sx=1.0, sy=1.0, sz=1.0, slices=14):
        glPushMatrix()
        glScalef(sx, sy, sz)
        gluSphere(self.quadric, r, slices, max(6, slices // 2))
        glPopMatrix()

    def _disc(self, inner, outer, slices=36):
        glPushMatrix()
        glRotatef(-90, 1, 0, 0)
        gluDisk(self.quadric, inner, outer, slices, 1)
        glPopMatrix()

    def _shadow(self, r, alpha=0.28):
        glDisable(GL_LIGHTING)
        glColor4f(0.05, 0.08, 0.02, alpha)
        glPushMatrix()
        glTranslatef(0.12, 0.03, 0.12)
        glScalef(1.0, 1.0, 0.8)
        self._disc(0, r, 20)
        glPopMatrix()
        glEnable(GL_LIGHTING)

    def _diamond(self, s):
        mid = [(s * 0.7, 0, 0), (0, 0, s * 0.7), (-s * 0.7, 0, 0), (0, 0, -s * 0.7)]
        glBegin(GL_TRIANGLES)
        for apex in ((0, s, 0), (0, -s, 0)):
            for i in range(4):
                a, b = mid[i], mid[(i + 1) % 4]
                n = np.cross(np.subtract(a, apex), np.subtract(b, apex))
                glNormal3f(*(-n if apex[1] < 0 else n))
                glVertex3f(*apex)
                glVertex3f(*a)
                glVertex3f(*b)
        glEnd()

    # ================================================================ static scene (display lists)
    def _build_static_scene(self):
        rng = np.random.default_rng(11)
        self.static_list = glGenLists(1)
        glNewList(self.static_list, GL_COMPILE)
        self._compile_terrain()
        self._compile_roads()
        self._compile_trees(rng)
        self._compile_villages(rng)
        self._compile_depot()
        glEndList()

    def _compile_terrain(self):
        half, n = WORLD / 2 + 4.0, 72
        xs = np.linspace(-half, half, n + 1)
        H = np.array([[self.height(x, z) for x in xs] for z in xs])
        step = xs[1] - xs[0]
        grass_dry, grass_green, soil = np.array([0.74, 0.70, 0.43]), np.array([0.48, 0.62, 0.30]), np.array([0.66, 0.55, 0.38])
        rng = np.random.default_rng(3)
        noise = rng.uniform(-0.035, 0.035, size=H.shape)
        # far plains out to the horizon (faded by fog)
        far = 220.0
        glNormal3f(0, 1, 0)
        glColor3f(0.72, 0.66, 0.42)
        glBegin(GL_QUADS)
        for x0, z0, x1, z1 in ((-far, -far, far, -half), (-far, half, far, far), (-far, -half, -half, half), (half, -half, far, half)):
            glVertex3f(x0, 0.58, z0)
            glVertex3f(x0, 0.58, z1)
            glVertex3f(x1, 0.58, z1)
            glVertex3f(x1, 0.58, z0)
        glEnd()
        for i in range(n):
            glBegin(GL_TRIANGLE_STRIP)
            for j in range(n + 1):
                for ii in (i, i + 1):
                    x, z = xs[j], xs[ii]
                    h = H[ii, j]
                    hx = H[ii, min(j + 1, n)] - H[ii, max(j - 1, 0)]
                    hz = H[min(ii + 1, n), j] - H[max(ii - 1, 0), j]
                    glNormal3f(-hx, 2 * step, -hz)
                    inside = max(abs(x), abs(z)) <= WORLD / 2
                    mix = np.clip((h - 0.2) / 1.4, 0, 1)
                    col = grass_dry * (1 - mix) + grass_green * mix
                    if not inside:
                        col = col * 0.8 + soil * 0.2
                    glColor3f(*(col + noise[ii, j]))
                    glVertex3f(x, h, z)
            glEnd()
        # district boundary
        glDisable(GL_LIGHTING)
        glColor4f(1, 1, 1, 0.45)
        glLineWidth(2.0)
        glBegin(GL_LINE_LOOP)
        b = WORLD / 2
        for t in np.linspace(0, 4, 160, endpoint=False):
            side, f = int(t), t - int(t)
            x, z = [(-b + 2 * b * f, -b), (b, -b + 2 * b * f), (b - 2 * b * f, b), (-b, b - 2 * b * f)][side]
            glVertex3f(x, self.height(x, z) + 0.03, z)
        glEnd()
        glEnable(GL_LIGHTING)

    def _road(self, a, b, width=0.22, colour=(0.60, 0.47, 0.32)):
        d = b - a
        length = np.linalg.norm(d)
        if length < 1e-6:
            return
        perp = np.array([-d[1], d[0]]) / length * width / 2
        steps = max(2, int(length / 0.25))
        glColor3f(*colour)
        glNormal3f(0, 1, 0)
        glBegin(GL_TRIANGLE_STRIP)
        for s in np.linspace(0, 1, steps):
            p = a + d * s
            for side in (1, -1):
                q = p + side * perp
                glVertex3f(q[0], self.height(q[0], q[1]) + 0.025, q[1])
        glEnd()

    def _compile_roads(self):
        depot = self.site_xz[self.env.n]
        for k in range(self.env.n):
            self._road(depot, self.site_xz[k])

    def _compile_trees(self, rng):
        placed = 0
        while placed < 140:
            x, z = rng.uniform(-WORLD / 2 - 3, WORLD / 2 + 3, size=2)
            if np.min(np.linalg.norm(self.site_xz - np.array([x, z]), axis=1)) < 2.6:
                continue
            # keep roads clear
            depot = self.site_xz[self.env.n]
            p = np.array([x, z])
            near_road = False
            for k in range(self.env.n):
                a, b = depot, self.site_xz[k]
                t = np.clip(np.dot(p - a, b - a) / np.dot(b - a, b - a), 0, 1)
                if np.linalg.norm(a + t * (b - a) - p) < 0.6:
                    near_road = True
                    break
            if near_road:
                continue
            placed += 1
            h = self.height(x, z)
            s = rng.uniform(0.7, 1.3)
            glPushMatrix()
            glTranslatef(x, h, z)
            glScalef(s, s, s)
            self._shadow(0.55, 0.22)
            glColor3f(*_rng_colour(rng, (0.38, 0.27, 0.17), 0.04))
            glPushMatrix()
            glRotatef(rng.uniform(-8, 8), 0, 0, 1)
            self._cyl(0.06, 0.75, top=0.045, slices=8)
            glPopMatrix()
            glColor3f(*_rng_colour(rng, (0.30, 0.47, 0.20), 0.06))
            glTranslatef(0, 0.82, 0)
            self._sphere(0.5, 1.0, 0.28, 1.0, slices=12)  # flat acacia canopy
            glTranslatef(rng.uniform(-0.2, 0.2), 0.08, rng.uniform(-0.2, 0.2))
            self._sphere(0.32, 1.0, 0.3, 1.0, slices=10)
            glPopMatrix()

    def _hut(self, rng):
        self._shadow(0.38, 0.2)
        glColor3f(*_rng_colour(rng, (0.70, 0.52, 0.36), 0.05))
        self._cyl(0.25, 0.32, slices=14)
        glColor3f(*_rng_colour(rng, (0.78, 0.66, 0.36), 0.05))
        glPushMatrix()
        glTranslatef(0, 0.30, 0)
        self._cone(0.36, 0.34, slices=14)
        glPopMatrix()
        glColor3f(0.25, 0.18, 0.12)  # doorway
        glPushMatrix()
        glTranslatef(0, 0.11, 0.245)
        self._box(0.1, 0.18, 0.02)
        glPopMatrix()

    def _compile_villages(self, rng):
        env = self.env
        for k in range(env.n):
            cx, cz = self.site_xz[k]
            n_huts = int(np.clip(env.pop[k] / 220, 2, 14))
            for i in range(n_huts):
                ang = 2 * math.pi * i / n_huts + rng.uniform(-0.2, 0.2)
                r = rng.uniform(1.35, 2.3)
                x, z = cx + r * math.cos(ang), cz + r * math.sin(ang)
                glPushMatrix()
                glTranslatef(x, self.height(x, z), z)
                glRotatef(-math.degrees(ang) + 90, 0, 1, 0)
                self._hut(rng)
                glPopMatrix()

    def _compile_depot(self):
        x, z = self.site_xz[self.env.n]
        h = self.height(x, z)
        glPushMatrix()
        glTranslatef(x, h, z)
        glDisable(GL_LIGHTING)
        glColor3f(0.55, 0.55, 0.52)  # concrete yard
        self._disc(0, 1.6, 32)
        glEnable(GL_LIGHTING)
        # main building
        glColor3f(0.86, 0.84, 0.78)
        glPushMatrix()
        glTranslatef(-0.25, 0.4, -0.4)
        self._box(1.5, 0.8, 0.9)
        glTranslatef(0, 0.45, 0)
        glColor3f(0.18, 0.36, 0.62)
        self._box(1.65, 0.1, 1.05)
        glPopMatrix()
        # windows and door
        glColor3f(0.35, 0.55, 0.75)
        for wx in (-0.75, -0.25):
            glPushMatrix()
            glTranslatef(wx, 0.5, 0.06)
            self._box(0.25, 0.2, 0.02)
            glPopMatrix()
        glColor3f(0.35, 0.25, 0.18)
        glPushMatrix()
        glTranslatef(0.25, 0.25, 0.06)
        self._box(0.22, 0.45, 0.02)
        glPopMatrix()
        # garage / store with roller door
        glColor3f(0.72, 0.74, 0.76)
        glPushMatrix()
        glTranslatef(0.95, 0.35, -0.35)
        self._box(0.8, 0.7, 0.8)
        glColor3f(0.5, 0.52, 0.55)
        glTranslatef(0, -0.08, 0.41)
        self._box(0.55, 0.5, 0.02)
        glPopMatrix()
        # fuel tank + pump
        glColor3f(0.92, 0.80, 0.15)
        glPushMatrix()
        glTranslatef(1.05, 0, 0.75)
        self._cyl(0.25, 0.7)
        glColor3f(0.8, 0.15, 0.1)
        glTranslatef(-0.45, 0, 0.05)
        glPushMatrix()
        glTranslatef(0, 0.25, 0)
        self._box(0.15, 0.5, 0.12)
        glPopMatrix()
        glPopMatrix()
        # parts crates
        glColor3f(0.55, 0.40, 0.22)
        for cx, cz, s in ((-1.1, 0.55, 0.22), (-0.85, 0.7, 0.18), (-1.05, 0.85, 0.16)):
            glPushMatrix()
            glTranslatef(cx, s / 2, cz)
            self._box(s, s, s)
            glPopMatrix()
        # flag pole
        glColor3f(0.9, 0.9, 0.9)
        glPushMatrix()
        glTranslatef(-1.25, 0, -0.95)
        self._cyl(0.025, 1.9, slices=8)
        glPopMatrix()
        # fence posts
        glColor3f(0.45, 0.35, 0.25)
        for a in np.linspace(0, 2 * math.pi, 28, endpoint=False):
            if abs(math.sin(a) - 1) < 0.15:
                continue  # gate facing the roads
            glPushMatrix()
            glTranslatef(1.6 * math.cos(a), 0, 1.6 * math.sin(a))
            self._cyl(0.03, 0.3, slices=6)
            glPopMatrix()
        glPopMatrix()

    # ================================================================ dynamic objects
    def _draw_water_point(self, k: int, highlight_label: str | None, pulse: float):
        env = self.env
        x, z = self.site_xz[k]
        h = self.site_height(k)
        status = int(env.status[k])
        col = STATUS_RGB[status]
        glPushMatrix()
        glTranslatef(x, h, z)
        # concrete apron
        glDisable(GL_LIGHTING)
        glColor3f(0.62, 0.62, 0.58)
        glPushMatrix()
        glTranslatef(0, 0.02, 0)
        self._disc(0, 0.75, 28)
        glPopMatrix()
        if status == FUNCTIONAL:  # water puddle
            glColor4f(0.25, 0.55, 0.95, 0.7)
            glPushMatrix()
            glTranslatef(0.45, 0.035, 0.25)
            glScalef(1.0, 1.0, 0.7)
            self._disc(0, 0.22, 20)
            glPopMatrix()
        glEnable(GL_LIGHTING)
        self._shadow(0.55, 0.3)
        # tower legs with cross braces
        glColor3f(0.42, 0.42, 0.45)
        legs = [(-0.26, -0.26), (0.26, -0.26), (0.26, 0.26), (-0.26, 0.26)]
        for dx, dz in legs:
            glPushMatrix()
            glTranslatef(dx, 0, dz)
            self._cyl(0.035, 1.25, slices=8)
            glPopMatrix()
        glDisable(GL_LIGHTING)
        glLineWidth(1.5)
        glColor3f(0.35, 0.35, 0.38)
        glBegin(GL_LINES)
        for i in range(4):
            (ax, az), (bx, bz) = legs[i], legs[(i + 1) % 4]
            for y0, y1 in ((0.15, 0.65), (0.65, 0.15), (0.65, 1.15), (1.15, 0.65)):
                glVertex3f(ax, y0, az)
                glVertex3f(bx, y1, bz)
        glEnd()
        glEnable(GL_LIGHTING)
        # tank: grey body + status band + roof
        glColor3f(0.80, 0.81, 0.83)
        glPushMatrix()
        glTranslatef(0, 1.25, 0)
        self._cyl(0.45, 0.62, slices=26)
        glColor3f(*col)
        glTranslatef(0, 0.18, 0)
        self._cyl(0.462, 0.26, slices=26, cap=False)
        glColor3f(0.55, 0.58, 0.62)
        glTranslatef(0, 0.44, 0)
        self._cone(0.48, 0.22, slices=26)
        glPopMatrix()
        # pipe down to the tap
        glColor3f(0.30, 0.45, 0.65)
        glPushMatrix()
        glTranslatef(0.30, 0.15, 0.0)
        self._cyl(0.035, 1.12, slices=8)
        glTranslatef(0.06, 0.25, 0)
        glRotatef(-90, 0, 0, 1)
        self._cyl(0.03, 0.2, slices=8)
        glPopMatrix()
        if status == FUNCTIONAL:  # flowing water
            glDisable(GL_LIGHTING)
            glColor4f(0.35, 0.65, 1.0, 0.75)
            glPushMatrix()
            glTranslatef(0.48, 0.04, 0.0)
            self._cyl(0.018, 0.34, slices=6, cap=False)
            glPopMatrix()
            glEnable(GL_LIGHTING)
        # warning beacon
        if status != FUNCTIONAL:
            rate = 6.0 if status == NON_FUNCTIONAL else 3.0
            on = 0.5 + 0.5 * math.sin(self.t * rate + k)
            glDisable(GL_LIGHTING)
            glColor4f(col[0], col[1], col[2], 0.35 + 0.65 * on)
            glPushMatrix()
            glTranslatef(0, 2.12, 0)
            self._sphere(0.09 + 0.04 * on, slices=10)
            glColor4f(col[0], col[1], col[2], 0.18 * on)
            self._sphere(0.25 + 0.1 * on, slices=12)
            glPopMatrix()
            glEnable(GL_LIGHTING)
        # villagers: queue at working taps, waiting by broken ones
        n_people = int(np.clip(env.pop[k] / 450, 1, 6))
        rng = np.random.default_rng(k)
        for i in range(n_people):
            if status == FUNCTIONAL:
                px, pz = 0.6 + 0.22 * i, 0.35 + 0.08 * math.sin(i)
                bob = 0.02 * math.sin(self.t * 3 + i)
            else:
                ang = 2.0 + 0.5 * i
                px, pz = 1.0 * math.cos(ang), 1.0 * math.sin(ang)
                bob = 0.0
            glPushMatrix()
            glTranslatef(px, bob, pz)
            glColor3f(*_rng_colour(rng, (0.6, 0.3, 0.5), 0.35))
            self._cyl(0.05, 0.18, top=0.035, slices=8)
            glColor3f(0.30, 0.20, 0.14)
            glTranslatef(0, 0.23, 0)
            self._sphere(0.045, slices=8)
            if status == FUNCTIONAL:  # bucket
                glColor3f(0.9, 0.75, 0.2)
                glTranslatef(0.07, -0.16, 0)
                self._cyl(0.03, 0.05, slices=8)
            glPopMatrix()
        # belief diamond (what the crew knows), fades with age
        age = min(env.days_since[k] / 30.0, 1.0)
        known_rgb = np.array(STATUS_RGB[int(env.known[k])])
        dcol = (1 - 0.75 * age) * known_rgb + 0.75 * age * np.array([0.6, 0.6, 0.6])
        glColor3f(*dcol)
        glPushMatrix()
        glTranslatef(0, 2.6 + 0.08 * math.sin(self.t * 2 + k), 0)
        glRotatef((self.t * 50 + k * 30) % 360, 0, 1, 0)
        self._diamond(0.22)
        glPopMatrix()
        # selection ring
        glDisable(GL_LIGHTING)
        if self.selected == k:
            glColor4f(1, 1, 1, 0.9)
            glPushMatrix()
            glTranslatef(0, 0.05, 0)
            glRotatef(self.t * 40, 0, 1, 0)
            self._disc(1.0, 1.08, 48)
            glPopMatrix()
        # action ring
        if highlight_label in ACTION_RGB:
            c = ACTION_RGB[highlight_label]
            for off in (0.0, 0.35):
                p = (pulse + off) % 1.0
                glColor4f(c[0], c[1], c[2], 0.9 * (1 - p))
                glPushMatrix()
                glTranslatef(0, 0.06 + (1.4 * p if highlight_label == "inspect" else 0), 0)
                self._disc(0.5 + 0.9 * p, 0.6 + 0.9 * p, 40)
                glPopMatrix()
        glEnable(GL_LIGHTING)
        glPopMatrix()

    def _draw_flag(self):
        x, z = self.site_xz[self.env.n]
        h = self.height(x, z)
        glDisable(GL_LIGHTING)
        glColor3f(0.12, 0.45, 0.85)
        glBegin(GL_QUAD_STRIP)
        for i in range(9):
            u = i / 8
            wave = 0.06 * math.sin(self.t * 5 - u * 5) * u
            glVertex3f(x - 1.25 + 0.5 * u, h + 1.88, z - 0.95 + wave)
            glVertex3f(x - 1.25 + 0.5 * u, h + 1.58, z - 0.95 + wave)
        glEnd()
        glEnable(GL_LIGHTING)

    def _draw_truck(self):
        x, y, z = self.truck_pos
        glPushMatrix()
        glTranslatef(x, y, z)
        glRotatef(self.truck_heading, 0, 1, 0)
        self._shadow(0.55, 0.3)
        # chassis
        glColor3f(0.2, 0.2, 0.22)
        glPushMatrix()
        glTranslatef(0, 0.17, 0)
        self._box(1.0, 0.08, 0.46)
        glPopMatrix()
        # cab
        glColor3f(0.93, 0.94, 0.96)
        glPushMatrix()
        glTranslatef(0.28, 0.37, 0)
        self._box(0.42, 0.34, 0.46)
        glPopMatrix()
        glColor3f(0.22, 0.32, 0.42)  # windscreen + side windows
        glPushMatrix()
        glTranslatef(0.495, 0.43, 0)
        self._box(0.02, 0.16, 0.38)
        glPopMatrix()
        for side in (-1, 1):
            glPushMatrix()
            glTranslatef(0.3, 0.44, side * 0.232)
            self._box(0.26, 0.13, 0.01)
            glPopMatrix()
        # blue stripe + bed
        glColor3f(0.15, 0.45, 0.85)
        glPushMatrix()
        glTranslatef(0.28, 0.27, 0)
        self._box(0.425, 0.05, 0.465)
        glPopMatrix()
        glColor3f(0.90, 0.91, 0.93)
        glPushMatrix()
        glTranslatef(-0.22, 0.29, 0)
        self._box(0.56, 0.16, 0.46)
        glPopMatrix()
        # cargo: spare parts crate + pipe
        glColor3f(0.55, 0.40, 0.22)
        glPushMatrix()
        glTranslatef(-0.28, 0.43, -0.08)
        self._box(0.22, 0.14, 0.2)
        glPopMatrix()
        glColor3f(0.3, 0.45, 0.65)
        glPushMatrix()
        glTranslatef(-0.45, 0.42, 0.14)
        glRotatef(90, 0, 0, 1)
        self._cyl(0.03, 0.5, slices=8)
        glPopMatrix()
        # headlights
        glDisable(GL_LIGHTING)
        glColor3f(1.0, 0.95, 0.7)
        for side in (-0.15, 0.15):
            glPushMatrix()
            glTranslatef(0.5, 0.27, side)
            self._sphere(0.035, slices=8)
            glPopMatrix()
        glEnable(GL_LIGHTING)
        # wheels (spinning)
        for dx in (-0.32, 0.3):
            for dz in (-0.24, 0.24):
                glPushMatrix()
                glTranslatef(dx, 0.11, dz)
                glRotatef(-self.wheel_angle, 0, 0, 1)
                glColor3f(0.08, 0.08, 0.08)
                glPushMatrix()
                glRotatef(90, 1, 0, 0)
                glTranslatef(0, -0.05, 0)
                self._cyl(0.11, 0.1, slices=14)
                glPopMatrix()
                glColor3f(0.7, 0.7, 0.72)  # hub marker shows rotation
                glPushMatrix()
                glTranslatef(0.05, 0, dz / abs(dz) * 0.052)
                self._box(0.06, 0.03, 0.005)
                glPopMatrix()
                glPopMatrix()
        glPopMatrix()

    def _draw_route(self):
        if len(self.route) < 2:
            return
        glDisable(GL_LIGHTING)
        glLineWidth(2.5)
        pts = self.route[-12:]
        glBegin(GL_LINE_STRIP)
        for i, (x, _, z) in enumerate(pts):
            glColor4f(0.15, 0.45, 0.95, 0.15 + 0.75 * i / len(pts))
            glVertex3f(x, self.height(x, z) + 0.08, z)
        glEnd()
        glEnable(GL_LIGHTING)

    def _spawn(self, pos, vel, life, rgb, size):
        self.particles.append([*pos, *vel, 0.0, life, *rgb, size])

    def _update_particles(self, dt):
        alive = []
        for p in self.particles:
            p[6] += dt
            if p[6] < p[7]:
                p[0] += p[3] * dt
                p[1] += p[4] * dt
                p[2] += p[5] * dt
                p[4] -= 1.5 * dt
                alive.append(p)
        self.particles = alive[-600:]

    def _draw_particles(self):
        if not self.particles:
            return
        glDisable(GL_LIGHTING)
        glDepthMask(GL_FALSE)
        for size in sorted({p[11] for p in self.particles}):
            glPointSize(size)
            glBegin(GL_POINTS)
            for p in self.particles:
                if p[11] == size:
                    glColor4f(p[8], p[9], p[10], max(0.0, 1 - p[6] / p[7]) * 0.8)
                    glVertex3f(p[0], p[1], p[2])
            glEnd()
        glDepthMask(GL_TRUE)
        glEnable(GL_LIGHTING)

    # ================================================================ camera
    def reset_camera(self):
        self.cam_target = np.array([0.0, 0.0, 0.0])
        self.cam_yaw = -35.0
        self.cam_pitch = 42.0
        self.cam_dist = 30.0

    def _sky(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, 1, 0, 1, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glDisable(GL_FOG)
        glBegin(GL_QUADS)
        glColor3f(*SKY_HORIZON)
        glVertex2f(0, 0)
        glVertex2f(1, 0)
        glColor3f(*SKY_TOP)
        glVertex2f(1, 1)
        glVertex2f(0, 1)
        glEnd()
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

    def _set_camera(self):
        glViewport(HUD_W, 0, WIDTH - HUD_W, HEIGHT)
        self._sky()
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(42.0, (WIDTH - HUD_W) / HEIGHT, 0.3, 300.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        if self.follow:
            self.cam_target += (self.truck_pos - self.cam_target) * 0.12
        yaw, pitch = math.radians(self.cam_yaw), math.radians(self.cam_pitch)
        tx, ty, tz = self.cam_target
        eye = (
            tx + self.cam_dist * math.cos(pitch) * math.sin(yaw),
            ty + self.cam_dist * math.sin(pitch),
            tz + self.cam_dist * math.cos(pitch) * math.cos(yaw),
        )
        gluLookAt(*eye, tx, ty, tz, 0, 1, 0)
        glLightfv(GL_LIGHT0, GL_POSITION, (0.45, 1.0, 0.3, 0.0))
        glEnable(GL_FOG)
        glFogf(GL_FOG_START, self.cam_dist * 1.2)
        glFogf(GL_FOG_END, self.cam_dist * 3.2 + 20)
        self._matrices = (
            glGetDoublev(GL_MODELVIEW_MATRIX),
            glGetDoublev(GL_PROJECTION_MATRIX),
            glGetIntegerv(GL_VIEWPORT),
        )

    def _project(self, p):
        if self._matrices is None:
            return None
        mv, pr, vp = self._matrices
        wx, wy, wz = gluProject(p[0], p[1], p[2], mv, pr, vp)
        return (wx, wy, wz) if 0 <= wz <= 1 else None

    def _pan(self, dx, dy):
        yaw = math.radians(self.cam_yaw)
        right = np.array([math.cos(yaw), 0, -math.sin(yaw)])
        fwd = np.array([-math.sin(yaw), 0, -math.cos(yaw)])
        k = self.cam_dist * 0.0018
        self.cam_target += (-dx * right + dy * fwd) * k
        self.cam_target[[0, 2]] = np.clip(self.cam_target[[0, 2]], -WORLD, WORLD)
        self.follow = False

    def _pick(self, mx, my):
        best, best_d = None, 45.0
        for k in range(self.env.n):
            x, z = self.site_xz[k]
            s = self._project((x, self.site_height(k) + 1.4, z))
            if s is None:
                continue
            d = math.hypot(s[0] - mx, s[1] - (HEIGHT - my))
            if d < best_d:
                best, best_d = k, d
        self.selected = best

    # ================================================================ HUD / overlays
    def _hud_surface(self) -> pygame.Surface:
        env, cfg = self.env, self.env.cfg
        info = env._info()
        s = pygame.Surface((HUD_W, HEIGHT), pygame.SRCALPHA)
        s.fill((22, 30, 42, 255))
        y = 12

        def text(t, font=None, color=(230, 235, 240), dy=23, x=16):
            nonlocal y
            s.blit((font or self.font).render(t, True, color), (x, y))
            y += dy

        def bar(label, frac, color):
            nonlocal y
            text(label, dy=19)
            pygame.draw.rect(s, (60, 70, 85), (16, y, HUD_W - 32, 9), border_radius=4)
            pygame.draw.rect(s, color, (16, y, int((HUD_W - 32) * max(0, min(1, frac))), 9), border_radius=4)
            y += 17

        text("WATER POINT DISPATCH", self.font_b, (120, 190, 255), 26)
        if self.caption_extra:
            text(self.caption_extra, self.font_s, (170, 180, 195), 20)
        status = "PAUSED" if self.paused else f"speed x{self.speed:g}"
        text(f"Day {env.day} / {cfg.horizon_days}      {status}", self.font_b, dy=28)
        bar(f"Fuel  {env.fuel:5.1f} / {cfg.fuel_capacity_l:.0f} L", env.fuel / cfg.fuel_capacity_l, (240, 200, 60))
        bar(f"Budget  ${env.budget:,.0f}", env.budget / cfg.start_budget, (90, 200, 120))
        text(f"Kits: minor {env.minor_kits}/{cfg.minor_kit_capacity}   major {env.major_kits}/{cfg.major_kit_capacity}", dy=21)
        text(f"Location: {env.location_name()}", dy=25)
        text("Last action", self.font_s, (150, 160, 175), 17)
        act = env.action_name(env.last_action) if env.last_action is not None else "-"
        text(act, self.font_b, (255, 255, 255), 24)
        for line in _wrap(env.last_event, 40):
            text(line, self.font_s, (200, 205, 215), 16)
        y += 4
        rc = (120, 230, 140) if env.last_reward >= 0 else (255, 130, 120)
        text(f"Step reward   {env.last_reward:+7.2f}", color=rc, dy=21)
        text(f"Episode return {env.episode_return:+8.1f}", dy=25)
        bar(f"Population with water  {info['pct_population_served']:.0f}%", info["pct_population_served"] / 100, (80, 160, 255))
        text(
            f"OK {info['n_functional']}   repair {info['n_needs_repair']}   broken {info['n_non_functional']}"
            f"    repairs {env.repairs_done}",
            self.font_s,
            dy=22,
        )
        self._minimap(s, y + 4)
        return s

    def _minimap(self, s, top):
        env = self.env
        size = min(HUD_W - 32, HEIGHT - top - 92)
        if size < 80:
            return
        left = 16
        pygame.draw.rect(s, (52, 66, 48), (left, top, size, size), border_radius=6)
        m = env.cfg.map_size_km

        def to_px(xy):
            return int(left + 8 + (xy[0] / m) * (size - 16)), int(top + 8 + (xy[1] / m) * (size - 16))

        depot = to_px(env.coords[env.n])
        for k in range(env.n):
            pygame.draw.line(s, (120, 100, 70), depot, to_px(env.coords[k]), 1)
        # view direction indicator
        pygame.draw.rect(s, (90, 140, 230), (depot[0] - 5, depot[1] - 5, 10, 10))
        for k in range(env.n):
            p = to_px(env.coords[k])
            r = int(4 + 6 * math.sqrt(env.pop[k] / env.pop.max()))
            c = tuple(int(255 * v) for v in STATUS_RGB[int(env.status[k])])
            kc = tuple(int(255 * v) for v in STATUS_RGB[int(env.known[k])])
            pygame.draw.circle(s, kc, p, r + 3, 2)
            pygame.draw.circle(s, c, p, r)
            if self.selected == k:
                pygame.draw.circle(s, (255, 255, 255), p, r + 7, 2)
            s.blit(self.font_s.render(f"{k}", True, (235, 235, 235)), (p[0] + r + 3, p[1] - 8))
        # truck
        tx = self.truck_pos[0] / self.scale + m / 2
        tz = self.truck_pos[2] / self.scale + m / 2
        tp = to_px((tx, tz))
        pygame.draw.circle(s, (255, 255, 255), tp, 5)
        pygame.draw.circle(s, (30, 110, 230), tp, 3)
        s.blit(self.font_s.render("map: fill = true status, ring = known", True, (170, 180, 195)), (left, top + size + 4))
        y = top + size + 24
        for st in (FUNCTIONAL, NEEDS_REPAIR, NON_FUNCTIONAL):
            c = tuple(int(255 * v) for v in STATUS_RGB[st])
            pygame.draw.rect(s, c, (left, y + 3, 12, 12), border_radius=3)
            s.blit(self.font_s.render(STATUS_NAMES[st], True, (210, 215, 225)), (left + 18, y))
            y += 18
        s.blit(self.font_s.render("diamond = crew's belief (greys with age)", True, (170, 180, 195)), (left, y))

    def _details_surface(self) -> pygame.Surface | None:
        k = self.selected
        if k is None:
            return None
        env, p = self.env, self.env.points[k]
        w, h = 300, 262
        s = pygame.Surface((w, h), pygame.SRCALPHA)
        s.fill((20, 28, 40, 225))
        pygame.draw.rect(s, (120, 190, 255), (0, 0, w, h), 2, border_radius=6)
        rows = [
            ("Water point", f"WP{k:02d}"),
            ("Population served", f"{p.population:,}"),
            ("Failure risk score", f"{p.risk:.2f}"),
            ("Pump age", f"{p.age_years} years"),
            ("Extraction type", p.extraction_type),
            ("Payment type", p.payment_type),
            ("TRUE status", STATUS_NAMES[int(env.status[k])]),
            ("Crew believes", STATUS_NAMES[int(env.known[k])]),
            ("Last observed", f"{int(env.days_since[k])} days ago"),
            ("Distance from crew", f"{env.dist[env.loc, k]:.1f} km"),
            ("Fuel to reach", f"{env.dist[env.loc, k] * env.cfg.fuel_per_km_l:.1f} L"),
        ]
        y = 10
        s.blit(self.font_sb.render("Selected water point  (TAB = next)", True, (120, 190, 255)), (12, y))
        y += 24
        for i, (a, b) in enumerate(rows):
            colour = (235, 240, 245)
            if i in (6, 7):
                st = int(env.status[k] if i == 6 else env.known[k])
                colour = tuple(int(255 * v) for v in STATUS_RGB[st])
            s.blit(self.font_s.render(a, True, (160, 170, 185)), (12, y))
            s.blit(self.font_sb.render(b, True, colour), (150, y - 1))
            y += 20
        return s

    def _help_surface(self) -> pygame.Surface:
        lines = (
            ["left-drag: orbit   right-drag / WASD: pan   scroll / Z X: zoom   Q E: rotate   T G: tilt",
             "click: inspect a water point   TAB: next point   F: follow truck   R: reset view",
             "SPACE: pause   + / -: speed   H: hide help   ESC: quit"]
            if self.show_help
            else ["H: show controls"]
        )
        w = max(self.font_s.size(line)[0] for line in lines) + 20
        s = pygame.Surface((w, 18 * len(lines) + 10), pygame.SRCALPHA)
        s.fill((20, 28, 40, 170))
        for i, line in enumerate(lines):
            s.blit(self.font_s.render(line, True, (225, 230, 240)), (10, 5 + 18 * i))
        return s

    def _blit(self, surf: pygame.Surface, x: int, y_top: int):
        """Draw a pygame surface at window position (x, y_top) with y measured from the top."""
        data = pygame.image.tostring(surf, "RGBA", True)
        glWindowPos2d(x, HEIGHT - y_top - surf.get_height())
        glDrawPixels(surf.get_width(), surf.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, data)

    def _draw_labels(self):
        env = self.env
        for k in range(env.n + 1):
            x, z = self.site_xz[k]
            s = self._project((x, self.height(x, z) + (3.0 if k < env.n else 2.3), z))
            if s is None:
                continue
            label = f"WP{k:02d}  {env.points[k].population:,} people" if k < env.n else "DEPOT"
            surf = self.font_s.render(label, True, (20, 25, 35))
            bg = pygame.Surface((surf.get_width() + 8, surf.get_height() + 2), pygame.SRCALPHA)
            bg.fill((255, 255, 255, 200 if k != self.selected else 255))
            bg.blit(surf, (4, 1))
            px = int(s[0] - bg.get_width() / 2)
            if px < HUD_W or px + bg.get_width() > WIDTH:
                continue
            data = pygame.image.tostring(bg, "RGBA", True)
            glWindowPos2d(px, s[1])
            glDrawPixels(bg.get_width(), bg.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, data)

    def _overlays(self):
        glViewport(0, 0, WIDTH, HEIGHT)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glDisable(GL_FOG)
        self._draw_labels()
        self._blit(self._hud_surface(), 0, 0)
        details = self._details_surface()
        if details is not None:
            self._blit(details, WIDTH - details.get_width() - 12, 12)
        if self.mode == "human":
            help_s = self._help_surface()
            self._blit(help_s, HUD_W + 12, HEIGHT - help_s.get_height() - 12)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

    # ================================================================ events / frame
    def _handle_events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                self.close()
                raise SystemExit("Render window closed by user")
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif e.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    self.speed = min(8.0, self.speed * 2)
                elif e.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self.speed = max(0.25, self.speed / 2)
                elif e.key == pygame.K_f:
                    self.follow = not self.follow
                elif e.key == pygame.K_r:
                    self.reset_camera()
                    self.follow = False
                elif e.key == pygame.K_h:
                    self.show_help = not self.show_help
                elif e.key == pygame.K_TAB:
                    self.selected = 0 if self.selected is None else (self.selected + 1) % self.env.n
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button in (1, 3):
                self._drag = (e.button, e.pos)
                self._drag_moved = 0
            elif e.type == pygame.MOUSEBUTTONUP and e.button in (1, 3):
                if e.button == 1 and self._drag_moved < 5 and e.pos[0] > HUD_W:
                    self._pick(*e.pos)
                self._drag = None
            elif e.type == pygame.MOUSEMOTION and self._drag is not None:
                dx, dy = e.rel
                self._drag_moved += abs(dx) + abs(dy)
                if self._drag[0] == 1:
                    self.cam_yaw -= dx * 0.35
                    self.cam_pitch = float(np.clip(self.cam_pitch + dy * 0.3, 8, 88))
                else:
                    self._pan(dx, dy)
            elif e.type == pygame.MOUSEWHEEL:
                self.cam_dist = float(np.clip(self.cam_dist * (0.9 ** e.y), 3.0, 70.0))
        keys = pygame.key.get_pressed()
        if keys[pygame.K_q]:
            self.cam_yaw += 1.5
        if keys[pygame.K_e]:
            self.cam_yaw -= 1.5
        if keys[pygame.K_t]:
            self.cam_pitch = min(88.0, self.cam_pitch + 0.8)
        if keys[pygame.K_g]:
            self.cam_pitch = max(8.0, self.cam_pitch - 0.8)
        if keys[pygame.K_z]:
            self.cam_dist = max(3.0, self.cam_dist * 0.98)
        if keys[pygame.K_x]:
            self.cam_dist = min(70.0, self.cam_dist * 1.02)
        pan = 6
        if keys[pygame.K_w]:
            self._pan(0, pan)
        if keys[pygame.K_s]:
            self._pan(0, -pan)
        if keys[pygame.K_a]:
            self._pan(pan, 0)
        if keys[pygame.K_d]:
            self._pan(-pan, 0)

    def _frame(self, label, pulse, dt):
        self.t += dt
        self._update_particles(dt)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self._set_camera()
        glCallList(self.static_list)
        self._draw_flag()
        self._draw_route()
        for k in range(self.env.n):
            self._draw_water_point(k, label if k == self.env.loc else None, pulse)
        self._draw_truck()
        self._draw_particles()
        self._overlays()

    def _effects(self, label, f):
        """Spawn particles for the current action."""
        env = self.env
        if env.loc < env.n and label in ("minor repair", "major repair") and self.mode == "human":
            x, z = self.site_xz[env.loc]
            c = ACTION_RGB[label]
            for _ in range(3):
                v = np.random.uniform(-1, 1, 3) * np.array([1.2, 0, 1.2]) + np.array([0, np.random.uniform(1.5, 3), 0])
                self._spawn((x, self.site_height(env.loc) + 1.3, z), v, 0.8, c, 4)
        if env.loc == env.n and label == "return to depot" and f > 0.7 and self.mode == "human":
            x, y, z = self.truck_pos
            self._spawn((x, y + 0.5, z), (np.random.uniform(-0.3, 0.3), 1.2, np.random.uniform(-0.3, 0.3)), 0.7, (1.0, 0.85, 0.2), 5)

    def draw(self):
        env = self.env
        start = self.truck_pos.copy()
        end = self._world(env.coords[env.loc])
        # park the truck beside the tank / in the depot yard
        end = end + np.array([0.75, 0.0, 0.55]) if env.loc < env.n else end + np.array([0.3, 0.0, 0.9])
        end[1] = self.height(end[0], end[2])
        dist = float(np.linalg.norm((end - start)[[0, 2]]))
        moved = dist > 1e-3
        if moved:
            d = end - start
            self.truck_heading = -math.degrees(math.atan2(d[2], d[0]))
        if not self.route:
            self.route.append(start.copy())
        label = env.action_name(env.last_action) if env.last_action is not None else None

        if self.mode != "human":
            self.truck_pos = end
            if moved:
                self.route.append(end.copy())
            self._frame(label, 0.5, 1 / self.fps)
            glReadBuffer(GL_BACK)
            px = glReadPixels(0, 0, WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE)
            return np.flipud(np.frombuffer(px, dtype=np.uint8).reshape(HEIGHT, WIDTH, 3)).copy()

        duration = (0.35 + 0.06 * dist) if moved else 0.45
        elapsed = 0.0
        while True:
            self._handle_events()
            dt = 1.0 / self.fps
            if not self.paused:
                elapsed += dt * self.speed
            f = min(elapsed / duration, 1.0)
            ease = f * f * (3 - 2 * f)
            prev = self.truck_pos.copy()
            if moved:
                p = start + (end - start) * ease
                p[1] = self.height(p[0], p[2])
                self.truck_pos = p
                step = float(np.linalg.norm(p - prev))
                self.wheel_angle = (self.wheel_angle + step * 520) % 360
                if step > 1e-3 and np.random.random() < 0.8:  # dust trail
                    back = prev + np.random.uniform(-0.1, 0.1, 3)
                    self._spawn((back[0], back[1] + 0.12, back[2]), (np.random.uniform(-0.2, 0.2), 0.35, np.random.uniform(-0.2, 0.2)), 1.2, (0.78, 0.66, 0.48), 7)
            else:
                self.truck_pos = end
            if not self.paused:
                self._effects(label, f)
            self._frame(label, (self.t * 1.2) % 1.0, dt * (0 if self.paused else self.speed))
            pygame.display.flip()
            self.clock.tick(self.fps)
            if f >= 1.0 and not self.paused:
                break
        self.truck_pos = end
        if moved:
            self.route.append(end.copy())
        return None

    def close(self):
        try:
            pygame.display.quit()
            pygame.quit()
        except Exception:
            pass


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines[:3]
