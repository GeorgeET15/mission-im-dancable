"""Screens for the dance-arcade UI: home, select, ready, play, results, settings."""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pygame

from ..audio import ensure_audio_wav
from ..beats import load_or_detect_beats
from ..camera import CameraError, open_camera
from ..config import AppConfig
from ..dance import build_dance
from ..editor import draft_moves, suggest_boundaries
from ..game import GamePhase, GameSession, PlaySession, grade_for, judge, Judgement
from ..geometry import SKELETON_EDGES
from ..moves import default_moves, load_dance_file
from ..persist import add_score, load_scores, load_settings, save_settings
from ..playback import PlaybackClock
from ..pose import PoseTracker
from ..reference import (
    ReferenceChoreography,
    ReferenceVideoError,
    load_or_extract,
    pose_at,
    window_indices,
)
from ..render import fit_keypoints
from ..scoring import (
    CorrectionTracker,
    ScoreSmoother,
    analyze_pose,
    best_window_score,
    describe_angle_correction,
    describe_correction,
    joint_angle_deg,
    mirror_keypoints,
    worst_mismatch,
)
from . import theme
from .nav import Navigator
from .widgets import Button, FocusGroup, ProgressBar, Stepper, TextField, TimelineBar, Toggle

W, H = 1280, 720
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_DETECTED = 4


class AppContext:
    """Shared mutable state passed to every screen."""

    def __init__(self, cfg: AppConfig, fonts: dict, manager: Navigator) -> None:
        self.cfg = cfg
        self.fonts = fonts
        self.manager = manager
        self.settings = load_settings()
        self.mirror = bool(self.settings.get("mirror", cfg.mirror))
        self.mode = "play"  # "play" (scored) or "learn" (free study)
        self.tracker: PoseTracker | None = None
        self.device = "cpu"
        self.video_path: Path | None = cfg.source
        self.moves_path: Path | None = cfg.moves_path
        self.choreo: ReferenceChoreography | None = None
        self.moves = None
        self.dance = None  # dance.Dance — metadata for the selected video
        self.beats = None  # BeatGrid set after analyze; None = no beat sync
        # Shared webcam capture: ReadyScreen opens it in the background,
        # PlayScreen consumes it, menus release it. One open per session.
        self.cap = None
        self.auto_start = False  # ReadyScreen's countdown finished → play now
        self.fullscreen = bool(self.settings.get("fullscreen", False))
        self.music = None        # audio.Music, created by the app loop
        self.audio_path: Path | None = None  # cached WAV of the picked video
        self.browse_result: Path | None = None  # set by BrowseScreen picks
        self.last_session = None
        self.last_results = None
        self.t = 0.0
        self.particles = theme.Particles()
        self.quit = False

    def window_s(self) -> float:
        """Match window half-width; user setting wins over the CLI default."""
        try:
            return float(self.settings.get("window_s", self.cfg.window_s))
        except (TypeError, ValueError):
            return self.cfg.window_s


def close_camera(ctx) -> None:
    """Release the shared webcam capture if one is open (None ctx = no-op)."""
    if ctx is None:
        return
    cap, ctx.cap = ctx.cap, None
    if cap is not None:
        cap.release()


class Screen:
    def enter(self) -> None: ...
    def exit(self) -> None: ...
    def handle_event(self, ctx, ev) -> None: ...
    def update(self, ctx, dt) -> None: ...
    def draw(self, ctx, surf) -> None: ...


_glow_cache = {}


def _glow_layer(pygame, size):
    """One reusable translucent scratch surface per size (avoids a fresh
    1280x720 allocation for every skeleton, every frame)."""
    layer = _glow_cache.get(size)
    if layer is None:
        layer = pygame.Surface(size, pygame.SRCALPHA)
        _glow_cache[size] = layer
    layer.fill((0, 0, 0, 0))
    return layer


def _bg(pygame, ctx, surf, t: float) -> None:
    surf.blit(ctx.bg, (0, 0))
    theme.draw_dance_floor(pygame, surf, t, horizon=int(H * 0.62))
    theme.draw_spotlights(pygame, surf, t)
    theme.draw_equalizer(pygame, surf, t, (0, H - 90, W, 60), bars=48)


def _title(pygame, ctx, surf, t: float, y: int = 90) -> None:
    img = theme.glow_text(pygame, ctx.fonts["title"], "MISSION I'M-DANCABLE", theme.PINK)
    surf.blit(img, img.get_rect(center=(W // 2, y)))
    tag = ctx.fonts["sm"].render("your mission, should you choose to accept it.", True, theme.DIM)
    surf.blit(tag, tag.get_rect(center=(W // 2, y + 52)))


def _dim_strip(pygame, surf, rect) -> None:
    """Soft dark band behind footer text so equalizer bars don't eat it."""
    strip = pygame.Surface((rect[2], rect[3]), pygame.SRCALPHA)
    strip.fill((8, 4, 18, 170))
    surf.blit(strip, rect[:2])


def _panel_label(pygame, ctx, surf, text: str, x: int, y: int, color=theme.DIM) -> None:
    img = ctx.fonts["sm"].render(text, True, color)
    surf.blit(img, (x, y))


def _chip(pygame, ctx, surf, text: str, center, fg, bg=(16, 10, 30, 220)) -> None:
    """Small rounded status pill (tracking state, camera status)."""
    img = ctx.fonts["sm"].render(text, True, fg)
    w, h = img.get_width() + 28, img.get_height() + 12
    rect = pygame.Rect(0, 0, w, h)
    rect.center = center
    box = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.rect(box, bg, box.get_rect(), border_radius=h // 2)
    pygame.draw.rect(box, fg, box.get_rect(), 1, border_radius=h // 2)
    surf.blit(box, rect)
    surf.blit(img, img.get_rect(center=rect.center))


def _torso_scale(k: np.ndarray, conf: float) -> float | None:
    """Shoulder-to-hip distance — sizing reference for decorative parts."""
    if all(k[i, 2] >= conf for i in (5, 6, 11, 12)):
        sh = ((k[5, 0] + k[6, 0]) / 2, (k[5, 1] + k[6, 1]) / 2)
        hip = ((k[11, 0] + k[12, 0]) / 2, (k[11, 1] + k[12, 1]) / 2)
        return math.hypot(sh[0] - hip[0], sh[1] - hip[1])
    return None


def _head_circle(k: np.ndarray, conf: float, torso: float | None):
    """(cx, cy, r) head ring from face keypoints, or None when no face."""
    ear_l, ear_r = k[3], k[4]
    if ear_l[2] >= conf and ear_r[2] >= conf:
        r = max(math.hypot(ear_l[0] - ear_r[0], ear_l[1] - ear_r[1]) * 0.58, 8.0)
        return (ear_l[0] + ear_r[0]) / 2, (ear_l[1] + ear_r[1]) / 2 - r * 0.3, r
    pts = [p for p in k[:5] if p[2] >= conf]
    if not pts:
        return None
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    return cx, cy - (torso or 90.0) * 0.1, (torso or 90.0) * 0.22


def _neon_skeleton(pygame, surf, keypoints: np.ndarray, conf: float, color=theme.CYAN) -> None:
    glow = _glow_layer(pygame, surf.get_size())
    torso = _torso_scale(keypoints, conf)
    head = _head_circle(keypoints, conf, torso)
    if head is not None:
        hx, hy, hr = int(head[0]), int(head[1]), int(head[2])
        pygame.draw.circle(glow, (*color, 50), (hx, hy), hr)
        pygame.draw.circle(surf, color, (hx, hy), hr, 3)
    for a, b in SKELETON_EDGES:
        pa, pb = keypoints[a], keypoints[b]
        if pa[2] >= conf and pb[2] >= conf:
            p1, p2 = (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1]))
            pygame.draw.line(glow, (*color, 70), p1, p2, 7)
            pygame.draw.line(surf, color, p1, p2, 2)
    for x, y, c in keypoints:
        if c >= conf:
            cpt = (int(x), int(y))
            pygame.draw.circle(glow, (*color, 70), cpt, 7)
            pygame.draw.circle(surf, theme.PINK, cpt, 4)
    # fists at the wrists, shoes at the ankles — reads as a dancer silhouette
    fist_r = max(int((torso or 90.0) * 0.07), 7)
    for i in (9, 10):
        if keypoints[i, 2] >= conf:
            cpt = (int(keypoints[i, 0]), int(keypoints[i, 1]))
            pygame.draw.circle(glow, (*theme.PINK, 90), cpt, fist_r + 4)
            pygame.draw.circle(surf, theme.PINK, cpt, fist_r)
            pygame.draw.circle(surf, color, cpt, fist_r, 2)
    shoe_w = max(int((torso or 90.0) * 0.24), 14)
    shoe_h = max(int(shoe_w * 0.45), 7)
    for i in (15, 16):
        if keypoints[i, 2] >= conf:
            cx, cy = int(keypoints[i, 0]), int(keypoints[i, 1])
            rect = (cx - shoe_w // 2, cy + shoe_h // 2, shoe_w, shoe_h)
            pygame.draw.ellipse(glow, (*color, 60), rect)
            pygame.draw.ellipse(surf, theme.GOLD, rect)
            pygame.draw.ellipse(surf, color, rect, 2)
    surf.blit(glow, (0, 0))


def _frame_surf(pygame, frame: np.ndarray, w: int, h: int):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return pygame.transform.scale(
        pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2))), (w, h)
    )


def _detect_videos(directory: Path) -> list[Path]:
    try:
        return sorted(
            p for p in directory.iterdir() if p.suffix.lower() in VIDEO_EXTS
        )
    except OSError:
        return []


def _probe_video(path: Path) -> tuple[float, int, int] | None:
    """(duration_s, width, height) or None when unreadable."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    if frames <= 0:
        return None
    return frames / fps, int(w), int(h)


def _pick_hint(analysis, live, reference):
    wa = analysis.worst_angle()
    wk = worst_mismatch(analysis.pose_score)
    if wa and wa[1] / 90.0 >= (analysis.pose_score.keypoint_errors[wk] if wk is not None else 0.0):
        ld, rd = joint_angle_deg(live, wa[0]), joint_angle_deg(reference, wa[0])
        if ld is not None and rd is not None:
            return describe_angle_correction(wa[0], ld, rd)
    if wk is not None:
        return describe_correction(wk, live, reference)
    return None


def eval_against_reference(ctx, choreo, t: float, live_keypoints):
    """Score one live pose against the reference window around time t.

    Returns (accuracy, reliable, hint, matched_ref_row). Both Learn and Play
    screens call this so the comparison rule is identical in the two modes.
    """
    rows = choreo.keypoints[window_indices(choreo, t, ctx.window_s())]
    if ctx.mirror and rows.size:
        rows = np.stack([mirror_keypoints(r, choreo.width) for r in rows])
    if live_keypoints is None or not rows.size:
        return None, False, None, None
    best = best_window_score(live_keypoints, rows, ctx.cfg.confidence)
    if best is None:
        return None, False, None, None
    analysis = analyze_pose(live_keypoints, rows[best[1]], ctx.cfg.confidence)
    if analysis is None:
        return None, False, None, None
    hint = _pick_hint(analysis, live_keypoints, rows[best[1]])
    return analysis.pose_score.score, True, hint, rows[best[1]]


def _open_camera_ctx(ctx) -> None:
    """Open the shared camera into ctx.cap (runs on a worker thread)."""
    cam_cfg = AppConfig(
        camera_index=int(ctx.settings.get("camera_index", ctx.cfg.camera_index)),
        frame_width=ctx.cfg.frame_width,
        frame_height=ctx.cfg.frame_height,
    )
    ctx.cap = open_camera(cam_cfg)


def _best_score_for(video_name: str, scores: list[dict]) -> dict | None:
    best = None
    for s in scores:
        if Path(s.get("video", "")).name == video_name and (
            best is None or s.get("score", 0) > best.get("score", 0)
        ):
            best = s
    return best


# --- HOME ------------------------------------------------------------------


class HomeScreen(Screen):
    LABELS = [("LEARN", "learn", True), ("PLAY", "play", True),
              ("HIGH SCORES", "scores", False), ("SETTINGS", "settings", False),
              ("QUIT", "quit", False)]

    def enter(self) -> None:
        self.buttons = [
            Button(pygame.Rect(W // 2 - 160, 280 + i * 78, 320, 56), lbl, act, prim)
            for i, (lbl, act, prim) in enumerate(self.LABELS)
        ]
        self.group = FocusGroup(self.buttons)
        self.confirm_quit = False
        self.last_line = ""
        self._need_close = True  # release a leftover webcam next update

    def _activate(self, ctx, action: str) -> None:
        if action in ("learn", "play"):
            ctx.mode = action
            ctx.manager.push(SelectScreen())
        elif action == "scores":
            ctx.manager.push(ScoresScreen())
        elif action == "settings":
            ctx.manager.push(SettingsScreen())
        elif action == "quit":
            self.confirm_quit = True

    def handle_event(self, ctx, ev) -> None:
        self._ctx = ctx
        if self.confirm_quit:
            if ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_RETURN, pygame.K_SPACE, pygame.K_q):
                    ctx.quit = True
                elif ev.key in (pygame.K_ESCAPE, pygame.K_BACKSPACE):
                    self.confirm_quit = False
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                self.confirm_quit = False
            return
        if ev.type == pygame.MOUSEMOTION:
            self.group.hover(ev.pos)
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            for b in self.buttons:
                if b.contains(ev.pos) and b.enabled:
                    b.press()
                    self._activate(ctx, b.action)
        elif ev.type == pygame.KEYDOWN:
            if ev.key in (pygame.K_DOWN, pygame.K_s):
                self.group.move(1)
            elif ev.key in (pygame.K_UP, pygame.K_w):
                self.group.move(-1)
            elif ev.key in (pygame.K_RETURN, pygame.K_SPACE):
                b = self.group.current
                if b:
                    b.press()
                    self._activate(ctx, b.action)
            elif ev.key == pygame.K_ESCAPE:
                self.confirm_quit = True

    def update(self, ctx, dt) -> None:
        self._ctx = ctx
        if self._need_close:
            self._need_close = False
            close_camera(ctx)
        for b in self.buttons:
            b.update(dt)
        if not self.last_line:
            video = ctx.settings.get("last_video", "")
            scores = load_scores()
            best = _best_score_for(Path(video).name, scores) if video else None
            if best:
                self.last_line = (
                    f"last mission: {Path(video).name} · best {best['score']} pts "
                    f"(grade {best['grade']})"
                )
            elif scores:
                top = scores[0]
                self.last_line = (
                    f"best run so far: {top['score']} pts "
                    f"(grade {top['grade']}) on {Path(top['video']).name}"
                )

    def draw(self, ctx, surf) -> None:
        _bg(pygame, ctx, surf, ctx.t)
        _title(pygame, ctx, surf, ctx.t)
        if self.last_line:
            info = ctx.fonts["sm"].render(self.last_line, True, theme.GOLD)
            surf.blit(info, info.get_rect(center=(W // 2, 244)))
        for b in self.buttons:
            b.draw(pygame, surf, ctx.fonts["med"], ctx.t)
        _dim_strip(pygame, surf, (W // 2 - 220, H - 44, 440, 34))
        hint = ctx.fonts["sm"].render("arrows + ENTER · or click · ESC quit", True, theme.DIM)
        surf.blit(hint, hint.get_rect(center=(W // 2, H - 26)))
        if self.confirm_quit:
            veil = pygame.Surface((W, H), pygame.SRCALPHA)
            veil.fill((8, 4, 18, 210))
            surf.blit(veil, (0, 0))
            img = theme.glow_text(pygame, ctx.fonts["big"], "QUIT MISSION?", theme.PINK)
            surf.blit(img, img.get_rect(center=(W // 2, 320)))
            t3 = ctx.fonts["med"].render("ENTER yes · ESC stay", True, theme.GOLD)
            surf.blit(t3, t3.get_rect(center=(W // 2, 400)))


# --- DANCE SELECT -----------------------------------------------------------


class SelectScreen(Screen):
    """Pick a local video, analyze it (background thread), start dancing."""

    def enter(self) -> None:
        self.field = TextField(pygame.Rect(60, 190, 760, 44), placeholder="type a video path, or click a dance below")
        self.btn_analyze = Button(pygame.Rect(840, 188, 150, 48), "ANALYZE", "analyze", True)
        self.btn_start = Button(pygame.Rect(1000, 188, 240, 48), "START DANCING", "start", True, enabled=False)
        self.tgl_mirror = Toggle(pygame.Rect(60, 260, 460, 30), "mirror choreography", "mirror", False)
        self.btn_back = Button(pygame.Rect(60, 645, 160, 48), "BACK", "back")
        self.btn_browse = Button(pygame.Rect(230, 645, 160, 48), "BROWSE…", "browse")
        self.btn_watch = Button(pygame.Rect(400, 645, 160, 48), "WATCH", "watch", enabled=False)
        self.cards: list[dict] = []
        self.status = "pick a video and press ANALYZE"
        self.status_color = theme.DIM
        self.info = ""
        self.progress = -1.0
        self.choreo = None
        self.moves = None
        self.dance = None
        self._thread = None
        self.error = ""
        self.bar = ProgressBar(pygame.Rect(60, 610, 700, 14), color=theme.PINK)
        self._synced = False
        self._scores = None

    # -- helpers ------------------------------------------------------------

    def _sync(self, ctx) -> None:
        self._ctx = ctx
        if not self._synced:
            self._synced = True
            close_camera(ctx)  # landing here ends any gameplay session
            self.tgl_mirror.value = ctx.mirror
            self.btn_start.label = "START LEARNING" if ctx.mode == "learn" else "START DANCING"
            if ctx.video_path and not self.field.text:
                self.field.text = str(ctx.video_path)
            elif not self.field.text and ctx.settings.get("last_video"):
                self.field.text = ctx.settings["last_video"]
            self._scores = load_scores()
            vids = _detect_videos(Path.cwd())[:MAX_DETECTED]
            self.cards = [self._make_card(v, i) for i, v in enumerate(vids)]
            if ctx.choreo is not None and ctx.video_path and Path(self.field.text) == ctx.video_path:
                self.choreo, self.moves, self.dance = ctx.choreo, ctx.moves, ctx.dance
                self.btn_start.enabled = True
                self.btn_watch.enabled = True
                self._fill_info(ctx.video_path)
                self._set_status("ready — press START or WATCH first", theme.GOOD)

    def _make_card(self, path: Path, i: int) -> dict:
        probe = _probe_video(path)
        dur = f"{probe[0]:.1f}s" if probe else "?s"
        best = _best_score_for(path.name, self._scores or [])
        meta = dur + (f" · best {best['score']} ({best['grade']})" if best else " · new")
        return {
            "path": path,
            "rect": pygame.Rect(60 + i * 300, 355, 290, 64),
            "name": path.name,
            "meta": meta,
            "hovered": False,
        }

    def _fill_info(self, path: Path) -> None:
        probe = _probe_video(path)
        if probe and self.choreo:
            dur, w, h = probe
            diff = f" · {self.dance.difficulty}" if self.dance and self.dance.difficulty else ""
            self.info = (
                f"{path.name} · {dur:.1f}s · {w}x{h}{diff} · "
                f"{self.choreo.detected_count()}/{self.choreo.num_samples} poses detected · "
                f"{len(self.moves) if self.moves else 0} move(s)"
            )

    def _set_status(self, msg: str, color=theme.DIM) -> None:
        self.status, self.status_color = msg, color

    def _analyze(self, ctx) -> None:
        if self._thread is not None and self._thread.is_alive():
            return  # one extraction per selection
        path = Path(self.field.text.strip()).expanduser()
        if not path.is_file():
            self._set_status(f"file not found: {path}", theme.MISS)
            self.btn_start.enabled = False
            return
        if _probe_video(path) is None:
            self._set_status("cannot decode this video — unsupported or corrupt", theme.MISS)
            self.btn_start.enabled = False
            return
        ctx.video_path = path
        ctx.moves_path = self._detect_moves(path) or ctx.cfg.moves_path
        self.choreo = None
        self.moves = None
        self.dance = None
        self.info = ""
        self.btn_start.enabled = False
        self.btn_watch.enabled = False
        self.progress = 0.0
        self._set_status("analyzing choreography…", theme.CYAN)

        def work() -> None:
            try:
                if ctx.tracker is None:
                    self._set_status("loading pose model…", theme.CYAN)
                    import torch
                    from ..config import resolve_device

                    ctx.device = resolve_device(ctx.cfg.device, torch.cuda.is_available())
                    ctx.tracker = PoseTracker(
                        ctx.cfg.model_path, ctx.device, ctx.cfg.confidence
                    )
                choreo = load_or_extract(
                    ctx.tracker, path, ctx.cfg.sample_fps, ctx.cfg.cache_dir,
                    ctx.cfg.confidence, model_name=ctx.cfg.model_path.name,
                    on_progress=lambda d, t: setattr(self, "progress", d / max(t, 1)),
                )
                moves, meta = (
                    load_dance_file(ctx.moves_path, choreo.duration)
                    if ctx.moves_path
                    else (default_moves(choreo.duration), {})
                )
            except Exception as exc:  # surface to UI thread
                self.error = str(exc)
                return
            ctx.beats = load_or_detect_beats(path, ctx.cfg.cache_dir)
            ctx.audio_path = ensure_audio_wav(path, ctx.cfg.cache_dir)
            self.choreo, self.moves = choreo, moves
            self.dance = build_dance(path, choreo, moves, meta)

        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()

    @staticmethod
    def _detect_moves(video: Path) -> Path | None:
        for cand in (video.with_suffix(".moves.json"), video.parent / f"{video.stem}.moves.json"):
            if cand.is_file():
                return cand
        return None

    def _start(self, ctx) -> None:
        if self.choreo is None or self.moves is None:
            return
        # No hand-written annotation file → draft real steps from the
        # video's own beat grid / pose motion so Learn navigation and
        # Play move results get segments instead of one giant blob.
        # These stay in-memory suggestions until saved in the editor.
        if ctx.moves_path is None and len(self.moves) <= 1:
            bounds = suggest_boundaries(ctx.choreo, ctx.beats, self.choreo.duration)
            if bounds:
                self.moves = draft_moves(self.choreo.duration, bounds)
                self.dance = build_dance(ctx.video_path, self.choreo, self.moves, {})
        ctx.choreo, ctx.moves, ctx.dance = self.choreo, self.moves, self.dance
        ctx.mirror = self.tgl_mirror.value
        ctx.settings["last_video"] = str(ctx.video_path)
        ctx.settings["mirror"] = ctx.mirror
        save_settings(ctx.settings)
        if ctx.mode == "learn":
            from .learn import LearnScreen

            ctx.manager.push(LearnScreen())
        else:
            ctx.manager.push(ReadyScreen())

    def _back(self, ctx) -> None:
        if ctx.manager.depth > 1:
            ctx.manager.pop()
        else:
            ctx.manager.replace(HomeScreen())

    # -- events -------------------------------------------------------------

    def handle_event(self, ctx, ev) -> None:
        self._ctx = ctx
        widgets = [self.field, self.btn_analyze, self.btn_start, self.btn_back,
                   self.btn_browse, self.btn_watch, self.tgl_mirror]
        if ev.type == pygame.MOUSEMOTION:
            for wdgt in widgets:
                if hasattr(wdgt, "hovered"):
                    wdgt.hovered = wdgt.contains(ev.pos)
            for c in self.cards:
                c["hovered"] = c["rect"].collidepoint(ev.pos)
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            self.field.focused = self.field.contains(ev.pos)
            for c in self.cards:
                if c["rect"].collidepoint(ev.pos):
                    self.field.text = str(c["path"])
                    self.field.focused = False
                    self._analyze(ctx)
                    return
            for wdgt in widgets:
                if isinstance(wdgt, TextField):
                    continue
                if wdgt.contains(ev.pos):
                    self._activate(ctx, wdgt.action)
        elif ev.type == pygame.KEYDOWN:
            if self.field.focused:
                if ev.key == pygame.K_RETURN:
                    self.field.focused = False
                    self._analyze(ctx)
                else:
                    self.field.handle_key(ev.key, ev.unicode)
            elif ev.key == pygame.K_ESCAPE:
                self._back(ctx)
            elif ev.key == pygame.K_RETURN and self.btn_start.enabled:
                self._start(ctx)

    def _activate(self, ctx, action: str) -> None:
        if action == "analyze":
            self._analyze(ctx)
        elif action == "start":
            self._start(ctx)
        elif action == "back":
            self._back(ctx)
        elif action == "browse":
            self._browse(ctx)
        elif action == "watch":
            ctx.choreo, ctx.moves, ctx.dance = self.choreo, self.moves, self.dance
            ctx.mirror = self.tgl_mirror.value
            ctx.manager.push(WatchScreen())
        elif action == "mirror":
            self.tgl_mirror.value = not self.tgl_mirror.value
            ctx.mirror = self.tgl_mirror.value

    def _browse(self, ctx) -> None:
        ctx.manager.push(BrowseScreen())

    def update(self, ctx, dt) -> None:
        picked = getattr(ctx, "browse_result", None)
        if picked is not None:
            ctx.browse_result = None
            self.field.text = str(picked)
            self.field.focused = False
            self._analyze(ctx)
        self._sync(ctx)
        for wdgt in (self.btn_analyze, self.btn_start, self.btn_back,
                     self.btn_browse, self.btn_watch):
            wdgt.update(dt)
        if self.error:
            self._set_status(f"error: {self.error}", theme.MISS)
            self.progress, self.error = -1.0, ""
        if self.choreo is not None and self.progress >= 0:
            self.progress = -1.0
            n = self.choreo.detected_count()
            if n == 0:
                self._set_status("no person detected in the video — try another clip", theme.MISS)
                self.btn_start.enabled = False
            else:
                self.btn_start.enabled = True
                self.btn_watch.enabled = True
                self._set_status("ready with reference skeleton — START DANCING or WATCH first", theme.GOOD)
                self._fill_info(ctx.video_path)

    # -- draw ---------------------------------------------------------------

    def _draw_card(self, ctx, surf, card: dict) -> None:
        selected = self.field.text.strip() in (str(card["path"]), card["name"])
        edge = theme.CYAN if selected else (theme.VIOLET if card["hovered"] else theme.PANEL_EDGE)
        pygame.draw.rect(surf, theme.PANEL, card["rect"], border_radius=8)
        pygame.draw.rect(surf, edge, card["rect"], 2 if selected or card["hovered"] else 1,
                         border_radius=8)
        name = ctx.fonts["sm"].render(card["name"][:30], True,
                                      theme.TEXT if selected else theme.DIM)
        surf.blit(name, (card["rect"].x + 12, card["rect"].y + 10))
        meta = ctx.fonts["sm"].render(card["meta"], True, theme.DIM)
        surf.blit(meta, (card["rect"].x + 12, card["rect"].y + 36))

    def draw(self, ctx, surf) -> None:
        _bg(pygame, ctx, surf, ctx.t)
        _title(pygame, ctx, surf, ctx.t, y=48)
        f_sm, f_med = ctx.fonts["sm"], ctx.fonts["med"]
        head = f_med.render(
            "CHOOSE YOUR DANCE" + (" — LEARN" if ctx.mode == "learn" else " — PLAY"),
            True, theme.CYAN,
        )
        surf.blit(head, (60, 150))
        self.field.draw(pygame, surf, f_sm, ctx.t)
        self.btn_analyze.draw(pygame, surf, f_med, ctx.t)
        self.btn_start.draw(pygame, surf, f_med, ctx.t)
        self.tgl_mirror.draw(pygame, surf, f_sm)
        surf.blit(f_sm.render("detected videos — click to analyze:", True, theme.DIM), (60, 332))
        for c in self.cards:
            self._draw_card(ctx, surf, c)
        if not self.cards:
            surf.blit(
                f_sm.render("no .mp4/.mov/.mkv/.webm found next to the game — type a path above", True, theme.DIM),
                (60, 375))
        if self.info:
            surf.blit(f_sm.render(self.info, True, theme.TEXT), (60, 450))
        surf.blit(f_sm.render(self.status, True, self.status_color), (60, 580))
        if 0 <= self.progress <= 1:
            self.bar.draw(pygame, surf, self.progress, dt=0.016)
        if self.choreo is not None:
            i = next(
                (k for k in range(self.choreo.num_samples)
                 if self.choreo.keypoints[k][:, 2].max() > 0),
                None,
            )
            if i is not None:
                rect = (1050, 355, 180, 220)
                pygame.draw.rect(surf, theme.PANEL, rect, border_radius=8)
                pygame.draw.rect(surf, theme.PANEL_EDGE, rect, 1, border_radius=8)
                fitted = fit_keypoints(self.choreo.keypoints[i], self.choreo.width,
                                       self.choreo.height, *rect)
                _neon_skeleton(pygame, surf, fitted, ctx.cfg.confidence)
                surf.blit(f_sm.render("first pose", True, theme.DIM), (1050, 580))
        self.btn_back.draw(pygame, surf, f_med, ctx.t)
        self.btn_browse.draw(pygame, surf, f_med, ctx.t)
        self.btn_watch.draw(pygame, surf, f_med, ctx.t)


# --- FILE BROWSER ------------------------------------------------------------


class BrowseScreen(Screen):
    """In-game file browser: pick a dance video from anywhere on disk.

    Left pane is the directory list (dirs + videos); the right pane shows a
    preview frame grabbed from the highlighted video plus its duration,
    resolution, and readiness badges (existing segment file / cached poses)
    so you can tell a usable dance clip from a random file before picking.
    """

    ROWS = 14
    ROW_H = 30
    LIST_X, LIST_Y, LIST_W = 60, 200, 620
    PANE = pygame.Rect(700, 190, 520, 430)
    THUMB = pygame.Rect(716, 206, 488, 275)

    def enter(self) -> None:
        self.cwd: Path | None = None  # resolved on first update (needs ctx)
        self.home_cwd: Path | None = None  # REPO button returns here
        self.entries: list[tuple[str, str, Path]] = []
        self.sel = 0
        self.offset = 0
        self.err = ""
        self._meta: dict[Path, dict] = {}    # video info probe cache
        self._thumbs: dict[Path, object] = {}  # thumbnail surface cache
        self._thumb_sel = -1                  # sel the pending thumb belongs to
        self._thumb_wait = 0.0                # debounce so scrolling stays fast
        self.btn_up = Button(pygame.Rect(60, 645, 90, 48), "UP", "up")
        self.btn_home = Button(pygame.Rect(160, 645, 110, 48), "HOME", "home")
        self.btn_repo = Button(pygame.Rect(280, 645, 110, 48), "REPO", "repo")
        self.btn_cancel = Button(pygame.Rect(400, 645, 140, 48), "CANCEL", "cancel")
        self._buttons = [self.btn_up, self.btn_home, self.btn_repo, self.btn_cancel]

    # -- data ------------------------------------------------------------------

    def _refresh(self) -> None:
        self.entries = []
        parent = self.cwd.parent
        if parent != self.cwd:
            self.entries.append(("dir", "..  (up)", parent))
        try:
            kids = sorted(self.cwd.iterdir(), key=lambda p: p.name.lower())
        except OSError as exc:
            self.err = f"cannot read directory: {exc}"
            kids = []
        dirs = [p for p in kids if p.is_dir() and not p.name.startswith(".")]
        vids = [p for p in kids if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
        self.entries += [("dir", p.name + "/", p) for p in dirs]
        self.entries += [("vid", p.name, p) for p in vids]
        self.sel = min(self.sel, max(len(self.entries) - 1, 0))
        self._thumb_sel = -1
        self._thumb_wait = 0.15
        self._clamp_scroll()

    def _probe(self, path: Path) -> dict:
        """Cached cv2 probe of a video's duration/resolution/size.

        Never raises — a file that won't open reports only its size so the
        row still tells the player something honest.
        """
        if path in self._meta:
            return self._meta[path]
        info: dict = {"dur": 0.0, "w": 0, "h": 0, "fps": 0.0, "size": 0}
        try:
            info["size"] = path.stat().st_size
        except OSError:
            pass
        try:
            cap = cv2.VideoCapture(str(path))
            try:
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
                    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
                    info.update(
                        w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        fps=fps,
                        dur=frames / fps if fps > 0 else 0.0,
                    )
            finally:
                cap.release()
        except Exception:
            pass
        self._meta[path] = info
        return info

    def _thumb_for(self, path: Path):
        """Cached preview frame grabbed ~15% into the video (skips black
        intros); falls back to the first frame, then None."""
        if path in self._thumbs:
            return self._thumbs[path]
        surf = None
        try:
            cap = cv2.VideoCapture(str(path))
            try:
                if cap.isOpened():
                    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
                    cap.set(cv2.CAP_PROP_POS_FRAMES, max(int(n * 0.15), 0))
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok, frame = cap.read()
                    if ok and frame is not None:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        surf = _frame_surf(pygame, frame, self.THUMB.w, self.THUMB.h)
            finally:
                cap.release()
        except Exception:
            surf = None
        self._thumbs[path] = surf
        return surf

    @staticmethod
    def _fmt_dur(sec: float) -> str:
        if sec <= 0:
            return "?"
        m, s = divmod(int(sec), 60)
        return f"{m}:{s:02d}"

    @staticmethod
    def _fmt_size(n: int) -> str:
        return f"{n / 1e6:.1f}MB" if n >= 1e6 else f"{max(n // 1024, 1)}KB"

    def _clamp_scroll(self) -> None:
        if self.sel < self.offset:
            self.offset = self.sel
        elif self.sel >= self.offset + self.ROWS:
            self.offset = self.sel - self.ROWS + 1
        self.offset = max(0, min(self.offset, max(len(self.entries) - self.ROWS, 0)))

    def _activate_row(self, ctx, idx: int) -> None:
        if not (0 <= idx < len(self.entries)):
            return
        kind, _, path = self.entries[idx]
        if kind == "dir":
            self.cwd = path
            self.sel = 0
            self.err = ""
            self._refresh()
        else:
            ctx.browse_result = path
            ctx.manager.pop()

    def handle_event(self, ctx, ev) -> None:
        if ev.type == pygame.KEYDOWN:
            if ev.key in (pygame.K_ESCAPE, pygame.K_q, pygame.K_BACKSPACE):
                ctx.manager.pop()
            elif ev.key in (pygame.K_DOWN, pygame.K_s):
                self.sel = min(self.sel + 1, max(len(self.entries) - 1, 0))
                self._thumb_wait = 0.2
                self._clamp_scroll()
            elif ev.key in (pygame.K_UP, pygame.K_w):
                self.sel = max(self.sel - 1, 0)
                self._thumb_wait = 0.2
                self._clamp_scroll()
            elif ev.key in (pygame.K_RETURN, pygame.K_SPACE, pygame.K_RIGHT):
                self._activate_row(ctx, self.sel)
            elif ev.key in (pygame.K_LEFT, pygame.K_a):
                self._go_parent()
            elif ev.key == pygame.K_PAGEDOWN:
                self.sel = min(self.sel + self.ROWS, max(len(self.entries) - 1, 0))
                self._thumb_wait = 0.2
                self._clamp_scroll()
            elif ev.key == pygame.K_PAGEUP:
                self.sel = max(self.sel - self.ROWS, 0)
                self._thumb_wait = 0.2
                self._clamp_scroll()
        elif ev.type == pygame.MOUSEMOTION:
            for b in self._buttons:
                b.hovered = b.contains(ev.pos)
        elif ev.type == pygame.MOUSEBUTTONDOWN:
            if ev.button == 4:  # wheel up
                self.sel = max(self.sel - 1, 0)
                self._thumb_wait = 0.2
                self._clamp_scroll()
            elif ev.button == 5:  # wheel down
                self.sel = min(self.sel + 1, max(len(self.entries) - 1, 0))
                self._thumb_wait = 0.2
                self._clamp_scroll()
            elif ev.button == 1:
                if self.btn_up.contains(ev.pos):
                    self._go_parent()
                elif self.btn_home.contains(ev.pos):
                    self._go(Path.home())
                elif self.btn_repo.contains(ev.pos) and self.home_cwd:
                    self._go(self.home_cwd)
                elif self.btn_cancel.contains(ev.pos):
                    ctx.manager.pop()
                else:
                    lx = self.LIST_X - 10
                    if lx <= ev.pos[0] < self.LIST_X + self.LIST_W and \
                            self.LIST_Y <= ev.pos[1] < self.LIST_Y + self.ROWS * self.ROW_H:
                        idx = int((ev.pos[1] - self.LIST_Y) // self.ROW_H) + self.offset
                        if idx == self.sel:
                            self._activate_row(ctx, idx)
                        elif 0 <= idx < len(self.entries):
                            self.sel = idx  # first click selects, second opens
                            self._thumb_wait = 0.2

    def _go(self, path: Path) -> None:
        if path.is_dir():
            self.cwd = path
            self.sel = 0
            self.err = ""
            self._refresh()

    def _go_parent(self) -> None:
        parent = self.cwd.parent
        if parent != self.cwd:
            self._go(parent)

    def update(self, ctx, dt) -> None:
        if self.cwd is None:
            start = ctx.video_path.parent if ctx.video_path else Path.cwd()
            self.cwd = start if start.is_dir() else Path.cwd()
            self.home_cwd = self.cwd
            self._refresh()
        # debounced thumbnail load for the highlighted video — keeps
        # arrow-key scrolling responsive
        if self._thumb_wait > 0:
            self._thumb_wait -= dt
            if self._thumb_wait <= 0:
                self._thumb_sel = self.sel
                if 0 <= self.sel < len(self.entries):
                    kind, _, path = self.entries[self.sel]
                    if kind == "vid":
                        self._thumb_for(path)
        for b in self._buttons:
            b.update(dt)

    def _draw_pane(self, ctx, surf, f_sm) -> None:
        """Right-hand preview pane: thumbnail + metadata + readiness badges."""
        pygame.draw.rect(surf, theme.PANEL, self.PANE, border_radius=12)
        pygame.draw.rect(surf, theme.PANEL_EDGE, self.PANE, 2, border_radius=12)
        sel = self.entries[self.sel] if self.entries else None
        _panel_label(pygame, ctx, surf, "PREVIEW", self.PANE.x + 14, self.PANE.y - 8)
        if sel is None or sel[0] != "vid":
            txt = f_sm.render(
                "select a video file" if sel is None else "open the folder to see inside",
                True, theme.DIM)
            surf.blit(txt, txt.get_rect(center=self.THUMB.center))
            return
        _, name, path = sel
        thumb = self._thumbs.get(path)  # never load in draw — update() does it
        pygame.draw.rect(surf, theme.PANEL, self.THUMB, border_radius=8)
        if thumb is not None:
            surf.blit(thumb, thumb.get_rect(center=self.THUMB.center))
        elif path in self._thumbs:
            none_img = f_sm.render("(no frame)", True, theme.DIM)
            surf.blit(none_img, none_img.get_rect(center=self.THUMB.center))
        else:
            loading = f_sm.render("loading…", True, theme.DIM)
            surf.blit(loading, loading.get_rect(center=self.THUMB.center))
        pygame.draw.rect(surf, theme.PANEL_EDGE, self.THUMB, 1, border_radius=8)
        info = self._probe(path)
        y = self.THUMB.bottom + 14
        lines = [
            (name[:46], theme.TEXT),
            (f"{self._fmt_dur(info['dur'])} · {info['w']}×{info['h']} @ {info['fps']:.0f}fps · {self._fmt_size(info['size'])}",
             theme.DIM),
        ]
        # readiness badges — things that make analysis fast or already done
        badges = []
        if path.with_suffix(".moves.json").is_file():
            badges.append(("SEGMENTS ✓", theme.GOLD))
        try:
            from ..reference import cache_path_for
            if cache_path_for(path, ctx.cfg.cache_dir).is_file():
                badges.append(("POSES CACHED ✓", theme.CYAN))
        except Exception:
            pass
        for text, color in lines:
            surf.blit(f_sm.render(text, True, color), (self.THUMB.x, y))
            y += 24
        for label, color in badges:
            surf.blit(f_sm.render(label, True, color), (self.THUMB.x, y))
            y += 22

    def draw(self, ctx, surf) -> None:
        _bg(pygame, ctx, surf, ctx.t)
        _title(pygame, ctx, surf, ctx.t, y=44)
        f_sm, f_med = ctx.fonts["sm"], ctx.fonts["med"]
        surf.blit(f_med.render("BROWSE — PICK A VIDEO FILE", True, theme.CYAN), (60, 140))
        cwd_img = f_sm.render(str(self.cwd), True, theme.GOLD)
        surf.blit(cwd_img, (60, 170))
        if self.err:
            surf.blit(f_sm.render(self.err[:80], True, theme.MISS), (60, 620))
        if not self.entries:
            surf.blit(f_sm.render("no videos or folders here", True, theme.DIM), (60, 210))
        y = self.LIST_Y
        for i in range(self.offset, min(self.offset + self.ROWS, len(self.entries))):
            kind, name, path = self.entries[i]
            selected = i == self.sel
            row = pygame.Rect(self.LIST_X - 10, y, self.LIST_W + 10, self.ROW_H - 4)
            if selected:
                pygame.draw.rect(surf, (*theme.PANEL_EDGE,), row, border_radius=6)
            elif kind == "dir":
                pygame.draw.rect(surf, theme.PANEL, row, border_radius=6)
            color = theme.CYAN if kind == "dir" else theme.TEXT
            txt = f_sm.render(("▸ " if kind == "dir" else "♪ ") + name, True, color)
            surf.blit(txt, (row.x + 10, row.y + (row.h - txt.get_height()) // 2))
            if kind == "vid":  # size + duration, right-aligned
                info = self._probe(path)
                meta = f"{self._fmt_dur(info['dur'])} · {self._fmt_size(info['size'])}"
                mi = f_sm.render(meta, True, theme.DIM)
                surf.blit(mi, mi.get_rect(right=row.right - 10,
                                          centery=row.y + row.h // 2))
            y += self.ROW_H
        if len(self.entries) > self.ROWS:
            track = pygame.Rect(self.LIST_X + self.LIST_W + 14, self.LIST_Y, 8,
                                self.ROWS * self.ROW_H)
            pygame.draw.rect(surf, theme.PANEL, track, border_radius=4)
            frac = self.offset / max(len(self.entries) - self.ROWS, 1)
            th = max(int(track.h * self.ROWS / len(self.entries)), 24)
            pygame.draw.rect(surf, theme.PANEL_EDGE,
                             (track.x, track.y + int((track.h - th) * frac), 8, th),
                             border_radius=4)
        self._draw_pane(ctx, surf, f_sm)
        _dim_strip(pygame, surf, (W // 2 - 320, H - 44, 640, 34))
        hint = f_sm.render("↑/↓ select · ENTER open · click selects, click again opens · Q back",
                           True, theme.DIM)
        surf.blit(hint, hint.get_rect(center=(W // 2, H - 26)))
        for b in self._buttons:
            b.draw(pygame, surf, f_med, ctx.t)


# --- READY (camera check + countdown) ----------------------------------------


class ReadyScreen(Screen):
    """Live camera check + countdown between dance select and gameplay.

    Phases: "opening" (camera opens on a worker thread) → "detect"
    (waiting for a tracked person) → "locked" (person found) →
    "countdown" → swap into PlayScreen with ctx.auto_start set.
    A failed open shows the error with retry/back instead of a dead end.
    """

    DETECT_HOLD = 0.8    # seconds of continuous detection before "locked"
    LOCK_BEAT = 0.7      # "got you" flash before the countdown
    COUNTDOWN = 3.0
    LOST_TIMEOUT = 1.5   # seconds without a pose that aborts the countdown

    def enter(self) -> None:
        self.phase = "opening"
        self.err: str | None = None
        self.frame = None
        self.live = None
        self.detect_t = 0.0
        self.lock_t = 0.0
        self.count_t = self.COUNTDOWN
        self.lost_t = 0.0
        self._canceled = False
        self._thread: threading.Thread | None = None
        self.btn_back = Button(pygame.Rect(60, 645, 170, 48), "BACK", "back")
        self.btn_retry = Button(pygame.Rect(240, 645, 170, 48), "RETRY", "retry")
        self.btn_skip = Button(pygame.Rect(1050, 645, 170, 48), "SKIP ▸", "skip")

    # -- camera -------------------------------------------------------------

    def _open(self, ctx) -> None:
        def work() -> None:
            try:
                _open_camera_ctx(ctx)
            except CameraError as exc:
                self.err = str(exc)
            except Exception as exc:  # e.g. backend init failure
                self.err = f"camera failed: {exc}"
            if self._canceled:
                close_camera(ctx)  # user backed out while the driver opened

        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()

    def _finish(self, ctx) -> None:
        # If the player skips while the camera thread is still opening, wait
        # for it so PlayScreen doesn't open a second capture device.
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        ctx.auto_start = True
        ctx.manager.swap(PlayScreen())

    # -- events -------------------------------------------------------------

    def _go_back(self, ctx) -> None:
        self._canceled = True  # tell the camera thread to release its result
        ctx.manager.pop()

    def handle_event(self, ctx, ev) -> None:
        if ev.type == pygame.KEYDOWN:
            if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                self._go_back(ctx)
            elif ev.key in (pygame.K_RETURN, pygame.K_SPACE):
                if self.phase == "error":
                    self._retry(ctx)
                else:
                    self._finish(ctx)
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            if self.btn_back.contains(ev.pos):
                self._go_back(ctx)
            elif self.phase == "error" and self.btn_retry.contains(ev.pos):
                self._retry(ctx)
            elif self.phase != "error" and self.btn_skip.contains(ev.pos):
                self._finish(ctx)

    def _retry(self, ctx) -> None:
        close_camera(ctx)
        self.err = None
        self._thread = None
        self.phase = "opening"

    # -- state --------------------------------------------------------------

    def update(self, ctx, dt) -> None:
        for b in (self.btn_back, self.btn_retry, self.btn_skip):
            b.update(dt)
        if self.phase == "opening":
            if ctx.cap is None and self.err is None and self._thread is None:
                self._open(ctx)
            if self.err:
                self.phase = "error"
            elif ctx.cap is not None:
                self.phase = "detect"
            return
        if self.phase == "error":
            return

        ok, frame = ctx.cap.read() if ctx.cap is not None else (False, None)
        if ok:
            self.frame = frame
            self.live = ctx.tracker.predict(frame) if ctx.tracker else None
        else:
            self.live = None
            if self.frame is None:
                self.err = "camera opened but no frames are arriving"
                self.phase = "error"
                return

        if self.phase == "detect":
            if self.live is not None:
                self.detect_t += dt
                if self.detect_t >= self.DETECT_HOLD:
                    self.phase = "locked"
                    self.lock_t = self.LOCK_BEAT
            else:
                self.detect_t = 0.0
        elif self.phase == "locked":
            self.lock_t -= dt
            if self.lock_t <= 0:
                self.phase = "countdown"
                self.count_t = self.COUNTDOWN
        elif self.phase == "countdown":
            if self.live is not None:
                self.lost_t = 0.0
                self.count_t -= dt
                if self.count_t <= 0:
                    self._finish(ctx)
            else:
                self.lost_t += dt
                if self.lost_t >= self.LOST_TIMEOUT:
                    self.phase = "detect"
                    self.detect_t = 0.0
                    self.lost_t = 0.0

    # -- draw ---------------------------------------------------------------

    def draw(self, ctx, surf) -> None:
        _bg(pygame, ctx, surf, ctx.t)
        _title(pygame, ctx, surf, ctx.t, y=48)
        f_sm, f_med, f_big = ctx.fonts["sm"], ctx.fonts["med"], ctx.fonts["big"]
        surf.blit(f_med.render("GET READY", True, theme.CYAN), (60, 150))
        rect = (W // 2 - 330, 180, 660, 400)
        pygame.draw.rect(surf, theme.PANEL, rect, border_radius=8)
        pygame.draw.rect(surf, theme.PANEL_EDGE, rect, 1, border_radius=8)

        if self.phase == "error":
            msg = f_sm.render((self.err or "camera unavailable")[:76], True, theme.MISS)
            surf.blit(msg, msg.get_rect(center=(rect[0] + rect[2] // 2, rect[1] + 170)))
            sub = f_sm.render("check the webcam is connected and free, then RETRY", True, theme.DIM)
            surf.blit(sub, sub.get_rect(center=(rect[0] + rect[2] // 2, rect[1] + 210)))
        elif self.phase == "opening":
            msg = f_med.render("opening camera…", True, theme.DIM)
            surf.blit(msg, msg.get_rect(center=(rect[0] + rect[2] // 2, rect[1] + 190)))
        else:
            if self.frame is not None:
                surf.blit(_frame_surf(pygame, self.frame, rect[2], rect[3]), rect[:2])
            if self.live is not None:
                fitted = fit_keypoints(self.live.keypoints, self.frame.shape[1],
                                       self.frame.shape[0], *rect)
                _neon_skeleton(pygame, surf, fitted, ctx.cfg.confidence, theme.CYAN)

            if self.phase == "detect":
                frac = self.detect_t / self.DETECT_HOLD
                _chip(pygame, ctx, surf,
                      "POSE FOUND — HOLD STILL" if self.live else "STEP INTO THE FRAME",
                      (rect[0] + rect[2] // 2, rect[1] + 30),
                      theme.GOOD if self.live else theme.GOLD)
                bar = pygame.Rect(rect[0] + 180, rect[1] + rect[3] - 28, rect[2] - 360, 8)
                pygame.draw.rect(surf, theme.PANEL_EDGE, bar, border_radius=4)
                pygame.draw.rect(surf, theme.CYAN,
                                 (bar.x, bar.y, int(bar.w * min(frac, 1.0)), bar.h),
                                 border_radius=4)
            elif self.phase == "locked":
                _chip(pygame, ctx, surf, "GOT YOU — GET READY",
                      (rect[0] + rect[2] // 2, rect[1] + 30), theme.GOOD)
            elif self.phase == "countdown":
                n = int(math.ceil(self.count_t))
                img = theme.glow_text(pygame, f_big, str(max(n, 1)), theme.GOLD)
                surf.blit(img, img.get_rect(center=(rect[0] + rect[2] // 2,
                                                    rect[1] + rect[3] // 2)))
                if self.lost_t > 0:
                    _chip(pygame, ctx, surf, "POSE LOST — COUNTDOWN PAUSED",
                          (rect[0] + rect[2] // 2, rect[1] + 30), theme.MISS)

        _dim_strip(pygame, surf, (160, 588, W - 320, 52))
        tip = f_sm.render(
            "stand back so head, hips and feet are visible · decent lighting helps the tracker",
            True, theme.DIM)
        surf.blit(tip, tip.get_rect(center=(W // 2, 602)))
        keys_img = f_sm.render("ENTER skip · Q back", True, theme.DIM)
        surf.blit(keys_img, keys_img.get_rect(center=(W // 2, 626)))
        self.btn_back.draw(pygame, surf, f_med, ctx.t)
        if self.phase != "error":
            self.btn_skip.draw(pygame, surf, f_med, ctx.t)
        else:
            self.btn_retry.draw(pygame, surf, f_med, ctx.t)


# --- PLAY --------------------------------------------------------------------


POPUP_TTL = 0.9


class PlayScreen(Screen):
    """Play Mode: continuous scored run through the whole dance.

    The reference plays start-to-finish while the session evaluates the
    player's pose at fixed intervals. No seeking and no mid-run retries —
    the only controls are pause/resume, restart, mirror, and quit.
    """

    def enter(self) -> None:
        self._ready = False
        self.cap = None
        self.cam_err = None
        self.frame = None
        self.live = None
        self.hint = None
        self.session = None
        self.frame_id = 0
        self.popups: list[list] = []
        self.flash = 0.0
        self.results_done = False
        self.no_pose_t = 0.0
        self.smoother = ScoreSmoother()
        self.corrections = CorrectionTracker()
        self._music_started = False  # soundtrack anchored at run start
        self._music_paused = False
        self._music_stopped = False
        self._last_phase = None

    def _ensure(self, ctx) -> None:
        """One-time init needing ctx (camera + session); safe to call often."""
        if self._ready:
            return
        self._ready = True
        if ctx.cap is None:
            try:
                _open_camera_ctx(ctx)
            except CameraError as exc:
                self.cam_err = str(exc)
        self.cap = ctx.cap
        if self.session is None:
            self.session = PlaySession(ctx.moves)
        ctx.auto_start = False

    def exit(self) -> None:
        # The camera is shared on ctx — leaving for results or a retry keeps
        # it warm; menus release it via close_camera().
        pass

    # -- soundtrack ---------------------------------------------------------

    def _update_music(self, ctx) -> None:
        """Keep the soundtrack aligned with the continuous session clock.

        Starts at the first move's start, pauses/unpauses with the session,
        re-anchors when wall-clock drift exceeds 0.25 s, and stops when the
        run completes.
        """
        music = getattr(ctx, "music", None)
        s = self.session
        if music is None or s is None:
            return
        if s.phase == GamePhase.COMPLETE:
            if not self._music_stopped:
                music.stop()
                self._music_stopped = True
            return
        if not music.load(getattr(ctx, "audio_path", None)):
            return  # no audio track for this video — silent run
        if not self._music_started:
            music.play_at(s.start_time)
            self._music_started = True
            self._music_paused = False
        if s.paused != self._music_paused:
            (music.pause if s.paused else music.unpause)()
            self._music_paused = s.paused
        if not s.paused:
            pos = music.position()
            if pos is not None and abs(pos - s.time) > 0.25:
                music.play_at(s.time)

    def _eval_feed(self, ctx) -> tuple[float | None, bool, np.ndarray | None, object]:
        """Compute accuracy + reliability + display hint against the window."""
        live_kpts = self.live.keypoints if self.live is not None else None
        raw, reliable, hint, ref_row = eval_against_reference(
            ctx, ctx.choreo, self.session.time, live_kpts
        )
        accuracy = self.smoother.update(raw) if reliable else self.smoother.update(None)
        return accuracy, reliable, hint, ref_row

    def _on_eval(self, ctx, ev) -> None:
        if ev is None:
            return
        if ev.accuracy is None:
            return  # tracking loss is not a player mistake — HUD chip covers it
        j = judge(ev.accuracy, ev.hit)
        color = theme.JUDGE_COLORS[j.value]
        self.popups.append([j.value, color, ctx.t])
        if ev.hit and ev.combo and ev.combo % self.session.config.combo_step == 0:
            mult = min(1 + ev.combo // self.session.config.combo_step,
                       self.session.config.max_multiplier)
            self.popups.append([f"COMBO x{mult}!", theme.PINK, ctx.t])
            ctx.particles.burst(W // 2, 140, theme.PINK, n=30)
            self.flash = 0.5

    def _back(self, ctx) -> None:
        if ctx.manager.depth > 1:
            ctx.manager.pop()
        else:
            ctx.manager.replace(SelectScreen())

    def handle_event(self, ctx, ev) -> None:
        self._ensure(ctx)
        if ev.type != pygame.KEYDOWN:
            return
        k = ev.key
        if k in (pygame.K_ESCAPE, pygame.K_p, pygame.K_SPACE):
            self.session.toggle_pause()
        elif k == pygame.K_q:
            music = getattr(ctx, "music", None)
            if music is not None:
                music.stop()
            self._back(ctx)
        elif k == pygame.K_r and self.cam_err:
            # dead camera → back through the readiness flow to reopen it
            ctx.manager.swap(ReadyScreen())
        elif k == pygame.K_r:
            self.session.restart()  # restart the run from the top
            self._music_started = False
            self._music_stopped = False
        elif k == pygame.K_m:
            ctx.mirror = not ctx.mirror

    def update(self, ctx, dt) -> None:
        self._ensure(ctx)
        self.cap = self.cap or ctx.cap
        if self.cap is None:
            return
        ok, frame = self.cap.read()
        self.frame = frame if ok else None
        self.live = ctx.tracker.predict(frame) if (ok and ctx.tracker) else None
        if self.live is None:
            self.no_pose_t += dt
        else:
            self.no_pose_t = 0.0
        self.frame_id += 1
        accuracy, reliable, hint, _ = self._eval_feed(ctx)
        ev = self.session.update(time.perf_counter(), accuracy, reliable, self.frame_id)
        if self.session.phase == GamePhase.GRACE and self._last_phase != GamePhase.GRACE:
            self.popups.append(["LAST CHANCE!", theme.GOLD, ctx.t])
        self._last_phase = self.session.phase
        if not self.session.paused:
            self._on_eval(ctx, ev)
        self.hint = self.corrections.update(hint)
        self.flash = max(0.0, self.flash - dt * 2)
        self._update_music(ctx)
        if self.session.phase == GamePhase.COMPLETE and not self.results_done:
            self.results_done = True
            ctx.last_session = self.session
            ctx.last_results = self.session.results
            ctx.manager.replace(ResultsScreen())

    def _panels(self, ctx, surf) -> tuple:
        ref_rect = (30, 110, 600, 470)
        cam_rect = (650, 110, 600, 470)
        for r, label, color in ((ref_rect, "REFERENCE", theme.VIOLET),
                                (cam_rect, "YOU", theme.CYAN)):
            pygame.draw.rect(surf, theme.PANEL, r)
            pygame.draw.rect(surf, theme.PANEL_EDGE, r, 1, border_radius=4)
            _panel_label(pygame, ctx, surf, label, r[0] + 12, r[1] - 24, color)
        t = self.session.time
        pose = pose_at(ctx.choreo, t)
        if pose is not None and pose[:, 2].max() > 0:
            shown = mirror_keypoints(pose, ctx.choreo.width) if ctx.mirror else pose
            fitted = fit_keypoints(shown, ctx.choreo.width, ctx.choreo.height, *ref_rect)
            _neon_skeleton(pygame, surf, fitted, ctx.cfg.confidence, theme.VIOLET)
        if self.frame is not None:
            surf.blit(_frame_surf(pygame, self.frame, cam_rect[2], cam_rect[3]), cam_rect[:2])
            if self.live is not None:
                fitted = fit_keypoints(self.live.keypoints, self.frame.shape[1], self.frame.shape[0], *cam_rect)
                _neon_skeleton(pygame, surf, fitted, ctx.cfg.confidence, theme.CYAN)
        else:
            msg = self.cam_err or "camera signal lost"
            txt = ctx.fonts["sm"].render(msg[:70], True, theme.MISS)
            surf.blit(txt, txt.get_rect(center=(cam_rect[0] + 300, cam_rect[1] + 235)))
        return ref_rect, cam_rect

    def draw(self, ctx, surf) -> None:
        _bg(pygame, ctx, surf, ctx.t)
        # Screens can be swapped in during another screen's update(), so
        # draw can run before this screen's first update() — init lazily.
        self._ensure(ctx)
        f_sm, f_med, f_big = ctx.fonts["sm"], ctx.fonts["med"], ctx.fonts["big"]
        s = self.session
        surf.blit(f_med.render(f"MOVE {s.move_index + 1}/{len(s.moves)}: {s.move.name}", True, theme.PINK), (30, 20))
        surf.blit(f_med.render(f"SCORE {s.total_score}", True, theme.GOLD), (900, 20))
        mult = min(1 + s.combo // s.config.combo_step, s.config.max_multiplier)
        surf.blit(f_med.render(f"COMBO x{mult} ({s.combo})", True, theme.CYAN), (900, 52))
        ref_rect, cam_rect = self._panels(ctx, surf)
        # beat-synced pulse: border glow on the reference panel + popup kick
        pulse = ctx.beats.pulse(s.time) if ctx.beats is not None else 0.0
        if pulse > 0.05:
            glow = tuple(int(a + (b - a) * pulse) for a, b in zip(theme.PANEL_EDGE, theme.CYAN))
            pygame.draw.rect(surf, glow, ref_rect, 1 + int(3 * pulse), border_radius=4)
        # tracking-loss chip — replaces the old MISS spam when the pose drops
        if self.frame is not None and self.live is None and self.no_pose_t > 0.6:
            _chip(pygame, ctx, surf, "POSE LOST — STEP INTO FRAME",
                  (cam_rect[0] + cam_rect[2] // 2, cam_rect[1] + 28), theme.GOLD)
        # grace window — the failing segment loops; finish it before it ends
        if s.phase == GamePhase.GRACE:
            _chip(pygame, ctx, surf,
                  f"COMPLETE THE MOVE — {s.grace_remaining:.0f}s",
                  (ref_rect[0] + ref_rect[2] // 2, ref_rect[1] + 28), theme.GOLD)
        # bottom HUD
        acc = s.current_accuracy or 0.0
        surf.blit(f_sm.render(f"ACCURACY {acc * 100:3.0f}%", True, theme.TEXT), (30, 600))
        pygame.draw.rect(surf, theme.PANEL_EDGE, (200, 602, 300, 14), border_radius=7)
        pygame.draw.rect(surf, theme.CYAN if acc >= 0.6 else theme.MISS, (200, 602, int(300 * min(acc, 1.0)), 14), border_radius=7)
        tick_x = 200 + int(300 * 0.6)  # pass threshold marker
        pygame.draw.line(surf, theme.GOLD, (tick_x, 599), (tick_x, 619), 2)
        move = s.move
        surf.blit(f_sm.render(f"{move.name[:22]:<22} {s.time - move.start:4.1f}/{move.duration:.1f}s", True, theme.TEXT), (540, 600))
        pygame.draw.rect(surf, theme.PANEL_EDGE, (780, 602, 300, 14), border_radius=7)
        pygame.draw.rect(surf, theme.PINK, (780, 602, int(300 * min(s.move_progress, 1.0)), 14), border_radius=7)
        # whole-run progress strip with segment ticks — display only, no seek
        strip = pygame.Rect(30, 666, 1220, 8)
        pygame.draw.rect(surf, theme.PANEL_EDGE, strip, border_radius=4)
        total = max(s.end_time - s.start_time, 1e-6)
        fill = int(strip.w * min(s.progress, 1.0))
        if fill:
            pygame.draw.rect(surf, theme.VIOLET, (strip.x, strip.y, fill, strip.h), border_radius=4)
        for m in s.moves[1:]:
            tx = strip.x + int(strip.w * (m.start - s.start_time) / total)
            pygame.draw.line(surf, theme.TEXT, (tx, strip.y - 2), (tx, strip.y + strip.h + 2), 1)
        _dim_strip(pygame, surf, (24, 628, 700, 40))
        hint_txt = self.hint or "match the pose"
        surf.blit(f_med.render(hint_txt, True, theme.MISS if self.hint else theme.DIM), (30, 636))
        _dim_strip(pygame, surf, (24, 686, 560, 30))
        surf.blit(f_sm.render("R restart · P pause · M mirror · Q back", True, theme.DIM), (30, 693))
        # judgement popups
        for text, color, born in self.popups:
            age = ctx.t - born
            if age > POPUP_TTL:
                continue
            scale = 1.0 + 0.6 * max(0.0, 1.0 - age * 4)
            img = theme.glow_text(pygame, f_big, text, color)
            img = pygame.transform.rotozoom(img, 0, scale * (1.0 + 0.25 * pulse))
            img.set_alpha(int(255 * max(0.0, 1.0 - age / POPUP_TTL)))
            surf.blit(img, img.get_rect(center=(W // 2, 90)))
        self.popups = [p for p in self.popups if ctx.t - p[2] < POPUP_TTL]
        if self.flash > 0:
            veil = pygame.Surface((W, H), pygame.SRCALPHA)
            veil.fill((*theme.CYAN, int(60 * self.flash)))
            surf.blit(veil, (0, 0))
        ctx.particles.draw(pygame, surf)
        if s.paused:
            self._overlay(ctx, surf, "PAUSED", "", "P resume · R restart · Q quit to menu", theme.GOLD)
        if self.cam_err and self.cap is None:
            self._overlay(ctx, surf, "CAMERA PROBLEM", self.cam_err[:60],
                          "R reopen camera · Q back to dances", theme.MISS)

    def _overlay(self, ctx, surf, line1, line2, line3, color) -> None:
        veil = pygame.Surface((W, H), pygame.SRCALPHA)
        veil.fill((8, 4, 18, 210))
        surf.blit(veil, (0, 0))
        img = theme.glow_text(pygame, ctx.fonts["big"], line1, color)
        surf.blit(img, img.get_rect(center=(W // 2, 280)))
        if line2:
            t2 = ctx.fonts["med"].render(line2, True, theme.TEXT)
            surf.blit(t2, t2.get_rect(center=(W // 2, 360)))
        t3 = ctx.fonts["med"].render(line3, True, theme.GOLD)
        surf.blit(t3, t3.get_rect(center=(W // 2, 430)))


# --- WATCH -------------------------------------------------------------------


class WatchScreen(Screen):
    """View the whole reference choreography as one continuous skeleton."""

    def enter(self) -> None:
        self.clock: PlaybackClock | None = None
        self.music = None          # ctx.music captured for exit()/stop
        self.music_paused = False

    def exit(self) -> None:
        if self.music is not None:
            self.music.stop()

    def _sync_music(self, ctx) -> None:
        """Keep audio aligned with the playback clock (pause, seek, end)."""
        music = getattr(ctx, "music", None)
        if music is None:
            return
        self.music = music
        if music.path != getattr(ctx, "audio_path", None):
            # fresh load (new video or first play) → start at the clock
            if not music.load(getattr(ctx, "audio_path", None)):
                return
            music.play_at(self.clock.time)
        if self.clock.playing and self.music_paused:
            music.unpause()
        elif not self.clock.playing and not self.music_paused:
            music.pause()
        self.music_paused = not self.clock.playing

    def _reanchor(self, ctx) -> None:
        """Re-sync audio after a seek/restart."""
        if self.music is not None:
            self.music.play_at(self.clock.time)

    def update(self, ctx, dt) -> None:
        if self.clock is None:
            self.clock = PlaybackClock(ctx.choreo.duration)
        self.clock.now(time.perf_counter())
        self._sync_music(ctx)

    def handle_event(self, ctx, ev) -> None:
        if ev.type != pygame.KEYDOWN:
            return
        if ev.key in (pygame.K_ESCAPE, pygame.K_q):
            ctx.manager.pop()
        elif ev.key == pygame.K_RETURN:
            ctx.manager.swap(ReadyScreen())
        elif ev.key == pygame.K_SPACE and self.clock is not None:
            self.clock.toggle()
        elif ev.key == pygame.K_LEFT and self.clock is not None:
            self.clock.skip(-2.0)
            self._reanchor(ctx)
        elif ev.key == pygame.K_RIGHT and self.clock is not None:
            self.clock.skip(2.0)
            self._reanchor(ctx)
        elif ev.key == pygame.K_r and self.clock is not None:
            self.clock.restart()
            self._reanchor(ctx)
        elif ev.key == pygame.K_m:
            ctx.mirror = not ctx.mirror

    def draw(self, ctx, surf) -> None:
        _bg(pygame, ctx, surf, ctx.t)
        rect = (140, 110, 1000, 470)
        pygame.draw.rect(surf, theme.PANEL, rect)
        pygame.draw.rect(surf, theme.PANEL_EDGE, rect, 1, border_radius=4)
        _panel_label(pygame, ctx, surf, "REFERENCE — FULL DANCE", rect[0] + 12, rect[1] - 24, theme.VIOLET)
        t = self.clock.time if self.clock is not None else 0.0
        pose = pose_at(ctx.choreo, t)
        if pose is not None and pose[:, 2].max() > 0:
            shown = mirror_keypoints(pose, ctx.choreo.width) if ctx.mirror else pose
            fitted = fit_keypoints(shown, ctx.choreo.width, ctx.choreo.height, *rect)
            _neon_skeleton(pygame, surf, fitted, ctx.cfg.confidence, theme.VIOLET)
        move = next((m.name for m in ctx.moves if m.start <= t < m.end), "")
        head = f"WATCH: {move}" if move else "WATCH THE FULL DANCE"
        surf.blit(ctx.fonts["med"].render(head, True, theme.CYAN), (140, 60))
        prog = min(t / max(ctx.choreo.duration, 1e-6), 1.0)
        pygame.draw.rect(surf, theme.PANEL_EDGE, (140, 600, 1000, 10))
        pygame.draw.rect(surf, theme.CYAN, (140, 600, int(1000 * prog), 10))
        if self.clock is not None and self.clock.finished:
            done = ctx.fonts["med"].render(
                "dance complete — ENTER to start dancing", True, theme.GOLD)
            surf.blit(done, done.get_rect(center=(W // 2, 560)))
        hint = "SPACE pause · ←/→ seek · R restart · M mirror · ENTER dance · Q back"
        surf.blit(ctx.fonts["sm"].render(hint, True, theme.DIM), (140, 640))


# --- RESULTS -----------------------------------------------------------------


class ResultsScreen(Screen):
    def enter(self) -> None:
        self.card = None
        self._saved = False
        self._checked = False
        self.new_best = False
        self.buttons = []
        self.group = None
        self.born = time.perf_counter()
        self.name_field = TextField(pygame.Rect(905, 248, 315, 38), placeholder="player name")
        self.name_field.focused = False

    def _sync(self, ctx) -> None:
        session = getattr(ctx, "last_session", None)
        if self.card is None and isinstance(session, (GameSession, PlaySession)):
            self.card = session.scorecard()
        if self.card is None:
            return
        if not self._checked:
            self._checked = True
            video = ctx.video_path.name if ctx.video_path else "?"
            prev = _best_score_for(video, load_scores())
            self.new_best = prev is None or self.card["total_score"] > prev["score"]
            self.name_field.text = ctx.settings.get("player_name", "PLAYER")

    def _commit(self, ctx) -> None:
        """Save the run under the entered name (once)."""
        if self._saved or self.card is None:
            return
        self._saved = True
        name = self.name_field.text.strip()[:16] or "PLAYER"
        ctx.settings["player_name"] = name
        save_settings(ctx.settings)
        video = ctx.video_path.name if ctx.video_path else "?"
        add_score(
            video,
            self.card["total_score"], self.card["mean_accuracy"],
            self.card["best_combo"], grade_for(self.card["mean_accuracy"]),
            name=name,
        )
        ctx.particles.burst(W // 2, 200, theme.GOLD, n=60)
        if not self.buttons:
            self.buttons = [
                Button(pygame.Rect(W // 2 - 340, 610, 210, 52), "RETRY", "retry", True),
                Button(pygame.Rect(W // 2 - 105, 610, 230, 52), "ANOTHER DANCE", "select"),
                Button(pygame.Rect(W // 2 + 150, 610, 190, 52), "HOME", "home"),
            ]
            self.group = FocusGroup(self.buttons)

    def _activate(self, ctx, action: str) -> None:
        self._commit(ctx)  # score saves before leaving
        if action == "retry":
            ctx.manager.swap(PlayScreen())
        elif action == "select":
            ctx.manager.replace(SelectScreen())
        elif action == "home":
            ctx.manager.replace(HomeScreen())

    def handle_event(self, ctx, ev) -> None:
        self._sync(ctx)
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            if self.name_field.contains(ev.pos):
                self.name_field.focused = True
                return
            if self.name_field.focused:
                self.name_field.focused = False
                self._commit(ctx)
        elif ev.type == pygame.KEYDOWN and self.name_field.focused:
            if ev.key == pygame.K_RETURN:
                self.name_field.focused = False
                self._commit(ctx)
            elif ev.key == pygame.K_ESCAPE:
                self.name_field.focused = False
            else:
                self.name_field.handle_key(ev.key, ev.unicode)
            return
        if not self.buttons:
            return
        if ev.type == pygame.MOUSEMOTION:
            self.group.hover(ev.pos)
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            for b in self.buttons:
                if b.contains(ev.pos):
                    b.press()
                    self._activate(ctx, b.action)
        elif ev.type == pygame.KEYDOWN:
            if ev.key in (pygame.K_LEFT, pygame.K_a):
                self.group.move(-1)
            elif ev.key in (pygame.K_RIGHT, pygame.K_d):
                self.group.move(1)
            elif ev.key in (pygame.K_RETURN, pygame.K_SPACE):
                self._activate(ctx, self.group.current.action)
            elif ev.key == pygame.K_r:
                self._activate(ctx, "retry")

    def update(self, ctx, dt) -> None:
        self._sync(ctx)
        for b in self.buttons:
            b.update(dt)

    def _stat(self, ctx, surf, x, y, w, label, value, color) -> None:
        rect = pygame.Rect(x, y, w, 74)
        pygame.draw.rect(surf, theme.PANEL, rect, border_radius=10)
        pygame.draw.rect(surf, theme.PANEL_EDGE, rect, 1, border_radius=10)
        v = ctx.fonts["med"].render(value, True, color)
        surf.blit(v, v.get_rect(center=(x + w // 2, y + 26)))
        lbl = ctx.fonts["sm"].render(label, True, theme.DIM)
        surf.blit(lbl, lbl.get_rect(center=(x + w // 2, y + 55)))

    def draw(self, ctx, surf) -> None:
        _bg(pygame, ctx, surf, ctx.t)
        if not self.card:
            surf.blit(ctx.fonts["med"].render("no results yet", True, theme.DIM), (500, 350))
            return
        card = self.card
        grade = grade_for(card["mean_accuracy"])
        pop = min(1.0, (time.perf_counter() - self.born) * 2.5)
        head = ctx.fonts["med"].render("MISSION COMPLETE", True, theme.DIM)
        surf.blit(head, head.get_rect(center=(W // 2, 56)))
        gi = theme.glow_text(pygame, ctx.fonts["big"], f"GRADE {grade}", theme.GOLD)
        gi = pygame.transform.rotozoom(gi, 0, 0.5 + 0.5 * pop)
        surf.blit(gi, gi.get_rect(center=(W // 2, 130)))
        score_img = theme.glow_text(pygame, ctx.fonts["big"], f"{card['total_score']:,} PTS", theme.TEXT)
        surf.blit(score_img, score_img.get_rect(center=(W // 2, 215)))
        if self.new_best:
            _chip(pygame, ctx, surf, "★ NEW PERSONAL BEST ★", (W // 2, 268), theme.GOLD)
        bw, gap = 240, 20
        x0 = W // 2 - (3 * bw + 2 * gap) // 2
        self._stat(ctx, surf, x0, 300, bw, "ACCURACY", f"{card['mean_accuracy'] * 100:.0f}%", theme.CYAN)
        self._stat(ctx, surf, x0 + bw + gap, 300, bw, "BEST COMBO", str(card['best_combo']), theme.PINK)
        self._stat(ctx, surf, x0 + 2 * (bw + gap), 300, bw, "MOVES PASSED",
                   f"{card['moves_passed']}/{card['moves_total']}", theme.GOOD)
        dance = getattr(ctx, "dance", None)
        if dance is not None or ctx.video_path:
            title = dance.title if dance is not None else ctx.video_path.name
            name = ctx.fonts["sm"].render(f"dance: {title}", True, theme.DIM)
            surf.blit(name, name.get_rect(center=(W // 2, 396)))
        y = 425
        for r in getattr(ctx, "last_results", [])[:6]:
            mark, color = ("PASS", theme.GOOD) if r.passed else ("FAIL", theme.MISS)
            row = ctx.fonts["sm"].render(
                f"[{mark}] {r.name:<24} acc {r.mean_accuracy * 100:3.0f}%  +{r.points} pts", True, color)
            surf.blit(row, row.get_rect(center=(W // 2, y)))
            y += 28
        if len(getattr(ctx, "last_results", [])) > 6:
            more = ctx.fonts["sm"].render(
                f"… and {len(ctx.last_results) - 6} more moves", True, theme.DIM)
            surf.blit(more, more.get_rect(center=(W // 2, y)))
        ctx.particles.draw(pygame, surf)
        lbl = ctx.fonts["sm"].render("PLAYER NAME", True, theme.GOLD)
        surf.blit(lbl, (905, 226))
        hint = ctx.fonts["sm"].render("ENTER to save score", True, theme.DIM)
        surf.blit(hint, (905, 292))
        self.name_field.draw(pygame, surf, ctx.fonts["sm"], ctx.t)
        for b in getattr(self, "buttons", []):
            b.draw(pygame, surf, ctx.fonts["med"], ctx.t)


# --- SETTINGS / SCORES -------------------------------------------------------


class SettingsScreen(Screen):
    WINDOW_MIN, WINDOW_MAX, WINDOW_STEP = 0.05, 0.50, 0.05

    def enter(self) -> None:
        self.tgl_mirror = Toggle(pygame.Rect(120, 285, 560, 34), "mirror choreography by default", "mirror", True)
        self.tgl_full = Toggle(pygame.Rect(120, 345, 560, 34), "fullscreen", "fullscreen", False)
        self.st_cam = Stepper(pygame.Rect(120, 405, 560, 40), "camera index", "cam", fmt=lambda v: str(int(v)))
        self.st_win = Stepper(pygame.Rect(120, 465, 560, 40), "match window (timing tolerance)", "win",
                              fmt=lambda v: f"{v:.2f}s")
        self.st_vol = Stepper(pygame.Rect(120, 525, 560, 40), "music volume", "vol",
                              fmt=lambda v: f"{v:.0f}%")
        self.btn_back = Button(pygame.Rect(120, 610, 160, 48), "BACK", "back")
        self.group = FocusGroup([self.tgl_mirror, self.tgl_full, self.st_cam, self.st_win,
                                 self.st_vol, self.btn_back])
        self._synced = False

    def _save(self, ctx) -> None:
        ctx.settings["mirror"] = self.tgl_mirror.value
        ctx.settings["fullscreen"] = self.tgl_full.value
        ctx.settings["camera_index"] = int(self.st_cam.value)
        ctx.settings["window_s"] = round(self.st_win.value, 3)
        ctx.settings["volume"] = int(self.st_vol.value)
        save_settings(ctx.settings)

    def _activate(self, ctx, action: str) -> None:
        if action == "back":
            self._save(ctx)
            ctx.manager.pop()
        elif action == "mirror":
            self.tgl_mirror.value = not self.tgl_mirror.value
            ctx.mirror = self.tgl_mirror.value
            self._save(ctx)
        elif action == "fullscreen":
            self.tgl_full.value = not self.tgl_full.value
            ctx.fullscreen = self.tgl_full.value
            self._save(ctx)
        elif action in ("cam-", "cam+"):
            self.st_cam.value = max(0, min(7, self.st_cam.value + (1 if action.endswith("+") else -1)))
            self._save(ctx)
        elif action in ("win-", "win+"):
            d = self.WINDOW_STEP if action.endswith("+") else -self.WINDOW_STEP
            self.st_win.value = round(max(self.WINDOW_MIN, min(self.WINDOW_MAX, self.st_win.value + d)), 2)
            self._save(ctx)
        elif action in ("vol-", "vol+"):
            d = 10 if action.endswith("+") else -10
            self.st_vol.value = max(0, min(100, self.st_vol.value + d))
            music = getattr(ctx, "music", None)
            if music is not None:
                music.set_volume(self.st_vol.value / 100.0)
            self._save(ctx)

    def handle_event(self, ctx, ev) -> None:
        items = [self.tgl_mirror, self.tgl_full, self.st_cam, self.st_win,
                 self.st_vol, self.btn_back]
        if ev.type == pygame.MOUSEMOTION:
            for i in items:
                i.hovered = i.contains(ev.pos)
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            for st in (self.st_cam, self.st_win, self.st_vol):
                act = st.action_at(ev.pos)
                if act:
                    self._activate(ctx, act)
                    return
            for i in items:
                if i.contains(ev.pos):
                    self._activate(ctx, i.action)
        elif ev.type == pygame.KEYDOWN:
            if ev.key in (pygame.K_DOWN, pygame.K_s):
                self.group.move(1)
            elif ev.key in (pygame.K_UP, pygame.K_w):
                self.group.move(-1)
            elif ev.key in (pygame.K_LEFT, pygame.K_a):
                self._adjust_focused(ctx, -1)
            elif ev.key in (pygame.K_RIGHT, pygame.K_d):
                self._adjust_focused(ctx, 1)
            elif ev.key in (pygame.K_RETURN, pygame.K_SPACE):
                self._adjust_focused(ctx, 1)
            elif ev.key == pygame.K_ESCAPE:
                self._activate(ctx, "back")

    def _adjust_focused(self, ctx, direction: int) -> None:
        cur = self.group.current
        if cur is self.tgl_mirror:
            self._activate(ctx, "mirror")
        elif cur is self.tgl_full:
            self._activate(ctx, "fullscreen")
        elif cur is self.st_cam:
            self._activate(ctx, "cam+" if direction > 0 else "cam-")
        elif cur is self.st_win:
            self._activate(ctx, "win+" if direction > 0 else "win-")
        elif cur is self.st_vol:
            self._activate(ctx, "vol+" if direction > 0 else "vol-")
        elif cur is self.btn_back:
            self._activate(ctx, "back")

    def update(self, ctx, dt) -> None:
        if not self._synced:
            self.tgl_mirror.value = bool(ctx.settings.get("mirror", True))
            self.tgl_full.value = bool(ctx.settings.get("fullscreen", False))
            self.st_cam.value = float(int(ctx.settings.get("camera_index", 0)))
            self.st_win.value = float(ctx.settings.get("window_s", ctx.cfg.window_s))
            self.st_vol.value = float(ctx.settings.get("volume", 80))
            self._synced = True
        for i in (self.st_cam, self.st_win, self.st_vol, self.btn_back):
            i.update(dt)

    def draw(self, ctx, surf) -> None:
        _bg(pygame, ctx, surf, ctx.t)
        _title(pygame, ctx, surf, ctx.t, y=64)
        f_sm, f_med = ctx.fonts["sm"], ctx.fonts["med"]
        surf.blit(f_med.render("SETTINGS", True, theme.CYAN), (120, 235))
        self.tgl_mirror.draw(pygame, surf, f_sm)
        self.tgl_full.draw(pygame, surf, f_sm)
        self.st_cam.draw(pygame, surf, f_sm)
        self.st_win.draw(pygame, surf, f_sm)
        self.st_vol.draw(pygame, surf, f_sm)
        note = f_sm.render(
            "camera + match window apply on the next dance · F11 toggles fullscreen anywhere",
            True, theme.DIM)
        surf.blit(note, (120, 578))
        _dim_strip(pygame, surf, (W // 2 - 260, H - 44, 520, 34))
        hint = f_sm.render("←/→ adjust · ENTER toggle · ESC save & back", True, theme.DIM)
        surf.blit(hint, hint.get_rect(center=(W // 2, H - 26)))
        self.btn_back.draw(pygame, surf, f_med, ctx.t)


class ScoresScreen(Screen):
    def enter(self) -> None:
        self.scores = load_scores()
        self.btn_back = Button(pygame.Rect(120, 620, 160, 48), "BACK", "back")

    def handle_event(self, ctx, ev) -> None:
        if ev.type == pygame.MOUSEBUTTONDOWN and self.btn_back.contains(ev.pos):
            ctx.manager.pop()
        elif ev.type == pygame.KEYDOWN and ev.key in (pygame.K_ESCAPE, pygame.K_RETURN, pygame.K_BACKSPACE):
            ctx.manager.pop()

    def draw(self, ctx, surf) -> None:
        _bg(pygame, ctx, surf, ctx.t)
        _title(pygame, ctx, surf, ctx.t, y=64)
        f_sm, f_med = ctx.fonts["sm"], ctx.fonts["med"]
        surf.blit(f_med.render("HIGH SCORES", True, theme.GOLD), (120, 190))
        if not self.scores:
            surf.blit(f_sm.render("no runs recorded yet — go dance!", True, theme.DIM), (120, 250))
            self.btn_back.draw(pygame, surf, f_med, ctx.t)
            return
        header = f_sm.render(f"{'#':>2}  {'SCORE':>6}  GRD   ACC   COMBO  NAME         DANCE", True, theme.DIM)
        surf.blit(header, (120, 240))
        y = 268
        for i, s in enumerate(self.scores):
            line = (f"{i + 1:>2}. {s['score']:>6}   [{s['grade']}]  {s['mean_accuracy'] * 100:3.0f}%"
                    f"   {s['best_combo']:>4}   {s.get('name', 'PLAYER'):<12} {Path(s['video']).name[:24]}")
            color = theme.GOLD if i == 0 else (theme.TEXT if i < 3 else theme.DIM)
            surf.blit(f_sm.render(line, True, color), (120, y))
            y += 32
        self.btn_back.draw(pygame, surf, f_med, ctx.t)
