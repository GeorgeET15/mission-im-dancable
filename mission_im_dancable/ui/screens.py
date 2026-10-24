"""Screens for the dance-arcade UI: home, select, play, results, settings, scores."""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pygame

from ..beats import load_or_detect_beats
from ..camera import CameraError, open_camera
from ..config import AppConfig
from ..game import GamePhase, GameSession, grade_for, judge, Judgement
from ..geometry import SKELETON_EDGES
from ..moves import MovesFileError, default_moves, load_moves
from ..persist import add_score, load_scores, load_settings, save_settings
from ..pose import PoseTracker
from ..practice import PracticeSession
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
from .widgets import Button, FocusGroup, ProgressBar, TextField, Toggle

W, H = 1280, 720
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


class AppContext:
    """Shared mutable state passed to every screen."""

    def __init__(self, cfg: AppConfig, fonts: dict, manager: Navigator) -> None:
        self.cfg = cfg
        self.fonts = fonts
        self.manager = manager
        self.settings = load_settings()
        self.mirror = bool(self.settings.get("mirror", cfg.mirror))
        self.mode = "performance"  # or "practice"
        self.tracker: PoseTracker | None = None
        self.device = "cpu"
        self.video_path: Path | None = cfg.source
        self.moves_path: Path | None = cfg.moves_path
        self.choreo: ReferenceChoreography | None = None
        self.moves = None
        self.beats = None  # BeatGrid set after analyze; None = no beat sync
        self.t = 0.0
        self.particles = theme.Particles()
        self.quit = False


class Screen:
    def enter(self) -> None: ...
    def exit(self) -> None: ...
    def handle_event(self, ctx, ev) -> None: ...
    def update(self, ctx, dt) -> None: ...
    def draw(self, ctx, surf) -> None: ...


def _bg(pygame, ctx, surf, t: float) -> None:
    surf.blit(ctx.bg, (0, 0))
    theme.draw_dance_floor(pygame, surf, t, horizon=int(H * 0.62))
    theme.draw_spotlights(pygame, surf, t)
    theme.draw_equalizer(pygame, surf, t, (0, H - 90, W, 60), bars=48)


def _title(pygame, ctx, surf, t: float, y: int = 90) -> None:
    bob = int(math.sin(t * 1.8) * 6)
    img = theme.glow_text(pygame, ctx.fonts["title"], "MISSION IM-DANCABLE", theme.PINK)
    surf.blit(img, img.get_rect(center=(W // 2, y + bob)))
    tag = ctx.fonts["sm"].render("your mission, should you choose to accept it.", True, theme.DIM)
    surf.blit(tag, tag.get_rect(center=(W // 2, y + 52 + bob)))


def _neon_skeleton(pygame, surf, keypoints: np.ndarray, conf: float, color=theme.CYAN) -> None:
    glow = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
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


# --- HOME ------------------------------------------------------------------


class HomeScreen(Screen):
    LABELS = [("LET'S DANCE", "dance", True), ("PRACTICE", "practice", False),
              ("HIGH SCORES", "scores", False), ("SETTINGS", "settings", False),
              ("QUIT", "quit", False)]

    def enter(self) -> None:
        self.buttons = [
            Button(pygame.Rect(W // 2 - 160, 280 + i * 78, 320, 56), lbl, act, prim)
            for i, (lbl, act, prim) in enumerate(self.LABELS)
        ]
        self.group = FocusGroup(self.buttons)

    def _activate(self, ctx, action: str) -> None:
        if action == "dance":
            ctx.mode = "performance"
            ctx.manager.push(SelectScreen())
        elif action == "practice":
            ctx.mode = "practice"
            ctx.manager.push(SelectScreen())
        elif action == "scores":
            ctx.manager.push(ScoresScreen())
        elif action == "settings":
            ctx.manager.push(SettingsScreen())
        elif action == "quit":
            ctx.quit = True

    def handle_event(self, ctx, ev) -> None:
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

    def update(self, ctx, dt) -> None:
        for b in self.buttons:
            b.update(dt)

    def draw(self, ctx, surf) -> None:
        _bg(pygame, ctx, surf, ctx.t)
        _title(pygame, ctx, surf, ctx.t)
        for b in self.buttons:
            b.draw(pygame, surf, ctx.fonts["med"], ctx.t)
        hint = ctx.fonts["sm"].render("arrows + ENTER · or click", True, theme.DIM)
        surf.blit(hint, hint.get_rect(center=(W // 2, H - 24)))


# --- DANCE SELECT -----------------------------------------------------------


class SelectScreen(Screen):
    """Pick a local video, analyze it (background thread), start dancing."""

    def enter(self) -> None:
        self.field = TextField(pygame.Rect(60, 300, 760, 44), placeholder="type a video path, or pick one below")
        self.btn_analyze = Button(pygame.Rect(840, 296, 190, 48), "ANALYZE", "analyze", True)
        self.btn_start = Button(pygame.Rect(1050, 296, 190, 48), "START DANCING", "start", True, enabled=False)
        self.tgl_mode = Toggle(pygame.Rect(60, 380, 400, 30), "mode: practice / performance", "mode", True)
        self.tgl_mirror = Toggle(pygame.Rect(60, 420, 400, 30), "mirror choreography", "mirror", False)
        self.btn_back = Button(pygame.Rect(60, 640, 160, 48), "BACK", "back")
        self.btn_browse = Button(pygame.Rect(230, 640, 160, 48), "BROWSE…", "browse")
        self.video_buttons: list[Button] = []
        self.status = "pick a video and press ANALYZE"
        self.status_color = theme.DIM
        self.info = ""
        self.progress = -1.0
        self.choreo = None
        self.moves = None
        self._thread = None
        self.error = ""
        self.bar = ProgressBar(pygame.Rect(60, 560, 700, 14), color=theme.PINK)
        self._synced = False

    def _sync(self, ctx) -> None:
        if not self._synced:
            self._synced = True
            self.tgl_mode.value = ctx.mode == "performance"
            self.tgl_mirror.value = ctx.mirror
            if ctx.video_path and not self.field.text:
                self.field.text = str(ctx.video_path)
            elif not self.field.text and ctx.settings.get("last_video"):
                self.field.text = ctx.settings["last_video"]
            vids = _detect_videos(Path.cwd())[:4]
            self.video_buttons = [
                Button(pygame.Rect(60 + i * 300, 500, 280, 40), v.name[:26], f"pick:{v.name}")
                for i, v in enumerate(vids)
            ]
            if ctx.choreo is not None and ctx.video_path and Path(self.field.text) == ctx.video_path:
                self.choreo, self.moves = ctx.choreo, ctx.moves
                self.btn_start.enabled = True
                self._fill_info(ctx.video_path)

    def _fill_info(self, path: Path) -> None:
        probe = _probe_video(path)
        if probe and self.choreo:
            dur, w, h = probe
            self.info = (
                f"{path.name} · {dur:.1f}s · {w}x{h} · "
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
        self.btn_start.enabled = False
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
                moves = (
                    load_moves(ctx.moves_path, choreo.duration)
                    if ctx.moves_path
                    else default_moves(choreo.duration)
                )
            except Exception as exc:  # surface to UI thread
                self.error = str(exc)
                return
            ctx.beats = load_or_detect_beats(path, ctx.cfg.cache_dir)
            self.choreo, self.moves = choreo, moves

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
        ctx.choreo, ctx.moves = self.choreo, self.moves
        ctx.mode = "performance" if self.tgl_mode.value else "practice"
        ctx.mirror = self.tgl_mirror.value
        ctx.settings["last_video"] = str(ctx.video_path)
        ctx.settings["mirror"] = ctx.mirror
        save_settings(ctx.settings)
        ctx.manager.replace(PlayScreen())

    def handle_event(self, ctx, ev) -> None:
        widgets = [self.field, self.btn_analyze, self.btn_start, self.btn_back,
                   self.btn_browse, self.tgl_mode, self.tgl_mirror, *self.video_buttons]
        if ev.type == pygame.MOUSEMOTION:
            for wdgt in widgets:
                if hasattr(wdgt, "hovered"):
                    wdgt.hovered = wdgt.contains(ev.pos)
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            self.field.focused = self.field.contains(ev.pos)
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
                ctx.manager.pop()
            elif ev.key == pygame.K_TAB:
                pass
            elif ev.key == pygame.K_RETURN and self.btn_start.enabled:
                self._start(ctx)

    def _activate(self, ctx, action: str) -> None:
        if action == "analyze":
            self._analyze(ctx)
        elif action == "start":
            self._start(ctx)
        elif action == "back":
            ctx.manager.pop()
        elif action == "browse":
            self._browse(ctx)
        elif action == "mode":
            self.tgl_mode.value = not self.tgl_mode.value
        elif action == "mirror":
            self.tgl_mirror.value = not self.tgl_mirror.value
        elif action.startswith("pick:"):
            self.field.text = action[5:]

    def _browse(self, ctx) -> None:
        try:
            import tkinter  # noqa: F401
            from tkinter import filedialog

            root = tkinter.Tk()
            root.withdraw()
            path = filedialog.askopenfilename(
                filetypes=[("videos", "*.mp4 *.mov *.avi *.mkv *.webm")]
            )
            root.destroy()
            if path:
                self.field.text = path
                return
        except Exception:
            pass
        self._set_status("no file dialog available — type the path or pick a detected video", theme.DIM)

    def update(self, ctx, dt) -> None:
        self._sync(ctx)
        for wdgt in (self.btn_analyze, self.btn_start, self.btn_back, self.btn_browse):
            wdgt.update(dt)
        if self.error:
            self._set_status(f"error: {self.error}", theme.MISS)
            self.progress, self.error = -1.0, ""
        if self.choreo is not None and self.progress >= 0:
            self.progress = -1.0
            self.btn_start.enabled = True
            n = self.choreo.detected_count()
            if n == 0:
                self._set_status("no person detected in the video — try another clip", theme.MISS)
                self.btn_start.enabled = False
            else:
                self._set_status("ready with reference skeleton — START DANCING", theme.GOOD)
                self._fill_info(ctx.video_path)

    def draw(self, ctx, surf) -> None:
        _bg(pygame, ctx, surf, ctx.t)
        _title(pygame, ctx, surf, ctx.t, y=54)
        f_sm, f_med = ctx.fonts["sm"], ctx.fonts["med"]
        head = f_med.render(
            "CHOOSE YOUR DANCE" + (" — PERFORMANCE" if self.tgl_mode.value else " — PRACTICE"),
            True, theme.CYAN,
        )
        surf.blit(head, (60, 250))
        self.field.draw(pygame, surf, f_sm, ctx.t)
        self.btn_analyze.draw(pygame, surf, f_med, ctx.t)
        self.btn_start.draw(pygame, surf, f_med, ctx.t)
        self.tgl_mode.draw(pygame, surf, f_sm)
        self.tgl_mirror.draw(pygame, surf, f_sm)
        for b in self.video_buttons:
            b.draw(pygame, surf, f_sm, ctx.t)
        if self.info:
            surf.blit(f_sm.render(self.info, True, theme.TEXT), (60, 465))
        surf.blit(f_sm.render(self.status, True, self.status_color), (60, 585))
        if 0 <= self.progress <= 1:
            self.bar.draw(pygame, surf, self.progress, dt=0.016)
        if self.choreo is not None:
            i = next(
                (k for k in range(self.choreo.num_samples)
                 if self.choreo.keypoints[k][:, 2].max() > 0),
                None,
            )
            if i is not None:
                pygame.draw.rect(surf, theme.PANEL, (1050, 380, 180, 240))
                pygame.draw.rect(surf, theme.PANEL_EDGE, (1050, 380, 180, 240), 1)
                fitted = fit_keypoints(self.choreo.keypoints[i], self.choreo.width, self.choreo.height, 1050, 380, 180, 240)
                _neon_skeleton(pygame, surf, fitted, ctx.cfg.confidence)
        self.btn_back.draw(pygame, surf, f_med, ctx.t)
        self.btn_browse.draw(pygame, surf, f_med, ctx.t)


# --- PLAY --------------------------------------------------------------------

POPUP_TTL = 0.9


class PlayScreen(Screen):
    """Gameplay: reference skeleton vs live camera + score/combo/judgements."""

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
        self.paused = False
        self.flash = 0.0
        self.results_done = False
        self.smoother = ScoreSmoother()
        self.corrections = CorrectionTracker()

    def _ensure(self, ctx) -> None:
        """One-time init needing ctx (camera + session); safe to call often."""
        if self._ready:
            return
        self._ready = True
        try:
            cam_cfg = AppConfig(
                camera_index=int(ctx.settings.get("camera_index", ctx.cfg.camera_index)),
                frame_width=ctx.cfg.frame_width,
                frame_height=ctx.cfg.frame_height,
            )
            self.cap = open_camera(cam_cfg)
        except CameraError as exc:
            self.cam_err = str(exc)
        if ctx.mode == "performance":
            self.session = GameSession(ctx.moves)
        else:
            self.session = PracticeSession(ctx.choreo, ctx.moves, ctx.cfg.window_s)

    def exit(self) -> None:
        if getattr(self, "cap", None) is not None:
            self.cap.release()
            self.cap = None

    def _eval_feed(self, ctx) -> tuple[float | None, bool, np.ndarray | None, object]:
        """Compute accuracy + reliability + display hint against the window."""
        rows = ctx.choreo.keypoints[window_indices(ctx.choreo, self.session.time, ctx.cfg.window_s)]
        if ctx.mirror and rows.size:
            rows = np.stack([mirror_keypoints(r, ctx.choreo.width) for r in rows])
        accuracy, reliable, hint, ref_row = None, False, None, None
        if self.live is not None and rows.size:
            best = best_window_score(self.live.keypoints, rows, ctx.cfg.confidence)
            if best is not None:
                analysis = analyze_pose(self.live.keypoints, rows[best[1]], ctx.cfg.confidence)
                accuracy = self.smoother.update(analysis.pose_score.score)
                reliable = True
                hint = _pick_hint(analysis, self.live.keypoints, rows[best[1]])
                ref_row = rows[best[1]]
        else:
            self.smoother.update(None)
        return accuracy, reliable, hint, ref_row

    def _on_eval(self, ctx, ev) -> None:
        if ev is None:
            return
        j = judge(ev.accuracy, ev.hit)
        color = theme.JUDGE_COLORS[j.value]
        self.popups.append([j.value, color, ctx.t])
        if ev.hit and ev.combo and ev.combo % self.session.config.combo_step == 0:
            mult = min(1 + ev.combo // self.session.config.combo_step,
                       self.session.config.max_multiplier)
            self.popups.append([f"COMBO x{mult}!", theme.PINK, ctx.t])
            ctx.particles.burst(W // 2, 140, theme.PINK, n=30)
            self.flash = 0.5

    def handle_event(self, ctx, ev) -> None:
        self._ensure(ctx)
        if ev.type != pygame.KEYDOWN:
            return
        k = ev.key
        if k in (pygame.K_ESCAPE, pygame.K_p):
            if isinstance(self.session, GameSession):
                self.session.toggle_pause()
            else:
                self.paused = not self.paused
        elif k == pygame.K_q:
            ctx.manager.replace(SelectScreen())
        elif k in (pygame.K_SPACE, pygame.K_RETURN):
            if isinstance(self.session, GameSession):
                if self.session.phase == GamePhase.BRIEFING:
                    self.session.start_move()
                elif self.session.phase == GamePhase.RESULT:
                    self.session.continue_from_result()
        elif k == pygame.K_r:
            if isinstance(self.session, GameSession):
                if self.session.phase == GamePhase.RESULT:
                    self.session.retry_move()
            else:
                self.session.set_move(0)
        elif k == pygame.K_n and isinstance(self.session, PracticeSession):
            self.session.set_move(self.session.move_index + 1)
        elif k == pygame.K_m:
            ctx.mirror = not ctx.mirror

    def update(self, ctx, dt) -> None:
        self._ensure(ctx)
        if self.cap is None:
            return
        ok, frame = self.cap.read()
        self.frame = frame if ok else None
        self.live = ctx.tracker.predict(frame) if (ok and ctx.tracker) else None
        self.frame_id += 1
        accuracy, reliable, hint, _ = self._eval_feed(ctx)
        if isinstance(self.session, GameSession):
            ev = self.session.update(time.perf_counter(), accuracy, reliable, self.frame_id)
            if not self.session.paused:
                self._on_eval(ctx, ev)
        elif not self.paused:
            self.session.advance(dt)
        self.hint = self.corrections.update(hint)
        self.flash = max(0.0, self.flash - dt * 2)
        if (
            isinstance(self.session, GameSession)
            and self.session.phase == GamePhase.COMPLETE
            and not self.results_done
        ):
            self.results_done = True
            ctx.last_session = self.session
            ctx.last_results = self.session.results
            ctx.manager.replace(ResultsScreen())

    def _panels(self, ctx, surf) -> tuple:
        ref_rect = (30, 110, 600, 470)
        cam_rect = (650, 110, 600, 470)
        for r in (ref_rect, cam_rect):
            pygame.draw.rect(surf, theme.PANEL, r)
            pygame.draw.rect(surf, theme.PANEL_EDGE, r, 1, border_radius=4)
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
        f_sm, f_med, f_big = ctx.fonts["sm"], ctx.fonts["med"], ctx.fonts["big"]
        perf = isinstance(self.session, GameSession)
        surf.blit(f_med.render(f"OBJECTIVE {self.session.move_index + 1}/{len(self.session.moves)}: {self.session.move.name}", True, theme.PINK), (30, 20))
        if perf:
            surf.blit(f_med.render(f"SCORE {self.session.total_score}", True, theme.GOLD), (900, 20))
            mult = min(1 + self.session.combo // self.session.config.combo_step, self.session.config.max_multiplier)
            surf.blit(f_med.render(f"COMBO x{mult} ({self.session.combo})", True, theme.CYAN), (900, 52))
        self._panels(ctx, surf)
        # beat-synced pulse: border glow on the reference panel + popup kick
        pulse = ctx.beats.pulse(self.session.time) if ctx.beats is not None else 0.0
        if pulse > 0.05:
            glow = tuple(int(a + (b - a) * pulse) for a, b in zip(theme.PANEL_EDGE, theme.CYAN))
            pygame.draw.rect(surf, glow, (30, 110, 600, 470), 1 + int(3 * pulse), border_radius=4)
        # bottom HUD
        acc = getattr(self.session, "current_accuracy", None) or 0.0
        surf.blit(f_sm.render(f"ACCURACY {acc * 100:3.0f}%", True, theme.TEXT), (30, 600))
        pygame.draw.rect(surf, theme.PANEL_EDGE, (200, 602, 300, 14))
        pygame.draw.rect(surf, theme.CYAN if acc >= 0.6 else theme.MISS, (200, 602, int(300 * min(acc, 1.0)), 14))
        prog = self.session.move_progress if perf else ((self.session.time - self.session.move.start) / max(self.session.move.duration, 1e-6))
        surf.blit(f_sm.render(f"MOVE {min(prog,1) * 100:3.0f}%", True, theme.TEXT), (540, 600))
        pygame.draw.rect(surf, theme.PANEL_EDGE, (680, 602, 300, 14))
        pygame.draw.rect(surf, theme.PINK, (680, 602, int(300 * min(prog, 1.0)), 14))
        hint_txt = self.hint or ("match the pose" if perf else "practice — copy the dancer")
        surf.blit(f_med.render(hint_txt, True, theme.MISS if self.hint else theme.DIM), (30, 636))
        keys = "P pause · M mirror · Q back" if not perf else "SPACE confirm · R retry · P pause · M mirror · Q back"
        surf.blit(f_sm.render(keys, True, theme.DIM), (30, 690))
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
        # phase overlays
        if perf and self.session.phase == GamePhase.BRIEFING:
            self._overlay(ctx, surf, f"MOVE {self.session.move_index + 1}", self.session.move.name, "press SPACE to dance", theme.CYAN)
        elif perf and self.session.phase == GamePhase.RESULT and self.session.last_result:
            r = self.session.last_result
            self._overlay(ctx, surf, "MOVE PASSED!" if r.passed else "MOVE FAILED",
                          f"accuracy {r.mean_accuracy * 100:.0f}% · +{r.points} pts · best combo {r.best_combo}",
                          "SPACE next move" if r.passed else "R retry move",
                          theme.GOOD if r.passed else theme.MISS)
        paused = self.session.paused if perf else self.paused
        if paused:
            self._overlay(ctx, surf, "PAUSED", "", "P resume · Q quit to menu", theme.GOLD)

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


# --- RESULTS -----------------------------------------------------------------


class ResultsScreen(Screen):
    def enter(self) -> None:
        self.card = None
        self._saved = False
        self.buttons = []
        self.group = None
        self.born = time.perf_counter()

    def _sync(self, ctx) -> None:
        if self.card is None and isinstance(getattr(ctx, "last_session", None), GameSession):
            self.card = ctx.last_session.scorecard()
        if self.card is None:
            return
        if not self._saved:
            self._saved = True
            add_score(
                ctx.video_path.name if ctx.video_path else "?",
                self.card["total_score"], self.card["mean_accuracy"],
                self.card["best_combo"], grade_for(self.card["mean_accuracy"]),
            )
            ctx.particles.burst(W // 2, 200, theme.GOLD, n=60)
        if not self.buttons:
            self.buttons = [
                Button(pygame.Rect(W // 2 - 340, 600, 210, 52), "RETRY", "retry", True),
                Button(pygame.Rect(W // 2 - 105, 600, 230, 52), "ANOTHER DANCE", "select"),
                Button(pygame.Rect(W // 2 + 150, 600, 190, 52), "HOME", "home"),
            ]
            self.group = FocusGroup(self.buttons)

    def _activate(self, ctx, action: str) -> None:
        if action == "retry":
            ctx.manager.replace(PlayScreen())
        elif action == "select":
            ctx.manager.replace(SelectScreen())
        elif action == "home":
            ctx.manager.replace(HomeScreen())

    def handle_event(self, ctx, ev) -> None:
        self._sync(ctx)
        if not self.buttons:
            return
        if ev.type == pygame.MOUSEMOTION:
            self.group.hover(ev.pos)
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            for b in self.buttons:
                if b.contains(ev.pos):
                    self._activate(ctx, b.action)
        elif ev.type == pygame.KEYDOWN:
            if ev.key in (pygame.K_LEFT, pygame.K_a):
                self.group.move(-1)
            elif ev.key in (pygame.K_RIGHT, pygame.K_d):
                self.group.move(1)
            elif ev.key in (pygame.K_RETURN, pygame.K_SPACE):
                self._activate(ctx, self.group.current.action)

    def update(self, ctx, dt) -> None:
        self._sync(ctx)

    def draw(self, ctx, surf) -> None:
        _bg(pygame, ctx, surf, ctx.t)
        if not self.card:
            surf.blit(ctx.fonts["med"].render("no results yet", True, theme.DIM), (500, 350))
            return
        card = self.card
        grade = grade_for(card["mean_accuracy"])
        pop = min(1.0, (time.perf_counter() - self.born) * 2.5)
        gi = theme.glow_text(pygame, ctx.fonts["big"], f"GRADE {grade}", theme.GOLD)
        gi = pygame.transform.rotozoom(gi, 0, 0.5 + 0.5 * pop)
        surf.blit(gi, gi.get_rect(center=(W // 2, 120)))
        stats = (f"score {card['total_score']}  ·  accuracy {card['mean_accuracy'] * 100:.0f}%  ·  "
                 f"best combo {card['best_combo']}  ·  {card['moves_passed']}/{card['moves_total']} moves")
        t1 = ctx.fonts["med"].render(stats, True, theme.TEXT)
        surf.blit(t1, t1.get_rect(center=(W // 2, 210)))
        y = 270
        for r in getattr(ctx, "last_results", []):
            mark, color = ("PASS", theme.GOOD) if r.passed else ("FAIL", theme.MISS)
            row = ctx.fonts["sm"].render(
                f"[{mark}] {r.name:<24} acc {r.mean_accuracy * 100:3.0f}%  +{r.points} pts", True, color)
            surf.blit(row, row.get_rect(center=(W // 2, y)))
            y += 30
        ctx.particles.draw(pygame, surf)
        for b in getattr(self, "buttons", []):
            b.draw(pygame, surf, ctx.fonts["med"], ctx.t)


# --- SETTINGS / SCORES -------------------------------------------------------


class SettingsScreen(Screen):
    def enter(self) -> None:
        self.cam = 0
        self.tgl_mirror = Toggle(pygame.Rect(120, 300, 420, 30), "mirror choreography default", "mirror", True)
        self.btn_cam_m = Button(pygame.Rect(120, 390, 60, 44), "−", "cam-")
        self.btn_cam_p = Button(pygame.Rect(300, 390, 60, 44), "+", "cam+")
        self.btn_back = Button(pygame.Rect(120, 560, 160, 48), "BACK", "back")
        self.group = FocusGroup([self.tgl_mirror, self.btn_cam_m, self.btn_cam_p, self.btn_back])

    def _activate(self, ctx, action: str) -> None:
        if action == "back":
            ctx.settings["camera_index"] = self.cam
            save_settings(ctx.settings)
            ctx.manager.pop()
        elif action == "mirror":
            self.tgl_mirror.value = not self.tgl_mirror.value
            ctx.settings["mirror"] = self.tgl_mirror.value
            ctx.mirror = self.tgl_mirror.value
        elif action == "cam-":
            self.cam = max(0, self.cam - 1)
        elif action == "cam+":
            self.cam = min(7, self.cam + 1)

    def handle_event(self, ctx, ev) -> None:
        self.cam = int(ctx.settings.get("camera_index", 0)) if not getattr(self, "_touched", False) else self.cam
        items = [self.tgl_mirror, self.btn_cam_m, self.btn_cam_p, self.btn_back]
        if ev.type == pygame.MOUSEMOTION:
            for i in items:
                i.hovered = i.contains(ev.pos)
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            for i in items:
                if i.contains(ev.pos):
                    self._touched = True
                    self._activate(ctx, i.action)
        elif ev.type == pygame.KEYDOWN:
            if ev.key in (pygame.K_DOWN, pygame.K_s):
                self.group.move(1)
            elif ev.key in (pygame.K_UP, pygame.K_w):
                self.group.move(-1)
            elif ev.key in (pygame.K_RETURN, pygame.K_SPACE):
                self._touched = True
                self._activate(ctx, self.group.current.action)
            elif ev.key == pygame.K_ESCAPE:
                self._activate(ctx, "back")

    def update(self, ctx, dt) -> None:
        if not getattr(self, "_synced", False):
            self.tgl_mirror.value = bool(ctx.settings.get("mirror", True))
            self.cam = int(ctx.settings.get("camera_index", 0))
            self._synced = True
        for b in (self.btn_cam_m, self.btn_cam_p, self.btn_back):
            b.update(dt)

    def draw(self, ctx, surf) -> None:
        _bg(pygame, ctx, surf, ctx.t)
        _title(pygame, ctx, surf, ctx.t, y=70)
        surf.blit(ctx.fonts["med"].render("SETTINGS", True, theme.CYAN), (120, 240))
        self.tgl_mirror.draw(pygame, surf, ctx.fonts["sm"])
        surf.blit(ctx.fonts["sm"].render("camera index", True, theme.TEXT), (120, 370))
        val = ctx.fonts["med"].render(str(self.cam), True, theme.GOLD)
        surf.blit(val, val.get_rect(center=(245, 412)))
        self.btn_cam_m.draw(pygame, surf, ctx.fonts["med"], ctx.t)
        self.btn_cam_p.draw(pygame, surf, ctx.fonts["med"], ctx.t)
        note = ctx.fonts["sm"].render("camera applies on next open", True, theme.DIM)
        surf.blit(note, (120, 450))
        self.btn_back.draw(pygame, surf, ctx.fonts["med"], ctx.t)


class ScoresScreen(Screen):
    def enter(self) -> None:
        self.scores = load_scores()
        self.btn_back = Button(pygame.Rect(120, 600, 160, 48), "BACK", "back")

    def handle_event(self, ctx, ev) -> None:
        if ev.type == pygame.MOUSEBUTTONDOWN and self.btn_back.contains(ev.pos):
            ctx.manager.pop()
        elif ev.type == pygame.KEYDOWN and ev.key in (pygame.K_ESCAPE, pygame.K_RETURN, pygame.K_BACKSPACE):
            ctx.manager.pop()

    def draw(self, ctx, surf) -> None:
        _bg(pygame, ctx, surf, ctx.t)
        _title(pygame, ctx, surf, ctx.t, y=70)
        surf.blit(ctx.fonts["med"].render("HIGH SCORES", True, theme.GOLD), (120, 200))
        if not self.scores:
            surf.blit(ctx.fonts["sm"].render("no runs recorded yet — go dance!", True, theme.DIM), (120, 260))
        y = 250
        for i, s in enumerate(self.scores):
            line = (f"{i + 1:>2}. {s['score']:>6}  [{s['grade']}]  {s['mean_accuracy'] * 100:3.0f}%  "
                    f"combo {s['best_combo']:>3}  {Path(s['video']).name[:28]}")
            color = theme.GOLD if i == 0 else theme.TEXT
            surf.blit(ctx.fonts["sm"].render(line, True, color), (120, y))
            y += 32
        self.btn_back.draw(pygame, surf, ctx.fonts["med"], ctx.t)
