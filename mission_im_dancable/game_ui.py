"""Pygame-ce mission-control game interface.

The game screen shows the reference skeleton (left) and the live camera
with the player's skeleton (right), plus score, combo, accuracy, and one
correction hint. All game rules live in game.GameSession — this module is
input + drawing only.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import torch

from .camera import CameraError, open_camera
from .config import AppConfig, resolve_device
from .game import GamePhase, GameSession
from .geometry import SKELETON_EDGES
from .moves import MovesFileError, default_moves, load_moves
from .pose import PoseTracker
from .reference import (
    ReferenceVideoError,
    load_or_extract,
    nearest_index,
    window_indices,
)
from .render import fit_keypoints
from .scoring import (
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

WIDTH, HEIGHT = 1280, 720
FPS_CAP = 60

BG = (12, 16, 22)
PANEL = (20, 26, 34)
BORDER = (46, 62, 78)
CYAN = (0, 220, 180)
RED = (230, 70, 70)
AMBER = (240, 180, 60)
TEXT = (210, 220, 225)
DIM = (110, 125, 135)

REF_RECT = (20, 90, 600, 480)
CAM_RECT = (660, 90, 600, 480)


def _font(pygame, size: int, bold: bool = False):
    return pygame.font.SysFont("dejavusansmono", size, bold=bold)


def _text(pygame, surface, font, s: str, x: int, y: int, color=TEXT):
    surface.blit(font.render(s, True, color), (x, y))


def _frame_to_surface(pygame, frame: np.ndarray, w: int, h: int):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
    return pygame.transform.scale(surf, (w, h))


def _draw_skeleton(pygame, surface, keypoints: np.ndarray, conf: float, color=CYAN) -> None:
    for a, b in SKELETON_EDGES:
        pa, pb = keypoints[a], keypoints[b]
        if pa[2] >= conf and pb[2] >= conf:
            pygame.draw.line(
                surface, color, (float(pa[0]), float(pa[1])), (float(pb[0]), float(pb[1])), 2
            )
    for x, y, c in keypoints:
        if c >= conf:
            pygame.draw.circle(surface, RED, (float(x), float(y)), 4)


def _bar(pygame, surface, rect, frac: float, color, bg=BORDER) -> None:
    x, y, w, h = rect
    pygame.draw.rect(surface, bg, rect)
    pygame.draw.rect(surface, color, (x, y, int(w * min(max(frac, 0.0), 1.0)), h))


def _overlay(pygame, surface, alpha: int = 200) -> None:
    veil = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    veil.fill((8, 10, 14, alpha))
    surface.blit(veil, (0, 0))


def _pick_hint(
    analysis, live: np.ndarray, reference: np.ndarray
) -> str | None:
    """Choose the single most useful correction: worst angle vs worst point."""
    wa = analysis.worst_angle()
    wk = worst_mismatch(analysis.pose_score)
    angle_norm = (wa[1] / 90.0) if wa else 0.0
    point_err = (
        analysis.pose_score.keypoint_errors[wk] if wk is not None else 0.0
    )
    if wa and angle_norm >= point_err:
        live_deg = joint_angle_deg(live, wa[0])
        ref_deg = joint_angle_deg(reference, wa[0])
        if live_deg is not None and ref_deg is not None:
            return describe_angle_correction(wa[0], live_deg, ref_deg)
    if wk is not None:
        return describe_correction(wk, live, reference)
    return None


def run_game(cfg: AppConfig) -> int:
    """Pygame game loop. Returns a process exit code."""
    import pygame

    device = resolve_device(cfg.device, torch.cuda.is_available())

    pygame.init()
    pygame.display.set_caption("MISSION IM-DANCABLE")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    f_title = _font(pygame, 30, bold=True)
    f_big = _font(pygame, 44, bold=True)
    f_med = _font(pygame, 20, bold=True)
    f_sm = _font(pygame, 16)

    def _quit_events() -> bool:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (
                e.type == pygame.KEYDOWN and e.key in (pygame.K_ESCAPE, pygame.K_q)
            ):
                return True
        return False

    def _fatal_screen(line1: str, line2: str = "press Q to abort") -> None:
        while True:
            if _quit_events():
                return
            screen.fill(BG)
            _overlay(pygame, screen, 120)
            _text(pygame, screen, f_title, line1, 60, 300, RED)
            _text(pygame, screen, f_med, line2, 60, 350, TEXT)
            pygame.display.flip()
            clock.tick(30)

    def _loading(lines: list[str]) -> None:
        screen.fill(BG)
        _text(pygame, screen, f_title, "MISSION IM-DANCABLE", 60, 60, CYAN)
        for i, line in enumerate(lines):
            _text(pygame, screen, f_med, line, 60, 160 + i * 34, TEXT)
        pygame.display.flip()

    # --- load model + reference before the active loop ---------------------
    _loading(["loading pose model...", f"device: {device.upper()}"])
    try:
        tracker = PoseTracker(cfg.model_path, device, cfg.confidence)
    except (FileNotFoundError, OSError) as exc:
        _fatal_screen(f"MODEL ERROR: {exc}")
        pygame.quit()
        return 2

    _loading(["extracting reference poses...", "this can take a moment on first run"])
    try:
        choreo = load_or_extract(
            tracker, cfg.source, cfg.sample_fps, cfg.cache_dir,
            cfg.confidence, model_name=cfg.model_path.name,
        )
    except (ReferenceVideoError, FileNotFoundError, OSError) as exc:
        _fatal_screen(f"REFERENCE ERROR: {exc}")
        pygame.quit()
        return 2

    try:
        moves = (
            load_moves(cfg.moves_path, choreo.duration)
            if cfg.moves_path
            else default_moves(choreo.duration)
        )
    except MovesFileError as exc:
        _fatal_screen(f"MOVES FILE ERROR: {exc}")
        pygame.quit()
        return 2

    try:
        cap = open_camera(cfg)
    except CameraError as exc:
        _fatal_screen(f"CAMERA OFFLINE: {exc}", "check connection, then relaunch")
        pygame.quit()
        return 2

    session = GameSession(moves)
    smoother = ScoreSmoother()
    corrections = CorrectionTracker()
    mirror = cfg.mirror
    frame_id = 0
    hit_flash = 0.0

    def _ref_pose_at(t: float) -> np.ndarray | None:
        i = nearest_index(choreo, t)
        if i is None:
            return None
        row = choreo.keypoints[i]
        return mirror_keypoints(row, choreo.width) if mirror else row

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif e.key in (pygame.K_SPACE, pygame.K_RETURN):
                    if session.phase == GamePhase.BRIEFING:
                        session.start_move()
                    elif session.phase == GamePhase.RESULT:
                        session.continue_from_result()
                elif e.key == pygame.K_r:
                    if session.phase == GamePhase.RESULT:
                        session.retry_move()
                    elif session.phase == GamePhase.COMPLETE:
                        session = GameSession(moves)
                elif e.key == pygame.K_p:
                    session.toggle_pause()
                elif e.key == pygame.K_m:
                    mirror = not mirror

        # --- capture + infer every frame; the session gates scoring --------
        ok, frame = cap.read()
        frame_id += 1
        live = tracker.predict(frame) if ok else None

        hint = None
        if session.phase == GamePhase.ACTIVE and not session.paused:
            rows = choreo.keypoints[
                window_indices(choreo, session.time, cfg.window_s)
            ]
            if mirror and rows.size:
                rows = np.stack(
                    [mirror_keypoints(r, choreo.width) for r in rows]
                )
            accuracy = None
            reliable = False
            if live is not None and rows.size:
                best = best_window_score(live.keypoints, rows, cfg.confidence)
                if best is not None:
                    analysis = analyze_pose(
                        live.keypoints, rows[best[1]], cfg.confidence
                    )
                    accuracy = smoother.update(analysis.pose_score.score)
                    reliable = True
                    hint = _pick_hint(analysis, live.keypoints, rows[best[1]])
            else:
                smoother.update(None)
            event = session.update(
                time.perf_counter(), accuracy, reliable, frame_id
            )
            if event and event.hit:
                hit_flash = 0.4
        else:
            session.update(time.perf_counter(), None, False, frame_id)

        hint = corrections.update(hint) if session.phase == GamePhase.ACTIVE else corrections.update(None)
        hit_flash = max(0.0, hit_flash - 1.0 / FPS_CAP)

        # --- draw ------------------------------------------------------------
        screen.fill(BG)
        _text(pygame, screen, f_title, "MISSION IM-DANCABLE", 20, 16, CYAN)
        _text(
            pygame, screen, f_med,
            f"OBJECTIVE {session.move_index + 1}/{len(moves)}: {session.move.name}",
            640, 24, TEXT,
        )
        _text(pygame, screen, f_med, f"SCORE {session.total_score}", 20, 56, AMBER)
        _text(
            pygame, screen, f_med,
            f"COMBO x{min(1 + session.combo // session.config.combo_step, session.config.max_multiplier)} ({session.combo})",
            320, 56, CYAN if session.combo else DIM,
        )

        # reference panel
        rx, ry, rw, rh = REF_RECT
        pygame.draw.rect(screen, PANEL, REF_RECT)
        pygame.draw.rect(screen, BORDER, REF_RECT, 1)
        ref_pose = _ref_pose_at(session.time)
        if ref_pose is not None and ref_pose[:, 2].max() > 0:
            fitted = fit_keypoints(ref_pose, choreo.width, choreo.height, rx, ry, rw, rh)
            _draw_skeleton(pygame, screen, fitted, cfg.confidence)
        else:
            _text(pygame, screen, f_med, "NO REFERENCE POSE", rx + 180, ry + rh // 2, DIM)
        _text(pygame, screen, f_sm, "REFERENCE" + (" (MIRRORED)" if mirror else ""), rx + 8, ry + 8, DIM)

        # camera panel
        cx, cy, cw, ch = CAM_RECT
        pygame.draw.rect(screen, PANEL, CAM_RECT)
        pygame.draw.rect(screen, BORDER, CAM_RECT, 1)
        if ok:
            screen.blit(_frame_to_surface(pygame, frame, cw, ch), (cx, cy))
            if live is not None:
                fitted = fit_keypoints(live.keypoints, frame.shape[1], frame.shape[0], cx, cy, cw, ch)
                _draw_skeleton(pygame, screen, fitted, cfg.confidence)
        else:
            _text(pygame, screen, f_med, "SIGNAL LOST", cx + 220, cy + ch // 2, RED)
        _text(pygame, screen, f_sm, "YOU", cx + 8, cy + 8, DIM)

        # bottom HUD: accuracy bar, progress bar, correction, controls
        acc = session.current_accuracy or 0.0
        _text(pygame, screen, f_sm, f"ACCURACY {acc * 100:3.0f}%", 20, 590, TEXT)
        _bar(pygame, screen, (160, 592, 280, 14), acc, CYAN if acc >= session.config.acc_threshold else RED)
        _text(pygame, screen, f_sm, f"MOVE {session.move_progress * 100:3.0f}%", 460, 590, TEXT)
        _bar(pygame, screen, (560, 592, 400, 14), session.move_progress, AMBER)
        _text(pygame, screen, f_med, hint or ("match the reference pose" if session.phase == GamePhase.ACTIVE else ""), 20, 630, RED if hint else DIM)
        _text(pygame, screen, f_sm, "SPACE confirm · R retry · P pause · M mirror · Q quit", 20, 680, DIM)
        if hit_flash > 0:
            _text(pygame, screen, f_med, "+HIT", 1180, 56, CYAN)

        # --- phase overlays --------------------------------------------------
        if session.phase == GamePhase.BRIEFING:
            _overlay(pygame, screen)
            _text(pygame, screen, f_big, f"MISSION {session.move_index + 1}", 430, 260, CYAN)
            _text(pygame, screen, f_title, session.move.name, 430, 320, TEXT)
            _text(pygame, screen, f_med, f"{session.move.start:.1f}s – {session.move.end:.1f}s of the track", 430, 370, DIM)
            _text(pygame, screen, f_med, "press SPACE to begin", 430, 430, AMBER)
        elif session.phase == GamePhase.RESULT and session.last_result:
            r = session.last_result
            _overlay(pygame, screen)
            color = CYAN if r.passed else RED
            _text(pygame, screen, f_big, "OBJECTIVE COMPLETE" if r.passed else "OBJECTIVE FAILED", 330, 240, color)
            _text(pygame, screen, f_med, f"accuracy {r.mean_accuracy * 100:.0f}%  ·  reliable {r.reliable_evaluations}/{r.evaluations}  ·  +{r.points} pts  ·  best combo {r.best_combo}", 300, 320, TEXT)
            prompt = "SPACE next objective" if r.passed else "R retry objective"
            _text(pygame, screen, f_med, prompt, 330, 380, AMBER)
        elif session.phase == GamePhase.COMPLETE:
            card = session.scorecard()
            _overlay(pygame, screen)
            _text(pygame, screen, f_big, "MISSION COMPLETE", 330, 160, CYAN)
            _text(pygame, screen, f_med, f"final score: {card['total_score']}", 380, 250, AMBER)
            _text(pygame, screen, f_med, f"mean accuracy: {card['mean_accuracy'] * 100:.0f}%   best combo: {card['best_combo']}", 380, 290, TEXT)
            y = 350
            for r in session.results:
                mark = "PASS" if r.passed else "FAIL"
                _text(pygame, screen, f_sm, f"[{mark}] {r.name:<24} acc {r.mean_accuracy * 100:3.0f}%  +{r.points} pts", 380, y, CYAN if r.passed else RED)
                y += 26
            _text(pygame, screen, f_med, "R run it again · Q quit", 380, y + 30, AMBER)
        if session.paused:
            _overlay(pygame, screen, 150)
            _text(pygame, screen, f_big, "PAUSED", 520, 320, AMBER)

        pygame.display.flip()
        clock.tick(FPS_CAP)

    cap.release()
    pygame.quit()
    return 0
