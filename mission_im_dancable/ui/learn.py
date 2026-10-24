"""Learn Mode screen: study a dance at your own pace.

Reference skeleton on the left, live camera on the right, a seekable
segment timeline on the bottom. No score, no combo, no fail state — only
match feedback and self-marked progress. Camera failure degrades to a
watch-only mode instead of blocking the lesson.
"""

from __future__ import annotations

import threading

import pygame

from ..camera import CameraError
from ..learn import LearnSession, SEEK_STEP
from ..persist import MOVE_STATUSES, move_statuses, set_move_status
from ..reference import pose_at
from ..render import fit_keypoints
from ..scoring import CorrectionTracker, ScoreSmoother, mirror_keypoints
from . import theme
from .screens import (
    Screen,
    SelectScreen,
    _bg,
    _chip,
    _dim_strip,
    _frame_surf,
    _neon_skeleton,
    _open_camera_ctx,
    _panel_label,
    _title,
    close_camera,
    eval_against_reference,
)
from .widgets import Button, TimelineBar

REF_RECT = (60, 190, 560, 350)
CAM_RECT = (660, 190, 560, 350)
STATUS_COLORS = {"learning": theme.VIOLET, "practicing": theme.CYAN, "mastered": theme.GOLD}


class LearnScreen(Screen):
    """Free-form dance study: seekable timeline, segments, loop, speed."""

    def enter(self) -> None:
        self._ready = False
        self.session = None
        self.statuses: dict[int, str] = {}
        self._last_selected = -1
        self.cam_state = "idle"  # idle → opening → ready | error
        self.cam_err: str | None = None
        self._canceled = False
        self._thread: threading.Thread | None = None
        self.frame = None
        self.live = None
        self.accuracy: float | None = None
        self.hint = None
        self.no_pose_t = 0.0
        self.smoother = ScoreSmoother()
        self.corrections = CorrectionTracker()
        self._reanchor = False
        self._music = None      # ctx.music captured for exit()
        self._music_on = False
        self._prev_t = 0.0

        self.timeline = TimelineBar(pygame.Rect(60, 610, 1160, 20))
        self.btn_play = Button(pygame.Rect(60, 650, 120, 44), "▶ PLAY", "play", True)
        self.btn_speed = Button(pygame.Rect(190, 650, 110, 44), "1×", "speed")
        self.btn_prev = Button(pygame.Rect(310, 650, 110, 44), "◀ SEG", "prev")
        self.btn_next = Button(pygame.Rect(430, 650, 110, 44), "SEG ▶", "next")
        self.btn_loop = Button(pygame.Rect(550, 650, 150, 44), "LOOP OFF", "loop")
        self.btn_mark = Button(pygame.Rect(710, 650, 220, 44), "MARK: —", "mark")
        self.btn_mirror = Button(pygame.Rect(940, 650, 110, 44), "MIRROR", "mirror")
        self.btn_back = Button(pygame.Rect(1060, 650, 160, 44), "BACK", "back")
        self._buttons = [
            self.btn_play, self.btn_speed, self.btn_prev, self.btn_next,
            self.btn_loop, self.btn_mark, self.btn_mirror, self.btn_back,
        ]

    # -- one-time setup --------------------------------------------------------

    def _ensure(self, ctx) -> None:
        if self._ready:
            return
        self._ready = True
        self.session = LearnSession(ctx.moves, ctx.choreo.duration, ctx.choreo.video_fps)
        if ctx.video_path is not None:
            self.statuses = move_statuses(ctx.video_path.name)
        if ctx.cap is None:
            self._open(ctx)
        else:
            self.cam_state = "ready"

    def _open(self, ctx) -> None:
        self.cam_state = "opening"
        self.cam_err = None

        def work() -> None:
            try:
                _open_camera_ctx(ctx)
            except CameraError as exc:
                self.cam_err = str(exc)
                self.cam_state = "error"
            except Exception as exc:
                self.cam_err = f"camera failed: {exc}"
                self.cam_state = "error"
            if self._canceled:
                close_camera(ctx)

        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()

    def _retry_camera(self, ctx) -> None:
        close_camera(ctx)
        self._open(ctx)

    # -- progress marks -----------------------------------------------------------

    def _video_key(self, ctx) -> str | None:
        return ctx.video_path.name if ctx.video_path is not None else None

    def _note_selected(self, ctx) -> None:
        """First visit to a segment marks it 'learning' — progress, not skill."""
        video = self._video_key(ctx)
        if video is None or self.session.selected in self.statuses:
            return
        self.statuses = set_move_status(video, self.session.selected, "learning")

    def _cycle_status(self, ctx) -> None:
        video = self._video_key(ctx)
        if video is None:
            return
        cur = self.statuses.get(self.session.selected)
        nxt = MOVE_STATUSES[0] if cur is None else (
            MOVE_STATUSES[MOVE_STATUSES.index(cur) + 1]
            if cur != MOVE_STATUSES[-1]
            else None
        )
        self.statuses = set_move_status(video, self.session.selected, nxt)

    # -- soundtrack -----------------------------------------------------------------

    def _sync_music(self, ctx, prev_t: float) -> None:
        """Audio follows the playhead; only real speed plays audible sound."""
        music = getattr(ctx, "music", None)
        if music is None:
            return
        self._music = music
        s = self.session
        if not (s.playing and s.speed == 1.0):
            if self._music_on:
                music.pause()
                self._music_on = False
            return
        if not music.loaded:
            if not music.load(getattr(ctx, "audio_path", None)):
                return
        wrapped = s.time < prev_t - 1e-3  # loop jumped back to segment start
        if not self._music_on or wrapped or self._reanchor:
            music.play_at(s.time)
            self._reanchor = False
        elif (pos := music.position()) is not None and abs(pos - s.time) > 0.3:
            music.play_at(s.time)
        self._music_on = True

    # -- events ----------------------------------------------------------------------

    def _back(self, ctx) -> None:
        if ctx.manager.depth > 1:
            ctx.manager.pop()
        else:
            ctx.manager.replace(SelectScreen())

    def _activate(self, ctx, action: str) -> None:
        s = self.session
        if action == "play":
            s.toggle()
            self._reanchor = True
        elif action == "speed":
            s.cycle_speed(1)
        elif action == "prev":
            s.prev_segment()
            self._reanchor = True
        elif action == "next":
            s.next_segment()
            self._reanchor = True
        elif action == "loop":
            s.cycle_loop()
        elif action == "mark":
            self._cycle_status(ctx)
        elif action == "mirror":
            ctx.mirror = not ctx.mirror
        elif action == "back":
            self._back(ctx)

    def handle_event(self, ctx, ev) -> None:
        self._ensure(ctx)
        s = self.session
        if ev.type == pygame.MOUSEMOTION:
            for b in self._buttons:
                b.hovered = b.contains(ev.pos)
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            frac = self.timeline.fraction_at(ev.pos)
            if frac is not None:
                s.seek(frac * s.duration)
                s.selected = s.active_index  # the block you clicked is selected
                self._reanchor = True
                return
            for b in self._buttons:
                if b.contains(ev.pos) and b.enabled:
                    b.press()
                    self._activate(ctx, b.action)
        elif ev.type == pygame.KEYDOWN:
            k = ev.key
            if k == pygame.K_SPACE:
                s.toggle()
                self._reanchor = True
            elif k == pygame.K_LEFT:
                s.skip(-SEEK_STEP)
                self._reanchor = True
            elif k == pygame.K_RIGHT:
                s.skip(SEEK_STEP)
                self._reanchor = True
            elif k == pygame.K_COMMA:
                s.step_frames(-1)
                self._reanchor = True
            elif k == pygame.K_PERIOD:
                s.step_frames(1)
                self._reanchor = True
            elif k == pygame.K_n:
                s.next_segment()
                self._reanchor = True
            elif k == pygame.K_p:
                s.prev_segment()
                self._reanchor = True
            elif k == pygame.K_l:
                s.cycle_loop()
            elif k == pygame.K_MINUS:
                s.cycle_speed(-1)
            elif k == pygame.K_EQUALS:
                s.cycle_speed(1)
            elif k == pygame.K_k:
                self._cycle_status(ctx)
            elif k == pygame.K_m:
                ctx.mirror = not ctx.mirror
            elif k == pygame.K_r and self.cam_state == "error":
                self._retry_camera(ctx)
            elif k in (pygame.K_q, pygame.K_ESCAPE):
                self._back(ctx)

    # -- frame ------------------------------------------------------------------------

    def update(self, ctx, dt) -> None:
        self._ensure(ctx)
        s = self.session
        for b in self._buttons:
            b.update(dt)
        if self.cam_state == "opening" and ctx.cap is not None:
            self.cam_state = "ready"
        elif self.cam_state == "ready" and ctx.cap is None:
            self.cam_state = "error"
            self.cam_err = "camera disconnected"
        prev_t = s.time
        if self.cam_state == "ready" and ctx.cap is not None:
            ok, frame = ctx.cap.read()
            self.frame = frame if ok else self.frame
            self.live = ctx.tracker.predict(frame) if (ok and ctx.tracker) else None
        else:
            self.live = None
        if self.live is None:
            self.no_pose_t += dt
        else:
            self.no_pose_t = 0.0

        s.advance(dt)
        if s.selected != self._last_selected:
            self._last_selected = s.selected
            self._note_selected(ctx)

        live_kpts = self.live.keypoints if self.live is not None else None
        raw, reliable, hint, _ = eval_against_reference(ctx, ctx.choreo, s.time, live_kpts)
        self.accuracy = self.smoother.update(raw) if reliable else self.smoother.update(None)
        self.hint = self.corrections.update(hint)
        self._sync_music(ctx, prev_t)
        self._prev_t = s.time

    def exit(self) -> None:
        # Tell the camera thread to drop a late open, and stop the
        # soundtrack. The capture itself is shared on ctx — the select
        # screen releases it when we land back there.
        self._canceled = True
        if self._music is not None:
            self._music.stop()
            self._music_on = False

    # -- draw ----------------------------------------------------------------------------

    def _draw_panels(self, ctx, surf) -> None:
        s = self.session
        for r, label, color in ((REF_RECT, "REFERENCE", theme.VIOLET),
                                (CAM_RECT, "YOU", theme.CYAN)):
            pygame.draw.rect(surf, theme.PANEL, r)
            pygame.draw.rect(surf, theme.PANEL_EDGE, r, 1, border_radius=4)
            _panel_label(pygame, ctx, surf, label, r[0] + 12, r[1] - 24, color)
        pose = pose_at(ctx.choreo, s.time)
        if pose is not None and pose[:, 2].max() > 0:
            shown = mirror_keypoints(pose, ctx.choreo.width) if ctx.mirror else pose
            fitted = fit_keypoints(shown, ctx.choreo.width, ctx.choreo.height, *REF_RECT)
            _neon_skeleton(pygame, surf, fitted, ctx.cfg.confidence, theme.VIOLET)
        if self.frame is not None:
            surf.blit(_frame_surf(pygame, self.frame, CAM_RECT[2], CAM_RECT[3]), CAM_RECT[:2])
            if self.live is not None:
                fitted = fit_keypoints(
                    self.live.keypoints, self.frame.shape[1], self.frame.shape[0], *CAM_RECT
                )
                _neon_skeleton(pygame, surf, fitted, ctx.cfg.confidence, theme.CYAN)
            if self.live is None and self.no_pose_t > 0.6:
                _chip(pygame, ctx, surf, "NO POSE — STEP INTO FRAME",
                      (CAM_RECT[0] + CAM_RECT[2] // 2, CAM_RECT[1] + 28), theme.GOLD)
        else:
            if self.cam_state == "opening":
                msg, color = "opening camera…", theme.DIM
            elif self.cam_state == "error":
                msg, color = f"camera unavailable — {self.cam_err or ''}"[:56], theme.MISS
            else:
                msg, color = "watch mode — no camera", theme.DIM
            txt = ctx.fonts["sm"].render(msg, True, color)
            surf.blit(txt, txt.get_rect(center=(CAM_RECT[0] + 280, CAM_RECT[1] + 175)))
            if self.cam_state == "error":
                sub = ctx.fonts["sm"].render("R retry — the reference still plays", True, theme.DIM)
                surf.blit(sub, sub.get_rect(center=(CAM_RECT[0] + 280, CAM_RECT[1] + 210)))

    def draw(self, ctx, surf) -> None:
        self._ensure(ctx)
        _bg(pygame, ctx, surf, ctx.t)
        _title(pygame, ctx, surf, ctx.t, y=44)
        f_sm, f_med = ctx.fonts["sm"], ctx.fonts["med"]
        s = self.session
        dance = getattr(ctx, "dance", None)
        title = dance.title if dance is not None else (
            ctx.video_path.name if ctx.video_path else "?"
        )
        diff = f" · {dance.difficulty}" if dance is not None and dance.difficulty else ""
        surf.blit(f_med.render(f"LEARN — {title}{diff}", True, theme.CYAN), (60, 148))
        keys = f_sm.render(
            "SPACE play · ←/→ seek · ,/. frame · N/P segment · L loop · -/= speed · K mark · M mirror · Q back",
            True, theme.DIM)
        _dim_strip(pygame, surf, (40, 694, 740, 24))
        surf.blit(keys, (60, 698))

        self._draw_panels(ctx, surf)

        # selected segment + its mark and instruction
        seg = s.moves[s.selected]
        status = self.statuses.get(s.selected)
        mark = f" · {status}" if status else ""
        line = f"SEGMENT {s.selected + 1}/{len(s.moves)} — {seg.name}  [{seg.start:.1f}–{seg.end:.1f}s]{mark}"
        surf.blit(f_sm.render(line, True, theme.TEXT), (60, 552))
        if s.active_index != s.selected:
            now = f_sm.render(f"▶ playing: {s.moves[s.active_index].name}", True, theme.GOLD)
            surf.blit(now, now.get_rect(topright=(1220, 552)))
        note = seg.instruction or "watch the reference, then copy the dancer"
        surf.blit(f_sm.render(note[:96], True, theme.DIM), (60, 576))

        # match feedback — guidance, not a score
        if self.accuracy is not None:
            _chip(pygame, ctx, surf, f"match {self.accuracy * 100:3.0f}%",
                  (CAM_RECT[0] + CAM_RECT[2] - 74, CAM_RECT[1] + 28), theme.CYAN)
        hint = self.hint or ("copy the pose — no score here" if self.cam_state == "ready" else "")
        if hint:
            surf.blit(f_sm.render(hint[:70], True, theme.MISS if self.hint else theme.DIM), (60, 594))

        # seekable segment timeline + progress marks
        self.timeline.draw(pygame, surf, f_sm, s.time, s.duration,
                           s.moves, s.segment_state)
        x, y, w, h = self.timeline.rect
        for i, st in self.statuses.items():
            if 0 <= i < len(s.moves):
                m = s.moves[i]
                cx = x + int(w * (m.start + m.end) / 2 / max(s.duration, 1e-6))
                pygame.draw.circle(surf, STATUS_COLORS.get(st, theme.DIM), (cx, y + h - 5), 3)
        if s.finished:
            _chip(pygame, ctx, surf, "DANCE COMPLETE — SPACE replays",
                  (CAM_RECT[0] + CAM_RECT[2] // 2, CAM_RECT[1] + CAM_RECT[3] - 30), theme.GOLD)

        # button labels follow live state
        self.btn_play.label = "❚❚ PAUSE" if s.playing else "▶ PLAY"
        self.btn_speed.label = f"{s.speed:g}×"
        self.btn_loop.label = f"LOOP {s.loop.upper()}"
        self.btn_mark.label = f"MARK: {(status or '—').upper()}"
        for b in self._buttons:
            b.draw(pygame, surf, f_med, ctx.t)
