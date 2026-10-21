"""
xbox.py

Xbox controller over Bluetooth, read through SDL's GameController API
(pygame._sdl2.controller) so the button/axis layout is the same on every
OS, unlike raw pygame.joystick indices.

`state_from_raw()` is pure (no pygame calls) so it can be unit tested with
made-up numbers; `XboxController` is the thin hardware layer around it.
"""

import os
from dataclasses import dataclass

# Keep reading the controller even when the terminal, not a pygame
# window, has focus.
os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame
from pygame._sdl2 import controller as sdl_controller

import config

AXIS_MAX = 32767.0

BUTTONS = {
    "a": pygame.CONTROLLER_BUTTON_A,
    "b": pygame.CONTROLLER_BUTTON_B,
    "x": pygame.CONTROLLER_BUTTON_X,
    "y": pygame.CONTROLLER_BUTTON_Y,
    "lb": pygame.CONTROLLER_BUTTON_LEFTSHOULDER,
    "rb": pygame.CONTROLLER_BUTTON_RIGHTSHOULDER,
    "back": pygame.CONTROLLER_BUTTON_BACK,
    "start": pygame.CONTROLLER_BUTTON_START,
}

AXES = {
    "lx": pygame.CONTROLLER_AXIS_LEFTX,
    "ly": pygame.CONTROLLER_AXIS_LEFTY,
    "rx": pygame.CONTROLLER_AXIS_RIGHTX,
    "ry": pygame.CONTROLLER_AXIS_RIGHTY,
    "lt": pygame.CONTROLLER_AXIS_TRIGGERLEFT,
    "rt": pygame.CONTROLLER_AXIS_TRIGGERRIGHT,
}


@dataclass(frozen=True)
class XboxState:
    connected: bool = False
    lx: float = 0.0   # -1 left  .. +1 right
    ly: float = 0.0   # -1 down  .. +1 up   (SDL's Y is flipped so up is +)
    rx: float = 0.0
    ry: float = 0.0
    lt: float = 0.0   # 0 released .. 1 fully pulled
    rt: float = 0.0
    a: bool = False
    b: bool = False
    x: bool = False
    y: bool = False
    lb: bool = False
    rb: bool = False
    back: bool = False
    start: bool = False


def apply_deadzone(value, deadzone):
    """Zero out small values, then rescale so the output still ramps
    smoothly from 0 right at the deadzone edge up to 1 at full travel."""
    if abs(value) < deadzone:
        return 0.0
    sign = 1.0 if value > 0 else -1.0
    return sign * min(1.0, (abs(value) - deadzone) / (1.0 - deadzone))


def state_from_raw(axes, buttons):
    """Turn raw SDL readings into an XboxState.

    axes:    {"lx": int, ...} raw SDL values (sticks -32768..32767, triggers 0..32767)
    buttons: {"a": bool, ...}
    """
    def stick(name):
        v = max(-1.0, min(1.0, axes.get(name, 0) / AXIS_MAX))
        return apply_deadzone(v, config.STICK_DEADZONE)

    def trigger(name):
        v = max(0.0, min(1.0, axes.get(name, 0) / AXIS_MAX))
        return apply_deadzone(v, config.TRIGGER_DEADZONE)

    return XboxState(
        connected=True,
        lx=stick("lx"),
        ly=-stick("ly"),
        rx=stick("rx"),
        ry=-stick("ry"),
        lt=trigger("lt"),
        rt=trigger("rt"),
        **{name: bool(buttons.get(name, False)) for name in BUTTONS},
    )


class ButtonEdges:
    """Which buttons were *just* pressed this tick (released → pressed), so
    holding a button fires its action once, not every tick."""

    def __init__(self):
        self._held = frozenset()

    def update(self, state):
        held = frozenset(n for n in BUTTONS if state.connected and getattr(state, n))
        pressed = held - self._held
        self._held = held
        return pressed


class XboxController:
    def __init__(self):
        pygame.init()
        sdl_controller.init()
        self._pad = None

    def connect(self, timeout_s=None):
        """Block until a game controller is found (or timeout_s passes).
        Returns True if connected."""
        clock = pygame.time.Clock()
        waited = 0.0
        while self._pad is None:
            pygame.event.pump()
            for i in range(sdl_controller.get_count()):
                if sdl_controller.is_controller(i):
                    self._pad = sdl_controller.Controller(i)
                    return True
            if timeout_s is not None and waited >= timeout_s:
                return False
            waited += clock.tick(10) / 1000.0
        return True

    @property
    def name(self):
        return self._pad.name if self._pad else None

    def read(self):
        """Current state; XboxState(connected=False) if the pad dropped out."""
        pygame.event.pump()
        if self._pad is None or not self._pad.attached():
            self._pad = None
            return XboxState(connected=False)
        axes = {name: self._pad.get_axis(axis) for name, axis in AXES.items()}
        buttons = {name: self._pad.get_button(btn) for name, btn in BUTTONS.items()}
        return state_from_raw(axes, buttons)

    def rumble(self, strength=1.0, duration_ms=300):
        """Haptic buzz on both motors. Returns False if the pad can't rumble."""
        if self._pad is None:
            return False
        return self._pad.rumble(strength, strength, duration_ms)

    def close(self):
        sdl_controller.quit()
        pygame.quit()
