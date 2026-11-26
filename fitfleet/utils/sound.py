"""The end-of-run sounds: synthesized as WAV bytes and played with the OS player.

The `_C5` ... `_C7` constants are note frequencies in Hz, and `_PARTIALS` lists the
(frequency ratio, loudness) pairs that give the notes their bell-like timbre.
"""
from __future__ import annotations

import os
from enum import StrEnum

import numpy as np


class Sound(StrEnum):
    """Sound played when a training object goes out of scope after a finished run.

    none is silent; ping is one soft bell note; double is two rising notes, a happy
    "ding-ding"; chime is three bell notes ringing into a major chord; success is a quick
    rising arpeggio ending on a bright chord ("it worked!"); sparkle is a fast, playful
    twinkling run up high; alarm is a repeated high/low tone that is hard to miss; bell
    is the terminal bell only.
    """
    none = "none"
    ping = "ping"
    double = "double"
    chime = "chime"
    success = "success"
    sparkle = "sparkle"
    alarm = "alarm"
    bell = "bell"

    @property
    def notes(self) -> list[tuple[float, float, float]]:
        """(frequency Hz, start s, length s) per note; notes may overlap and ring together."""
        return _SOUND_NOTES.get(self, [])

    def wav_bytes(self, rate: int = 44_100) -> bytes:
        """The sound rendered as an in-memory 16-bit mono WAV file (b"" for none / bell).

        `rate` is the sample rate in Hz. Each note gets a bell / marimba timbre from a few
        partials, the higher ones dying away faster, with a 4 ms fade-in (no click) and a
        20 ms fade-out. The mix is normalized to half of full scale for a comfortable volume.
        """
        import io
        import wave
        notes = self.notes
        if not notes:
            return b""
        total = max(start + length for _, start, length in notes) + 0.05
        mix = np.zeros(int(total * rate))
        for freq, start, length in notes:
            t = np.arange(int(length * rate)) / rate
            tone = np.zeros_like(t)
            for k, (ratio, amp) in enumerate(_PARTIALS, start=1):
                tone += amp * np.sin(2 * np.pi * freq * ratio * t) * np.exp(-t * k / (length * 0.35))
            attack = np.minimum(1, t / 0.004)
            release = np.minimum(1, (length - t) / 0.02)
            begin = int(start * rate)
            mix[begin:begin + len(t)] += tone * attack * release
        mix *= 0.5 / np.abs(mix).max()
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(rate)
            f.writeframes((mix * 32767).astype("<i2").tobytes())
        return buffer.getvalue()


_C5, _E5, _G5, _A5 = 523.25, 659.25, 783.99, 880.00
_C6, _D6, _E6, _G6, _A6, _C7 = 1046.50, 1174.66, 1318.51, 1567.98, 1760.00, 2093.00
_PARTIALS = [(1, 1.0), (2, 0.35), (3, 0.12), (4.2, 0.06)]

_SOUND_NOTES: dict[Sound, list[tuple[float, float, float]]] = {
    Sound.ping: [(_A5, 0, 0.8)],
    Sound.double: [(_G5, 0, 0.5), (_C6, 0.14, 0.9)],
    Sound.chime: [(_C6, 0, 1.4), (_E6, 0.15, 1.3), (_G6, 0.30, 1.5)],
    Sound.success: [(_C5, 0, 0.4), (_E5, 0.09, 0.4), (_G5, 0.18, 0.4), (_C6, 0.27, 0.5),
                    (_C5, 0.40, 1.3), (_E5, 0.40, 1.3), (_G5, 0.40, 1.3), (_C6, 0.40, 1.5), (_E6, 0.40, 1.5)],
    Sound.sparkle: [(f, i * 0.06, 0.6) for i, f in enumerate([_C6, _D6, _E6, _G6, _A6, _C7])]
                   + [(_C7, 0.42, 1.0), (_G6, 0.42, 1.0)],
    Sound.alarm: [(f, i * 0.2, 0.18) for i, f in enumerate([1200, 800] * 4)],
}


def play_wav(data: bytes) -> bool:
    """Play WAV bytes to the end; True if a player was found and worked."""
    try:
        import winsound
        winsound.PlaySound(data, winsound.SND_MEMORY)  # waits until done, so exit doesn't cut it off
        return True
    except ImportError:
        pass
    import shutil
    import subprocess
    import tempfile
    for player in ("afplay", "paplay", "aplay"):
        if shutil.which(player):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(data)
            try:
                return subprocess.run([player, f.name], capture_output=True, timeout=10).returncode == 0
            finally:
                os.unlink(f.name)
    return False
