# Sparky Improvement & Feature Roadmap: Making Sparky Fun & Alive

This document outlines creative feature concepts, architecture improvements, and design blueprints to transform **Sparky** from a passive desktop animation into a delightful, expressive, and interactive AI desktop companion.

---

## Table of Contents

1. [Vision & Personality Core](#1-vision--personality-core)
2. [Gemma 4 AI Emotion & Intelligence Engine](#2-gemma-4-ai-emotion--intelligence-engine)
3. [Physical Interactions & Desktop Physics](#3-physical-interactions--desktop-physics)
4. [Ambient Awareness & Context-Driven Reactions](#4-ambient-awareness--context-driven-reactions)
5. [Speech, Audio & Dialogue Bubbles](#5-speech-audio--dialogue-bubbles)
6. [Gamification & Virtual Pet Mechanics (Tamagotchi Elements)](#6-gamification--virtual-pet-mechanics-tamagotchi-elements)
7. [Extensibility, Developer Hooks & IPC](#7-extensibility-developer-hooks--ipc)
8. [Technical Architecture Roadmap](#8-technical-architecture-roadmap)

---

## 1. Vision & Personality Core

Currently, Sparky displays looped video animations (`angry`, `happy1`, `happy2`, `naughty`, `sleepy`, `tired`) selected at random. The goal is to give Sparky:
* **Sentience & Continuity**: An emotional state that evolves based on how you treat it, what you are working on, and the time of day.
* **Physical Presence**: Natural desktop physics (reacting to cursor motion, being dragged or tossed, resting on top of active windows).
* **Playfulness**: Mischief, celebrations, mini-games, and quirky Easter eggs.

---

## 2. Gemma AI Emotion & Intelligence Engine (Gemma 3 270M Local & Gemma 4 API)

Sparky supports dual intelligence backends:
1. **Local In-Process Gemma 3 270M**: Runs completely locally on CPU using Hugging Face Candle in Rust on a background Tokio runtime. Requiring only ~270M parameters and ~300MB RAM, it operates offline without an external API key or heavy server runtimes.
2. **Cloud Gemma 4 API**: Connects to Google AI Studio for larger multimodal reasoning capabilities.

### A. Dynamic Emotional State Machine (Valence / Arousal / Mood)
Instead of pure random picking, maintain an internal psychological model driven by Gemma:
* **Emotion Vectors**: Energy (0-100%), Mood (-100% to +100%), Affection (0-100%), Boredom (0-100%).
* **Gemma-Driven Transitions**:
  * Periodically or upon events, a background prompt feeds recent interaction history to Gemma.
  * Gemma returns structured output:
    ```json
    {
      "target_emotion": "naughty",
      "dialogue": "Hehe, are you sure you want to run `git push --force`?",
      "energy_delta": -5,
      "mood_delta": 15,
      "duration_sec": 4.5
    }
    ```
  * The video playback smoothly transitions into the chosen state.

### B. Custom Personality Archetypes
Allow configuring Sparky’s personality via Gemma system prompts:
* **The Chaos Gremlin**: Constantly distracts you, cheers when code breaks, acts smug and naughty.
* **The Hype Pet**: Celebrates every git commit and typing streak like a stadium mascot.
* **The Zen Monk**: Reminds you to breathe, stretch, drink water, and sleep when it gets late.
* **The Tsundere Companion**: Pretends not to like being petted or dragged, but acts sleepy and affectionate when you stay focused.

### C. Multimodal Screen Understanding (Gemma Vision)
* Take low-res periodic screenshots or window title checks (with privacy toggles).
* Gemma 4 observes what you're doing:
  * Watching cat videos? Sparky gets playful or curious.
  * Late night debugging? Sparky yawns and holds up a coffee cup.
  * Passing all unit tests? Sparky does a victory loop.

---

## 3. Physical Interactions & Desktop Physics

### A. Cursor Interaction & Petting Mechanics
* **Petting / Scratching**: Moving the cursor back-and-forth across Sparky triggers a "happy/purring" reaction with heart particles.
* **Poking / Teasing**: Fast repeated clicks poke Sparky, causing it to dodge, get annoyed (`angry`), or do a cheeky dodge animation (`naughty`).
* **Mouse Chasing**: Sparky’s eyes or head track cursor position when the mouse lingers nearby.

### B. Throwing & Momentum Physics
* When dragged and released while moving, Sparky inherits cursor velocity:
  * Floats or slides with friction across the screen.
  * Bounces off screen borders or window titles with a slight squish animation.
  * Gets dizzy or surprised after a high-speed toss!

### C. Desktop Roaming & Window Perching
* **Window Sitting**: Sparky detects the active window boundary (via X11/Wayland/Win32 APIs) and "sits" on top of your browser or code editor title bar.
* **Screen Edge Peeking**: Occasionally wanders off to the side and peeks out from the left or bottom monitor edge.
* **Click-Through Mode ("Ghost Mode")**: Toggleable shortcut (`Ctrl+Shift+P`) to make Sparky transparent to mouse clicks when you need unhindered screen space.

---

## 4. Ambient Awareness & Context-Driven Reactions

Make Sparky react to your real computing environment:

### A. Developer & System Events
* **Git Commit & Push**: Sparky cheers (`happy1`/`happy2`) whenever you commit or push code.
* **Build / Test Failure**: Sparky turns `angry` at the compiler or offers a consoling look.
* **Battery & Hardware Load**:
  * Battery under 15%: Sparky turns `tired` or `sleepy`, looking for a power plug.
  * High CPU / fan noise: Sparky wipes sweat off its forehead.

### B. Daily Rhythms & Time of Day
* **Late Night (1:00 AM - 5:00 AM)**: Sparky nods off into `sleepy`, occasionally falling over.
* **Morning Greeting**: A bright morning stretch and wave on the first mouse movement of the day.
* **Long Inactivity**: After 45 minutes of no keyboard/mouse input, Sparky falls into deep sleep with animated "Zzz" floating up.

### C. Music & Media Detection
* Detect system audio output or Spotify status via MPRIS:
  * Sparky bobs its head to the rhythm of music.
  * If a sad song plays, Sparky looks contemplative.

---

## 5. Speech, Audio & Dialogue Bubbles

### A. Floating Comic Speech Bubbles
* A lightweight OpenGL/Cairo overlay rendering speech/thought bubbles above Sparky.
* Gemma 4 generates short, witty one-liners tailored to the situation.
* Bubbles can show:
  * Status remarks ("Phew, that was a big file!").
  * Random trivia, witty jokes, or coding encouragement.
  * Pomodoro break reminders ("Time for water, human!").

### B. Voice & Sound Effects
* **Audio FX**: Cute sound effects for emotion changes (playful giggle for `naughty`, soft purr for `happy`, cute snore for `sleepy`).
* **Optional Text-to-Speech (TTS)**: Integration with local neural TTS (e.g. Piper or Coqui) or Gemma speech output with cute pitch-shifted voices (Animal Crossing "animalese" gibberish or clear cute voice).

### C. Quick-Chat Prompt Bar
* Press a global hotkey (e.g., `Super + Sparky` or `Ctrl + Space` on Sparky) to summon a sleek mini text input.
* Chat directly with Sparky! Ask coding questions, brainstorm ideas, or just banter.

---

## 6. Gamification & Virtual Pet Mechanics (Tamagotchi Elements)

### A. Needs & Status Indicators
* **Hunger & Energy**: Sparky gets hungry over time. Drag and drop any file or URL onto Sparky to "feed" it (eating compiled binaries, log files, or cookies).
* **Affection Meter**: Grows through petting, spending time together, and daily logins.

### B. Mini-Games
* **Staring Contest**: Don't move the cursor off Sparky's eyes for 15 seconds.
* **Fetch**: Sparky steals an icon or spawns a tiny ball you have to click and throw back.
* **Trivia / Typing Duel**: Sparky challenges you to type a random witty quote quickly.

### C. Accessories & Customization
* Unlockable hats, glasses, and seasonal accessories (detecting Halloween, Christmas, or your birthday):
  * Party hat on milestones.
  * Tiny detective hat or programmer glasses.

---

## 7. Extensibility, Developer Hooks & IPC

Provide simple ways for users and external scripts to control Sparky:

### A. Local IPC Server (Unix Socket / Named Pipe / HTTP)
Run a lightweight local daemon endpoint:
```bash
# Trigger an emotion directly:
sparky-ctl emotion angry --duration 3s

# Make Sparky say something:
sparky-ctl say "Build succeeded in 2.4s!" --mood happy

# Adjust emotion weight:
sparky-ctl weights --happy 0.8 --naughty 0.2
```

### B. Shell & Git Hook Integration
* Add hooks to `.git/hooks/post-commit` or shell prompts (`precmd` in zsh):
  * When a command fails with non-zero exit code, Sparky acts surprised or frustrated.
  * When a long command (like `cargo build --release`) finishes, Sparky taps your shoulder.

### C. VS Code / JetBrains / Neovim Plugin
* A lightweight companion extension that sends events to Sparky:
  * Sparky reacts when saving files, fixing linter errors, or triggering tests.

---

## 8. Technical Architecture Roadmap

```
+-------------------------------------------------------------+
|                        Sparky Core                          |
|                                                             |
|  +-------------------+               +-------------------+  |
|  |  GLFW / OpenGL    | <-----------  |  Video Animation  |  |
|  |  Transparent Win  |  Frame Blend  |  Streaming Engine |  |
|  +-------------------+               +-------------------+  |
|           ^                                    ^            |
|           | User Input / Mouse                 | State Sync |
|           v                                    v            |
|  +-------------------------------------------------------+  |
|  |             Interactive Behavior Engine               |  |
|  |     (Physics, Drag/Toss, Petting, Hit Detection)      |  |
|  +-------------------------------------------------------+  |
|           ^                                    ^            |
+-----------|------------------------------------|------------+
            | Async Event Channel                | IPC / Hooks
            v                                    v
+-----------------------+              +----------------------+
|  Gemma 4 AI Brain     |              |  System & Shell IPC  |
|  - Personality State  |              |  - Git / IDE Hooks   |
|  - Multimodal Vision  |              |  - System Health     |
|  - Dialogue & Speech  |              |  - Unix Domain Socket|
+-----------------------+              +----------------------+
```

### Proposed Milestones

* **Phase 1: Direct Interactivity & Polish**
  * Mouse petting detection (velocity/back-and-forth gestures).
  * Momentum-based drag and toss physics with screen-edge bounce.
  * Tray menu expansion (force emotion selection, mute sounds, ghost mode).

* **Phase 2: Gemma 4 API Connection & Brain Service**
  * Asynchronous background worker using Tokio/channels so OpenGL 60fps rendering never blocks.
  * System prompt configuration with personality settings (`config.toml`).
  * Gemma 4 emotion controller updating playback weights based on events.

* **Phase 3: Visual Dialogue & Audio Experience**
  * Comic speech bubble overlay rendered directly with text layout (e.g., cosmic-text / freetype).
  * Audio sound effect player (using `rodio` or `kira` crate) matching animation frames.

* **Phase 4: Ecosystem & Ambient Presence**
  * Local CLI (`sparky-cli`) and Unix socket listener.
  * Git hooks, battery, and music watcher integration.
  * Mini-games and cosmetic accessories.
