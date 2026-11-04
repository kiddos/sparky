---
name: sparky-visual-debugging
description: >-
  Use this skill when inspecting Sparky's screen, taking screenshots of the desktop pet,
  capturing rendered frames via JSON-RPC, or debugging OpenGL/egui visual and layout issues.
---

# Sparky Visual Inspection & Framebuffer Debugging Runbook

This guide covers how to capture, inspect, and debug the visual output of the Sparky desktop pet using the built-in JSON-RPC 2.0 interface and native agent tools.

## 1. Visual Capture Principle: Always Use `sparky.get_frame`

> [!IMPORTANT]
> Never run external screenshot utilities (`import`, `scrot`, `xwd`, `maim`). They capture extraneous desktop background pixels, fail in sandboxed environments, and require manual permission overrides. Sparky's built-in `sparky.get_frame` directly reads the transparent OpenGL framebuffer (`gl::ReadPixels`) with pixel-perfect fidelity.

## 2. Multi-Step Capture & Inspection Workflow

### Step 1: Ensure Application is Running
If Sparky is not running, launch it in the background:
```bash
./target/release/sparky
```
Verify JSON-RPC availability on port 6006:
```bash
curl -s -X POST http://127.0.0.1:6006 \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "id": 1, "method": "sparky.get_state"}'
```

### Step 2: Prepare Visual State (Optional)
Toggle UI overlays or emotions as needed:
- Open chat window:
  ```bash
  curl -s -X POST http://127.0.0.1:6006 \
    -H "Content-Type: application/json" \
    -d '{"jsonrpc": "2.0", "id": 2, "method": "sparky.set_chat_visible", "params": {"visible": true}}'
  ```
- Trigger an emotion (e.g. `happy`):
  ```bash
  curl -s -X POST http://127.0.0.1:6006 \
    -H "Content-Type: application/json" \
    -d '{"jsonrpc": "2.0", "id": 3, "method": "sparky.set_emotion", "params": {"emotion": "happy"}}'
  ```

### Step 3: Capture Framebuffer to File
Capture the rendered RGBA framebuffer directly to a PNG in your conversation artifact or scratch directory:
```bash
curl -s -X POST http://127.0.0.1:6006 \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 4,
    "method": "sparky.get_frame",
    "params": {
      "source": "framebuffer",
      "save_path": "/home/kiddos/.gemini/antigravity/brain/<conversation-id>/frame.png",
      "include_data": false
    }
  }'
```

### Step 4: Inspect Visually with `read_file`
Call the agent `read_file` tool with the path to the saved PNG:
- `read_file(Path: "/home/kiddos/.gemini/antigravity/brain/<conversation-id>/frame.png")`
- Antigravity natively parses and displays the PNG image directly in the conversation context for visual verification.

---

## 3. Visual Debugging Checklist

When analyzing the captured frame, verify these components:

1. **Window Dimensions & Viewport Split**:
   - Pet only: $288 \times 162\text{ px}$.
   - Chat open: $288 \times 466\text{ px}$ (Pet at bottom 162px, Chat overlay occupying top 304px).
2. **Egui Chat Layout Integrity**:
   - Title bar (`🐾 Sparky Chat`), mood badge (`Mood: 🐱 ...`), and thinking indicator.
   - Message scroll history.
   - Emotion chips row.
   - **Input Bar**: The text box must NOT be clipped or pushed outside the window.
3. **Focus & Cursor Indicators**:
   - Active input displays a cyan/blue outline.
   - Blinking cursor (`|`) indicates keyboard focus is ready for immediate text entry.
4. **Chroma-Keying & Fur Alpha**:
   - Transparent background around kitten; no residual bright green edge halos.
   - No black letterboxing artifacts along window boundaries.
