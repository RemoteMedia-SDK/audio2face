# audio2face — NVIDIA Audio2Face-style lipsync

Standalone Path 3 Rust cdylib that registers `Audio2FaceLipSyncNode` into the
[RemoteMedia SDK](https://github.com/RemoteMedia-SDK/remotemedia-sdk)
streaming pipeline registry.

This plugin coordinates the NVIDIA Audio2Face bundle (`network.onnx`,
`bs_skin_<Identity>.npz`, `model_data_<Identity>.npz`, `bs_skin_config_<Identity>.json`,
`model_config_<Identity>.json`) into a streaming lip-sync node:

1. consumes `RuntimeData::Audio` at 16 kHz in 1-second windows,
2. runs ONNX inference (CPU or CUDA / CoreML when `use_gpu = true`),
3. solves the masked vertex delta to 39-D blendshape weights with
   PGD (default) or BVLS (scipy-equivalent reference),
4. expands to 52-D ARKit, optionally smooths via uniform EMA,
5. and emits one `RuntimeData::Json {kind: "blendshapes", arkit_52, pts_ms, ...}`
   per ~33 ms output frame (30 fps).

It also passes the input `Audio` chunk through unchanged so downstream
consumers (`CcRenderNode`, audio sender) stay synchronised with the
lip-sync inference stream.

## Use from a manifest

```json
{
  "version": "v1",
  "plugins": ["audio2face@v0.1.0"],
  "nodes": [
    {
      "id": "lipsync",
      "node_type": "Audio2FaceLipSyncNode",
      "params": {
        "bundlePath": "/path/to/audio2face-bundle",
        "identity": "Claire",
        "solver": "pgd",
        "useGpu": false,
        "smoothingAlpha": 0.0,
        "paceRealtime": true
      }
    }
  ]
}
```

The SDK resolver expands `audio2face@v0.1.0` to
`github.com/RemoteMedia-SDK/audio2face`, fetches `plugin.toml`, then
falls through to `release-manifest.json` for the platform-specific
prebuilt `.so` / `.dylib` / `.dll` asset.

## Build the cdylib locally

```bash
git clone https://github.com/RemoteMedia-SDK/audio2face
cd audio2face
cargo build --release
# → target/release/libaudio2face_plugin.so
```

## What it exports

| Node type                | Input                       | Output                                                                       |
|--------------------------|-----------------------------|------------------------------------------------------------------------------|
| `Audio2FaceLipSyncNode`  | `Audio` (16 kHz f32 mono)   | `Audio` passthrough + `Json{kind:"blendshapes", arkit_52, pts_ms, turn_id?}` |

## Barge handling

The node accepts a `RuntimeData::Json {kind: "barge_in"}` envelope on
its input port and additionally responds to aux-port `barge_in`
control messages from the session router. On receipt it clears:

- the ONNX session's recurrent GRU state,
- the solver's temporal pull,
- the ARKit smoother's EMA state,
- the audio accumulator, the cumulative-window-ms counter, and the
  realtime pacing anchor.

## License

See `LICENSE.md`. Governed by the RemoteMedia SDK Community License 1.0.
