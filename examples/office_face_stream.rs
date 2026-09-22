//! Paced JSONL Audio2Face source for cc-render office_acceptance.
//! Waits for a start line after readiness; loops raw f32 mono 16 kHz PCM.
use audio2face_plugin::{Audio2FaceLipSyncConfig, Audio2FaceLipSyncNode};
use remotemedia_plugin_sdk::{
    traits::streaming::AsyncStreamingNode,
    types::{AudioSamples, RuntimeData},
};
use std::io::{BufRead, Write};
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: office_face_stream bundle-dir speech-16khz-mono.f32".into());
    }
    let node = Audio2FaceLipSyncNode::load(Audio2FaceLipSyncConfig {
        bundle_path: args[1].clone().into(),
        use_gpu: true,
        ..Default::default()
    })?;
    let bytes = std::fs::read(&args[2])?;
    let samples: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    println!("{{\"event\":\"ready\"}}");
    std::io::stdout().flush()?;
    let _ = std::io::stdin().lock().lines().next();
    loop {
        for chunk in samples.chunks(16000) {
            node.process_streaming(
                RuntimeData::Audio {
                    samples: AudioSamples::Vec(chunk.to_vec()),
                    sample_rate: 16000,
                    channels: 1,
                    stream_id: None,
                    timestamp_us: None,
                    arrival_ts_us: None,
                    metadata: None,
                },
                None,
                |d| {
                    if let RuntimeData::Json(v) = d {
                        println!("{}", v);
                        let _ = std::io::stdout().flush();
                    }
                    Ok(())
                },
            )
            .await?;
        }
    }
}
