use flat2rig_lib::{core, inference};
use serde_json::json;
use std::path::PathBuf;

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        return Err(
            "Usage: flat2rig-cli INPUT.png OUTPUT_DIR [MODEL.onnx] [LANDMARKS.json]".into(),
        );
    }
    let start = std::time::Instant::now();
    let img = image::open(&args[0])?.into_rgba8();
    let (w, h) = img.dimensions();
    core::check_size(w as usize, h as usize)?;
    let landmarks = if let Some(path) = args.get(3) {
        serde_json::from_slice(&std::fs::read(path)?)?
    } else {
        core::Landmarks {
            eye_l: core::Point {
                x: w as f64 * 0.61,
                y: h as f64 * 0.31,
            },
            eye_r: core::Point {
                x: w as f64 * 0.39,
                y: h as f64 * 0.31,
            },
            mouth: core::Point {
                x: w as f64 * 0.5,
                y: h as f64 * 0.415,
            },
            chin: core::Point {
                x: w as f64 * 0.5,
                y: h as f64 * 0.48,
            },
            crown: core::Point {
                x: w as f64 * 0.5,
                y: h as f64 * 0.08,
            },
        }
    };
    landmarks.validate(w as usize, h as usize)?;
    let output = PathBuf::from(&args[1]);
    std::fs::create_dir_all(output.join("parts"))?;
    let (alpha, model_hash) = if let Some(path) = args.get(2) {
        let (a, h) = inference::background_alpha(
            std::path::Path::new(path),
            img.as_raw(),
            w as usize,
            h as usize,
        )?;
        (Some(a), Some(h))
    } else {
        (None, None)
    };
    let masks = core::separate(
        img.as_raw(),
        w as usize,
        h as usize,
        &landmarks,
        32.,
        alpha.as_deref(),
    );
    let mut meshes = Vec::new();
    let mut composite = image::RgbaImage::new(w, h);
    for (id, mask) in core::IDS.iter().zip(&masks) {
        let mut part = img.clone();
        for (i, pixel) in part.pixels_mut().enumerate() {
            pixel[3] = mask[i];
            if mask[i] > 0 {
                composite.as_mut()[i * 4..i * 4 + 4].copy_from_slice(&pixel.0);
            }
        }
        part.save(output.join("parts").join(format!("{id}.png")))?;
        meshes.push(json!({"id":id,"mesh":core::mesh_for(mask,w as usize,h as usize,14)}));
    }
    composite.save(output.join("foreground.png"))?;
    let stats = json!({"width":w,"height":h,"parts":masks.len(),"landmarks":landmarks,"modelHash":model_hash,"elapsedMs":start.elapsed().as_millis(),"engine":"Rust / ort / delaunator / glam"});
    std::fs::write(output.join("rig.json"), serde_json::to_vec_pretty(&meshes)?)?;
    std::fs::write(
        output.join("stats.json"),
        serde_json::to_vec_pretty(&stats)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&stats)?);
    Ok(())
}
