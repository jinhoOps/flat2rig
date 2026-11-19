//! B4 evaluation tool (not part of the app): renders envelope poses of an existing CLI run's rig
//! (`RUN_DIR/rig.json` meshes + `RUN_DIR/parts/<id>.png` textures) with the shared renderer
//! `rig::raster`, so meshes of two runs (e.g. before / after a mesh change) can be compared under
//! the same rendering rules.
//!
//! ```text
//! cargo run --example b4_render -- LAYER_DIR RUN_DIR OUT_DIR [options]
//!   --poses a,b,c        envelope pose names (default: every pose of the eye / eyeSeal groups)
//!   --group g1,g2        envelope groups instead of names
//!   --opacity app|one    per-frame layer opacity like the app (default) or 1 for every layer
//!                        (the geometric view `eye_seal` / `exposure` measure)
//!   --drop id,id         leave these layers out (`eye-interior` = every eye white / iris layer)
//!   --gl-export          also write OUT_DIR/<pose>.json (posed geometry for the WebGL comparison)
//! ```
//! `LAYER_DIR` supplies the visible / hidden split (eye frames and the clipping rule), exactly as
//! `flat2rig-cli --from-layers` derives them.

use flat2rig_lib::analysis::layer_input;
use flat2rig_lib::core::{self, Landmarks, Mesh, Point};
use flat2rig_lib::rig::{exposure, raster, seams};
use serde_json::json;
use std::path::{Path, PathBuf};

fn pose_file(name: &str) -> String {
    name.replace([':', '/', '='], "_").replace('+', "p").replace('-', "m")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        return Err("usage: b4_render LAYER_DIR RUN_DIR OUT_DIR [--poses ..|--group ..] [--opacity app|one] [--drop ..] [--gl-export]".into());
    }
    let (layer_dir, run_dir, out_dir) = (PathBuf::from(&args[0]), PathBuf::from(&args[1]), PathBuf::from(&args[2]));
    let mut names: Option<Vec<String>> = None;
    let mut groups: Vec<String> = vec!["eye".into(), "eyeSeal".into()];
    let mut opacity_one = false;
    let mut drop: Vec<String> = Vec::new();
    let mut gl_export = false;
    let mut i = 3;
    while i < args.len() {
        let next = |i: usize| args.get(i + 1).cloned().ok_or_else(|| format!("{} needs a value", args[i]));
        match args[i].as_str() {
            "--poses" => {
                names = Some(next(i)?.split(',').map(String::from).collect());
                i += 1;
            }
            "--group" => {
                groups = next(i)?.split(',').map(String::from).collect();
                i += 1;
            }
            "--opacity" => {
                opacity_one = match next(i)?.as_str() {
                    "one" => true,
                    "app" => false,
                    o => return Err(format!("--opacity app|one, got {o}").into()),
                };
                i += 1;
            }
            "--drop" => {
                drop = next(i)?.split(',').map(String::from).collect();
                i += 1;
            }
            "--gl-export" => gl_export = true,
            a => return Err(format!("unknown option {a}").into()),
        }
        i += 1;
    }
    std::fs::create_dir_all(&out_dir)?;

    // Rig of the run: part order, meshes, textures, landmarks.
    let rig: Vec<serde_json::Value> = serde_json::from_str(&std::fs::read_to_string(run_dir.join("rig.json"))?)?;
    let ids: Vec<String> = rig.iter().map(|e| e["id"].as_str().unwrap_or_default().to_string()).collect();
    let meshes: Vec<Mesh> = rig
        .iter()
        .map(|e| serde_json::from_value(e["mesh"].clone()))
        .collect::<Result<_, _>>()?;
    let stats: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(run_dir.join("stats.json"))?)?;
    let lm: Landmarks = serde_json::from_value(stats["landmarks"].clone())?;

    let set = layer_input::load_layer_dir(&layer_dir)?;
    let (w, h) = (set.width, set.height);
    let (vis_by_layer, fill_by_layer) = layer_input::split_visible_fill(&set);
    let pick = |src: &[Vec<u8>]| -> Result<Vec<Vec<u8>>, String> {
        ids.iter()
            .map(|id| {
                set.ids
                    .iter()
                    .position(|x| x == id)
                    .map(|p| src[p].clone())
                    .ok_or_else(|| format!("layer dir has no '{id}'"))
            })
            .collect()
    };
    let visible = pick(&vis_by_layer)?;
    let fills = pick(&fill_by_layer)?;
    let lm = lm.with_eye_frames(&ids, &visible, w, h);
    let clip = exposure::default_clip_targets(&ids, &fills);

    let tex_paths: Vec<PathBuf> = ids.iter().map(|id| run_dir.join("parts").join(format!("{id}.png"))).collect();
    let textures: Vec<raster::Texture> = tex_paths
        .iter()
        .map(|p| -> Result<raster::Texture, Box<dyn std::error::Error>> {
            let img = image::open(p)?.into_rgba8();
            Ok(raster::Texture::from_straight_rgba8(w, h, img.as_raw()))
        })
        .collect::<Result<_, _>>()?;
    let dropped = |id: &str| {
        drop.iter().any(|d| d == id)
            || (drop.iter().any(|d| d == "eye-interior") && core::is_eye_interior_layer(id))
    };

    let env = &seams::pose_envelope().poses;
    let poses: Vec<&seams::EnvelopePose> = match &names {
        Some(n) => n
            .iter()
            .map(|name| env.iter().find(|p| &p.name == name).ok_or_else(|| format!("no pose '{name}'")))
            .collect::<Result<_, _>>()?,
        None => env.iter().filter(|p| groups.contains(&p.group)).collect(),
    };

    let mut rows = Vec::new();
    for pe in poses {
        let posed: Vec<Option<Vec<Point>>> = ids
            .iter()
            .zip(&meshes)
            .map(|(id, mesh)| {
                if mesh.triangles.is_empty() {
                    return None;
                }
                let pts = raster::posed_vertices(mesh, id, &lm, &pe.pose);
                let moved = pts
                    .iter()
                    .zip(&mesh.vertices)
                    .map(|(d, s)| (d.x - s.x).abs() + (d.y - s.y).abs())
                    .fold(0.0f64, f64::max);
                (moved > 1e-3).then_some(pts)
            })
            .collect();
        let opacity: Vec<f32> = ids
            .iter()
            .map(|id| if dropped(id) { 0.0 } else if opacity_one { 1.0 } else { raster::layer_opacity(id, &pe.pose) })
            .collect();
        let layers: Vec<raster::Layer> = (0..ids.len())
            .map(|k| raster::Layer {
                texture: &textures[k],
                mesh: &meshes[k],
                posed: posed[k].as_deref(),
                opacity: opacity[k],
                blend: raster::Blend::Normal,
                clip_to: clip[k],
            })
            .collect();
        let file = pose_file(&pe.name);
        raster::render_layers(w, h, &layers).to_image().save(out_dir.join(format!("{file}.png")))?;
        if gl_export {
            let clip_map: serde_json::Map<String, serde_json::Value> = (0..ids.len())
                .filter_map(|k| clip[k].map(|t| (ids[k].clone(), json!(ids[t]))))
                .collect();
            let gl_layers: Vec<serde_json::Value> = (0..ids.len())
                .map(|k| {
                    json!({
                        "id": ids[k],
                        "texture": abs(&tex_paths[k]),
                        "opacity": opacity[k],
                        "vertices": meshes[k].vertices.iter().map(|v| [v.x, v.y]).collect::<Vec<_>>(),
                        "posed": posed[k].as_ref().map(|p| p.iter().map(|v| [v.x, v.y]).collect::<Vec<_>>()),
                        "triangles": meshes[k].triangles,
                    })
                })
                .collect();
            let doc = json!({ "pose": pe.name, "width": w, "height": h, "clip": clip_map, "layers": gl_layers });
            std::fs::write(out_dir.join(format!("{file}.json")), serde_json::to_string(&doc)?)?;
        }
        rows.push(json!({
            "pose": pe.name,
            "group": pe.group,
            "file": format!("{file}.png"),
            "eyeL": pe.pose.eye_l,
            "eyeR": pe.pose.eye_r,
            "opacity": ids.iter().zip(&opacity).filter(|(_, &o)| o != 1.0).map(|(id, &o)| json!([id, o])).collect::<Vec<_>>(),
        }));
    }
    let summary = json!({
        "layerDir": abs(&layer_dir),
        "runDir": abs(&run_dir),
        "opacity": if opacity_one { "one" } else { "app" },
        "drop": drop,
        "landmarks": lm,
        "eyeFrames": lm.eye_frames,
        "poses": rows,
    });
    std::fs::write(out_dir.join("summary.json"), serde_json::to_string_pretty(&summary)?)?;
    println!("{} poses -> {}", summary["poses"].as_array().map_or(0, |a| a.len()), out_dir.display());
    Ok(())
}

fn abs(p: &Path) -> String {
    std::fs::canonicalize(p).map_or_else(|_| p.display().to_string(), |c| c.display().to_string())
}
