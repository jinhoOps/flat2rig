use image::{imageops::FilterType, GrayImage, ImageBuffer, RgbaImage};
use ort::{session::Session, value::Tensor};
use sha2::{Digest, Sha256};
use std::path::Path;

/// U2Net/U2NetP: NCHW RGB ImageNet-normalized 320², first NCHW alpha output.
/// Models are explicitly selected local files and are never downloaded by the app.
pub fn background_alpha(
    path: &Path,
    rgba: &[u8],
    w: usize,
    h: usize,
) -> Result<(Vec<u8>, String), String> {
    if path.extension().and_then(|v| v.to_str()) != Some("onnx") {
        return Err("Select a U2Net/U2NetP .onnx file".into());
    }
    let metadata = std::fs::metadata(path).map_err(|e| e.to_string())?;
    if metadata.len() > 1024 * 1024 * 1024 {
        return Err("Model larger than 1 GiB is unsupported".into());
    }
    let hash = format!(
        "{:x}",
        Sha256::digest(std::fs::read(path).map_err(|e| e.to_string())?)
    );
    let img: RgbaImage =
        ImageBuffer::from_raw(w as u32, h as u32, rgba.to_vec()).ok_or("Invalid input pixels")?;
    let resized = image::imageops::resize(&img, 320, 320, FilterType::Triangle);
    let mut input = vec![0f32; 3 * 320 * 320];
    let max_value = resized
        .pixels()
        .flat_map(|p| p.0[..3].to_vec())
        .max()
        .unwrap_or(255)
        .max(1) as f32;
    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];
    for (i, p) in resized.pixels().enumerate() {
        for c in 0..3 {
            input[c * 320 * 320 + i] = (p[c] as f32 / max_value - mean[c]) / std[c];
        }
    }
    let mut session = Session::builder()
        .map_err(|e| e.to_string())?
        .with_intra_threads(4)
        .map_err(|e| e.to_string())?
        .commit_from_file(path)
        .map_err(|e| e.to_string())?;
    let tensor = Tensor::from_array(([1usize, 3, 320, 320], input)).map_err(|e| e.to_string())?;
    let outputs = session
        .run(ort::inputs![tensor])
        .map_err(|e| e.to_string())?;
    let (shape, values) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| e.to_string())?;
    if shape.as_ref() != [1, 1, 320, 320]
        || values.len() != 320 * 320
        || values.iter().any(|v| !v.is_finite())
    {
        return Err("Expected first model output [1,1,320,320] finite float alpha".into());
    }
    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let data = values
        .iter()
        .map(|v| ((v - min) / (max - min).max(1e-8) * 255.).clamp(0., 255.) as u8)
        .collect();
    let alpha = GrayImage::from_raw(320, 320, data).ok_or("Invalid output alpha")?;
    let mut result =
        image::imageops::resize(&alpha, w as u32, h as u32, FilterType::Triangle).into_raw();
    for (i, a) in result.iter_mut().enumerate() {
        *a = (*a as u16 * rgba[i * 4 + 3] as u16 / 255) as u8;
    }
    Ok((result, hash))
}
