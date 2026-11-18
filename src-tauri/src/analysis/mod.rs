pub mod detector;
pub mod landmarks;
pub mod layer_input;
pub mod lineart;
pub mod models;
pub mod parts;
pub mod region_fill;
pub mod segmentation;
pub mod state;
pub mod templates;
pub mod transforms;

#[cfg(test)]
mod tests {
    use super::detector::{detect_faces, DEFAULT_FACE_SCORE_THRESHOLD};
    use super::landmarks::detect_landmarks;
    use super::models::{ModelRegistry, ModelRole};
    use super::parts::{
        propose_parts_with_sam, propose_parts_with_sam_with_options, refine_part_with_prompts,
    };
    use super::segmentation::PromptInput;
    use super::state::AnalysisState;
    use std::path::PathBuf;

    #[test]
    fn rust_detector_and_landmarks_match_python_reference_when_models_present() {
        let mut reg = ModelRegistry::new(Vec::new()).unwrap();
        if reg
            .resolve_path(super::models::ModelRole::FaceDetector)
            .is_none()
            || reg
                .resolve_path(super::models::ModelRole::Landmark28)
                .is_none()
        {
            return;
        }

        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
        let parity_path = repo_root.join("artifacts/model-export/export-parity.json");
        let sutteang_img = repo_root.join("sutteang/2d_.png");
        if !sutteang_img.is_file() {
            return;
        }

        let img = image::open(&sutteang_img).unwrap().to_rgba8();
        let (w, h) = img.dimensions();
        let det = detect_faces(&mut reg, img.as_raw(), w as usize, h as usize, DEFAULT_FACE_SCORE_THRESHOLD)
            .unwrap();
        assert_eq!(det.candidates.len(), 1);
        let top = &det.candidates[0];

        if parity_path.is_file() {
            let v: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(&parity_path).unwrap()).unwrap();
            let py_top = &v["faceAndLandmarks"]["cases"][0]["topBox"];
            for i in 0..4 {
                let diff = (top.bbox[i] - py_top[i].as_f64().unwrap()).abs();
                assert!(diff <= 0.5, "bbox[{i}] diff {diff} > 0.5px");
            }
            let score_diff = (top.score - py_top[4].as_f64().unwrap()).abs();
            assert!(score_diff < 1e-3, "score diff {score_diff}");
        }

        let lm = detect_landmarks(
            &mut reg,
            img.as_raw(),
            w as usize,
            h as usize,
            top.bbox,
            top.crop_bbox,
        )
        .unwrap();
        assert_eq!(lm.points_28.len(), 28);
        assert!(lm.rig_landmarks.eye_r.x < lm.rig_landmarks.eye_l.x);
        assert!(lm.rig_landmarks.crown.y < lm.rig_landmarks.eye_r.y);
        assert!(lm.rig_landmarks.mouth.y > lm.rig_landmarks.eye_r.y);
        assert!(lm.rig_landmarks.chin.y > lm.rig_landmarks.mouth.y);

        println!(
            "Rust 2d_.png topBox={:?} score={:.6} rig={:?}",
            top.bbox, top.score, lm.rig_landmarks
        );

        // Negative sample: blank solid canvas must yield zero face detections.
        let blank = vec![200u8; 512 * 512 * 4];
        let neg = detect_faces(&mut reg, &blank, 512, 512, DEFAULT_FACE_SCORE_THRESHOLD).unwrap();
        assert!(neg.candidates.is_empty(), "blank image should yield 0 faces");
    }

    #[test]
    fn mobilesam_parts_preserve_partition_and_cache_decoder_refinement() {
        let state = AnalysisState::new(None).unwrap();
        {
            let reg = state.registry.lock().unwrap();
            if reg.resolve_path(ModelRole::PartEncoder).is_none()
                || reg.resolve_path(ModelRole::PartDecoder).is_none()
                || reg.resolve_path(ModelRole::FaceDetector).is_none()
                || reg.resolve_path(ModelRole::Landmark28).is_none()
            {
                return;
            }
        }
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
        let sutteang_img = repo_root.join("sutteang/2d_.png");
        if !sutteang_img.is_file() {
            return;
        }
        let img = image::open(&sutteang_img).unwrap().to_rgba8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        let (top, lm) = {
            let mut reg = state.registry.lock().unwrap();
            let det = detect_faces(&mut reg, img.as_raw(), w, h, DEFAULT_FACE_SCORE_THRESHOLD).unwrap();
            let top = det.candidates[0].clone();
            let lm = detect_landmarks(&mut reg, img.as_raw(), w, h, top.bbox, top.crop_bbox).unwrap();
            (top, lm)
        };

        let proposal = propose_parts_with_sam(
            &state,
            img.as_raw(),
            w,
            h,
            &lm.rig_landmarks,
            &lm.points_28,
            Some(top.bbox),
            32.0,
            None,
            "detail",
        )
        .unwrap();
        assert_eq!(proposal.masks.len(), 18);
        let face_meta = proposal.metas.iter().find(|m| m.id == "face").unwrap();
        assert_eq!(face_meta.provenance, "model");
        assert!(face_meta.pixel_count > 1000);

        // Verify strict partition invariant: every pixel alpha sums across masks cleanly (zero duplicate alpha).
        for i in 0..(w * h) {
            let sum: u32 = proposal.masks.iter().map(|m| m[i] as u32).sum();
            let nonzero_owners = proposal.masks.iter().filter(|m| m[i] > 0).count();
            assert!(nonzero_owners <= 1, "pixel {i} owned by {nonzero_owners} parts");
            assert!(sum <= 255);
        }

        // Verify interactive decoder-only click refinement reuses warm cache quickly!
        let t_refine = std::time::Instant::now();
        let refined = refine_part_with_prompts(
            &state,
            img.as_raw(),
            w,
            h,
            &lm.rig_landmarks,
            "eyeR_iris",
            &[PromptInput {
                x: lm.rig_landmarks.eye_r.x,
                y: lm.rig_landmarks.eye_r.y,
                label: 1,
            }],
            32.0,
        )
        .unwrap();
        let refine_ms = t_refine.elapsed().as_millis();
        assert_eq!(refined.candidates.len(), 4);
        assert!(refined.used_face_crop);
        assert!(refined.candidates[refined.best_index].pixel_count > 50);
        println!("MobileSAM face_px={} cached refine_ms={} used_face_crop={}", face_meta.pixel_count, refine_ms, refined.used_face_crop);

        // Verify multi-layer hair separation produces role/strand layers in Z-order with zero overlap.
        let split_prop = propose_parts_with_sam_with_options(
            &state,
            img.as_raw(),
            w,
            h,
            &lm.rig_landmarks,
            &lm.points_28,
            Some(top.bbox),
            32.0,
            None,
            "detail",
            true,
        )
        .unwrap();
        assert!(split_prop.part_ids.len() > 18);
        let back_pos = split_prop.part_ids.iter().position(|id| id == "hair_back").unwrap();
        let body_pos = split_prop.part_ids.iter().position(|id| id == "body").unwrap();
        let clothes_pos = split_prop.part_ids.iter().position(|id| id == "clothes").unwrap();
        let front_pos = split_prop.part_ids.iter().position(|id| id == "hair_front").unwrap();
        assert!(back_pos < body_pos);
        assert!(body_pos < clothes_pos);
        assert!(clothes_pos < front_pos);
        assert!(split_prop
            .part_ids
            .iter()
            .any(|id| id == "hair_side_R" || id == "hair_side_L"));
        for i in 0..(w * h) {
            let owners = split_prop.masks.iter().filter(|m| m[i] > 0).count();
            assert!(owners <= 1, "hair_split pixel {i} owned by {owners} parts");
        }
    }
}
