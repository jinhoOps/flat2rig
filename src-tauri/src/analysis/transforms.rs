use glam::{DAffine2, DMat3, DVec2};
use serde::{Deserialize, Serialize};

/// Serializable 2D affine transform (`[a, b, c, d, tx, ty]`) representing:
/// `x' = a * x + c * y + tx`, `y' = b * x + d * y + ty` (column-major `DAffine2`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AffineTransform2D {
    pub matrix2: [f64; 4],
    pub translation: [f64; 2],
}

impl AffineTransform2D {
    pub fn identity() -> Self {
        Self::from_daffine2(DAffine2::IDENTITY)
    }

    pub fn from_daffine2(t: DAffine2) -> Self {
        let cols = t.matrix2.to_cols_array();
        Self {
            matrix2: cols,
            translation: t.translation.to_array(),
        }
    }

    pub fn to_daffine2(self) -> DAffine2 {
        DAffine2::from_cols_array(&[
            self.matrix2[0],
            self.matrix2[1],
            self.matrix2[2],
            self.matrix2[3],
            self.translation[0],
            self.translation[1],
        ])
    }

    pub fn map_point(self, x: f64, y: f64) -> [f64; 2] {
        self.to_daffine2().transform_point2(DVec2::new(x, y)).to_array()
    }

    pub fn unmap_point(self, x: f64, y: f64) -> Option<[f64; 2]> {
        let m = self.to_daffine2();
        if m.matrix2.determinant().abs() < 1e-12 {
            return None;
        }
        Some(m.inverse().transform_point2(DVec2::new(x, y)).to_array())
    }

    pub fn compose(self, first: Self) -> Self {
        Self::from_daffine2(self.to_daffine2() * first.to_daffine2())
    }
}

/// Computes `DAffine2` mapping a source rectangle (`src_w x src_h`) to destination (`dst_w x dst_h`).
pub fn scale_transform(src_w: usize, src_h: usize, dst_w: usize, dst_h: usize) -> DAffine2 {
    let sx = dst_w as f64 / src_w.max(1) as f64;
    let sy = dst_h as f64 / src_h.max(1) as f64;
    DAffine2::from_scale(DVec2::new(sx, sy))
}

/// Exact port of `mmpose.datasets.datasets.face` / `_face` rescale logic:
/// preserves aspect ratio inside `(max_w, max_h)` and pads bottom-right to divisor multiple.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct YoloResizeSpec {
    pub orig_w: usize,
    pub orig_h: usize,
    pub resized_w: usize,
    pub resized_h: usize,
    pub padded_w: usize,
    pub padded_h: usize,
    pub scale_x: f64,
    pub scale_y: f64,
}

impl YoloResizeSpec {
    pub fn new(orig_w: usize, orig_h: usize, max_size: usize, pad_divisor: usize) -> Self {
        let max_long = max_size as f64;
        let long_edge = orig_w.max(orig_h).max(1) as f64;
        let short_edge = orig_w.min(orig_h).max(1) as f64;
        let scale_factor = (max_long / long_edge).min(max_long / short_edge);
        let resized_w = ((orig_w as f64) * scale_factor + 0.5) as usize;
        let resized_h = ((orig_h as f64) * scale_factor + 0.5) as usize;
        let padded_w = resized_w.div_ceil(pad_divisor) * pad_divisor;
        let padded_h = resized_h.div_ceil(pad_divisor) * pad_divisor;
        Self {
            orig_w,
            orig_h,
            resized_w: resized_w.max(1),
            resized_h: resized_h.max(1),
            padded_w: padded_w.max(pad_divisor),
            padded_h: padded_h.max(pad_divisor),
            scale_x: resized_w.max(1) as f64 / orig_w.max(1) as f64,
            scale_y: resized_h.max(1) as f64 / orig_h.max(1) as f64,
        }
    }

    pub fn orig_to_model(self) -> DAffine2 {
        DAffine2::from_scale(DVec2::new(self.scale_x, self.scale_y))
    }

    pub fn model_to_orig(self) -> DAffine2 {
        self.orig_to_model().inverse()
    }
}

/// Converts `[x1, y1, x2, y2]` box into `(center, scale)` matching `anime-face-detector`'s
/// `bbox_xyxy2xywh` -> `bbox_xywh2cs(aspect_ratio=1.0, padding=1.25, pixel_std=200.0)`.
pub fn bbox_to_center_scale(
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    aspect_ratio: f64,
    padding: f64,
) -> ([f64; 2], [f64; 2]) {
    let mut w = (x2 - x1).max(1.0);
    let mut h = (y2 - y1).max(1.0);
    let cx = x1 + w * 0.5;
    let cy = y1 + h * 0.5;
    if w > aspect_ratio * h {
        h = w / aspect_ratio;
    } else if w < aspect_ratio * h {
        w = h * aspect_ratio;
    }
    let pixel_std = 200.0;
    let sx = (w / pixel_std) * padding;
    let sy = (h / pixel_std) * padding;
    ([cx, cy], [sx, sy])
}

/// Solves a 2D affine transform mapping three non-collinear source points to destination points.
pub fn affine_from_three_points(src: [DVec2; 3], dst: [DVec2; 3]) -> Option<DAffine2> {
    let src_mat = DMat3::from_cols(
        DVec2::new(src[0].x, src[0].y).extend(1.0),
        DVec2::new(src[1].x, src[1].y).extend(1.0),
        DVec2::new(src[2].x, src[2].y).extend(1.0),
    );
    if src_mat.determinant().abs() < 1e-12 {
        return None;
    }
    let dst_mat = DMat3::from_cols(
        dst[0].extend(1.0),
        dst[1].extend(1.0),
        dst[2].extend(1.0),
    );
    let m = dst_mat * src_mat.inverse();
    Some(DAffine2::from_cols_array(&[
        m.col(0).x,
        m.col(0).y,
        m.col(1).x,
        m.col(1).y,
        m.col(2).x,
        m.col(2).y,
    ]))
}

fn rotate_point(pt: DVec2, rot_rad: f64) -> DVec2 {
    let (sn, cs) = rot_rad.sin_cos();
    DVec2::new(pt.x * cs - pt.y * sn, pt.x * sn + pt.y * cs)
}

fn third_point(a: DVec2, b: DVec2) -> DVec2 {
    let dir = a - b;
    b + DVec2::new(-dir.y, dir.x)
}

/// Exact port of `mmpose.core.post_processing.get_affine_transform`.
/// If `inv` is true, returns `dst -> src`; otherwise returns `src -> dst`.
pub fn hrnet_affine_transform(
    center: [f64; 2],
    scale: [f64; 2],
    rot_deg: f64,
    output_size: [f64; 2],
    shift: [f64; 2],
    inv: bool,
) -> Option<DAffine2> {
    let scale_tmp = [scale[0] * 200.0, scale[1] * 200.0];
    let src_w = scale_tmp[0];
    let dst_w = output_size[0];
    let dst_h = output_size[1];

    let rot_rad = rot_deg.to_radians();
    let src_dir = rotate_point(DVec2::new(0.0, src_w * -0.5), rot_rad);
    let dst_dir = DVec2::new(0.0, dst_w * -0.5);

    let c = DVec2::new(center[0], center[1]);
    let shift_vec = DVec2::new(scale_tmp[0] * shift[0], scale_tmp[1] * shift[1]);
    let src0 = c + shift_vec;
    let src1 = c + src_dir + shift_vec;
    let src2 = third_point(src0, src1);

    let dst0 = DVec2::new(dst_w * 0.5, dst_h * 0.5);
    let dst1 = dst0 + dst_dir;
    let dst2 = third_point(dst0, dst1);

    if inv {
        affine_from_three_points([dst0, dst1, dst2], [src0, src1, src2])
    } else {
        affine_from_three_points([src0, src1, src2], [dst0, dst1, dst2])
    }
}

/// Exact MobileSAM `ResizeLongestSide(target_length)` dimensions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamResizeSpec {
    pub orig_w: usize,
    pub orig_h: usize,
    pub resized_w: usize,
    pub resized_h: usize,
    pub input_size: usize,
}

impl SamResizeSpec {
    pub fn new(orig_w: usize, orig_h: usize, input_size: usize) -> Self {
        let scale = input_size as f64 / orig_w.max(orig_h).max(1) as f64;
        let resized_w = ((orig_w as f64) * scale + 0.5) as usize;
        let resized_h = ((orig_h as f64) * scale + 0.5) as usize;
        Self {
            orig_w,
            orig_h,
            resized_w: resized_w.max(1),
            resized_h: resized_h.max(1),
            input_size,
        }
    }

    pub fn scale_to_dst(self) -> DAffine2 {
        scale_transform(self.orig_w, self.orig_h, self.resized_w, self.resized_h)
    }

    pub fn orig_to_sam(self) -> DAffine2 {
        self.scale_to_dst()
    }

    pub fn sam_to_orig(self) -> DAffine2 {
        self.orig_to_sam().inverse()
    }
}

/// Samples RGBA pixels bilinearly at floating-point `(sx, sy)` matching OpenCV `BORDER_CONSTANT(0)`
/// outside `[0, w - 1]` x `[0, h - 1]` (or clamping inside border for resize when `clamp_edge = true`).
/// Returns normalized RGB values in `0.0..=255.0`.
pub fn sample_bilinear_rgb(
    rgba: &[u8],
    w: usize,
    h: usize,
    sx: f64,
    sy: f64,
    clamp_edge: bool,
) -> [f32; 3] {
    if w == 0 || h == 0 || rgba.len() < w * h * 4 {
        return [0.0, 0.0, 0.0];
    }
    let (fx, fy) = if clamp_edge {
        (sx.clamp(0.0, (w - 1) as f64), sy.clamp(0.0, (h - 1) as f64))
    } else {
        if sx < -0.5 || sy < -0.5 || sx > (w as f64 - 0.5) || sy > (h as f64 - 0.5) {
            return [0.0, 0.0, 0.0];
        }
        (sx, sy)
    };

    let x0 = fx.floor() as isize;
    let y0 = fy.floor() as isize;
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let dx = (fx - x0 as f64) as f32;
    let dy = (fy - y0 as f64) as f32;

    let read_px = |px: isize, py: isize| -> [f32; 3] {
        let (cx, cy) = if clamp_edge {
            (
                px.clamp(0, w as isize - 1) as usize,
                py.clamp(0, h as isize - 1) as usize,
            )
        } else if px < 0 || py < 0 || px >= w as isize || py >= h as isize {
            return [0.0, 0.0, 0.0];
        } else {
            (px as usize, py as usize)
        };
        let i = (cy * w + cx) * 4;
        // Composite over white or raw RGB? Raw RGB channels match PIL/cv2.imread RGB behavior;
        // transparent PNG pixels keep RGB unless caller already flattened.
        [rgba[i] as f32, rgba[i + 1] as f32, rgba[i + 2] as f32]
    };

    let p00 = read_px(x0, y0);
    let p10 = read_px(x1, y0);
    let p01 = read_px(x0, y1);
    let p11 = read_px(x1, y1);

    let w00 = (1.0 - dx) * (1.0 - dy);
    let w10 = dx * (1.0 - dy);
    let w01 = (1.0 - dx) * dy;
    let w11 = dx * dy;

    [
        p00[0] * w00 + p10[0] * w10 + p01[0] * w01 + p11[0] * w11,
        p00[1] * w00 + p10[1] * w10 + p01[1] * w01 + p11[1] * w11,
        p00[2] * w00 + p10[2] * w10 + p01[2] * w01 + p11[2] * w11,
    ]
}

/// Converts a single `[x1, y1, x2, y2]` box with `scale_factor` matching
/// `AnimeFaceDetector._postprocess_face_detection_result`.
pub fn scale_face_box(box_xyxy: [f64; 4], scale_factor: f64) -> [f64; 4] {
    let [x0, y0, x1, y1] = box_xyxy;
    let cx = (x0 + x1) * 0.5;
    let cy = (y0 + y1) * 0.5;
    let w = (x1 - x0 + 1.0) * scale_factor;
    let h = (y1 - y0 + 1.0) * scale_factor;
    [cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_non_square_and_crop_transforms() {
        let orig_to_work = scale_transform(1440, 2160, 512, 768);
        let yolo = YoloResizeSpec::new(1440, 2160, 608, 32);
        assert_eq!(yolo.resized_w, 405);
        assert_eq!(yolo.resized_h, 608);
        assert_eq!(yolo.padded_w, 416);
        assert_eq!(yolo.padded_h, 608);

        let work_to_orig = orig_to_work.inverse();
        let orig_to_yolo = yolo.orig_to_model();
        let work_to_yolo = orig_to_yolo * work_to_orig;
        let serialized = AffineTransform2D::from_daffine2(work_to_yolo);

        for (wx, wy) in [(0.0, 0.0), (256.0, 384.0), (489.25, 712.75)] {
            let [mx, my] = serialized.map_point(wx, wy);
            let [wx_back, wy_back] = serialized.unmap_point(mx, my).unwrap();
            assert!((wx - wx_back).abs() < 1e-9);
            assert!((wy - wy_back).abs() < 1e-9);
        }
    }

    #[test]
    fn hrnet_affine_round_trip_matches_output_corners() {
        let ([cx, cy], [sx, sy]) = bbox_to_center_scale(120.0, 80.0, 360.0, 340.0, 1.0, 1.25);
        let fwd = hrnet_affine_transform([cx, cy], [sx, sy], 0.0, [256.0, 256.0], [0.0, 0.0], false)
            .unwrap();
        let inv = hrnet_affine_transform([cx, cy], [sx, sy], 0.0, [256.0, 256.0], [0.0, 0.0], true)
            .unwrap();
        let center_dst = fwd.transform_point2(DVec2::new(cx, cy));
        assert!((center_dst.x - 128.0).abs() < 1e-9);
        assert!((center_dst.y - 128.0).abs() < 1e-9);

        let pt_back = inv.transform_point2(DVec2::new(64.5, 193.25));
        let pt_again = fwd.transform_point2(pt_back);
        assert!((pt_again.x - 64.5).abs() < 1e-9);
        assert!((pt_again.y - 193.25).abs() < 1e-9);
    }

    #[test]
    fn sam_resize_matches_longest_side() {
        let spec = SamResizeSpec::new(1200, 1600, 1024);
        assert_eq!(spec.resized_w, 768);
        assert_eq!(spec.resized_h, 1024);
        let p = spec.orig_to_sam().transform_point2(DVec2::new(600.0, 800.0));
        let back = spec.sam_to_orig().transform_point2(p);
        assert!((back.x - 600.0).abs() < 1e-9);
        assert!((back.y - 800.0).abs() < 1e-9);
    }
}
