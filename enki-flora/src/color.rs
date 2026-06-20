//! Pigment / bark color derivation — port of dryad `colorRamp.js`.
//!
//! `pigment ∈ [0,1]` sweeps the full HSL hue wheel (0→360°) at S=0.55, L=0.42.
//! Single source of truth for the foliage color ramp.

/// HSL→RGB. `h` in `[0,1]` (0=red, 1/3=green, 2/3=blue), `s`,`l` in `[0,1]`.
/// Returns `[r, g, b]` each in `[0,1]`. Mirrors the JS closure exactly.
fn hsl_to_rgb(h: f64, s: f64, l: f64) -> [f64; 3] {
    let a = s * l.min(1.0 - l);
    // f(n) = l - a * clamp(min(k-3, 9-k, 1), -1, +inf) where k = (n + h*12) % 12
    let f = |n: f64| -> f64 {
        let k = (n + h * 12.0) % 12.0;
        // JS: Math.max(-1, Math.min(k - 3, 9 - k, 1))
        let inner = (k - 3.0).min(9.0 - k).min(1.0);
        l - a * inner.max(-1.0)
    };
    [f(0.0), f(8.0), f(4.0)]
}

/// `pigmentToColor(pigment) -> [r, g, b]` in `[0,1]`.
///
/// Full hue sweep: pigment 0→1 maps hue 0°→360°. S=0.55, L=0.42. `pigment` is
/// clamped to `[0,1]` before lookup.
pub fn pigment_to_color(pigment: f64) -> [f64; 3] {
    let p = if pigment < 0.0 {
        0.0
    } else if pigment > 1.0 {
        1.0
    } else {
        pigment
    };
    hsl_to_rgb(p, 0.55, 0.42)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoints_and_clamp() {
        // pigment 0 and 1 are both hue 0°/360° → identical red.
        let zero = pigment_to_color(0.0);
        let one = pigment_to_color(1.0);
        for i in 0..3 {
            assert!((zero[i] - one[i]).abs() < 1e-12);
        }
        // clamp: out-of-range equals the endpoint.
        assert_eq!(pigment_to_color(-3.0), pigment_to_color(0.0));
        assert_eq!(pigment_to_color(7.0), pigment_to_color(1.0));
    }

    /// Cross-check against dryad's `pigmentToColor` (node reference values).
    #[test]
    fn matches_js_reference() {
        let cases = [
            (0.0, [0.651, 0.18899999999999997, 0.18899999999999997]),
            (0.33, [0.19823999999999997, 0.651, 0.18899999999999997]),
            (0.75, [0.42, 0.18899999999999997, 0.651]),
        ];
        for (p, exp) in cases {
            let got = pigment_to_color(p);
            for i in 0..3 {
                assert!(
                    (got[i] - exp[i]).abs() < 1e-12,
                    "pigment {p} ch {i}: got {}, expect {}",
                    got[i],
                    exp[i]
                );
            }
        }
    }
}
