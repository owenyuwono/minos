//! Elevation, slope, and climate-biome vertex coloring.

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// Smooth Hermite interpolation: maps x from [e0, e1] to [0, 1] with smooth
/// ease-in / ease-out at both ends (the CSS/GLSL `smoothstep`).
fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = clamp01((x - e0) / (e1 - e0));
    t * t * (3.0 - 2.0 * t)
}

/// Map elevation `e` (normalized, -1..1 range) and `slope` (0–1) to an RGB color.
///
/// Seabed (e < 0): geological sediment bands — abyssal to sandy shelf.
/// Land (e >= 0): sand through highland/rock, with snow at high elevation.
/// Slope override (land only): steep slopes blend toward rock grey.
pub fn elevation_color(e: f32, slope: f32) -> [f32; 3] {
    let r;
    let g;
    let b;

    if e < 0.0 {
        // ---- SEABED --------------------------------------------------------
        // Abyssal sediment #232220
        let r_aby = 0x23_u8 as f32 / 255.0;
        let g_aby = 0x22_u8 as f32 / 255.0;
        let b_aby = 0x20_u8 as f32 / 255.0;
        // Deep sediment #3a352c
        let r_dep = 0x3a_u8 as f32 / 255.0;
        let g_dep = 0x35_u8 as f32 / 255.0;
        let b_dep = 0x2c_u8 as f32 / 255.0;
        // Slope sediment #605442
        let r_slp = 0x60_u8 as f32 / 255.0;
        let g_slp = 0x54_u8 as f32 / 255.0;
        let b_slp = 0x42_u8 as f32 / 255.0;
        // Sandy shelf #9c8a66
        let r_shf = 0x9c_u8 as f32 / 255.0;
        let g_shf = 0x8a_u8 as f32 / 255.0;
        let b_shf = 0x66_u8 as f32 / 255.0;

        if e < -0.55 {
            // Pure abyssal
            r = r_aby;
            g = g_aby;
            b = b_aby;
        } else if e < -0.18 {
            // Abyssal → deep sediment (blend window 0.015 around −0.55)
            let t = clamp01((e + 0.55) / 0.015);
            r = lerp(r_aby, r_dep, t);
            g = lerp(g_aby, g_dep, t);
            b = lerp(b_aby, b_dep, t);
        } else if e < -0.045 {
            // Deep → slope sediment (blend window 0.015 around −0.18)
            let t = clamp01((e + 0.18) / 0.015);
            r = lerp(r_dep, r_slp, t);
            g = lerp(g_dep, g_slp, t);
            b = lerp(b_dep, b_slp, t);
        } else {
            // Slope → sandy shelf (blend window 0.015 around −0.045)
            let t = clamp01((e + 0.045) / 0.015);
            r = lerp(r_slp, r_shf, t);
            g = lerp(g_slp, g_shf, t);
            b = lerp(b_slp, b_shf, t);
        }
    } else {
        // ---- LAND ----------------------------------------------------------
        // Sandy shelf at waterline #9c8a66
        let r_shf = 0x9c_u8 as f32 / 255.0;
        let g_shf = 0x8a_u8 as f32 / 255.0;
        let b_shf = 0x66_u8 as f32 / 255.0;
        // Beach sand #b8a36e
        let r_snd = 0xb8_u8 as f32 / 255.0;
        let g_snd = 0xa3_u8 as f32 / 255.0;
        let b_snd = 0x6e_u8 as f32 / 255.0;
        // Lowland #5a7a4a
        let r_low = 0x5a_u8 as f32 / 255.0;
        let g_low = 0x7a_u8 as f32 / 255.0;
        let b_low = 0x4a_u8 as f32 / 255.0;
        // Highland #8a7a55
        let r_hig = 0x8a_u8 as f32 / 255.0;
        let g_hig = 0x7a_u8 as f32 / 255.0;
        let b_hig = 0x55_u8 as f32 / 255.0;
        // Bare rock #7a7060
        let r_rck = 0x7a_u8 as f32 / 255.0;
        let g_rck = 0x70_u8 as f32 / 255.0;
        let b_rck = 0x60_u8 as f32 / 255.0;
        // Snow #e6e8eb
        let r_snw = 0xe6_u8 as f32 / 255.0;
        let g_snw = 0xe8_u8 as f32 / 255.0;
        let b_snw = 0xeb_u8 as f32 / 255.0;

        let (mut cr, mut cg, mut cb);

        if e < 0.012 {
            // Sandy shelf → beach sand (blend over full 0..0.012 range)
            let t = clamp01(e / 0.012);
            cr = lerp(r_shf, r_snd, t);
            cg = lerp(g_shf, g_snd, t);
            cb = lerp(b_shf, b_snd, t);
        } else if e < 0.18 {
            // Sand → lowland (blend window 0.015 around 0.012)
            let t = clamp01((e - 0.012) / 0.015);
            cr = lerp(r_snd, r_low, t);
            cg = lerp(g_snd, g_low, t);
            cb = lerp(b_snd, b_low, t);
        } else if e < 0.42 {
            // Lowland → highland (blend window 0.015 around 0.18)
            let t = clamp01((e - 0.18) / 0.015);
            cr = lerp(r_low, r_hig, t);
            cg = lerp(g_low, g_hig, t);
            cb = lerp(b_low, b_hig, t);
        } else if e < 0.62 {
            // Highland → bare rock (blend window 0.015 around 0.42)
            let t = clamp01((e - 0.42) / 0.015);
            cr = lerp(r_hig, r_rck, t);
            cg = lerp(g_hig, g_rck, t);
            cb = lerp(b_hig, b_rck, t);
        } else {
            // Pure bare rock
            cr = r_rck;
            cg = g_rck;
            cb = b_rck;
        }

        // Snow blend: start at 0.55, fully white by 0.62
        if e > 0.55 {
            let snow_t = clamp01((e - 0.55) / (0.62 - 0.55));
            cr = lerp(cr, r_snw, snow_t);
            cg = lerp(cg, g_snw, snow_t);
            cb = lerp(cb, b_snw, snow_t);
        }

        // Slope override: blend toward rock grey #6e6a64 on cliffs
        if slope > 0.22 {
            let rock_blend = clamp01((slope - 0.22) / 0.20);
            cr = lerp(cr, 0x6e_u8 as f32 / 255.0, rock_blend);
            cg = lerp(cg, 0x6a_u8 as f32 / 255.0, rock_blend);
            cb = lerp(cb, 0x64_u8 as f32 / 255.0, rock_blend);
        }

        r = cr;
        g = cg;
        b = cb;
    }

    [r, g, b]
}

// Temperature below which land trends to snow/ice (blended over SNOW_BLEND °C).
const SNOW_TEMP: f32 = -2.0;
const SNOW_BLEND: f32 = 5.0;

/// Climate-driven biome coloring.
///
/// - Seabed (`e < 0`): identical geological sediment bands to [`elevation_color`].
///   The water shell draws the ocean; climate does not recolor the seabed.
/// - Land (`e >= 0`): biome from `(temp_c, moisture)`.
///   - Dry axis (moisture < ~0.22): desert (hot) → steppe (cold).
///   - Vegetated axis by temperature: tundra → temperate forest → tropical.
///   - Grassland/steppe belt for middling moisture (~0.20..0.48).
///   - Polar/peak snow where temperature < SNOW_TEMP (blended over SNOW_BLEND °C).
///   - High elevation (e > 0.5) blends toward bare rock.
///   - Beach sand at waterline (e < 0.012), seamless with seabed shelf.
///   - Cliff slopes (> 0.22) blend toward rock grey #6e6a64.
///
/// Operation order matches demiurge's `biomeColor` exactly (required for the
/// 4.3 chunk golden hash).
pub fn biome_color(temp_c: f32, moisture: f32, elevation: f32, slope: f32) -> [f32; 3] {
    let e = elevation;

    if e < 0.0 {
        // ---- SEABED: verbatim copy of elevation_color seabed branch ----------
        let r_aby = 0x23_u8 as f32 / 255.0;
        let g_aby = 0x22_u8 as f32 / 255.0;
        let b_aby = 0x20_u8 as f32 / 255.0;
        let r_dep = 0x3a_u8 as f32 / 255.0;
        let g_dep = 0x35_u8 as f32 / 255.0;
        let b_dep = 0x2c_u8 as f32 / 255.0;
        let r_slp = 0x60_u8 as f32 / 255.0;
        let g_slp = 0x54_u8 as f32 / 255.0;
        let b_slp = 0x42_u8 as f32 / 255.0;
        let r_shf = 0x9c_u8 as f32 / 255.0;
        let g_shf = 0x8a_u8 as f32 / 255.0;
        let b_shf = 0x66_u8 as f32 / 255.0;

        let (r, g, b) = if e < -0.55 {
            (r_aby, g_aby, b_aby)
        } else if e < -0.18 {
            let t = clamp01((e + 0.55) / 0.015);
            (lerp(r_aby, r_dep, t), lerp(g_aby, g_dep, t), lerp(b_aby, b_dep, t))
        } else if e < -0.045 {
            let t = clamp01((e + 0.18) / 0.015);
            (lerp(r_dep, r_slp, t), lerp(g_dep, g_slp, t), lerp(b_dep, b_slp, t))
        } else {
            let t = clamp01((e + 0.045) / 0.015);
            (lerp(r_slp, r_shf, t), lerp(g_slp, g_shf, t), lerp(b_slp, b_shf, t))
        };

        [r, g, b]
    } else {
        // ---- LAND: biome by (temperature, moisture) --------------------------

        // Biome palette colours (generic — Earth's look emerges from Earth-like params)
        let sand_hot  = [0xc2_u8 as f32 / 255.0, 0xa8_u8 as f32 / 255.0, 0x78_u8 as f32 / 255.0];
        let sand_cold = [0x9a_u8 as f32 / 255.0, 0x8c_u8 as f32 / 255.0, 0x70_u8 as f32 / 255.0];
        let grass     = [0x8f_u8 as f32 / 255.0, 0x95_u8 as f32 / 255.0, 0x55_u8 as f32 / 255.0];
        let tropical  = [0x2f_u8 as f32 / 255.0, 0x6b_u8 as f32 / 255.0, 0x34_u8 as f32 / 255.0];
        let temperate = [0x4f_u8 as f32 / 255.0, 0x7a_u8 as f32 / 255.0, 0x45_u8 as f32 / 255.0];
        let tundra    = [0x6b_u8 as f32 / 255.0, 0x6f_u8 as f32 / 255.0, 0x57_u8 as f32 / 255.0];
        let rock      = [0x7a_u8 as f32 / 255.0, 0x70_u8 as f32 / 255.0, 0x60_u8 as f32 / 255.0];
        let snow      = [0xe6_u8 as f32 / 255.0, 0xe8_u8 as f32 / 255.0, 0xeb_u8 as f32 / 255.0];

        // Shelf colour for beach waterline blend
        let r_shf = 0x9c_u8 as f32 / 255.0;
        let g_shf = 0x8a_u8 as f32 / 255.0;
        let b_shf = 0x66_u8 as f32 / 255.0;

        // --- 2. Dry vs vegetated axis -----------------------------------------
        // dryness: 1 when very dry (M→0), 0 once moisture clears desert line.
        let dry = 1.0 - smoothstep(0.14, 0.22, moisture);
        // Within dry: hot deserts vs cold steppe (by temperature).
        let hot_dry = smoothstep(2.0, 14.0, temp_c);
        let desert_r = lerp(sand_cold[0], sand_hot[0], hot_dry);
        let desert_g = lerp(sand_cold[1], sand_hot[1], hot_dry);
        let desert_b = lerp(sand_cold[2], sand_hot[2], hot_dry);

        // --- 3. Vegetated biome by temperature --------------------------------
        let t_warm = smoothstep(2.0, 8.0, temp_c);     // 0 cold (tundra), 1 by 8°C (forest)
        let t_hot  = smoothstep(18.0, 24.0, temp_c);   // 0 mild, 1 by 24°C
        let trop_wet = smoothstep(0.42, 0.58, moisture); // tropical needs moisture too
        let trop_mix = t_hot * trop_wet;
        let mild_r = lerp(temperate[0], tropical[0], trop_mix);
        let mild_g = lerp(temperate[1], tropical[1], trop_mix);
        let mild_b = lerp(temperate[2], tropical[2], trop_mix);
        let mut veg_r = lerp(tundra[0], mild_r, t_warm);
        let mut veg_g = lerp(tundra[1], mild_g, t_warm);
        let mut veg_b = lerp(tundra[2], mild_b, t_warm);

        // --- 4. Grassland/steppe for middling moisture ------------------------
        let grass_w = smoothstep(0.20, 0.30, moisture)
            * (1.0 - smoothstep(0.38, 0.48, moisture))
            * (1.0 - trop_mix);
        veg_r = lerp(veg_r, grass[0], grass_w);
        veg_g = lerp(veg_g, grass[1], grass_w);
        veg_b = lerp(veg_b, grass[2], grass_w);

        // --- 5. Combine dry (desert) with vegetated along dryness axis --------
        let mut r = lerp(veg_r, desert_r, dry);
        let mut g = lerp(veg_g, desert_g, dry);
        let mut b = lerp(veg_b, desert_b, dry);

        // --- 6. Beach sand at the immediate waterline -------------------------
        if e < 0.012 {
            let t = clamp01(e / 0.012);
            r = lerp(r_shf, r, t);
            g = lerp(g_shf, g, t);
            b = lerp(b_shf, b, t);
        }

        // --- 7. High elevation trends rocky -----------------------------------
        if e > 0.5 {
            let rock_t = smoothstep(0.5, 0.72, e);
            r = lerp(r, rock[0], rock_t);
            g = lerp(g, rock[1], rock_t);
            b = lerp(b, rock[2], rock_t);
        }

        // --- 8. Snow / ice by temperature (latitude- AND altitude-dependent) --
        if temp_c < SNOW_TEMP + SNOW_BLEND {
            let snow_t = 1.0 - smoothstep(SNOW_TEMP, SNOW_TEMP + SNOW_BLEND, temp_c);
            r = lerp(r, snow[0], snow_t);
            g = lerp(g, snow[1], snow_t);
            b = lerp(b, snow[2], snow_t);
        }

        // --- 9. Slope override: cliffs → rock grey #6e6a64 --------------------
        if slope > 0.22 {
            let rock_blend = clamp01((slope - 0.22) / 0.20);
            r = lerp(r, 0x6e_u8 as f32 / 255.0, rock_blend);
            g = lerp(g, 0x6a_u8 as f32 / 255.0, rock_blend);
            b = lerp(b, 0x64_u8 as f32 / 255.0, rock_blend);
        }

        [r, g, b]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seabed_deep_is_dark() {
        let c = elevation_color(-0.8, 0.0);
        assert!(c[0] < 0.2 && c[1] < 0.2 && c[2] < 0.2);
    }

    #[test]
    fn land_green() {
        let c = elevation_color(0.1, 0.0);
        assert!(c[1] > c[0] && c[1] > c[2]); // green channel dominant
    }

    #[test]
    fn snow_at_high_elevation() {
        let c = elevation_color(0.65, 0.0);
        assert!(c[0] > 0.8 && c[1] > 0.8 && c[2] > 0.8);
    }

    #[test]
    fn slope_override_darkens_land() {
        let flat = elevation_color(0.1, 0.0);
        let steep = elevation_color(0.1, 0.9);
        // slope grey is darker/more neutral than green lowland
        assert!(steep[0] > steep[1] * 0.9); // R and G closer on steep grey
        let _ = flat;
    }

    #[test]
    fn colors_in_range() {
        for ei in -20i32..=20 {
            let e = ei as f32 / 20.0;
            for si in 0..=10 {
                let slope = si as f32 / 10.0;
                let c = elevation_color(e, slope);
                assert!(c[0] >= 0.0 && c[0] <= 1.0);
                assert!(c[1] >= 0.0 && c[1] <= 1.0);
                assert!(c[2] >= 0.0 && c[2] <= 1.0);
            }
        }
    }

    // -- biome_color tests -------------------------------------------------------

    #[test]
    fn biome_seabed_deep_is_dark() {
        let c = biome_color(20.0, 0.5, -0.8, 0.0);
        assert!(c[0] < 0.2 && c[1] < 0.2 && c[2] < 0.2, "deep seabed should be near-black: {c:?}");
    }

    #[test]
    fn biome_tropical_is_dark_green() {
        // Hot + wet → tropical jungle
        let c = biome_color(28.0, 0.8, 0.1, 0.0);
        // tropical[1] = 0x6b/255 ≈ 0.42; green channel should lead
        assert!(c[1] > c[0] && c[1] > c[2], "tropical should be greenish: {c:?}");
    }

    #[test]
    fn biome_polar_ice_is_white() {
        // Very cold → snow/ice
        let c = biome_color(-20.0, 0.5, 0.05, 0.0);
        assert!(c[0] > 0.8 && c[1] > 0.8 && c[2] > 0.8, "polar ice should be near-white: {c:?}");
    }

    #[test]
    fn biome_desert_vs_cold_wet_are_distinct() {
        let desert = biome_color(30.0, 0.05, 0.2, 0.0); // hot-dry
        let cold_wet = biome_color(-5.0, 0.9, 0.2, 0.0); // cold-wet (tundra/snow)
        // They should differ by at least 0.1 in some channel
        let diff = (desert[0] - cold_wet[0]).abs()
            .max((desert[1] - cold_wet[1]).abs())
            .max((desert[2] - cold_wet[2]).abs());
        assert!(diff > 0.1, "desert and cold-wet should differ: desert={desert:?} cold_wet={cold_wet:?}");
    }

    #[test]
    fn biome_deterministic() {
        let a = biome_color(15.0, 0.5, 0.2, 0.1);
        let b = biome_color(15.0, 0.5, 0.2, 0.1);
        assert_eq!(a, b, "biome_color must be deterministic");
    }

    #[test]
    fn biome_colors_in_range() {
        // Sweep elevation, slope, temperature, moisture to check [0,1] invariant.
        for ei in -20i32..=20 {
            let e = ei as f32 / 20.0;
            for si in 0..=5 {
                let slope = si as f32 / 5.0;
                for ti in [-30i32, -10, 0, 15, 30] {
                    for mi in [0i32, 25, 50, 75, 100] {
                        let temp = ti as f32;
                        let moist = mi as f32 / 100.0;
                        let c = biome_color(temp, moist, e, slope);
                        assert!(c[0] >= 0.0 && c[0] <= 1.0, "R out of range e={e} s={slope} t={temp} m={moist}: {c:?}");
                        assert!(c[1] >= 0.0 && c[1] <= 1.0, "G out of range e={e} s={slope} t={temp} m={moist}: {c:?}");
                        assert!(c[2] >= 0.0 && c[2] <= 1.0, "B out of range e={e} s={slope} t={temp} m={moist}: {c:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn biome_hot_dry_vs_cold_wet_land_differ() {
        // Above-water: hot-dry desert vs cold-wet tundra/ice
        let hot_dry  = biome_color(35.0, 0.05, 0.3, 0.0);
        let cold_wet = biome_color(-8.0, 0.8,  0.3, 0.0);
        let diff = (hot_dry[0] - cold_wet[0]).abs()
            .max((hot_dry[1] - cold_wet[1]).abs())
            .max((hot_dry[2] - cold_wet[2]).abs());
        assert!(diff > 0.15, "hot-dry and cold-wet should be visually distinct: {hot_dry:?} vs {cold_wet:?}");
    }
}
