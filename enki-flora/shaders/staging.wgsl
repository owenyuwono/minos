// staging.wgsl — SCENE staging for the standalone flora viewer (NOT the tree).
//
// ponytail: the cheapest believable "studio" around one tree, reusing enki's
// EXISTING non-pulling 4×vec3 vertex layout, reversed-Z, FrameUniforms lights,
// and the camera-relative f64->f32 push scheme — NO net-new RHI (no second color
// target, no depth-sampler, no shadow-map descriptors, no post pass). This file
// hosts THREE pipelines, selected by entry point:
//   (1) SKY    : vs_sky    / fs_sky    — fullscreen horizon→zenith gradient
//   (2) GROUND : vs_ground / fs_ground — large lit ground plane at y=0
//   (3) SHADOW : vs_shadow / fs_shadow — soft radial contact disc (alpha-blended)
//
// All three share the SAME 4×vec3 vertex bindings (pos / nrm / uv / attr) that
// terrain, branches and leaves use, so they need no pipeline knobs beyond enki's
// `blend`/`fill` bools. Shaders validate in enki-flora/tests/shader.rs (naga 29).
//
// ── Shared uniforms (set 0, binding 0) — verbatim from flora.wgsl / terrain ──

struct FrameUniforms {
    view_proj  : mat4x4<f32>,
    camera_pos : vec4<f32>,   // always (0,0,0,1) in camera-relative space
    sun0_dir   : vec4<f32>,   // primary sun, points TOWARD the light; w unused
    sun0_color : vec4<f32>,   // sun color × intensity
    sun1_dir   : vec4<f32>,   // fill light dir
    sun1_color : vec4<f32>,   // fill color × intensity
    hemi_sky   : vec4<f32>,   // hemisphere sky color
    hemi_ground: vec4<f32>,   // hemisphere ground color
    ambient    : vec4<f32>,   // x=y=z=ambient scalar; w unused
}

@group(0) @binding(0) var<uniform> frame: FrameUniforms;

// ── SUN SHADOWS (set 1) — identical layout to flora.wgsl so the GROUND receives
//    the tree's cast shadow. dryad ground: receiveShadow = true, castShadow =
//    false — so the ground is NOT rendered into the shadow map (a flat y=0 plane
//    casts nothing useful and would fight the receivers), it only READS it here.
//    // ponytail: ground-doesn't-cast matches dryad and avoids a coplanar depth
//    // fight; only branch+leaf are drawn into the depth target. ──
struct ShadowUniforms {
    light_view_proj : mat4x4<f32>,
    params          : vec4<f32>,    // (1/shadowMapSize, normalBias, enabled, 0)
}
@group(1) @binding(0) var shadow_map     : texture_depth_2d;
@group(1) @binding(1) var shadow_sampler : sampler_comparison;
@group(1) @binding(2) var<uniform> shadow : ShadowUniforms;

// ── IBL (set 1, bindings 3-5) — identical layout to flora.wgsl so the GROUND
//    receives the same image-based lighting (diffuse SH + equirect specular). ──
struct ShCoeffs {
    c      : array<vec4<f32>, 9>,
    params : vec4<f32>,            // x = max LOD
}
@group(1) @binding(3) var ibl_spec    : texture_2d<f32>;
@group(1) @binding(4) var ibl_sampler : sampler;
@group(1) @binding(5) var<uniform> sh : ShCoeffs;

const PI : f32 = 3.14159265359;

fn sh_irradiance(n: vec3<f32>) -> vec3<f32> {
    let x = n.x; let y = n.y; let z = n.z;
    var r = sh.c[0].xyz;
    r += sh.c[1].xyz * y;
    r += sh.c[2].xyz * z;
    r += sh.c[3].xyz * x;
    r += sh.c[4].xyz * (x * y);
    r += sh.c[5].xyz * (y * z);
    r += sh.c[6].xyz * (3.0 * z * z - 1.0);
    r += sh.c[7].xyz * (x * z);
    r += sh.c[8].xyz * (x * x - y * y);
    return max(r, vec3<f32>(0.0));
}

fn equirect_uv(d: vec3<f32>) -> vec2<f32> {
    let u = atan2(d.z, d.x) * (0.5 / PI) + 0.5;
    let v = acos(clamp(d.y, -1.0, 1.0)) / PI;
    return vec2<f32>(u, v);
}

fn ibl_specular(refl: vec3<f32>, roughness: f32) -> vec3<f32> {
    let lod = roughness * sh.params.x;
    return textureSampleLevel(ibl_spec, ibl_sampler, equirect_uv(refl), lod).rgb;
}

fn env_brdf_approx(roughness: f32, ndv: f32) -> vec2<f32> {
    let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4<f32>( 1.0,  0.0425,  1.04, -0.04);
    let r = roughness * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * ndv)) * r.x + r.y;
    return vec2<f32>(-1.04, 1.04) * a004 + r.zw;
}

// Cook-Torrance GGX direct(sun0,shadowed) + fill(sun1) + IBL(SH diffuse + equirect
// specular). Identical to flora.wgsl::pbr_shade so the ground sits in the SAME
// lit range as the tree at their contact line.
fn pbr_shade(n: vec3<f32>, v: vec3<f32>, albedo: vec3<f32>, roughness: f32,
             shadow_f: f32, ao: f32) -> vec3<f32> {
    let F0 = vec3<f32>(0.04);
    let ndv = max(dot(n, v), 1e-4);

    let L = frame.sun0_dir.xyz;
    let H = normalize(L + v);
    let ndl = max(dot(n, L), 0.0);
    let ndh = max(dot(n, H), 0.0);
    let vdh = max(dot(v, H), 0.0);
    let a  = roughness * roughness;
    let a2 = a * a;
    let d_denom = (ndh * ndh * (a2 - 1.0) + 1.0);
    let D = a2 / max(PI * d_denom * d_denom, 1e-6);
    let lv = ndl * sqrt(ndv * ndv * (1.0 - a2) + a2);
    let ll = ndv * sqrt(ndl * ndl * (1.0 - a2) + a2);
    let Vis = 0.5 / max(lv + ll, 1e-5);
    let F = F0 + (vec3<f32>(1.0) - F0) * pow(1.0 - vdh, 5.0);
    let spec = D * Vis * F;
    let kd = (vec3<f32>(1.0) - F);
    let direct = (kd * albedo / PI + spec) * frame.sun0_color.xyz * ndl * shadow_f;

    let fill = albedo / PI * frame.sun1_color.xyz * max(dot(n, frame.sun1_dir.xyz), 0.0);

    let diff_ibl = sh_irradiance(n) * albedo;
    let R = reflect(-v, n);
    let prefiltered = ibl_specular(R, roughness);
    let ab = env_brdf_approx(roughness, ndv);
    let spec_ibl = prefiltered * (F0 * ab.x + ab.y);

    // dryad: reflectedLight.indirectDiffuse *= mix(1.0, vAo, 0.85) — AO floored
    // at 15% and applied to the INDIRECT (IBL/SH) term ONLY, never the direct sun.
    // (ground passes ao=1.0 → mix(1,1,0.85)=1, no visual change; keeps the two
    // pbr_shade copies byte-identical with flora.wgsl.)
    let ao_indirect = mix(1.0, ao, 0.85);
    return direct + fill + (diff_ibl + spec_ibl) * ao_indirect;
}

const SHADOW_DEPTH_BIAS : f32 = 0.0005;
const SHADOW_NORMAL_BIAS : f32 = 0.02;

// PCF 3×3 over the compare sampler — byte-identical to flora.wgsl::sample_shadow
// so the tree and ground agree on the shadow term at their contact line.
fn sample_shadow(world_pos: vec3<f32>, world_normal: vec3<f32>, ndl: f32) -> f32 {
    if (shadow.params.z < 0.5) {
        return 1.0;
    }
    let slope = clamp(1.0 - ndl, 0.0, 1.0);
    let biased = world_pos + world_normal * (SHADOW_NORMAL_BIAS * (0.4 + slope));
    let clip = shadow.light_view_proj * vec4<f32>(biased, 1.0);
    if (clip.w <= 0.0) {
        return 1.0;
    }
    let ndc = clip.xyz / clip.w;
    if (ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0) {
        return 1.0;
    }
    let uv = ndc.xy * 0.5 + vec2<f32>(0.5, 0.5);
    // Reversed-Z: ADD the bias (push receiver toward the light) — see flora.wgsl.
    let ref_depth = clamp(ndc.z + SHADOW_DEPTH_BIAS, 0.0, 1.0);
    let texel = shadow.params.x;
    var sum = 0.0;
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let ofs = vec2<f32>(f32(dx), f32(dy)) * texel;
            sum = sum + textureSampleCompare(shadow_map, shadow_sampler, uv + ofs, ref_depth);
        }
    }
    return sum / 9.0;
}

// ── Shared lighting helpers (identical to flora.wgsl so the ground sits in the
//    same exposure range as the tree it grows under). The hemisphere/half-Lambert
//    helpers were removed with the move to IBL+PBR (see pbr_shade above); the sky
//    FS still reads hemi_sky/hemi_ground straight for its background gradient. ──

// ACES filmic tonemap (Narkowicz 2015), matching flora.wgsl / terrain.wgsl.
fn aces_filmic(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

// Shared push: just a camera-relative model matrix (ground/shadow); the sky
// ignores it. 64 bytes, well under the 128-byte immediate limit.
struct StagePush {
    model : mat4x4<f32>,
}
var<immediate> stage_pc: StagePush;

// Shared 4×vec3 vertex input (only the streams each entry needs are read).
struct StageVsIn {
    @location(0) position : vec3<f32>,  // ground: world-XZ plane corner; sky: NDC.xy in .xy
    @location(1) normal   : vec3<f32>,  // ground: +Y
    @location(2) uv       : vec3<f32>,  // .xy = [0,1]² corner uv
    @location(3) attr     : vec3<f32>,  // unused (padding stream)
}

// ─────────────────────────────────────────────────────────────────────────────
// (1) SKY — analytic Preetham daylight model (ported from three.js Sky.js).
//
// ponytail: replaces the flat clear without touching the RHI's private
// clear_color. A fullscreen quad is emitted at clip z = 0.0 (the reversed-Z FAR
// plane) and drawn FIRST, so every later fragment (ground/tree, depth > 0)
// passes the GREATER depth test over it. position.xy carries the NDC corner
// directly (-1..1); the FS reconstructs a world-space view ray by inverting
// frame.view_proj (rotation-only, camera at origin in camera-relative space) so
// the sun disc lines up where FrameUniforms.sun0_dir points. The raw linear
// Preetham radiance feeds the SAME aces_filmic the ground/tree use, so the sky
// composites in the identical space (dryad: ACES @ exposure 1.0).
//
// dryad Sky addon params: turbidity 6.0, rayleigh 2.0, mieCoefficient 0.005,
// mieDirectionalG 0.8, up = (0,1,0), sunPosition = normalized sun dir.
// ─────────────────────────────────────────────────────────────────────────────

const SKY_UP        : vec3<f32> = vec3<f32>(0.0, 1.0, 0.0);
// ponytail: HDR scale on the raw Preetham radiance (see fs_sky) — brings the
// daytime sky into a tonemap range that reads as blue without blowing out the
// frame or tripping the bloom high-pass.
const SKY_INTENSITY : f32 = 0.25;
const SKY_TURBIDITY : f32 = 6.0;
const SKY_RAYLEIGH  : f32 = 2.0;
const SKY_MIE_COEFF : f32 = 0.005;
const SKY_MIE_G     : f32 = 0.8;

const SKY_E : f32 = 2.718281828459045;
// totalRayleigh (Sky.js:82, precomputed for the default lambda/n/N/pn).
const SKY_TOTAL_RAYLEIGH : vec3<f32> =
    vec3<f32>(5.804542996261093e-6, 1.3562911419845635e-5, 3.0265902468824876e-5);
// MieConst (Sky.js:89).
const SKY_MIE_CONST : vec3<f32> =
    vec3<f32>(1.8399918514433978e14, 2.7798023919660528e14, 4.0790479543861094e14);
const SKY_CUTOFF_ANGLE    : f32 = 1.6110731556870734;
const SKY_STEEPNESS       : f32 = 1.5;
const SKY_EE              : f32 = 1000.0;
const SKY_RAYLEIGH_ZENITH : f32 = 8.4e3;
const SKY_MIE_ZENITH      : f32 = 1.25e3;
const SKY_SUN_ANGULAR_COS : f32 = 0.9999566769464485;
const SKY_THREE_OVER_16PI : f32 = 0.05968310365946075;
const SKY_ONE_OVER_4PI    : f32 = 0.07957747154594767;

fn sky_sun_intensity(zenith_cos: f32) -> f32 {
    let z = clamp(zenith_cos, -1.0, 1.0);
    return SKY_EE * max(0.0, 1.0 - pow(SKY_E, -((SKY_CUTOFF_ANGLE - acos(z)) / SKY_STEEPNESS)));
}

fn sky_total_mie(t: f32) -> vec3<f32> {
    let c = (0.2 * t) * 10e-18;
    return 0.434 * c * SKY_MIE_CONST;
}

fn sky_rayleigh_phase(cos_theta: f32) -> f32 {
    return SKY_THREE_OVER_16PI * (1.0 + cos_theta * cos_theta);
}

fn sky_hg_phase(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let inv = 1.0 / pow(1.0 - 2.0 * g * cos_theta + g2, 1.5);
    return SKY_ONE_OVER_4PI * ((1.0 - g2) * inv);
}

// `dir` = unit world-space view ray; `sun_dir` = frame.sun0_dir.xyz (normalized).
// Returns LINEAR radiance; the caller applies aces_filmic.
fn preetham_sky(dir: vec3<f32>, sun_dir: vec3<f32>) -> vec3<f32> {
    let sun = normalize(sun_dir);

    // ── VS-side terms (Sky.js:101-128), now evaluated per-fragment. ──
    let sunE = sky_sun_intensity(dot(sun, SKY_UP));
    // vSunfade: with a unit sun dir, sun.y is the elevation sine (Sky.js feeds a
    // huge sunPosition so its /450000 ratio → ~0; the unit-dir form is faithful).
    let sunfade = 1.0 - clamp(1.0 - exp(sun.y), 0.0, 1.0);
    let rayleighCoeff = SKY_RAYLEIGH - (1.0 * (1.0 - sunfade));
    let betaR = SKY_TOTAL_RAYLEIGH * rayleighCoeff;
    let betaM = sky_total_mie(SKY_TURBIDITY) * SKY_MIE_COEFF;

    // ── FS body (Sky.js:170-208). ──
    let zenithAngle = acos(max(0.0, dot(SKY_UP, dir)));
    let inv = 1.0 / (cos(zenithAngle)
        + 0.15 * pow(93.885 - (zenithAngle * 180.0 / PI), -1.253));
    let sR = SKY_RAYLEIGH_ZENITH * inv;
    let sM = SKY_MIE_ZENITH * inv;

    let Fex = exp(-(betaR * sR + betaM * sM));

    let cosTheta = dot(dir, sun);
    let rPhase = sky_rayleigh_phase(cosTheta * 0.5 + 0.5);
    let betaRTheta = betaR * rPhase;
    let mPhase = sky_hg_phase(cosTheta, SKY_MIE_G);
    let betaMTheta = betaM * mPhase;

    let bsum = betaR + betaM;
    var Lin = pow(sunE * ((betaRTheta + betaMTheta) / bsum) * (1.0 - Fex), vec3<f32>(1.5));
    Lin = Lin * mix(
        vec3<f32>(1.0),
        pow(sunE * ((betaRTheta + betaMTheta) / bsum) * Fex, vec3<f32>(0.5)),
        clamp(pow(1.0 - dot(SKY_UP, sun), 5.0), 0.0, 1.0));

    var L0 = vec3<f32>(0.1) * Fex;
    let sundisk = smoothstep(SKY_SUN_ANGULAR_COS, SKY_SUN_ANGULAR_COS + 0.00002, cosTheta);
    L0 = L0 + (sunE * 19000.0 * Fex) * sundisk;

    let texColor = (Lin + L0) * 0.04 + vec3<f32>(0.0, 0.0003, 0.00075);
    let retColor = pow(texColor, vec3<f32>(1.0 / (1.2 + (1.2 * sunfade))));
    return retColor;   // linear; caller applies aces_filmic
}

// Inverse of a 4×4 matrix (cofactor method). WGSL has no built-in inverse; we
// only call this once per sky fragment over the rotation-only view_proj, so the
// cost is irrelevant. Used to turn an NDC corner into a world-space view ray.
fn mat4_inverse(m: mat4x4<f32>) -> mat4x4<f32> {
    let a00 = m[0][0]; let a01 = m[0][1]; let a02 = m[0][2]; let a03 = m[0][3];
    let a10 = m[1][0]; let a11 = m[1][1]; let a12 = m[1][2]; let a13 = m[1][3];
    let a20 = m[2][0]; let a21 = m[2][1]; let a22 = m[2][2]; let a23 = m[2][3];
    let a30 = m[3][0]; let a31 = m[3][1]; let a32 = m[3][2]; let a33 = m[3][3];

    let b00 = a00 * a11 - a01 * a10;
    let b01 = a00 * a12 - a02 * a10;
    let b02 = a00 * a13 - a03 * a10;
    let b03 = a01 * a12 - a02 * a11;
    let b04 = a01 * a13 - a03 * a11;
    let b05 = a02 * a13 - a03 * a12;
    let b06 = a20 * a31 - a21 * a30;
    let b07 = a20 * a32 - a22 * a30;
    let b08 = a20 * a33 - a23 * a30;
    let b09 = a21 * a32 - a22 * a31;
    let b10 = a21 * a33 - a23 * a31;
    let b11 = a22 * a33 - a23 * a32;

    let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
    let invDet = 1.0 / det;

    return mat4x4<f32>(
        vec4<f32>(
            ( a11 * b11 - a12 * b10 + a13 * b09) * invDet,
            (-a01 * b11 + a02 * b10 - a03 * b09) * invDet,
            ( a31 * b05 - a32 * b04 + a33 * b03) * invDet,
            (-a21 * b05 + a22 * b04 - a23 * b03) * invDet),
        vec4<f32>(
            (-a10 * b11 + a12 * b08 - a13 * b07) * invDet,
            ( a00 * b11 - a02 * b08 + a03 * b07) * invDet,
            (-a30 * b05 + a32 * b02 - a33 * b01) * invDet,
            ( a20 * b05 - a22 * b02 + a23 * b01) * invDet),
        vec4<f32>(
            ( a10 * b10 - a11 * b08 + a13 * b06) * invDet,
            (-a00 * b10 + a01 * b08 - a03 * b06) * invDet,
            ( a30 * b04 - a31 * b02 + a33 * b00) * invDet,
            (-a20 * b04 + a21 * b02 - a23 * b00) * invDet),
        vec4<f32>(
            (-a10 * b09 + a11 * b07 - a12 * b06) * invDet,
            ( a00 * b09 - a01 * b07 + a02 * b06) * invDet,
            (-a30 * b03 + a31 * b01 - a32 * b00) * invDet,
            ( a20 * b03 - a21 * b01 + a22 * b00) * invDet),
    );
}

struct SkyVsOut {
    @builtin(position) clip_pos : vec4<f32>,
    @location(0)       ndc_xy   : vec2<f32>,   // NDC corner (±1) for ray rebuild
}

@vertex
fn vs_sky(v: StageVsIn) -> SkyVsOut {
    var out: SkyVsOut;
    // position.xy already holds NDC corners (±1). z = 0 → reversed-Z far plane.
    out.clip_pos = vec4<f32>(v.position.x, v.position.y, 0.0, 1.0);
    out.ndc_xy = v.position.xy;
    return out;
}

@fragment
fn fs_sky(in: SkyVsOut) -> @location(0) vec4<f32> {
    // Reconstruct a world-space view ray from the NDC corner. In camera-relative
    // space the camera sits at the origin, so the unprojected far point IS the
    // ray direction. view_proj is rotation-only → inverse maps NDC → world dir.
    let inv_vp = mat4_inverse(frame.view_proj);
    let world = inv_vp * vec4<f32>(in.ndc_xy, 1.0, 1.0);  // z=1: far on reversed-Z
    let dir = normalize(world.xyz / world.w);
    // ponytail: the raw three.js Preetham radiance is HDR-bright (linear luma
    // ~0.8–2.1) — at our composite exposure it blew the whole frame to near-white
    // (sky overexposed, tree/ground washed out) AND most of the sky crossed the
    // bloom threshold (1.1), bleeding glow everywhere. Scale it to SKY_INTENSITY
    // so the gradient lands as a believable daytime blue (zenith ~rgb 83,113,141)
    // that stays under the bloom gate and leaves headroom for the lit tree.
    let col = preetham_sky(dir, frame.sun0_dir.xyz) * SKY_INTENSITY;
    // LINEAR HDR out — the flora OutputPass applies ACES ONCE on scene+bloom.
    return vec4<f32>(col, 1.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// (2) GROUND — large lit plane at y=0 (the tree base sits at y=0).
//
// Same ambient + hemisphere + 2×half-Lambert directional model as the bark FS,
// over a subtle earthy albedo (dryad ground 0x4a3e2a, linear-ish). Camera-
// relative: stage_pc.model carries (−camera) translation so the ±200 m quad is
// drawn near the origin in f32. Opaque (blend:false) → writes reversed-Z depth,
// so the tree and contact disc test against it.
// ─────────────────────────────────────────────────────────────────────────────

struct GroundVsOut {
    @builtin(position) clip_pos     : vec4<f32>,
    @location(0)       world_normal : vec3<f32>,
    @location(1)       uv           : vec2<f32>,  // 0..1 across the plane (radial fade)
    @location(2)       shadow_pos   : vec3<f32>,  // tree-local world pos (shadow lookup)
    @location(3)       view_pos     : vec3<f32>,  // camera-relative world pos (PBR view dir)
}

@vertex
fn vs_ground(v: StageVsIn) -> GroundVsOut {
    var out: GroundVsOut;
    let world_pos = stage_pc.model * vec4<f32>(v.position, 1.0);
    out.clip_pos = frame.view_proj * world_pos;
    let m3 = mat3x3<f32>(
        stage_pc.model[0].xyz,
        stage_pc.model[1].xyz,
        stage_pc.model[2].xyz,
    );
    out.world_normal = m3 * v.normal;
    out.uv = v.uv.xy;
    // Tree-local world pos (camera-relative minus the model translation). The
    // ground sits at world y=0, so this recovers the true world position the
    // light matrix expects (ground origin == tree origin == world origin).
    out.shadow_pos = world_pos.xyz - stage_pc.model[3].xyz;
    out.view_pos = world_pos.xyz; // camera-relative (camera at origin)
    return out;
}

@fragment
fn fs_ground(in: GroundVsOut) -> @location(0) vec4<f32> {
    // Earthy albedo (dryad 0x4a3e2a ≈ srgb(74,62,42) → ~linear). Subtle, neutral.
    let albedo = vec3<f32>(0.072, 0.052, 0.026);
    let n = normalize(in.world_normal);

    // ── Cook-Torrance PBR: matte ground (roughness ≈ 0.9), dielectric. The ground
    //    RECEIVES the tree's cast shadow on sun0 and the same IBL as the tree. ──
    let roughness = 0.9;
    let v = normalize(-in.view_pos);
    let ndl0     = dot(n, frame.sun0_dir.xyz);
    let shadow_f = sample_shadow(in.shadow_pos, n, ndl0);

    // Radial vignette toward the plane edge so the ground fades into the sky
    // gradient instead of ending in a hard line (uv centred at 0.5,0.5).
    let r = length(in.uv - vec2<f32>(0.5, 0.5)) * 2.0;
    let edge = 1.0 - smoothstep(0.55, 1.0, r);

    var lit = pbr_shade(n, v, albedo, roughness, shadow_f, 1.0);
    // Fade the lit ground toward the horizon color at the rim.
    let horizon = frame.hemi_ground.xyz * 1.6 + vec3<f32>(0.10, 0.09, 0.07);
    lit = mix(horizon, lit, edge);
    // LINEAR HDR out — the flora OutputPass applies ACES ONCE on scene+bloom.
    return vec4<f32>(lit, 1.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// (3) SHADOW — soft radial contact disc (the grounding shadow).
//
// ponytail: the recon's CHEAPEST grounding shadow with NO new descriptors and no
// flattened-mesh redraw — a dark alpha disc on a small quad lifted to y+epsilon
// under the trunk, drawn alpha-blended (blend:true → depth-write OFF) AFTER the
// ground and BEFORE the tree. Alpha falls off radially from uv centre, so the
// tree reads as contacting the ground at orbit distance.
// // ponytail: a real PCF sun-shadow-map is the upgrade — it needs a depth
// // render target + a depth-sampler descriptor set that bind_pipeline does not
// // expose today, i.e. net-new RHI, which this viewer deliberately avoids.
// ─────────────────────────────────────────────────────────────────────────────

struct ShadowVsOut {
    @builtin(position) clip_pos : vec4<f32>,
    @location(0)       uv       : vec2<f32>,  // 0..1 across the disc quad
}

@vertex
fn vs_shadow(v: StageVsIn) -> ShadowVsOut {
    var out: ShadowVsOut;
    let world_pos = stage_pc.model * vec4<f32>(v.position, 1.0);
    out.clip_pos = frame.view_proj * world_pos;
    out.uv = v.uv.xy;
    return out;
}

@fragment
fn fs_shadow(in: ShadowVsOut) -> @location(0) vec4<f32> {
    // Radial distance from disc centre (uv 0.5,0.5), 0 at centre → 1 at the rim.
    let r = length(in.uv - vec2<f32>(0.5, 0.5)) * 2.0;
    // Soft falloff: opaque-ish core, feathered edge. Squared for a denser centre.
    let a = (1.0 - smoothstep(0.0, 1.0, r));
    let alpha = a * a * 0.55; // peak contact darkness ~0.55
    // Dark, slightly warm contact color (not pure black — reads as soft AO).
    return vec4<f32>(vec3<f32>(0.02, 0.018, 0.014), alpha);
}

// ─────────────────────────────────────────────────────────────────────────────
// (4) ARROW — flora-owned WIND-DIRECTION DEBUG GIZMO (unlit, solid color).
//
// ponytail: the cheapest possible direction indicator — a CPU-built shaft+head
// mesh (4×vec3 layout, like the ground/shadow quads) drawn UNLIT with one solid
// color. The viewer rotates/scales it via `arrow_pc.model` (camera-relative · a
// yaw aligning local +X to windDir, length scaled by strength) and passes the
// strength-mapped color (green→red) in `arrow_pc.color`. Opaque, depth-tested,
// CULL-NONE (a thin box reads from any orbit angle). It lives in the SAME scene
// HDR pass as the tree, so it composites/depth-tests against ground+tree. The
// raw Preetham/ground go through ACES once in the OutputPass, so we emit a LINEAR
// HDR color here too (the push color is already in a pleasant linear range).
// ─────────────────────────────────────────────────────────────────────────────

// 80-byte push: camera-relative model + a solid RGBA color. Under the 128-byte
// shared flora push range. Mirrors `ArrowPush` in staging.rs.
struct ArrowPush {
    model : mat4x4<f32>,
    color : vec4<f32>,
}
var<immediate> arrow_pc: ArrowPush;

struct ArrowVsOut {
    @builtin(position) clip_pos : vec4<f32>,
}

@vertex
fn vs_arrow(v: StageVsIn) -> ArrowVsOut {
    var out: ArrowVsOut;
    let world_pos = arrow_pc.model * vec4<f32>(v.position, 1.0);
    out.clip_pos = frame.view_proj * world_pos;
    return out;
}

@fragment
fn fs_arrow(in: ArrowVsOut) -> @location(0) vec4<f32> {
    // LINEAR HDR out — the flora OutputPass applies ACES ONCE on scene+bloom.
    return arrow_pc.color;
}

// ─────────────────────────────────────────────────────────────────────────────
// (5) IMPOSTOR — far-distance TEXTURED TREE BILLBOARD (the leaf-LOD impostor tier).
//
// Once a tree is tiny on screen, its thousands of alpha-cutout leaf cards are
// pure overdraw for a few pixels. The leaf LOD crossfades the cards OUT and this
// IN: ONE camera-facing quad (built CPU-side — model cols = right·r, up·r) that
// samples a TWO-TILE atlas baked from the real tree (SIDE + TOP), so the disc is
// correct both at horizon-level and straight down. Alpha-blended, so the
// crossfade is just `alpha = coverage * blend`. The atlas (binding 12) is baked
// ONCE at startup (linear HDR, pre-ACES, cleared to alpha 0 → leaf-card coverage
// IS the silhouette). // ponytail: just 2 views (TOP+SIDE), blended by view
// elevation — no full octahedral hemisphere. At impostor distance a tree is a few
// px, so 2 views read identically to many; the only correctness that matters is
// "not bare red sticks from above", which TOP fixes.
// ─────────────────────────────────────────────────────────────────────────────

// 96-byte push: model + (rgb pigment tint, crossfade alpha) + tree radial up
// (xyz; w unused). Mirrors `ImpostorPush` in flora_view.rs. The radial up lets
// the FS reconstruct the view ELEVATION (top-down vs side-on) → tile blend.
struct ImpostorPush {
    model : mat4x4<f32>,
    color : vec4<f32>,   // rgb = canopy pigment tint (linear HDR); w = crossfade alpha
    up    : vec4<f32>,   // xyz = tree radial up (camera-relative world space); w unused
}
var<immediate> impostor_pc: ImpostorPush;

// The baked impostor atlas (LINEAR HDR, alpha = silhouette coverage). Two tiles
// side-by-side: SIDE = u ∈ [0, 0.5), TOP = u ∈ [0.5, 1]. Sampled with the LINEAR
// CLAMP `ibl_sampler` (binding 4) — a flat atlas wants exactly that.
@group(1) @binding(12) var impostor_atlas : texture_2d<f32>;

struct ImpostorVsOut {
    @builtin(position) clip_pos   : vec4<f32>,
    @location(0)       uv         : vec2<f32>,
    @location(1)       view_dir   : vec3<f32>,  // billboard centre → camera (cam-rel)
}

@vertex
fn vs_impostor(v: StageVsIn) -> ImpostorVsOut {
    var out: ImpostorVsOut;
    let world_pos = impostor_pc.model * vec4<f32>(v.position, 1.0);
    out.clip_pos = frame.view_proj * world_pos;
    out.uv = v.uv.xy;
    // Camera-relative: the camera sits at the origin, so the billboard centre's
    // world position (model translation) IS the camera→billboard vector; the view
    // direction (billboard→camera) is its negation.
    out.view_dir = normalize(-impostor_pc.model[3].xyz);
    return out;
}

@fragment
fn fs_impostor(in: ImpostorVsOut) -> @location(0) vec4<f32> {
    let up = normalize(impostor_pc.up.xyz);
    // view_dir = billboard→camera (cam-rel). Looking straight DOWN the tree, the
    // camera is above the canopy so view_dir ≈ +up → dot ≈ 1 (TOP tile). Side-on,
    // view_dir is roughly tangent → dot ≈ 0 (SIDE tile).
    let topness = clamp(dot(in.view_dir, up), 0.0, 1.0);
    let blend = smoothstep(0.35, 0.8, topness);

    // Map the quad uv into each tile's sub-rect. SIDE = left half, TOP = right.
    let side_uv = vec2<f32>(in.uv.x * 0.5,        in.uv.y);
    let top_uv  = vec2<f32>(in.uv.x * 0.5 + 0.5,  in.uv.y);
    let side = textureSampleLevel(impostor_atlas, ibl_sampler, side_uv, 0.0);
    let top  = textureSampleLevel(impostor_atlas, ibl_sampler, top_uv,  0.0);

    let rgb   = mix(side.rgb, top.rgb, blend);
    let cover = mix(side.a,   top.a,   blend);     // baked silhouette coverage
    if (cover < 0.3) { discard; }                  // alpha cutout
    // The baked atlas already carries the real tree colours; just fade by the
    // crossfade alpha (push color.w). LINEAR HDR — the OutputPass ACES once.
    return vec4<f32>(rgb, cover * impostor_pc.color.w);
}
