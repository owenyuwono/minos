// flora.wgsl — minimal lit tree (one branch mesh + instanced leaves).
//
// ponytail: smallest thing that renders one lit tree. NO wind (no bone /
// windWeight path — wind is a later phase). NO leaf-texture bake and NO
// node-canvas dep — leaf alpha is computed in-shader from quad UV. Bark is a
// cheap procedural wood tint; full ridged-FBM bark is deferred (see // ponytail:
// below). Reuses enki's FrameUniforms, reversed-Z, and camera-relative
// f64->f32 scheme — no IBL / HDRI / EffectComposer / post.
//
// This module hosts TWO pipelines, selected by entry point:
//   (1) BRANCH : vs_branch / fs_branch  (4×vec3 vertex layout, mirrors terrain)
//   (2) LEAF   : vs_leaf   / fs_leaf    (4×vec3 layout, CPU-expanded quads)
// ponytail: both pipelines share enki's ONE hard-coded non-pulling 4×vec3
// vertex layout, so neither needs net-new RHI (no storage buffers, no
// instancing, no extra descriptor sets).
//
// ── Shared uniforms (set 0, binding 0) — verbatim from terrain.wgsl ──────────
//   Reversed-Z is handled entirely by frame.view_proj (near→1, far→0) plus the
//   pipeline's GREATER depth compare — no per-vertex depth math here.

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

// ── SUN SHADOWS (set 1) — dryad DirectionalLight + PCFSoft, 1:1. ──────────────
//
// set1 holds everything the shadow lookup needs, kept OUT of set0 so the
// FrameUniforms layout stays byte-identical to enki's:
//   binding 0 : the depth shadow map (D32), texture_depth_2d.
//   binding 1 : sampler_comparison — the COMPARE sampler (reversed-Z:
//               GREATER_OR_EQUAL). A sampler-compare gives hardware 2×2 PCF per
//               tap "for free"; the 3×3 grid below stacks on top of that.
//   binding 2 : ShadowUniforms — the light-space view-proj that the shadow PASS
//               rendered with, plus the texel size + normalBias + enable flag.
struct ShadowUniforms {
    light_view_proj : mat4x4<f32>,  // world → light clip (same matrix the depth pass used)
    params          : vec4<f32>,    // (1/shadowMapSize, normalBias, enabled, 0)
}
@group(1) @binding(0) var shadow_map     : texture_depth_2d;
@group(1) @binding(1) var shadow_sampler : sampler_comparison;
@group(1) @binding(2) var<uniform> shadow : ShadowUniforms;

// ── IBL (set 1, bindings 3-5) — dryad's HDRI lighting, baked on the CPU. ──────
//   binding 3 : the equirect HDRI as a roughness-indexed MIP CHAIN (texture_2d).
//               Sampled dir→uv at roughness→LOD for specular reflections.
//   binding 4 : a LINEAR trilinear sampler (NOT comparison) for the equirect.
//   binding 5 : ShCoeffs — 9 RGB SH irradiance coeffs (cosine-convolved AND
//               ×environmentIntensity=0.6 on the CPU), + max LOD. Diffuse IBL is
//               a single 9-term dot; no irradiance cubemap.
// // ponytail: equirect-2D mips (no cubemap), SH9 (no irradiance map), analytic
// // env-BRDF (no LUT). The faithful-but-lazy IBL the recon specifies.
struct ShCoeffs {
    c      : array<vec4<f32>, 9>,  // .xyz = SH coeff; .w pad
    params : vec4<f32>,            // x = max LOD (roughness=1 → this LOD)
}
@group(1) @binding(3) var ibl_spec    : texture_2d<f32>;
@group(1) @binding(4) var ibl_sampler : sampler;
@group(1) @binding(5) var<uniform> sh : ShCoeffs;

const PI : f32 = 3.14159265359;

// ── IBL evaluation ───────────────────────────────────────────────────────────

// SH9 diffuse irradiance (Ramamoorthi/Sloan polynomial form). The coeffs are
// already cosine-convolved + intensity-scaled on the CPU, so this returns the
// diffuse radiance (irradiance/π) directly — multiply by albedo at the call site.
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

// Equirect dir→uv (Y-up). MUST match the CPU bake's convention
// (flora_ibl::sh9 / mip_chain): u = atan2(z,x)/2π + 0.5, v = acos(y)/π.
fn equirect_uv(d: vec3<f32>) -> vec2<f32> {
    let u = atan2(d.z, d.x) * (0.5 / PI) + 0.5;
    let v = acos(clamp(d.y, -1.0, 1.0)) / PI;
    return vec2<f32>(u, v);
}

// Roughness→LOD specular sample of the equirect mip chain.
fn ibl_specular(refl: vec3<f32>, roughness: f32) -> vec3<f32> {
    let lod = roughness * sh.params.x;
    return textureSampleLevel(ibl_spec, ibl_sampler, equirect_uv(refl), lod).rgb;
}

// Karis/Lazarov "Mobile" analytic env-BRDF fit — replaces the BRDF LUT texture.
// Returns (scale, bias) so specular = prefiltered·(F0·scale + bias).
fn env_brdf_approx(roughness: f32, ndv: f32) -> vec2<f32> {
    let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4<f32>( 1.0,  0.0425,  1.04, -0.04);
    let r = roughness * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * ndv)) * r.x + r.y;
    return vec2<f32>(-1.04, 1.04) * a004 + r.zw;
}

// Cook-Torrance GGX direct + fill + IBL(diffuse SH + specular equirect), matching
// dryad's MeshStandardMaterial (metalness=0 dielectric, F0=0.04). The shadowed sun
// is sun0; sun1 is an unshadowed diffuse fill; AO multiplies ONLY the indirect
// (IBL) terms (dryad applies AO to indirectDiffuse, not the direct sun). The
// caller still applies ACES afterward, exactly as the half-Lambert path did.
fn pbr_shade(n: vec3<f32>, v: vec3<f32>, albedo: vec3<f32>, roughness: f32,
             shadow_f: f32, ao: f32) -> vec3<f32> {
    let F0 = vec3<f32>(0.04);
    let ndv = max(dot(n, v), 1e-4);

    // ── Direct key light (sun0, PCF-shadowed). ──
    let L = frame.sun0_dir.xyz;
    let H = normalize(L + v);
    let ndl = max(dot(n, L), 0.0);
    let ndh = max(dot(n, H), 0.0);
    let vdh = max(dot(v, H), 0.0);
    let a  = roughness * roughness;
    let a2 = a * a;
    let d_denom = (ndh * ndh * (a2 - 1.0) + 1.0);
    let D = a2 / max(PI * d_denom * d_denom, 1e-6);
    // Smith-GGX height-correlated visibility.
    let lv = ndl * sqrt(ndv * ndv * (1.0 - a2) + a2);
    let ll = ndv * sqrt(ndl * ndl * (1.0 - a2) + a2);
    let Vis = 0.5 / max(lv + ll, 1e-5);
    let F = F0 + (vec3<f32>(1.0) - F0) * pow(1.0 - vdh, 5.0);
    let spec = D * Vis * F;
    let kd = (vec3<f32>(1.0) - F);   // metalness 0 → albedo not killed
    let direct = (kd * albedo / PI + spec) * frame.sun0_color.xyz * ndl * shadow_f;

    // ── Fill light (sun1, unshadowed, diffuse only — cheap). ──
    let fill = albedo / PI * frame.sun1_color.xyz * max(dot(n, frame.sun1_dir.xyz), 0.0);

    // ── IBL: diffuse SH9 + specular equirect mip × analytic env-BRDF. ──
    let diff_ibl = sh_irradiance(n) * albedo;           // SH already ×0.6
    let R = reflect(-v, n);
    let prefiltered = ibl_specular(R, roughness);
    let ab = env_brdf_approx(roughness, ndv);
    let spec_ibl = prefiltered * (F0 * ab.x + ab.y);

    // dryad: reflectedLight.indirectDiffuse *= mix(1.0, vAo, 0.85) — AO floored
    // at 15% and applied to the INDIRECT (IBL/SH) term ONLY, never the direct sun.
    let ao_indirect = mix(1.0, ao, 0.85);
    return direct + fill + (diff_ibl + spec_ibl) * ao_indirect;
}

// dryad shadow.bias = -0.0005 (NDC depth units). enki is reversed-Z (GREATER,
// far=0), so a "pull the comparison toward the light" bias flips sign vs forward
// depth: we ADD a small epsilon to the receiver's compare depth so true surfaces
// don't self-shadow. normalBias (dryad 0.02 world units) is applied in the VS by
// offsetting the world position along the surface normal before the light-clip
// transform (done per-fragment-input below via the passed world pos + normal).
// ponytail: depth-compare bias is a single literal (dryad's -0.0005 magnitude),
// not a slope-scaled bias — the shadow map is a tight ortho fit so constant bias
// matches dryad's look without the extra cmd_set_depth_bias plumbing.
const SHADOW_DEPTH_BIAS : f32 = 0.0005;
const SHADOW_NORMAL_BIAS : f32 = 0.02;   // dryad shadow.normalBias (world units)

// PCF 3×3 over the compare sampler. Each textureSampleCompare already does a
// hardware 2×2 bilinear compare, so a 3×3 grid of compares ≈ a 6×6 effective
// kernel — the smallest tap count that reproduces dryad's PCFSoft softness.
// // ponytail: 3×3 (9 taps) is the smallest grid that reads as "soft" not
// // "stair-stepped"; dryad's PCFSoftShadowMap is a ~5-tap poisson-ish kernel of
// // similar radius, so 3×3 over a 2× hardware compare matches its blur closely.
fn sample_shadow(world_pos: vec3<f32>, world_normal: vec3<f32>, ndl: f32) -> f32 {
    if (shadow.params.z < 0.5) {
        return 1.0;                       // shadows disabled → fully lit
    }
    // normalBias: push the receiver point out along its normal before projecting,
    // proportional to the grazing angle (more offset when the sun is glancing).
    let slope = clamp(1.0 - ndl, 0.0, 1.0);
    let biased = world_pos + world_normal * (SHADOW_NORMAL_BIAS * (0.4 + slope));

    let clip = shadow.light_view_proj * vec4<f32>(biased, 1.0);
    if (clip.w <= 0.0) {
        return 1.0;
    }
    let ndc = clip.xyz / clip.w;
    // Outside the light frustum (XY) → unshadowed. Vulkan clip XY is [-1,1].
    if (ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0) {
        return 1.0;
    }
    // Light-space UV: NDC [-1,1] → [0,1]; Vulkan +Y is down so V is not flipped.
    let uv = ndc.xy * 0.5 + vec2<f32>(0.5, 0.5);
    // Reversed-Z: depth INCREASES toward the light (near→1, far→0), and the map
    // keeps the closest (GREATEST) caster depth. The receiver is lit where its
    // depth >= the stored caster depth. To kill self-shadow acne we push the
    // receiver's compare depth TOWARD the light (ADD the bias) so true surfaces
    // sit just in front of their own caster sample. (Forward-Z would subtract;
    // reversed-Z flips the sign — matching dryad's -0.0005 magnitude.)
    let ref_depth = clamp(ndc.z + SHADOW_DEPTH_BIAS, 0.0, 1.0);

    let texel = shadow.params.x;
    var sum = 0.0;
    // 3×3 grid (literal-bounded → naga-safe).
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let ofs = vec2<f32>(f32(dx), f32(dy)) * texel;
            // Reversed-Z compare: lit where receiver depth >= caster depth, i.e.
            // textureSampleCompare with GREATER_OR_EQUAL semantics (sampler op).
            sum = sum + textureSampleCompare(shadow_map, shadow_sampler, uv + ofs, ref_depth);
        }
    }
    return sum / 9.0;
}

// ── Shared lighting helpers ──────────────────────────────────────────────────
//
// NOTE: the old hemisphere_ambient + half_lambert helpers were removed when the
// lighting moved to IBL+PBR (pbr_shade above) — the hemisphere fill is now the
// SH9 diffuse irradiance, and the directional terms are Cook-Torrance GGX.

// ACES filmic tonemap (Narkowicz 2015), matching terrain.wgsl so flora sits in
// the same exposure range as the ground it grows on.
fn aces_filmic(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

// ── HIERARCHICAL SKELETAL WIND (1:1 dryad windSkinGlsl.js) ────────────────────
//
// The wind solver (CPU, port of windSolver.js) composes ONE column-major mat4
// per wind bone — T(pivot)·R(axis,angle)·T(-pivot), accumulated TOP-DOWN parent
// → child — and uploads them into this set1/binding 6 storage buffer each frame
// (per frame-in-flight; read in the VERTEX stage only). Each branch vertex is
// transformed by its bone matrix; leaves bone-FOLLOW their nearest branch bone
// and add a high-freq flutter. `wind = (time, strength, dirX, dirZ)`: the push
// lane is kept only for the `strength` calm-gate + clock — the actual motion is
// entirely in the bone matrices. strength==0 ⇒ the solver emits identities ⇒
// these helpers return the exact rest pose (byte-identical static tree).
//
// Storage element layout = a WGSL mat4x4<f32> (column-major), matching the
// solver's `out[c*16 .. c*16+16]`. naga-valid: `var<storage, read>` of a runtime
// array, indexed by a uniform-flow dynamic index (fixed binding, guarded fetch).
@group(1) @binding(6) var<storage, read> bones: array<mat4x4<f32>>;

// ── LEAF CLUSTER TEXTURES (set 1, bindings 7-9) — CPU-baked per genome from the
//    resolved leaf genes (enki_flora::leaf_texture::bake_leaf_cluster), the dryad
//    makeLeafClusterTexture sprite + buildLeafNormalMap. ──────────────────────
//   binding 7 : leaf COLOR sprite (RGBA8 sRGB), alpha = silhouette coverage.
//   binding 8 : leaf tangent-space NORMAL map (RGBA8 UNORM), alpha = the cutout.
//   binding 9 : LINEAR CLAMP sampler shared by both.
// fs_leaf samples color (alpha-cutout at 0.5) + the normal map (perturb the lit
// normal); fs_leaf_depth alpha-tests the same color alpha so the cast shadow
// silhouette matches the lit cutout exactly (replacing the in-shader Gielis test).
@group(1) @binding(7) var leaf_color   : texture_2d<f32>;
@group(1) @binding(8) var leaf_normal  : texture_2d<f32>;
@group(1) @binding(9) var leaf_sampler : sampler;

// Guarded bone fetch: an unwritten/degenerate slot has m[3][3]==0 (a valid affine
// transform always has m[3][3]==1). Treat it as identity so a vertex/leaf stays
// at rest instead of collapsing to the origin. (windSkinGlsl.js fetchBone:51-66.)
fn fetchBone(idx: f32) -> mat4x4<f32> {
    let m = bones[u32(idx + 0.5)];
    if (m[3][3] == 0.0) {
        return mat4x4<f32>(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0);
    }
    return m;
}

// BRANCH skin (windSkinGlsl.js:84-106). PURE composed skinning: the bone matrix
// already encodes the full chain rotation about its pivot, composed with every
// ancestor, so applying it directly makes each vertex follow its own branch AND
// all ancestors with no stretch. // ponytail: `frac` (boneFraction) is kept in
// the signature for parity but UNUSED — the final dryad form dropped the
// boneFraction/windWeight blend (it fought the hierarchical composition and was
// the documented exploding-spikes bug). strength==0 → exact rest pos.
fn windSkinPosition(restPos: vec3<f32>, boneIdx: f32, frac: f32, strength: f32) -> vec3<f32> {
    if (strength == 0.0) { return restPos; }
    let m = fetchBone(boneIdx);
    return (m * vec4<f32>(restPos, 1.0)).xyz;
}

// LEAF bone-follow delta (windSkinGlsl.js:123-133): the world-space displacement
// of the leaf anchor under its nearest branch bone. At an identity matrix the
// delta is exactly vec3(0). Calm gate first so leaves never collapse if the
// buffer is somehow unset.
fn windBoneFollowDelta(anchor: vec3<f32>, boneIdx: f32, strength: f32) -> vec3<f32> {
    if (strength == 0.0) { return vec3<f32>(0.0); }
    let m = fetchBone(boneIdx);
    return (m * vec4<f32>(anchor, 1.0)).xyz - anchor;
}

// ─────────────────────────────────────────────────────────────────────────────
// (1) BRANCH pipeline
//
// Vertex layout (mirrors terrain's separate-buffer SoA convention; integration
// must bind these 4 buffers in this order):
//   slot 0 : @location(0) positions vec3<f32>   (BranchMesh.positions, 3V)
//   slot 1 : @location(1) normals   vec3<f32>   (BranchMesh.normals,   3V)
//   slot 2 : @location(2) uv        vec3<f32>   (BranchMesh.uvs as .xy; .z pad=0)
//   slot 3 : @location(3) ao        vec3<f32>   (BranchMesh.ao in .x;  .yz pad)
// Index buffer: u32 triangle-list, CCW from outside.
//
// NOTE: enki's non-pulling pipeline hard-codes 4× vec3<f32> bindings. BranchMesh
// stores uv as 2V and ao as 1V; the integration stage must widen these to vec3
// streams (uv.z=0, ao.y=ao.z=0) when packing the buffers, OR build them as
// vec3 directly. Only uv.xy and ao.x are read here.
// ─────────────────────────────────────────────────────────────────────────────

// 128 bytes (== the wgpu/Vulkan guaranteed immediate-data minimum). The bark
// genes are PER-TREE, so they live here (not in a vertex stream). naga 29
// validates `var<immediate>` push constants in tests/shader.rs.
struct BranchPush {
    model       : mat4x4<f32>,  // 64  camera-relative: translation col = origin_f64 - cam_f64, f32
    wind        : vec4<f32>,    // 16  (time, strength, dirX, dirZ) — global-field wind (was the
                                //     never-sampled `wood_tint` lane; repurposed, push stays 128B)
    bark0       : vec4<f32>,    // 16  (barkHue, barkLightness, barkRelief, barkLenticels)
    bark1       : vec4<f32>,    // 16  (barkScale, barkOrient, barkPlates, barkShed)
    bark2       : vec4<f32>,    // 16  (barkUnderHue, woodiness, debugMode, 0)
}

// ── Debug render mode (carried in a spare push lane; keeps push ≤128B). ──
// 0 = lit (full), 1 = unlit (albedo only), 2 = normals (n*0.5+0.5),
// 3 = ao (branch vertex ao / leaf exposure as grey). Wireframe is a separate
// fill:false pipeline, so it never selects an FS arm here (falls through to lit).
// Read as u32(lane + 0.5) — naga-safe (literal compares, every arm returns).

var<immediate> branch_pc: BranchPush;

struct BranchVsIn {
    @location(0) position : vec3<f32>,
    @location(1) normal   : vec3<f32>,
    @location(2) uv       : vec3<f32>,  // .xy = (angle, world arc-length); .z = tube radius
    @location(3) attr     : vec3<f32>,  // .x = baked AO; .y = wind boneIndex; .z = boneFraction
}

struct BranchVsOut {
    @builtin(position) clip_pos     : vec4<f32>,
    @location(0)       world_normal : vec3<f32>,
    @location(1)       uv           : vec3<f32>,  // .xy = (angle, arc-len); .z = tube radius
    @location(2)       ao           : f32,
    @location(3)       obj_pos      : vec3<f32>,  // object-space pos (bark samples this)
    // Tree-LOCAL world position (NOT camera-relative) for the shadow lookup: the
    // light matrix is built in the tree-local frame (tree at world origin in the
    // viewer), so the receiver must project with the same un-relative coords.
    @location(4)       shadow_pos   : vec3<f32>,
    // CAMERA-RELATIVE world position — camera is at the origin in this space, so
    // the view dir is normalize(-view_pos). Needed for the PBR specular/Fresnel.
    @location(5)       view_pos     : vec3<f32>,
}

// Branch SKIN evaluated in TREE-LOCAL space. The bone matrices are tree-local
// (object space, like dryad), so we SKIN the rest position FIRST (about the
// bone's pivot, composed top-down with ancestors), THEN apply the model rotation
// m3 — keeping the result in the SAME frame the shadow depth pass projects from
// (so the caster silhouette and lit receiver stay in lockstep). The trunk base
// bone is pinned (identity) by the mesher, so the base stays put. strength==0 →
// skinned == rest → byte-identical static pose.
fn branch_local_skinned(position: vec3<f32>, boneIdx: f32, frac: f32, m3: mat3x3<f32>, wind: vec4<f32>) -> vec3<f32> {
    let skinned = windSkinPosition(position, boneIdx, frac, wind.y);
    return m3 * skinned;   // rotated into the surface frame, NO translation
}

@vertex
fn vs_branch(v: BranchVsIn) -> BranchVsOut {
    var out: BranchVsOut;
    let m3 = mat3x3<f32>(
        branch_pc.model[0].xyz,
        branch_pc.model[1].xyz,
        branch_pc.model[2].xyz,
    );

    // ── HIERARCHICAL WIND skin in tree-local space (see branch_local_skinned).
    // The final camera-relative world pos = local-skinned + model translation. ──
    let local = branch_local_skinned(v.position, v.attr.y, v.attr.z, m3, branch_pc.wind);
    let world_pos = vec4<f32>(local + branch_pc.model[3].xyz, 1.0);
    out.clip_pos = frame.view_proj * world_pos;

    out.world_normal = m3 * v.normal;
    out.uv = v.uv;            // carry radius through in .z
    out.ao = v.attr.x;
    out.obj_pos = v.position; // rest/object space — bark relief is sampled here
    // Tree-local displaced pos — the SAME value the shadow depth pass projects.
    out.shadow_pos = local;
    // Camera-relative world pos (camera at origin) for the PBR view vector.
    out.view_pos = world_pos.xyz;
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// BARK TOOLKIT — dryad procedural bark (barkMaterial.js) ported to WGSL.
//
// ridged-FBM furrows + voronoi plate network → albedo (flat base + colour
// features) and a height field whose GRADIENT perturbs the world normal. All
// loops are LITERAL-bounded (8 octaves, 3×3 voronoi) → naga-safe. GLSL→WGSL
// gotchas handled: mod(x,y)→x-y*floor(x/y) (hsl2rgb), atan(y,x)→atan2 (lenticel),
// scalar+vector broadcast made explicit, pow bases kept >=0.
// ─────────────────────────────────────────────────────────────────────────────

const BARK_FISSURE_FREQ_SCALE : f32 = 0.42;
const BARK_FISSURE_WEIGHT     : f32 = 0.55;
const BARK_RIDGE_FREQ_SCALE   : f32 = 0.5;
const BARK_PLATE_FREQ         : f32 = 1.4;
const BARK_CRACK_WIDTH        : f32 = 6.0;
const BARK_PLATE_ASPECT       : f32 = 0.30;
const BARK_WARP_Y_DAMP        : f32 = 0.30;
const BARK_ORIENT_DEFAULT     : f32 = 0.70;
const BARK_ANISO_SPREAD       : f32 = 1.20;
const BARK_SHED_FREQ          : f32 = 0.6;
const BARK_SHED_LIFT          : f32 = 0.35;
const BARK_Y_STRETCH          : f32 = 3.0;
const BARK_BUMP_MIN           : f32 = 0.18;
const BARK_BUMP_MAX           : f32 = 0.55;

// GLSL mod(x,y) for the hsl2rgb hue wrap (WGSL % differs for negatives).
fn bark_mod6(x: vec3<f32>) -> vec3<f32> {
    return x - 6.0 * floor(x / 6.0);
}

// Compact branchless HSL→RGB.
fn hsl2rgb(h: f32, s: f32, l: f32) -> vec3<f32> {
    let rgb = clamp(abs(bark_mod6(h * 6.0 + vec3<f32>(0.0, 4.0, 2.0)) - 3.0) - 1.0,
                    vec3<f32>(0.0), vec3<f32>(1.0));
    return vec3<f32>(l) + s * (rgb - 0.5) * (1.0 - abs(2.0 * l - 1.0));
}

// Value-noise hash in [-1,1] for a vec3 seed.
fn barkHash(p_in: vec3<f32>) -> f32 {
    var p = fract(p_in * vec3<f32>(127.1, 311.7, 74.7));
    p = p + vec3<f32>(dot(p, p.yzx + vec3<f32>(19.19)));
    return fract((p.x + p.y) * p.z) * 2.0 - 1.0;
}

// 3D value noise: trilinear interpolation of lattice hashes (smoothstep kernel).
fn barkNoise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);

    let n000 = barkHash(i + vec3<f32>(0.0, 0.0, 0.0));
    let n100 = barkHash(i + vec3<f32>(1.0, 0.0, 0.0));
    let n010 = barkHash(i + vec3<f32>(0.0, 1.0, 0.0));
    let n110 = barkHash(i + vec3<f32>(1.0, 1.0, 0.0));
    let n001 = barkHash(i + vec3<f32>(0.0, 0.0, 1.0));
    let n101 = barkHash(i + vec3<f32>(1.0, 0.0, 1.0));
    let n011 = barkHash(i + vec3<f32>(0.0, 1.0, 1.0));
    let n111 = barkHash(i + vec3<f32>(1.0, 1.0, 1.0));

    return mix(
        mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y),
        mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y),
        u.z);
}

// Single ridge octave: (1-|n|)^2 → flat-top ridge, sharp valley.
fn ridgeOctave(p: vec3<f32>) -> f32 {
    let n = 1.0 - abs(barkNoise(p));
    return n * n;
}

// 2-octave domain-warp vector (cheap).
fn barkWarpFBM(p: vec3<f32>) -> vec3<f32> {
    let a = barkNoise(p);
    let b = barkNoise(p * 2.07 + vec3<f32>(3.17, 1.43, 2.71));
    return vec3<f32>(a + 0.5 * b, b + 0.5 * a, a * b) * 0.5;
}

// Ridged FBM, FIXED 8 octaves (literal bound → naga-safe). Per-octave AA fade
// drops sub-pixel octaves. fw = caller's screen-space footprint of pw.
fn ridgedFBM(pw: vec3<f32>, freq_in: f32, fw: f32) -> f32 {
    var amp   = 1.0;
    var total = 0.0;
    var norm  = 0.0;
    var freq  = freq_in;
    for (var oct = 0; oct < 8; oct = oct + 1) {
        let octFw  = fw * freq;
        let aaFade = 1.0 - smoothstep(0.4, 1.0, octFw);
        total = total + ridgeOctave(pw * freq) * amp * aaFade;
        norm  = norm + amp;
        freq  = freq * 2.13;   // irrational lacunarity
        amp   = amp * 0.52;
    }
    let h = clamp(total / norm, 0.0, 1.0);
    return pow(h, 0.65);       // h>=0 guaranteed by clamp
}

// Orientation gain: 1.0 at barkOrient default (legacy byte-identity coord).
fn barkAnisoGain(orient: f32) -> f32 {
    var span = BARK_ORIENT_DEFAULT;
    if (orient >= BARK_ORIENT_DEFAULT) {
        span = 1.0 - BARK_ORIENT_DEFAULT;
    }
    let u = (orient - BARK_ORIENT_DEFAULT) / max(span, 1e-3);
    return exp2(u * BARK_ANISO_SPREAD);
}

// Tilt a legacy-stretched coord by the gain (XZ shrink as Y grows). gain==1 → input.
fn barkAnisoCoord(p: vec3<f32>, gain: f32) -> vec3<f32> {
    return vec3<f32>(p.x / gain, p.y * gain, p.z / gain);
}

// Exfoliation field in [0,1]. STRICT no-op at barkShed==0.
fn barkShedField(p: vec3<f32>, barkShed: f32) -> f32 {
    if (barkShed < 0.001) {
        return 0.0;
    }
    let patchVal = barkNoise(p * BARK_SHED_FREQ + vec3<f32>(13.7, 4.2, 8.1)) * 0.5 + 0.5;
    let shedT = mix(1.05, 0.02, barkShed);
    return smoothstep(shedT, shedT + 0.20, patchVal);
}

// Combined ridged height field (coarse fissures + fine grain). 0 = furrow, 1 = ridge.
fn barkHeightField(p_grain: vec3<f32>, featureScale: f32, fw: f32, barkScale: f32) -> f32 {
    var warpOfs = barkWarpFBM(p_grain * 0.45) * 0.55;
    warpOfs.y = warpOfs.y * BARK_WARP_Y_DAMP;
    let pw = p_grain + warpOfs;

    let baseFreq = (BARK_RIDGE_FREQ_SCALE * mix(0.6, 1.8, barkScale)) / max(featureScale, 0.01);
    let coarseH = ridgedFBM(pw, baseFreq * BARK_FISSURE_FREQ_SCALE, fw);
    let fineH   = ridgedFBM(pw, baseFreq, fw);
    var combined = mix(fineH, coarseH, BARK_FISSURE_WEIGHT);
    combined = combined * (0.4 + 0.6 * coarseH);
    return clamp(combined, 0.0, 1.0);
}

// Voronoi/cellular nearest-edge: 0 on a crack, 1 inside a plate. 3×3 fixed loop.
fn barkVoronoiEdge(q: vec2<f32>) -> f32 {
    let qi = floor(q);
    let qf = fract(q);

    var minDist1 = 8.0;
    var minDist2 = 8.0;

    for (var jy = -1; jy <= 1; jy = jy + 1) {
        for (var jx = -1; jx <= 1; jx = jx + 1) {
            let neighbor = vec2<f32>(f32(jx), f32(jy));
            let cellHash2 = vec2<f32>(
                barkHash(vec3<f32>(qi + neighbor, 0.0)),
                barkHash(vec3<f32>(qi + neighbor, 1.0))
            ) * 0.45 + 0.5;
            let diff = neighbor + cellHash2 - qf;
            let d = dot(diff, diff);
            if (d < minDist1) {
                minDist2 = minDist1;
                minDist1 = d;
            } else if (d < minDist2) {
                minDist2 = d;
            }
        }
    }
    let edge = sqrt(minDist2) - sqrt(minDist1);
    let crackW = 1.0 / BARK_CRACK_WIDTH;
    let aa = max(fwidth(edge), 1e-5);
    return smoothstep(crackW - aa, crackW + aa, edge);
}

// Plate UV for a grain coord (crack orientation; aspect → vertical furrows).
fn barkPlateUV(p_grain: vec3<f32>, featureScale: f32, barkScale: f32) -> vec2<f32> {
    let plateFreq = (BARK_PLATE_FREQ * mix(0.6, 1.8, barkScale)) / max(featureScale, 0.04);
    return vec2<f32>(p_grain.x, (p_grain.y / BARK_Y_STRETCH) * BARK_PLATE_ASPECT) * plateFreq;
}

// Horizontal lenticel dashes (birch). 1 inside a dash. No-op at prominence 0.
fn barkLenticel(p_grain: vec3<f32>, prominence: f32) -> f32 {
    if (prominence < 0.001) {
        return 0.0;
    }
    let bandY     = p_grain.y / 3.0;
    let bandIndex = floor(bandY * 5.0);
    let bandFrac  = fract(bandY * 5.0);
    let angle     = atan2(p_grain.z, p_grain.x);
    let bandHash  = fract(sin(bandIndex * 127.1 + 311.7) * 43758.5);
    let sectors   = 5.0;
    let sectorAngle = 6.28318 / sectors;
    let sectorFrac  = fract((angle / sectorAngle) + bandHash);

    let dashWidth  = 0.35;
    let dashHeight = 0.18;
    let dashCenterH = smoothstep(0.5 - dashWidth, 0.5 - dashWidth * 0.3, sectorFrac)
                    * (1.0 - smoothstep(0.5 + dashWidth * 0.3, 0.5 + dashWidth, sectorFrac));
    let dashCenterV = smoothstep(0.5 - dashHeight, 0.5 - dashHeight * 0.3, bandFrac)
                    * (1.0 - smoothstep(0.5 + dashHeight * 0.3, 0.5 + dashHeight, bandFrac));
    let dashMask = dashCenterH * dashCenterV;

    let presenceHash = fract(sin(bandIndex * 73.13 + floor(sectorFrac * sectors + bandHash) * 57.3) * 91027.3);
    let present = step(0.45, presenceHash);
    return dashMask * present * prominence;
}

// Bark albedo — FLAT base + colour-only features. The furrow/crack PATTERN lives
// in the normal, not here. barkHue/barkLightness drive the palette; lenticels,
// lichen, blotch, micro-variation are independent colour features.
fn barkAlbedo(p_grain: vec3<f32>, h: f32, worldY: f32,
              barkHue: f32, barkLightness: f32, barkLenticels: f32) -> vec3<f32> {
    let bL = mix(0.10, 0.92, barkLightness);
    let bS = barkHue * 0.55;
    let bH = mix(0.09, 0.02, barkHue);

    let paletteRidge  = hsl2rgb(bH, bS,        clamp(bL + 0.07, 0.0, 1.0));
    let paletteFurrow = hsl2rgb(bH, bS * 1.15, clamp(bL - 0.20, 0.0, 1.0));
    let paletteLichen = hsl2rgb(0.26, 0.28,    clamp(0.32 + bL * 0.18, 0.0, 1.0));
    let paletteBlotchWarm = hsl2rgb(clamp(bH - 0.015, 0.0, 1.0), bS,       clamp(bL - 0.05, 0.0, 1.0));
    let paletteBlotchCool = hsl2rgb(clamp(bH + 0.05, 0.0, 1.0),  bS * 0.7, clamp(bL - 0.07, 0.0, 1.0));

    let baseSat = bS * 0.50;
    let baseHue = mix(bH, 0.07, 0.40);
    var baseCol = hsl2rgb(baseHue, baseSat, bL);

    // Lichen: deep crevices, biased to the trunk base.
    let lichenCrevice = clamp((0.22 - h) * 6.0, 0.0, 1.0);
    let lichenBase    = clamp(1.0 - worldY * 0.55, 0.0, 1.0);
    let lichenSpatter = clamp(barkNoise(p_grain * 3.7 + vec3<f32>(7.3, 2.9, 5.1)) * 0.5 + 0.5, 0.0, 1.0);
    baseCol = mix(baseCol, paletteLichen, lichenCrevice * lichenBase * lichenSpatter * 0.70);

    // Low-freq warm/cool blotch.
    let blotch = barkNoise(p_grain * 0.18 + vec3<f32>(5.5, 1.3, 3.7)) * 0.5 + 0.5;
    baseCol = mix(baseCol, mix(paletteBlotchCool, paletteBlotchWarm, blotch), 0.22);

    // Lenticels (independent axis).
    let lenticelMask = barkLenticel(p_grain, clamp(barkLenticels * 1.2, 0.0, 1.0));
    let lenticelCol  = mix(paletteRidge, paletteFurrow * 0.55, 0.75);
    baseCol = mix(baseCol, lenticelCol, lenticelMask);

    // Micro variation.
    baseCol = clamp(baseCol + vec3<f32>(barkNoise(p_grain * 8.3 + vec3<f32>(1.1, 4.4, 2.2)) * 0.04),
                    vec3<f32>(0.0), vec3<f32>(1.0));
    return baseCol;
}

// Combined RELIEF height the normal follows: ridge field MINUS voronoi cracks
// (depth = barkRelief × barkPlates) PLUS shed curl-lip.
fn barkReliefHeight(p_grain: vec3<f32>, featureScale: f32, fw: f32,
                    barkScale: f32, barkRelief: f32, barkPlates: f32, barkShed: f32) -> f32 {
    var h = barkHeightField(p_grain, featureScale, fw, barkScale);
    let vEdge = barkVoronoiEdge(barkPlateUV(p_grain, featureScale, barkScale));
    h = h - barkRelief * barkPlates * (1.0 - vEdge);
    let shed = barkShedField(p_grain, barkShed);
    let shedEdge = shed * (1.0 - shed) * 4.0;
    h = h + barkRelief * BARK_SHED_LIFT * shedEdge;
    return h;
}

// Perturb the world normal by the relief gradient (finite differences).
fn barkPerturbNormal(p_grain: vec3<f32>, worldNormal: vec3<f32>, featureScale: f32, fw: f32,
                     barkScale: f32, barkRelief: f32, barkPlates: f32, barkShed: f32) -> vec3<f32> {
    let eps = clamp(featureScale * 0.04, 0.003, 0.04);
    let h0 = barkReliefHeight(p_grain,                              featureScale, fw, barkScale, barkRelief, barkPlates, barkShed);
    let hx = barkReliefHeight(p_grain + vec3<f32>(eps, 0.0, 0.0),   featureScale, fw, barkScale, barkRelief, barkPlates, barkShed);
    let hy = barkReliefHeight(p_grain + vec3<f32>(0.0, eps, 0.0),   featureScale, fw, barkScale, barkRelief, barkPlates, barkShed);
    let hz = barkReliefHeight(p_grain + vec3<f32>(0.0, 0.0, eps),   featureScale, fw, barkScale, barkRelief, barkPlates, barkShed);

    let grad = vec3<f32>((hx - h0) / eps, (hy - h0) / (eps * BARK_Y_STRETCH), (hz - h0) / eps);

    var bumpStr = clamp(featureScale * 1.4, BARK_BUMP_MIN, BARK_BUMP_MAX);
    bumpStr = bumpStr * (1.0 - smoothstep(0.7, 1.6, fw));
    return normalize(worldNormal - grad * bumpStr);
}

@fragment
fn fs_branch(in: BranchVsOut) -> @location(0) vec4<f32> {
    // ── Bark genes (per-tree push constant). ──
    let barkHue       = branch_pc.bark0.x;
    let barkLightness = branch_pc.bark0.y;
    let barkRelief    = branch_pc.bark0.z;
    let barkLenticels = branch_pc.bark0.w;
    let barkScale     = branch_pc.bark1.x;
    let barkOrient    = branch_pc.bark1.y;
    let barkPlates    = branch_pc.bark1.z;
    let barkShed      = branch_pc.bark1.w;
    let barkUnderHue  = branch_pc.bark2.x;
    let woodiness     = branch_pc.bark2.y;

    // ── Bark coordinate stack (object space, Y-up). ──
    // featureScale = TRUE local tube radius (packed into uv.z by the mesher), so
    // furrows size to actual thickness. orient tilts the grain (gain==1 at the
    // 0.70 default → legacy vec3(x, 3y, z) coord byte-for-byte).
    let featureScale = clamp(in.uv.z, 0.14, 0.55);
    let orientGain   = barkAnisoGain(barkOrient);
    let pGrain       = barkAnisoCoord(vec3<f32>(in.obj_pos.x, in.obj_pos.y * BARK_Y_STRETCH, in.obj_pos.z), orientGain);
    let barkFw       = max(max(fwidth(pGrain.x), fwidth(pGrain.y)), fwidth(pGrain.z));

    // Height field (for albedo lichen pooling) flattened toward smooth as relief→0.
    var barkH = barkHeightField(pGrain, featureScale, barkFw, barkScale);
    barkH = mix(0.55, barkH, mix(0.25, 1.0, barkRelief));

    // ── Albedo: procedural bark, blended toward a herbaceous green by woodiness.
    var albedo = barkAlbedo(pGrain, barkH, in.obj_pos.y, barkHue, barkLightness, barkLenticels);
    let herbAlbedo = mix(vec3<f32>(0.16, 0.34, 0.12), vec3<f32>(0.30, 0.52, 0.20), barkH);
    albedo = mix(herbAlbedo, albedo, woodiness);

    // Shed under-bark overlay (recolour by barkUnderHue), gated by woodiness.
    let shedAmt = barkShedField(pGrain, barkShed);
    let ubL = clamp(mix(0.10, 0.92, barkLightness) + 0.12, 0.0, 1.0);
    let ubS = barkUnderHue * 0.55;
    let ubH = mix(0.09, 0.02, barkUnderHue);
    let underCol = hsl2rgb(ubH, ubS, ubL);
    albedo = mix(albedo, underCol, shedAmt * woodiness);
    albedo = clamp(albedo, vec3<f32>(0.0), vec3<f32>(1.0));

    // ── Normal: perturb the world normal by the relief gradient. featureScale is
    // scaled by woodiness so herbaceous stems stay smooth.
    let n = barkPerturbNormal(pGrain, normalize(in.world_normal),
                              featureScale * woodiness, barkFw,
                              barkScale, barkRelief, barkPlates, barkShed);

    // ── Debug render modes (spare push lane). All arms return → naga-safe. ──
    let mode = u32(branch_pc.bark2.z + 0.5);
    if (mode == 1u) { return vec4<f32>(albedo, 1.0); }              // unlit
    if (mode == 2u) { return vec4<f32>(n * 0.5 + 0.5, 1.0); }       // normals (perturbed)
    if (mode == 3u) { let a = clamp(in.ao, 0.0, 1.0); return vec4<f32>(a, a, a, 1.0); } // ao

    // ── Cook-Torrance PBR (dryad bark MeshStandardMaterial, metalness 0). ──
    // Per-pixel roughness from dryad's FRAG_ROUGHNESS_REPLACEMENT: furrows
    // mix(0.86,0.96,relief), ridge tops mix(0.74,0.86,relief), blended by barkH;
    // herbaceous (woodiness→0) → 0.78. + Toksvig specular-AA compensation.
    let rough_furrow = mix(0.86, 0.96, barkRelief);
    let rough_ridge  = mix(0.74, 0.86, barkRelief);
    var roughness = mix(rough_furrow, rough_ridge, clamp(barkH, 0.0, 1.0));
    roughness = mix(0.78, roughness, woodiness);
    // Toksvig: widen roughness where the perturbed normal varies fast (anti-alias).
    roughness = roughness + clamp(length(fwidth(n)) * 4.0, 0.0, 0.3);
    roughness = clamp(roughness, 0.04, 1.0);

    let ao = clamp(in.ao, 0.0, 1.0);
    let v  = normalize(-in.view_pos);
    let ndl0   = dot(n, frame.sun0_dir.xyz);
    let shadow_f = sample_shadow(in.shadow_pos, n, ndl0);

    let lit = pbr_shade(n, v, albedo, roughness, shadow_f, ao);
    // LINEAR HDR out — the flora OutputPass applies ACES ONCE on scene+bloom.
    return vec4<f32>(lit, 1.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// (2) LEAF pipeline — CPU-expanded oriented quads (NON-pulling, 4×vec3 layout).
//
// ponytail: leaves reuse enki's EXISTING non-pulling 4×vec3 vertex pipeline
// (the same hard-coded layout terrain/branches use) instead of the
// storage-buffer + vertex-pulling + instanced-draw path. That path would need
// net-new RHI (an instanced draw call, a second set-0 binding, and descriptor
// wiring that `bind_pipeline` does not expose) — too much surface for "render
// ONE lit tree". Instead the integration stage expands every leaf card into 4
// world-space quad corners on the CPU (one indexed mesh, ONE draw) using the
// FoliageSoA basis (normal/tangent/roll/scale). Swap for true GPU instancing
// when leaf counts make CPU expansion costly.
//
// Vertex layout (4 separate vec3<f32> streams, bound in this order):
//   slot 0 : @location(0) position vec3<f32>  (tree-local quad-corner position)
//   slot 1 : @location(1) normal   vec3<f32>  (leaf face normal, unit)
//   slot 2 : @location(2) uv       vec3<f32>  (.xy = corner UV in [0,1]²; .z pad)
//   slot 3 : @location(3) attr     vec3<f32>  (.x = age [0,1], .y = exposure;
//                                              .z pad)
// Index buffer: u32 triangle-list, 6 indices per leaf ([0,1,2, 2,1,3]).
// ─────────────────────────────────────────────────────────────────────────────

struct LeafPush {
    // model translation = (tree_origin_f64 - camera_f64) cast f32. Quad-corner
    // positions are already relative to the tree origin, so the final
    // camera-relative position is model · vec4(corner, 1).
    model        : mat4x4<f32>,  // 64
    pigment      : vec4<f32>,    // 16  xyz = per-tree base leaf color; w unused
    leaf_params  : vec4<f32>,    // 16  (leafTip, leafWidth, leafSerration, leafLobing)
    leaf_params2 : vec4<f32>,    // 16  (leafSkew, leafLength, debugMode, _)
    wind         : vec4<f32>,    // 16  (time, strength, dirX, dirZ) — 112→128B, exact fit
}

var<immediate> leaf_pc: LeafPush;

struct LeafVsIn {
    @location(0) position : vec3<f32>,  // tree-local world-space quad corner
    @location(1) normal   : vec3<f32>,  // CANOPY sphere-normal (80% canopy/20% card)
    @location(2) uv        : vec3<f32>, // .xy = corner UV [0,1]²; .z = wind boneIndex (nearest branch bone)
    @location(3) attr      : vec3<f32>, // .x = age, .y = exposure, .z = variation seed
}

struct LeafVsOut {
    @builtin(position) clip_pos : vec4<f32>,
    @location(0)       uv       : vec2<f32>,   // [0,1]² across the quad
    @location(1)       normal   : vec3<f32>,   // canopy sphere-normal (object space)
    @location(2)       age      : f32,
    @location(3)       exposure : f32,
    @location(4)       seed     : f32,         // per-leaf variation seed [0,1)
    @location(5)       shadow_pos : vec3<f32>, // tree-local world pos (shadow lookup)
    @location(6)       view_pos : vec3<f32>,   // camera-relative world pos (PBR view dir)
}

// Leaf wind displacement in TREE-LOCAL space: bone-FOLLOW the nearest branch
// bone (so the leaf rides its twig's hierarchical sway) + a high-freq per-leaf
// flutter. Same frame as the shadow depth pass → matched silhouette. The follow
// delta is computed on the (rotated) tree-local anchor `local`, then added to it.
// `seed` is attr.z (per-leaf flutter-phase decorrelation); `boneIdx` is uv.z.
// strength==0 → follow delta is vec3(0) AND flutter amplitude is 0 → exact rest.
fn leaf_local_displaced(position: vec3<f32>, seed: f32, boneIdx: f32, m3: mat3x3<f32>, wind: vec4<f32>) -> vec3<f32> {
    let t   = wind.x;
    let s   = wind.y;
    // Bone-follow in BONE space (the rest anchor), then rotate the delta into the
    // surface frame — the bone matrices are tree-local/un-rotated like the branch
    // skin, so the follow must be evaluated on `position`, not the rotated anchor.
    let follow_obj = windBoneFollowDelta(position, boneIdx, s);
    let local  = m3 * position;                // rotated, NO translation
    let follow = m3 * follow_obj;
    let ph    = seed * 6.2831853;
    let fl    = s * 0.06;
    let flutter = vec3<f32>(
        sin(t * 8.0  + ph)        * fl,
        sin(t * 11.3 + ph * 1.7)  * fl * 0.5,
        cos(t * 9.1  + ph * 0.6)  * fl);
    return local + follow + flutter;
}

@vertex
fn vs_leaf(v: LeafVsIn) -> LeafVsOut {
    var out: LeafVsOut;
    let m3 = mat3x3<f32>(
        leaf_pc.model[0].xyz,
        leaf_pc.model[1].xyz,
        leaf_pc.model[2].xyz,
    );

    // ── WIND in tree-local space (see leaf_local_displaced). Final
    // camera-relative world pos adds back the model translation. ──
    let local = leaf_local_displaced(v.position, v.attr.z, v.uv.z, m3, leaf_pc.wind);
    let world_pos = vec4<f32>(local + leaf_pc.model[3].xyz, 1.0);
    out.clip_pos = frame.view_proj * world_pos;

    out.normal = m3 * v.normal;
    out.uv = v.uv.xy;
    out.age = v.attr.x;
    out.exposure = v.attr.y;
    out.seed = v.attr.z;
    out.shadow_pos = local;     // SAME value the shadow depth pass projects
    out.view_pos = world_pos.xyz; // camera-relative (PBR view dir = normalize(-view_pos))
    return out;
}

// ── In-shader leaf SILHOUETTE: true Gielis superformula (dryad leafTexture.js).
//
// Evaluates r(theta) radially around the leaf and tests the uv point against it,
// then folds in the gene→param mapping (lobing→m + sinus depth, tip→n1, width→
// xScale, length→axial stretch, skew→gamma) and outward serration teeth. Soft
// anti-aliased edge via smoothstep on the signed distance (no hard discard at
// the boundary). naga-valid: bounded math, |.| before pow, clamped bases.
// ─────────────────────────────────────────────────────────────────────────────

// Gielis radius r(theta) for symmetric params (a=b=1, n2=n3). Clamped to finite.
fn superformula_r(theta: f32, m: f32, n1: f32, n2: f32) -> f32 {
    let t = m * theta * 0.25;
    // |cos|^n2 + |sin|^n2 ; bases are non-negative (abs) so pow is naga-safe.
    let c = pow(abs(cos(t)), n2);
    let s = pow(abs(sin(t)), n2);
    let inner = c + s;
    // r = inner^(-1/n1). Guard inner==0 and a near-zero n1.
    let n1s = max(n1, 1e-3);
    return pow(max(inner, 1e-4), -1.0 / n1s);
}

// Returns soft coverage in [0,1] for a uv inside the leaf card.
fn leaf_alpha(uv: vec2<f32>, leafTip: f32, leafWidth: f32,
              leafSerration: f32, leafLobing: f32, leafSkew: f32) -> f32 {
    // Signed cross coord u (midrib at 0), axial v (0 base .. 1 tip).
    let u = 2.0 * uv.x - 1.0;
    var v = uv.y;

    // Skew: reposition the widest point. skewGamma = log(clamp)/log(0.5).
    let skewClamped = clamp(leafSkew, 0.08, 0.92);
    let skewGamma = log(skewClamped) / log(0.5);
    v = pow(clamp(v, 0.0, 1.0), max(skewGamma, 1e-3));

    // Gene → superformula params (dryad leafBaseParams).
    let m  = 2.0 + leafLobing * 3.0;                 // [2,5] lobe count
    let n2 = 4.0 + leafLobing * 10.0;                // [4,14] sinus depth
    let n1Base = 0.4 + leafTip * 2.1;                // [0.4,2.5] apex roundness
    let n1 = n1Base * (1.0 - leafLobing * 0.35);     // lobing sharpens lobe tips
    let xScale = 0.4 + leafWidth * 1.2;              // [0.4,1.6] half-width

    // Map the card uv to a polar point about the leaf center (v=0.5 axis). The
    // superformula is normalized so its max radius ~1; we shape an ovate blade
    // by sampling r(theta) for the angle of (u, centered-v) and comparing.
    let cy = v - 0.5;                                 // -0.5 base .. +0.5 tip
    let px = u / max(xScale, 1e-3);                   // width-normalized x
    let py = cy * 2.0;                                // stretch axial to [-1,1]
    let rad = sqrt(px * px + py * py) + 1e-5;
    let theta = atan2(py, px);
    let rmax = superformula_r(theta, m, n1, n2);
    // Normalize rmax (its peak is ~1 for these params); guard the divide.
    var w = rmax / (1.0 + 0.0001);

    // Base axial taper so the tip and base draw to a point (ovate envelope on
    // top of the radial test): pull the boundary in near v=0 and v=1.
    let tipPow = mix(0.95, 0.45, clamp(leafTip, 0.0, 1.0));
    let taper = pow(clamp(v, 0.0, 1.0), 0.6) * pow(clamp(1.0 - v, 0.0, 1.0), tipPow);
    // Blend the superformula margin with the taper envelope.
    w = w * (0.55 + 0.9 * taper);

    // Serration: outward teeth near the margin, faded at midrib & apex.
    let freq = 8.0 + leafSerration * 24.0;
    let teeth = max(0.0, sin(freq * 3.14159265 * v));
    let edgeFade = smoothstep(0.0, max(w, 1e-3), abs(px));
    w = w + min(0.30, leafSerration * 3.5) * w * teeth * edgeFade;

    // Soft anti-aliased cutout on the signed distance d = w - rad.
    let d = w - rad;
    let e = max(fwidth(rad), 1e-4);
    return clamp(d / e, 0.0, 1.0);
}

// ── Cheap procedural veins: a bright midrib + a few sweeping laterals, returned
// as an intensity used to subtly lighten the blade (mirrors dryad's normal-map
// ridges). ponytail: no atlas, no normal map — a small albedo perturbation.
fn leaf_veins(uv: vec2<f32>, pairs: f32) -> f32 {
    let u = 2.0 * uv.x - 1.0;
    let v = uv.y;
    // Midrib: bright ridge at u=0, fading toward the tip.
    let midrib = (1.0 - smoothstep(0.0, 0.045, abs(u))) * (1.0 - v * 0.3);
    // Laterals: repeating veins that sweep outward + up from the midrib.
    let n  = max(pairs, 1.0);
    let v0 = floor(v * n) / n;
    let slope = 1.6;
    let expected = slope * max(0.0, v - v0);
    let lat = (1.0 - smoothstep(0.0, 0.05, abs(abs(u) - expected)))
              * smoothstep(0.0, 0.15, v - v0)
              * (1.0 - v);
    return clamp(midrib + lat * 0.6, 0.0, 1.0);
}

// ponytail: leaf radiance trim — keeps the sunlit canopy under the bloom gate so
// individual leaf silhouettes survive instead of bleeding into one bloom blob.
// 0.55→0.48: with TREE_DEFAULT's full sphere-normal crown the sun-facing tops
// still bloomed into pale glassy blobs (no opaque-foliage read); the extra trim
// pulls the sunlit crown under the gate so leaves read as solid green foliage.
const LEAF_EXPOSURE : f32 = 0.48;

@fragment
fn fs_leaf(in: LeafVsOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    // ── Silhouette + albedo from the CPU-baked leaf CLUSTER sprite (binding 7).
    // dryad's leaf InstancedMesh samples a generated cluster texture as an
    // alpha-cutout sprite. We do the same: sample the color sprite and HARD-cutout
    // on its alpha at 0.5 — the SAME threshold the depth/shadow caster now uses
    // (fs_leaf_depth samples leaf_color.a too), so overlapping leaves OCCLUDE each
    // other (opaque) and the crown reads as dense foliage. The procedural in-shader
    // Gielis silhouette + vein math is GONE — the painted veins + base→tip gradient
    // already live in the sprite. ──
    let tex = textureSample(leaf_color, leaf_sampler, in.uv);
    if tex.a < 0.5 {
        discard;
    }

    // Albedo IS the sampled sprite color. The per-tree pigment is already baked
    // into it; we keep the per-leaf AGE ramp + jitter as MULTIPLIERS so canopy
    // tonal variation survives (dryad applies instanceColor the same way).
    let age = clamp(in.age, 0.0, 1.0);
    var albedo = tex.rgb;
    // Age ramp (young lighter/warmer → mature deeper), as a multiplicative tint.
    let age_tint = mix(vec3<f32>(1.12, 1.16, 1.02), vec3<f32>(0.85), age);
    albedo = clamp(albedo * age_tint, vec3<f32>(0.0), vec3<f32>(1.0));

    // Per-leaf jitter from the baked seed (dryad's LCG-jitter analogue).
    let seed = in.seed;
    // ponytail: WIDEN the per-leaf lightness jitter (±0.20 → ±0.32). The canopy
    // uses a soft sphere-normal, so a whole cluster of leaves faces the sun and
    // gets near-IDENTICAL lighting — overlapping sunlit leaves then have no tonal
    // step to signal one occluding the other, so the opaque crown READS as glassy
    // see-through. Stronger per-leaf tonal variation gives adjacent opaque cards a
    // visible brightness step, so the overlaps read as solid occlusion (dense
    // foliage) — and being symmetric about 0 it's mean-preserving: it does NOT
    // brighten or darken the canopy overall (no exposure/bloom regression).
    let jL = (seed - 0.5) * 0.64;                 // ±0.32 lightness
    let jR = (fract(seed * 17.0 + 0.13) - 0.5) * 0.16;  // ±0.08 warm/cool
    albedo = albedo * (1.0 + jL);
    albedo.x = clamp(albedo.x + jR, 0.0, 1.0);
    albedo.z = clamp(albedo.z - jR * 0.5, 0.0, 1.0);

    // (Veins are PAINTED into the sprite + carried in the normal map — no
    //  procedural vein math here anymore.)

    // ── Canopy volume normal: the slot1 normal is the soft sphere-normal. ──
    // Double-sided: flip it on back faces so both sides shade outward.
    var n = normalize(in.normal);
    if !front {
        n = -n;
    }

    // ── Leaf NORMAL MAP (binding 8): perturb the canopy normal by the baked
    //    tangent-space relief (midrib/veins/blade gradient catch light). dryad
    //    applies the normal map FIRST, then blends 80% toward the canopy sphere
    //    normal — flora's slot1 normal is already pre-blended 80/20, so we perturb
    //    that. Tangent frame is reconstructed from screen-space derivatives of the
    //    UV + view position (no extra vertex stream needed — the 4×vec3 leaf layout
    //    stays intact). LEAF_NORMAL_SCALE=2.5 matches dryad's normalScale. ──
    let nt_raw = textureSample(leaf_normal, leaf_sampler, in.uv).xyz * 2.0 - 1.0;
    let tn = normalize(vec3<f32>(nt_raw.xy * 2.5, max(nt_raw.z, 0.05)));
    // Cotangent frame from derivatives (Mikkelsen). dpdx/dpdy are uniform-control-
    // flow safe here (computed before any further branching).
    let dp1 = dpdx(in.view_pos);
    let dp2 = dpdy(in.view_pos);
    let duv1 = dpdx(in.uv);
    let duv2 = dpdy(in.uv);
    let denom = duv1.x * duv2.y - duv1.y * duv2.x;
    if abs(denom) > 1e-8 {
        let r = 1.0 / denom;
        let tangent   = normalize((dp1 * duv2.y - dp2 * duv1.y) * r);
        let bitangent = normalize((dp2 * duv1.x - dp1 * duv2.x) * r);
        // Re-orthonormalize against n (Gram-Schmidt) so the perturbation rides the
        // canopy normal, then map the tangent-space normal into world space.
        let t = normalize(tangent - n * dot(n, tangent));
        let b = cross(n, t);
        let perturbed = normalize(t * tn.x + b * tn.y + n * tn.z);
        // Blend toward the canopy sphere normal (dryad's 80% canopy weight) so the
        // relief reads as surface detail, not a per-card facet break.
        n = normalize(mix(perturbed, n, 0.8));
    }

    // ── Debug render modes (spare push lane). Silhouette cutout already applied
    //    above (discard), so unlit/normals/ao keep the leaf shape. naga-safe. ──
    let mode = u32(leaf_pc.leaf_params2.z + 0.5);
    if (mode == 1u) { return vec4<f32>(albedo, 1.0); }             // unlit
    if (mode == 2u) { return vec4<f32>(n * 0.5 + 0.5, 1.0); }      // normals (canopy/face)
    if (mode == 3u) { let e = clamp(in.exposure, 0.0, 1.0); return vec4<f32>(e, e, e, 1.0); } // ao=exposure

    // ── Cook-Torrance PBR (dryad leaf MeshStandardMaterial, metalness 0,
    //    roughness 0.88 matte). The leaf normal map / exposure-AO are folded in
    //    as before; AO≈exposure attenuates the indirect IBL term only. ──
    let roughness = 0.88;
    // dryad attenuates ONLY indirect by exposure/AO, never the direct sun. Fold
    // BOTH the canopy-depth AO (exposure*0.5+0.5) AND the former whole-term
    // interior-shadow factor (mix(0.45,1.0,exposure), was `shadowMul` below) into
    // the single `ao` arg → pbr_shade restricts it to the indirect IBL term.
    let exp_leaf  = clamp(in.exposure, 0.0, 1.0);
    let ao_leaf   = (exp_leaf * 0.5 + 0.5) * mix(0.45, 1.0, exp_leaf);
    let v = normalize(-in.view_pos);
    // SUN0 (key) is the shadow-caster: PCF-gate it. The leaf casts cutout shadows
    // (alpha-tested depth pass), so leaves shadow each other + the ground.
    let ndl0 = dot(n, frame.sun0_dir.xyz);
    let shadow_f = sample_shadow(in.shadow_pos, n, ndl0);
    let pbr = pbr_shade(n, v, albedo, roughness, shadow_f, ao_leaf);

    // ── Backlit / transmission (dryad two-lobe): wrap-diffuse + forward scatter.
    // ponytail: the OLD weights stacked the bright key sun (intensity ~3.0) onto
    // EVERY leaf via the 0.4-floored backWrap, pushing the whole canopy's linear
    // HDR luma above the bloom threshold (1.1) → a solid bloom flame with no leaf
    // silhouettes. Cut the always-on wrap base hard (0.5→0.12) and only let the
    // true forward-scatter (back-facing, sun behind the leaf) glow, so backlight
    // reads as edge translucency on rim leaves — not a uniform canopy bloom.
    let L = frame.sun0_dir.xyz;
    let backWrap = clamp(dot(-n, L) * 0.6 + 0.4, 0.0, 1.0);
    // Forward-scatter sharp lobe, gated to back faces to avoid blown-out green.
    var sc = 0.0;
    if !front {
        sc = pow(max(-dot(n, L), 0.0), 3.0) * 0.35;
    }
    let transAlbedo = clamp(albedo * vec3<f32>(1.15, 1.25, 1.05), vec3<f32>(0.0), vec3<f32>(1.0));
    // ponytail: trim the always-on wrap base (0.12→0.06) and the overall transmission
    // gain (0.35→0.22). The wrap lobe lit EVERY front-facing sunlit leaf with the
    // intensity-3.0 key sun, so the whole crown read as waxy/translucent; cut to
    // near-zero so transmission glows only on true back-lit rim leaves (the `sc`
    // forward-scatter), letting the canopy read as opaque green foliage.
    let backlitGlow = transAlbedo * frame.sun0_color.xyz * (backWrap * 0.06 + sc) * 0.22;

    // ── Exposure sunlift (dryad): canopy-tip warm lift. The former whole-term
    //    `* shadowMul` (interior shadow) was folded into ao_leaf above so it now
    //    attenuates ONLY the indirect IBL term, leaving the direct sun at full
    //    intensity (matching dryad's indirectDiffuse *= mix(1,vAo,0.85)). ──
    // ponytail: halved the warm sunlift — at full strength it desaturated the
    // most sun-exposed canopy tips toward white; a gentler lift keeps the leaf
    // pigment reading as foliage while still warming the sunlit crown.
    let sunlift = mix(vec3<f32>(0.0), vec3<f32>(0.03, 0.09, 0.04), exp_leaf * exp_leaf);

    // ponytail: the canopy uses a soft SPHERE normal, so a whole cluster of leaves
    // faces the intensity-3.0 key sun at once → their direct+IBL linear HDR pinned
    // well above the bloom threshold (1.1), blooming the canopy into a solid white
    // blob with no leaf silhouettes. Trim the leaf radiance into dryad's range
    // (LEAF_EXPOSURE) so sunlit leaves read as lit foliage just under the bloom
    // gate — only the very brightest specular tips still bloom, the way dryad's do.
    let lit = (pbr + sunlift + backlitGlow) * LEAF_EXPOSURE;
    // LINEAR HDR out — the flora OutputPass applies ACES ONCE on scene+bloom.
    return vec4<f32>(lit, 1.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// (3) SHADOW DEPTH pipelines — depth-only casters for the sun shadow map.
//
// dryad casts both branch + leaf into a 2048² depth target with the SAME wind
// displacement as the lit geometry (so the silhouette sways in lockstep) and the
// leaf respects its alpha cutout (alphaTest 0.5). We do the same: these VS
// reproduce the tree-local wind-displaced position (identical to the main pass'
// branch_local_displaced / leaf_local_displaced) and project it with the LIGHT'S
// view-proj instead of the camera's. The leaf FS alpha-tests the Gielis cutout.
//
// The model ROTATION is passed as a quaternion (16B) so the push fits 128B while
// still carrying the light matrix + wind (+ leaf shape for the leaf cutout). It
// rotates position into the same tree-local frame the receiver projects from.
// ─────────────────────────────────────────────────────────────────────────────

// Rotate a vector by a unit quaternion q = (x,y,z,w). Standard formula.
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let u = q.xyz;
    return v + 2.0 * cross(u, cross(u, v) + q.w * v);
}

// Branch depth caster push (96B used of 128): light matrix + rotation quat + wind.
struct BranchDepthPush {
    light_view_proj : mat4x4<f32>,  // 64  world(tree-local) → light clip
    rot             : vec4<f32>,    // 16  model rotation quaternion (x,y,z,w)
    wind            : vec4<f32>,    // 16  (time, strength, dirX, dirZ)
}
var<immediate> branch_depth_pc: BranchDepthPush;

@vertex
fn vs_branch_depth(v: BranchVsIn) -> @builtin(position) vec4<f32> {
    // IDENTICAL skin to the lit branch (so the cast silhouette sways in lockstep):
    // skin the rest pos by its bone matrix FIRST, then apply the model rotation
    // (here the quaternion form of m3). strength==0 → skinned == rest.
    let skinned = windSkinPosition(v.position, v.attr.y, v.attr.z, branch_depth_pc.wind.y);
    let local = quat_rotate(branch_depth_pc.rot, skinned);
    return branch_depth_pc.light_view_proj * vec4<f32>(local, 1.0);
}

// Leaf depth caster push (128B exactly): light matrix + rot quat + wind +
// leaf shape (for the alpha cutout) + skew.
struct LeafDepthPush {
    light_view_proj : mat4x4<f32>,  // 64
    rot             : vec4<f32>,    // 16  rotation quaternion
    wind            : vec4<f32>,    // 16  (time, strength, dirX, dirZ)
    leaf_params     : vec4<f32>,    // 16  (leafTip, leafWidth, leafSerration, leafLobing)
    leaf_params2    : vec4<f32>,    // 16  (leafSkew, _, _, _)
}
var<immediate> leaf_depth_pc: LeafDepthPush;

struct LeafDepthVsOut {
    @builtin(position) clip_pos : vec4<f32>,
    @location(0)       uv       : vec2<f32>,
}

@vertex
fn vs_leaf_depth(v: LeafVsIn) -> LeafDepthVsOut {
    var out: LeafDepthVsOut;
    // IDENTICAL skin to the lit leaf: rotate the anchor into tree-local, bone-
    // follow the nearest branch bone, then add the same per-leaf flutter.
    let t   = leaf_depth_pc.wind.x;
    let s   = leaf_depth_pc.wind.y;
    // Bone-follow in BONE space, then rotate (mirrors leaf_local_displaced).
    let follow_obj = windBoneFollowDelta(v.position, v.uv.z, s);
    let rotated = quat_rotate(leaf_depth_pc.rot, v.position);
    let follow  = quat_rotate(leaf_depth_pc.rot, follow_obj);
    let ph    = v.attr.z * 6.2831853;
    let fl    = s * 0.06;
    let flutter = vec3<f32>(
        sin(t * 8.0  + ph)        * fl,
        sin(t * 11.3 + ph * 1.7)  * fl * 0.5,
        cos(t * 9.1  + ph * 0.6)  * fl);
    let local = rotated + follow + flutter;
    out.clip_pos = leaf_depth_pc.light_view_proj * vec4<f32>(local, 1.0);
    out.uv = v.uv.xy;
    return out;
}

@fragment
fn fs_leaf_depth(in: LeafDepthVsOut) {
    // dryad leaf shadow material: alphaTest 0.5 against the leaf cutout, so the
    // shadow silhouette respects the leaf shape (not a solid quad). Sample the SAME
    // baked cluster sprite alpha the lit pass cuts on (binding 7 + sampler 9) so
    // caster and receiver cutouts match exactly. The leaf_params lanes are now only
    // carried for layout compatibility (the in-shader Gielis test is gone).
    let a = textureSample(leaf_color, leaf_sampler, in.uv).a;
    if (a < 0.5) {
        discard;
    }
    // No color target — depth-only. Reaching here writes depth.
}
