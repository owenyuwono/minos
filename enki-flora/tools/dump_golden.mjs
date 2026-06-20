// =============================================================================
// dump_golden.mjs — GOLDEN-VECTOR producer for the enki-flora determinism gate.
//
// Imports the REAL dryad generator (genome.js: randomGenome + resolve) and dumps
// a fixed set of cases to tests/golden.json. The Rust port (enki-flora) is then
// asserted bit-for-bit (rng/genes) and within a cross-language trig epsilon
// (geometry) against this file by tests/golden.rs.
//
// RUN (node v22, from anywhere):
//   node C:/Work/Prototype/enki/enki-flora/tools/dump_golden.mjs
//
// -----------------------------------------------------------------------------
// WHY THE ENVS ARE FULLY SPECIFIED (and not literal `{}`):
//   The task brief uses `env={}` as shorthand for "the neutral environment". But
//   dryad's solveProportions destructures `const { gravity, medium } = envelope`
//   with NO default for gravity, so resolve({}) yields gravPow(undefined,…) = NaN
//   for EVERY node radius and ~half the node positions. NaN geometry is neither a
//   meaningful golden vector nor representable in the Rust port's `Env` struct
//   (which carries gravity=1.0 for the neutral env). So each case here passes the
//   EXPLICIT neutral env — gravity=1, medium='air', light=0.6, sunAngle=0.25,
//   wind=0.2, aridity=0.35, temperature=0.5 — which is byte-identical to what the
//   Rust `Env::default()` represents, and produces finite, comparable geometry.
//   randomGenome's draw schedule is unaffected (it only reads gravity/medium/etc.
//   through computeEnvOffset, all of which DO default), so the genes are identical
//   to the `{}` shorthand regardless.
//
//   NOTE: graph node pos/radius are plain f64 in JS; the foliage SoA arrays are
//   Float32Array, so their values are ALREADY f32-rounded — the Rust port stores
//   them as f32 too, matching. The JS foliage arrays over-allocate to MAX_LEAVES;
//   only [0, count) slots are valid, so we slice to count here.
// =============================================================================

import { writeFileSync, mkdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const { randomGenome, resolve } = await import(
  'file:///C:/Work/Prototype/dryad/src/genome.js'
);

const __dirname = dirname(fileURLToPath(import.meta.url));
const OUT = join(__dirname, '..', 'tests', 'golden.json');

// Cap on captured foliage instances per case (full count is still recorded).
const LEAF_CAP = 200;

// The explicit neutral env (== Rust Env::default()). See header note.
const NEUTRAL_ENV = {
  gravity: 1.0,
  medium: 'air',
  light: 0.6,
  sunAngle: 0.25,
  wind: 0.2,
  aridity: 0.35,
  temperature: 0.5,
};

// A non-default env: heavy gravity, dry/hot, dim. Stresses computeEnvOffset
// (succulence/spininess/root biases) AND proportions (gravity droop/girth).
const HARSH_ENV = {
  gravity: 1.6,
  medium: 'air',
  light: 0.3,
  sunAngle: 0.25,
  wind: 0.2,
  aridity: 0.7,
  temperature: 0.5,
};

// 5 fixed cases: 3 neutral-env seeds + 2 harsh-env seeds.
const CASES = [
  { seed: 1, env: NEUTRAL_ENV },
  { seed: 42, env: NEUTRAL_ENV },
  { seed: 1337, env: NEUTRAL_ENV },
  { seed: 42, env: HARSH_ENV },
  { seed: 1337, env: HARSH_ENV },
];

// The full gene set captured per case (JS gene name → value). Mirrors the
// randomGenome return object 1:1 (structuralSeed is a uint32, the rest f64).
const GENE_KEYS = [
  'branchiness', 'branchFactorN', 'tillering', 'radialOrder', 'appendageBreadth',
  'appendageDensity', 'segmentation', 'succulence', 'stemGirth', 'taper',
  'rigidity', 'verticality', 'ribbing', 'spininess', 'branchAngle', 'lengthRatio',
  'apicalBias', 'droopBias', 'pigment', 'leafSize', 'jitter', 'leafWidth',
  'leafLength', 'leafTip', 'leafSerration', 'leafLobing', 'leafSkew', 'barkHue',
  'barkLightness', 'barkRelief', 'barkLenticels', 'barkScale', 'barkOrient',
  'barkPlates', 'barkShed', 'barkUnderHue', 'weep', 'trunkHeight', 'flatness',
  'stemSpread', 'rosette', 'woodiness', 'whorl', 'crownStart', 'tipTuft',
  'leafDivision', 'frondFan', 'phototropism', 'trunkTaper', 'structuralSeed',
  'rootCount', 'rootDepth', 'rootSpread', 'rootFlare', 'rootButtress',
  'rootBranchiness', 'rootTaper',
];

function dumpCase({ seed, env }) {
  const genome = randomGenome(env, seed);
  const r = resolve(genome, env);

  // --- Genes (the bit-exact contract) ---
  const genes = {};
  for (const k of GENE_KEYS) {
    if (!(k in genome)) throw new Error(`gene '${k}' missing from genome (seed ${seed})`);
    genes[k] = genome[k];
  }

  // --- Graph: nodes (pos[3], radius), bones (a,b), meta.lightDir ---
  const nodes = r.graph.nodes.map((n) => ({
    pos: [n.pos[0], n.pos[1], n.pos[2]],
    radius: n.radius,
  }));
  const bones = r.graph.bones.map((b) => ({ a: b.a, b: b.b }));

  // --- Foliage SoA (sliced to count; arrays already f32 in JS) ---
  const f = r.foliage;
  const cap = Math.min(f.count, LEAF_CAP);
  const position = Array.from(f.position.slice(0, cap * 3));
  const scale = Array.from(f.scale.slice(0, cap));
  const rotation = Array.from(f.rotation.slice(0, cap));

  return {
    seed,
    env,
    genes,
    graph: {
      nodeCount: r.graph.nodes.length,
      boneCount: r.graph.bones.length,
      nodes,
      bones,
      lightDir: [
        r.graph.meta.lightDir[0],
        r.graph.meta.lightDir[1],
        r.graph.meta.lightDir[2],
      ],
    },
    foliage: {
      count: f.count,
      shape: f.shape,
      cap,            // number of leaves captured below (first `cap` of `count`)
      position,       // 3 * cap
      scale,          // cap
      rotation,       // cap
    },
  };
}

const golden = {
  // Provenance so a stale golden.json is obvious.
  source: 'dryad/src/genome.js randomGenome+resolve',
  generator: 'enki-flora/tools/dump_golden.mjs',
  leafCap: LEAF_CAP,
  cases: CASES.map(dumpCase),
};

mkdirSync(dirname(OUT), { recursive: true });
// Pretty but compact: 0-space arrays would be huge; default 2-space is fine and
// JSON round-trips f64 losslessly (Number → shortest exact decimal).
writeFileSync(OUT, JSON.stringify(golden, null, 1));

const summary = golden.cases
  .map(
    (c) =>
      `  seed=${c.seed} env.gravity=${c.env.gravity} aridity=${c.env.aridity} light=${c.env.light}` +
      ` → nodes=${c.graph.nodeCount} bones=${c.graph.boneCount} foliage=${c.foliage.count}` +
      ` (cap ${c.foliage.cap}) structuralSeed=${c.genes.structuralSeed}`
  )
  .join('\n');
console.log(`wrote ${OUT}\n${summary}`);
