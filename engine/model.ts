// Vertigo inference in plain TypeScript. int8 weights, one scale per row. KV cache per layer.

export type Manifest = {
  name: string
  run: string
  d: number
  layers: number
  heads: number
  hidden: number
  context: number
  vocab: number
  ropeBase: number
  normEps: number
  qkNorm?: boolean   // q et k normalisés par tête avant l'attention (essai du labo, Qwen3)
  repeat?: number    // chaque couche passe `repeat` fois de suite avec les mêmes poids (la boucle)
  ngram?: number     // mémoire de bigrammes hachée : une table de `ngram` lignes ajoutée à l'embedding (labo tour 7-8)
  parallel?: boolean // attention et couche dense côte à côte, lues sur la même entrée
  untied?: boolean   // sortie séparée de l'embedding (matrice `head`)
  tokens: Record<string, number>
  shards: string[]
  bytes: number
  tensors: { name: string; shape: number[]; dtype: 'int8' | 'f32'; offset: number; scales?: number }[]
}

type Matrix = { q: Int8Array; s: Float32Array; rows: number; cols: number }
type Layer = { norm1: Float32Array; qkv: Matrix; out: Matrix; norm2: Float32Array; gateUp: Matrix; down: Matrix; qn?: Float32Array; kn?: Float32Array }

// out[r] = sum(q[r][c] * x[c]) * s[r]
function matvec(m: Matrix, x: Float32Array, out: Float32Array) {
  const { q, s, rows, cols } = m
  for (let r = 0; r < rows; r++) {
    const base = r * cols
    let a0 = 0, a1 = 0, a2 = 0, a3 = 0
    let c = 0
    for (; c + 3 < cols; c += 4) {
      a0 += q[base + c] * x[c]
      a1 += q[base + c + 1] * x[c + 1]
      a2 += q[base + c + 2] * x[c + 2]
      a3 += q[base + c + 3] * x[c + 3]
    }
    for (; c < cols; c++) a0 += q[base + c] * x[c]
    out[r] = (a0 + a1 + a2 + a3) * s[r]
  }
  return out
}

function rmsNorm(x: Float32Array, w: Float32Array, eps: number, out: Float32Array) {
  let ss = 0
  for (let i = 0; i < x.length; i++) ss += x[i] * x[i]
  const k = 1 / Math.sqrt(ss / x.length + eps)
  for (let i = 0; i < x.length; i++) out[i] = x[i] * k * w[i]
  return out
}

// shards concatenated in order
export function joinShards(manifest: Manifest, shards: Uint8Array[]): ArrayBuffer {
  const all = new Uint8Array(manifest.bytes)
  let at = 0
  for (const shard of shards) { all.set(shard, at); at += shard.length }
  if (at !== manifest.bytes) throw new Error(`Vertigo: ${at} bytes of weights, ${manifest.bytes} expected`)
  return all.buffer
}

export class VertigoModel {
  readonly manifest: Manifest
  readonly headDim: number
  readonly tok: Matrix
  readonly layers: Layer[] = []
  readonly norm: Float32Array
  readonly cos: Float32Array
  readonly sin: Float32Array
  ngram?: Matrix
  head?: Matrix

  constructor(manifest: Manifest, buffer: ArrayBuffer) {
    this.manifest = manifest
    const byName = new Map(manifest.tensors.map(t => [t.name, t]))
    const matrix = (name: string): Matrix => {
      const t = byName.get(name)
      if (!t || t.dtype !== 'int8' || t.scales === undefined) throw new Error(`Vertigo: missing ${name}`)
      const [rows, cols] = t.shape
      return { q: new Int8Array(buffer, t.offset, rows * cols), s: new Float32Array(buffer, t.scales, rows), rows, cols }
    }
    const vector = (name: string) => {
      const t = byName.get(name)
      if (!t || t.dtype !== 'f32') throw new Error(`Vertigo: missing ${name}`)
      return new Float32Array(buffer, t.offset, t.shape[0])
    }
    this.tok = matrix('tok')
    if (manifest.ngram) this.ngram = matrix('ngram')
    if (manifest.untied) this.head = matrix('head')
    for (let i = 0; i < manifest.layers; i++) {
      const p = `blocks.${i}.`
      this.layers.push({ norm1: vector(p + 'norm1'), qkv: matrix(p + 'qkv'), out: matrix(p + 'out'),
        norm2: vector(p + 'norm2'), gateUp: matrix(p + 'gate_up'), down: matrix(p + 'down'),
        ...(manifest.qkNorm ? { qn: vector(p + 'qn'), kn: vector(p + 'kn') } : {}) })
    }
    this.norm = vector('norm')

    // rope: pos * base^(-2i/headSize)
    this.headDim = manifest.d / manifest.heads
    const half = this.headDim / 2
    this.cos = new Float32Array(manifest.context * half)
    this.sin = new Float32Array(manifest.context * half)
    for (let pos = 0; pos < manifest.context; pos++) {
      for (let i = 0; i < half; i++) {
        const angle = Math.fround(pos * Math.fround(1 / Math.pow(manifest.ropeBase, (2 * i) / this.headDim)))
        this.cos[pos * half + i] = Math.cos(angle)
        this.sin[pos * half + i] = Math.sin(angle)
      }
    }
  }

  session() { return new Session(this) }
}

export class Session {
  private m: VertigoModel
  private keys: Float32Array[]
  private values: Float32Array[]
  position = 0
  // previous token (0 at start), bigram key
  private previous = 0
  // scratch buffers
  private x: Float32Array
  private h: Float32Array
  private qkv: Float32Array
  private att: Float32Array
  private proj: Float32Array
  private gu: Float32Array
  private act: Float32Array
  private scores: Float32Array
  private hp: Float32Array
  private attOut: Float32Array
  readonly logits: Float32Array

  constructor(model: VertigoModel) {
    this.m = model
    const { d, hidden, context, layers, vocab } = model.manifest
    // one KV cache per layer pass
    const passes = layers * (model.manifest.repeat ?? 1)
    this.keys = Array.from({ length: passes }, () => new Float32Array(context * d))
    this.values = Array.from({ length: passes }, () => new Float32Array(context * d))
    this.x = new Float32Array(d)
    this.h = new Float32Array(d)
    this.qkv = new Float32Array(3 * d)
    this.att = new Float32Array(d)
    this.proj = new Float32Array(d)
    this.gu = new Float32Array(2 * hidden)
    this.act = new Float32Array(hidden)
    this.scores = new Float32Array(context)
    this.hp = new Float32Array(d)
    this.attOut = new Float32Array(d)
    this.logits = new Float32Array(vocab)
  }

  get room() { return this.m.manifest.context - this.position }

  // feed one token, get logits for the next
  step(token: number): Float32Array {
    const m = this.m
    const { d, heads, hidden, normEps } = m.manifest
    const hd = m.headDim, half = hd / 2, pos = this.position
    if (pos >= m.manifest.context) throw new Error('Vertigo: context is full')
    const { x, h, qkv, att, proj, gu, act, scores } = this

    const row = token * d, scale = m.tok.s[token]
    for (let i = 0; i < d; i++) x[i] = m.tok.q[row + i] * scale
    if (m.ngram) {
      // (prev * 1000003 + tok) % N, exact in doubles
      const slot = (this.previous * 1_000_003 + token) % m.ngram.rows
      const nrow = slot * d, nscale = m.ngram.s[slot]
      for (let i = 0; i < d; i++) x[i] += m.ngram.q[nrow + i] * nscale
    }
    this.previous = token
    const parallel = !!m.manifest.parallel

    const repeat = m.manifest.repeat ?? 1
    for (let pass = 0; pass < m.layers.length * repeat; pass++) {
      const layer = m.layers[Math.floor(pass / repeat)]
      matvec(layer.qkv, rmsNorm(x, layer.norm1, normEps, h), qkv)
      if (parallel) this.hp.set(h)  // the parallel block feeds the dense layer with the same normalised input
      if (layer.qn && layer.kn) {
        // qk-norm per head, before rope
        for (let head = 0; head < heads; head++) {
          for (const [base, w] of [[head * hd, layer.qn], [d + head * hd, layer.kn]] as const) {
            let ss = 0
            for (let i = 0; i < hd; i++) ss += qkv[base + i] * qkv[base + i]
            const k = 1 / Math.sqrt(ss / hd + normEps)
            for (let i = 0; i < hd; i++) qkv[base + i] = qkv[base + i] * k * w[i]
          }
        }
      }
      // rope on q and k, then cache k/v
      const keys = this.keys[pass], values = this.values[pass]
      for (let head = 0; head < heads; head++) {
        for (let i = 0; i < half; i++) {
          const c = m.cos[pos * half + i], s = m.sin[pos * half + i]
          const qi = head * hd + 2 * i, ki = d + qi
          const q1 = qkv[qi], q2 = qkv[qi + 1], k1 = qkv[ki], k2 = qkv[ki + 1]
          qkv[qi] = q1 * c - q2 * s
          qkv[qi + 1] = q1 * s + q2 * c
          keys[pos * d + head * hd + 2 * i] = k1 * c - k2 * s
          keys[pos * d + head * hd + 2 * i + 1] = k1 * s + k2 * c
        }
      }
      values.set(qkv.subarray(2 * d, 3 * d), pos * d)

      const k = 1 / Math.sqrt(hd)
      for (let head = 0; head < heads; head++) {
        const o = head * hd
        let max = -Infinity
        for (let t = 0; t <= pos; t++) {
          let sum = 0
          const base = t * d + o
          for (let i = 0; i < hd; i++) sum += qkv[o + i] * keys[base + i]
          scores[t] = sum * k
          if (scores[t] > max) max = scores[t]
        }
        let total = 0
        for (let t = 0; t <= pos; t++) { scores[t] = Math.exp(scores[t] - max); total += scores[t] }
        for (let i = 0; i < hd; i++) att[o + i] = 0
        for (let t = 0; t <= pos; t++) {
          const p = scores[t] / total, base = t * d + o
          for (let i = 0; i < hd; i++) att[o + i] += p * values[base + i]
        }
      }
      matvec(layer.out, att, proj)
      if (parallel) {
        this.attOut.set(proj)
        matvec(layer.gateUp, this.hp, gu)
      } else {
        for (let i = 0; i < d; i++) x[i] += proj[i]
        matvec(layer.gateUp, rmsNorm(x, layer.norm2, normEps, h), gu)
      }
      for (let i = 0; i < hidden; i++) {
        const g = gu[i]
        act[i] = (g / (1 + Math.exp(-g))) * gu[hidden + i]
      }
      matvec(layer.down, act, proj)
      for (let i = 0; i < d; i++) x[i] += proj[i] + (parallel ? this.attOut[i] : 0)
    }
    this.position++
    return matvec(m.head ?? m.tok, rmsNorm(x, m.norm, normEps, h), this.logits)
  }

  // prefill, logits after the last token
  read(ids: number[]): Float32Array {
    let logits = this.logits
    for (const id of ids) logits = this.step(id)
    return logits
  }
}

// mulberry32
export function seeded(seed: number) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

// temperature + top-p; temperature 0 = argmax
export function sample(logits: Float32Array, temperature: number, topP: number, random: () => number) {
  if (temperature <= 0) {
    let best = 0
    for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i
    return best
  }
  let max = -Infinity
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i]
  const order: number[] = []
  const probs = new Float64Array(logits.length)
  let total = 0
  for (let i = 0; i < logits.length; i++) {
    probs[i] = Math.exp((logits[i] - max) / temperature)
    total += probs[i]
  }
  for (let i = 0; i < logits.length; i++) {
    probs[i] /= total
    if (probs[i] > 1e-6) order.push(i)
  }
  order.sort((a, b) => probs[b] - probs[a])
  let kept = 0, mass = 0
  while (kept < order.length && mass < topP) mass += probs[order[kept++]]
  let r = random() * mass
  for (let i = 0; i < kept; i++) {
    r -= probs[order[i]]
    if (r <= 0) return order[i]
  }
  return order[0]
}
