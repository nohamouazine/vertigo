// Byte-level BPE, same split and merge order as training. Checked by engine/test.ts.

const SPLIT = / ?[\p{L}\p{Nl}\p{No}]+| ?\p{Nd}{1,3}| ?_+| ?[^\s\p{L}\p{N}_]+|\s+(?!\S)|\s+/gu

export type TokenizerData = { merges: [number, number][]; specials: string[] }

const utf8 = new TextEncoder()

export class Tokenizer {
  readonly specials: Record<string, number> = {}
  readonly size: number
  private ranks = new Map<number, number>()
  private bytes: Uint8Array[] = []
  private names = new Map<number, string>()
  private specialSplit: RegExp
  private cache = new Map<string, number[]>()

  constructor(data: TokenizerData) {
    for (let i = 0; i < 256; i++) this.bytes.push(Uint8Array.of(i))
    data.merges.forEach(([a, b], rank) => {
      this.ranks.set(a * 65536 + b, rank)
      const joined = new Uint8Array(this.bytes[a].length + this.bytes[b].length)
      joined.set(this.bytes[a])
      joined.set(this.bytes[b], this.bytes[a].length)
      this.bytes.push(joined)
    })
    data.specials.forEach((name, i) => {
      this.specials[name] = this.bytes.length + i
      this.names.set(this.bytes.length + i, name)
    })
    this.size = this.bytes.length + data.specials.length
    this.specialSplit = new RegExp(`(${data.specials.map(s => s.replace(/[|]/g, '\\|')).join('|')})`)
  }

  // lowest-rank pair first, all occurrences left to right
  private piece(text: string): number[] {
    const cached = this.cache.get(text)
    if (cached) return cached
    let ids = Array.from(utf8.encode(text))
    while (ids.length > 1) {
      let best = Infinity
      let pair = -1
      for (let i = 0; i + 1 < ids.length; i++) {
        const rank = this.ranks.get(ids[i] * 65536 + ids[i + 1])
        if (rank !== undefined && rank < best) { best = rank; pair = ids[i] * 65536 + ids[i + 1] }
      }
      if (pair < 0) break
      const merged = 256 + best, a = Math.floor(pair / 65536), b = pair % 65536
      const next: number[] = []
      for (let j = 0; j < ids.length; j++) {
        if (j + 1 < ids.length && ids[j] === a && ids[j + 1] === b) { next.push(merged); j++ } else next.push(ids[j])
      }
      ids = next
    }
    if (this.cache.size < 200_000) this.cache.set(text, ids)
    return ids
  }

  encode(text: string): number[] {
    const out: number[] = []
    for (const part of text.split(this.specialSplit)) {
      if (!part) continue
      const special = this.specials[part]
      if (special !== undefined) { out.push(special); continue }
      for (const match of part.matchAll(SPLIT)) out.push(...this.piece(match[0]))
    }
    return out
  }

  decode(ids: number[]): string {
    const parts: Uint8Array[] = []
    let length = 0
    for (const id of ids) {
      const chunk = id < this.bytes.length ? this.bytes[id] : utf8.encode(this.names.get(id) ?? '')
      parts.push(chunk)
      length += chunk.length
    }
    const all = new Uint8Array(length)
    let at = 0
    for (const p of parts) { all.set(p, at); at += p.length }
    return new TextDecoder().decode(all)
  }
}
