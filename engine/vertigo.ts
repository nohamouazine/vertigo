// Vertigo as a library: load the model folder, write an answer, read Wikipedia when the model asks for it.
//
// The model learnt one tool call: when it writes [outil:cherche|query] (or [outil:wiki|query]) it wants to read
// before answering. We fetch the opening of the matching French Wikipedia article and ask again with the extract
// in front of it: <|doc|>extract<|user|>question<|bot|>. The answer then comes with a source, which is the point.
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { joinShards, sample, seeded, VertigoModel, type Manifest } from './model.ts'
import { Tokenizer } from './tokenizer.ts'

export const TOOL = /\[outil:(\w+)(?:\|([^\]]*))?\]/
export const MAX_NEW = 160
const SMALL_TALK = /^\s*(salut|slt|coucou|bonjour|bonsoir|hello|hey|yo|wesh|merci|cc|ça va|ca va|tu vas bien|bonne (nuit|journée|soirée))\b[^]{0,30}$/i

export type Engine = { model: VertigoModel; tokenizer: Tokenizer; end: number }

export function load(dir = join(import.meta.dirname, '..', 'model')): Engine {
  const manifest: Manifest = JSON.parse(readFileSync(join(dir, 'manifest.json'), 'utf8'))
  const tokenizer = new Tokenizer(JSON.parse(readFileSync(join(dir, 'tokenizer.json'), 'utf8')))
  const model = new VertigoModel(manifest, joinShards(manifest, manifest.shards.map(s => readFileSync(join(dir, s)))))
  return { model, tokenizer, end: tokenizer.encode('<|end|>')[0] }
}

export type Options = { temperature?: number; topP?: number; seed?: number; maxNew?: number; onText?: (text: string) => void }

/** Writes up to `maxNew` tokens after `prompt`. Temperature 0 always takes the most likely token. */
export function generate(engine: Engine, prompt: string, { temperature = 0.5, topP = 0.9, seed = 1, maxNew = MAX_NEW, onText }: Options = {}) {
  const { model, tokenizer, end } = engine
  const session = model.session()
  const random = seeded(seed)
  const ids = tokenizer.encode(prompt)
  const kept = ids.slice(-(model.manifest.context - maxNew))
  let logits: Float32Array = new Float32Array(0)
  for (const id of kept) logits = session.step(id)
  const out: number[] = []
  for (let i = 0; i < maxNew; i++) {
    // A small model goes round in circles: the words it just wrote become less likely (×1.15), and one word
    // never appears more than 3 times in the last 24.
    const recent = out.slice(-64)
    for (const t of new Set(recent)) logits[t] = logits[t] > 0 ? logits[t] / 1.15 : logits[t] * 1.15
    const last = out.slice(-24)
    for (const t of new Set(last)) if (last.filter(x => x === t).length >= 3) logits[t] = -Infinity
    const next = sample(logits, temperature, topP, random)
    if (next === end) break
    out.push(next)
    if (onText && out.length % 4 === 0) onText(tokenizer.decode(out))
    logits = session.step(next)
  }
  const text = tokenizer.decode(out).replace(/�/g, '').trim()
  onText?.(text)
  return text
}

export type Page = { title: string; extract: string; url: string }

/** The opening sentences of the first French Wikipedia article matching `query`. */
export async function wikipedia(query: string): Promise<Page | null> {
  const url = 'https://fr.wikipedia.org/w/api.php?' + new URLSearchParams({ action: 'query', generator: 'search', gsrsearch: query, gsrlimit: '1',
    prop: 'extracts', exintro: '1', explaintext: '1', format: 'json' })
  const r = await fetch(url, { headers: { 'user-agent': 'Vertigo/1.0 (https://paradoxalai.com)' }, signal: AbortSignal.timeout(8000) })
  if (!r.ok) return null
  const data = await r.json() as { query?: { pages?: Record<string, { title: string; extract?: string }> } }
  const page = Object.values(data.query?.pages ?? {})[0]
  if (!page?.extract) return null
  return { title: page.title, extract: firstSentences(page.extract, 450), url: `https://fr.wikipedia.org/wiki/${encodeURIComponent(page.title.replace(/ /g, '_'))}` }
}

/** The opening sentences, up to `max` characters: the model reads one token at a time, a shorter extract answers sooner. */
export function firstSentences(text: string, max: number) {
  const clean = text.replace(/\s+/g, ' ').trim()
  if (clean.length <= max) return clean
  const cut = clean.slice(0, max)
  const end = Math.max(cut.lastIndexOf('. '), cut.lastIndexOf('! '), cut.lastIndexOf('? '))
  return end > 80 ? cut.slice(0, end + 1) : cut
}

export type Answer = { text: string; source: Page | null; asked: string | null }

/** Asks the question; reads Wikipedia if the model asks to (unless `wiki` is false). */
export async function answer(engine: Engine, question: string, { wiki = true, seed = Date.now() % 2147483647, temperature = 0.5, lookup = wikipedia }:
  { wiki?: boolean; seed?: number; temperature?: number; lookup?: (query: string) => Promise<Page | null> } = {}): Promise<Answer> {
  const q = question.replace(/<\|[a-z]+\|>/g, '').slice(0, 1000)
  let first = generate(engine, `<|user|>${q}<|bot|>`, { seed, temperature })
  for (let attempt = 1; attempt <= 2 && SMALL_TALK.test(q) && TOOL.test(first); attempt++) first = generate(engine, `<|user|>${q}<|bot|>`, { seed: seed + 10 + attempt, temperature })
  const call = SMALL_TALK.test(q) ? null : TOOL.exec(first)
  if (!wiki || !call || (call[1] !== 'cherche' && call[1] !== 'wiki')) {
    return { text: first.replace(TOOL, '').trim() || 'Je préfère ne pas répondre de tête à ça.', source: null, asked: call ? (call[2] ?? '').trim() : null }
  }
  const asked = (call[2] ?? '').trim()
  // Its query must share a word with the question, otherwise the question itself is searched.
  const words = (x: string) => new Set(x.toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g, '').match(/[a-z0-9]{4,}/g) ?? [])
  const shared = [...words(asked)].some(w => words(q).has(w))
  const query = !asked || asked === '~' || !shared ? q : asked
  const page = await lookup(query).catch(() => null)
  if (!page) return { text: 'Je n’ai rien trouvé sur Wikipédia pour répondre à ça.', source: null, asked: query }
  let text = ''
  for (let attempt = 1; attempt <= 2 && !text; attempt++) {
    text = generate(engine, `<|doc|>${page.extract}<|user|>${q}<|bot|>`, { seed: seed + attempt, temperature })
    if (TOOL.test(text)) text = ''
  }
  if (!text) text = `D’après Wikipédia : ${firstSentences(page.extract, 300)}`
  return { text, source: page, asked: query }
}
