// The French bench: 150 questions in six categories, scored by rules, no judge model.
//   node --experimental-strip-types bench/score.ts            (writes bench/results.json, prints the table row)
//   node --experimental-strip-types bench/score.ts --limit 20 (a quick look)
//
// faits        40  a fact; the model may read Wikipedia first (its [outil:cherche] call is honoured)
// outils       21  the right tool call for the question (weather, time, calculator, Wikipedia...)
// calcul        9  the exact number, computed by its calculator call or written in the answer
// lecture      10  a question about a short text given with it
// demandes     40  a poem, a list, a message, a translation: an actual attempt, long enough
// conversation 30  small talk: an answer, not a refusal and not a tool call
//
// The score is the mean of the six category percentages. Decoding is greedy, so a run is reproducible except
// for what Wikipedia returns on the day (looked up once and cached in bench/cache-wiki.json).
import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { generate, load, TOOL, wikipedia, type Page } from '../engine/vertigo.ts'

type Item = { id: string; categorie: string; question: string; attendu?: string[]; extrait?: string; outil?: string; valeur?: number | string; min_mots?: number }

const here = import.meta.dirname
const limit = Number(process.argv[process.argv.indexOf('--limit') + 1]) || Infinity
const items: Item[] = readFileSync(join(here, 'questions.jsonl'), 'utf8').split('\n').filter(Boolean).map(l => JSON.parse(l)).slice(0, limit)
const CACHE = join(here, 'cache-wiki.json')
const cache: Record<string, Page | null> = existsSync(CACHE) ? JSON.parse(readFileSync(CACHE, 'utf8')) : {}
const REFUS = /je ne (sais|peux)|pas encore appris|en tant qu'ia|je n'ai pas acces/i

const plain = (t: string) => t.toLowerCase().replace(/’/g, "'").replace(/œ/g, 'oe').replace(/æ/g, 'ae').normalize('NFD').replace(/[̀-ͯ]/g, '')
const escape = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

/** One of the expected words is there. Short ones (« au », « 6 ») must be whole; longer ones may start a word. */
function contains(text: string, keywords: string[]) {
  const t = plain(text)
  return keywords.some(kw => {
    const k = escape(plain(kw))
    return new RegExp(kw.length <= 3 ? `(?<![a-z0-9])${k}(?![a-z0-9])` : `(?<![a-z0-9])${k}`).test(t)
  })
}

/** A calculator expression: numbers, + - × ÷ ^ %, racine, parentheses. Nothing is executed. */
function evaluate(input: string): number | null {
  let s = input.replace(/×/g, '*').replace(/x/g, '*').replace(/÷/g, '/').replace(/,/g, '.').replace(/\^/g, '**')
  s = s.replace(/(\d+(?:\.\d+)?)\s*%\s*(?:de|of)?\s*(\d+(?:\.\d+)?)/g, '($1*$2/100)')
  s = s.replace(/racine\s*(?:carr[ée]e)?\s*(?:de)?\s*\(?(\d+(?:\.\d+)?)\)?/g, 'sqrt($1)')
  const tokens = s.match(/\d+(?:\.\d+)?|\*\*|[-+*/%()]|sqrt/g)
  if (!tokens || tokens.join('') !== s.replace(/\s+/g, '')) return null
  let i = 0
  const peek = () => tokens[i]
  const next = () => tokens[i++]
  const primary = (): number => {
    const t = next()
    if (t === undefined) throw new Error()
    if (t === '(') { const v = expr(); if (next() !== ')') throw new Error(); return v }
    if (t === '-') return -primary()
    if (t === '+') return primary()
    if (t === 'sqrt') { if (next() !== '(') throw new Error(); const v = expr(); if (next() !== ')') throw new Error(); return Math.sqrt(v) }
    if (/^\d/.test(t)) return Number(t)
    throw new Error()
  }
  const power = (): number => { const b = primary(); if (peek() === '**') { next(); return b ** power() } return b }
  const term = (): number => {
    let v = power()
    while (peek() === '*' || peek() === '/' || peek() === '%') { const op = next(); const r = power(); v = op === '*' ? v * r : op === '/' ? v / r : v % r }
    return v
  }
  const expr = (): number => {
    let v = term()
    while (peek() === '+' || peek() === '-') { const op = next(); const r = term(); v = op === '+' ? v + r : v - r }
    return v
  }
  try { const v = expr(); return i === tokens.length && Number.isFinite(v) ? v : null } catch { return null }
}

const numbers = (t: string) => (t.match(/\d[\d ]*(?:[.,]\d+)?/g) ?? []).map(n => Number(n.replace(',', '.').replace(/ /g, '')))

async function lookup(query: string) {
  if (!(query in cache)) {
    for (const wait of [1000, 30000, 90000]) {
      await new Promise(r => setTimeout(r, wait))
      try { cache[query] = await wikipedia(query); break } catch { continue }
    }
    if (!(query in cache)) return null
    writeFileSync(CACHE, JSON.stringify(cache, null, 1))
  }
  return cache[query]
}

const engine = load()
const say = (prompt: string) => generate(engine, prompt, { temperature: 0 })
const results: { id: string; categorie: string; question: string; reponse: string; finale: string; ok: boolean }[] = []
const t0 = Date.now()
for (const it of items) {
  const { categorie: cat, question: q } = it
  const kw = it.attendu ?? []
  const prompt = cat === 'lecture' ? `<|doc|>${it.extrait}<|user|>${q}<|bot|>` : `<|user|>${q}<|bot|>`
  const first = say(prompt)
  const call = TOOL.exec(first)
  let final = first, ok = false
  if (cat === 'faits') {
    if (call && (call[1] === 'cherche' || call[1] === 'wiki')) {
      const query = (call[2] ?? '').trim()
      const page = await lookup(!query || query === '~' ? q : query)
      if (page) final = say(`<|doc|>${page.extract}<|user|>${q}<|bot|>`)
    }
    ok = contains(final, kw)
  } else if (cat === 'outils') {
    ok = Boolean(call) && (it.outil ?? '').split('|').includes(call![1]) && (kw.length === 0 || contains(call![2] ?? '', kw))
  } else if (cat === 'calcul') {
    const target = Number(it.valeur)
    if (call && call[1] === 'calcul') { const v = evaluate(call[2] ?? ''); ok = v !== null && Math.abs(v - target) < 1e-6 }
    else ok = numbers(first).some(n => Math.abs(n - target) < 1e-6)
  } else if (cat === 'lecture') {
    ok = contains(first, kw)
  } else {
    const words = (first.match(/\w+/g) ?? []).length
    ok = !first.includes('[outil:') && !REFUS.test(first) && words >= (it.min_mots ?? 2) && (kw.length === 0 || contains(first, kw))
  }
  results.push({ id: it.id, categorie: cat, question: q, reponse: first, finale: final, ok })
  process.stderr.write(`${ok ? '✓' : '✗'} ${it.id}\n`)
}

const cats = ['faits', 'outils', 'calcul', 'lecture', 'demandes', 'conversation']
const score = Object.fromEntries(cats.map(c => {
  const of = results.filter(r => r.categorie === c)
  return [c, of.length ? Math.round(1000 * of.filter(r => r.ok).length / of.length) / 10 : 0]
}))
const total = Math.round(10 * cats.reduce((s, c) => s + score[c], 0) / cats.length) / 10
writeFileSync(join(here, 'results.json'), JSON.stringify({ total, categories: score, seconds: Math.round((Date.now() - t0) / 1000), results }, null, 1))
console.log(`| ${new Date().toISOString().slice(0, 10)} | Vertigo 3 | **${total}** | ${cats.map(c => score[c]).join(' | ')} |`)
