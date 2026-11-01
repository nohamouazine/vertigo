// same tokens and logits (1e-3) as the training framework on model/vectors.json
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { load } from './vertigo.ts'

const dir = join(import.meta.dirname, '..', 'model')
const { model, tokenizer } = load(dir)
const { vectors } = JSON.parse(readFileSync(join(dir, 'vectors.json'), 'utf8'))

let failed = false
for (const v of vectors as { prompt: string; ids: number[]; top: number[]; logits: Record<string, number> }[]) {
  const ids = tokenizer.encode(v.prompt)
  const sameIds = ids.length === v.ids.length && ids.every((id, i) => id === v.ids[i])
  const t0 = performance.now()
  const logits = model.session().read(ids)
  const ms = (performance.now() - t0) / ids.length
  let maxDiff = 0
  for (const [i, expected] of Object.entries(v.logits)) maxDiff = Math.max(maxDiff, Math.abs(expected - logits[Number(i)]))
  let best = 0
  for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i
  const ok = sameIds && maxDiff < 1e-3 && best === v.top[0] && tokenizer.decode(ids) === v.prompt
  if (!ok) failed = true
  console.log(`« ${v.prompt.slice(8, 40)}… »: ${ids.length} tokens ${sameIds ? 'same' : 'DIFFERENT'}, max logit diff ${maxDiff.toExponential(2)}, ${ms.toFixed(0)} ms/token ${ok ? 'OK' : 'FAIL'}`)
}
process.exit(failed ? 1 : 0)
