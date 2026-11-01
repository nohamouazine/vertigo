// Ask Vertigo from the terminal:
//   node --experimental-strip-types engine/ask.ts "Qui a écrit Les Misérables ?"
//   node --experimental-strip-types engine/ask.ts --no-wiki "Donne-moi trois idées de repas rapides"
// It prints the answer as it is written, then the source it read, if any.
import { answer, load } from './vertigo.ts'

const args = process.argv.slice(2)
const wiki = !args.includes('--no-wiki')
const question = args.filter(a => !a.startsWith('--')).join(' ').trim()
if (!question) {
  console.error('usage: node --experimental-strip-types engine/ask.ts [--no-wiki] "your question"')
  process.exit(2)
}

const t0 = performance.now()
const engine = load()
console.error(`model loaded in ${((performance.now() - t0) / 1000).toFixed(1)} s`)

const t1 = performance.now()
const result = await answer(engine, question, { wiki })
console.log(result.text)
if (result.source) console.log(`\nSource : ${result.source.title} · ${result.source.url}`)
else if (result.asked) console.log(`\n(asked to read Wikipedia for « ${result.asked} », answered without it)`)
console.error(`\n${((performance.now() - t1) / 1000).toFixed(1)} s`)
