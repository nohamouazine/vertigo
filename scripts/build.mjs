// Strips the types out of engine/*.ts into dist/*.js. npm refuses to run .ts files from node_modules.
import { readdirSync, readFileSync, writeFileSync, mkdirSync } from 'node:fs'
import { stripTypeScriptTypes } from 'node:module'

mkdirSync('dist', { recursive: true })
for (const f of readdirSync('engine').filter(f => f.endsWith('.ts'))) {
  const js = stripTypeScriptTypes(readFileSync(`engine/${f}`, 'utf8'), { mode: 'strip' })
    .replace(/from '\.\/(\w+)\.ts'/g, "from './$1.js'")
  writeFileSync(`dist/${f.replace(/\.ts$/, '.js')}`, js)
}
