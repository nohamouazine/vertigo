#!/usr/bin/env node
// npx vertigo-fr "question"
import { spawnSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import { join, dirname } from 'node:path'

const ask = join(dirname(fileURLToPath(import.meta.url)), '..', 'dist', 'ask.js')
const r = spawnSync(process.execPath, ['--no-warnings', ask, ...process.argv.slice(2)], { stdio: 'inherit' })
process.exit(r.status ?? 1)
