# Vertigo 3

Small French language model by [ParadoxalAI](https://paradoxalai.com). Own training code, own data, own engine.
Runs with Node.js, no GPU, no dependency.

When it doesn't know, it asks to read: it writes `[outil:cherche|…]`, the engine fetches the intro of the
matching French Wikipedia article, and the answer comes with the source.

```
$ npm run ask "Qui a écrit Les Misérables ?"
Il s'agit d'un roman de Victor Hugo publié en 1862.

Source : Les Misérables · https://fr.wikipedia.org/wiki/Les_Misérables
```

## Run

Node.js 22.6+.

```bash
npm test                                   # engine vs training framework, same logits
npm run ask "Pourquoi le ciel est bleu ?"
npm run ask -- --no-wiki "Donne-moi trois idées de repas rapides"
npm run bench                              # 150 questions, up to an hour on a laptop
```

~60 ms/token on a MacBook, 100 MB of RAM.

## Layout

- `model/` weights (98 MB, int8 + one scale per row), tokenizer, test vectors
- `engine/` `tokenizer.ts`, `model.ts`, `vertigo.ts` (generation + Wikipedia tool), `ask.ts`, `test.ts`
- `bench/` 150 questions and `score.ts`

97M parameters, 12 layers, 1024 context, 16k vocab. Pretrained on French Wikipedia and French web text, then
trained to chat and to call its tool. Trained on one PC with an 8 GB GPU.

## Bench

150 French questions, six categories, graded by rules (no judge model). Score = mean of the categories.

| Model | Total | faits | outils | calcul | lecture | demandes | conversation |
|---|---|---|---|---|---|---|---|
| Vertigo 3 | **56.4** | 55.0 | 100.0 | 33.3 | 60.0 | 30.0 | 60.0 |

Run of 1 November 2026 with this engine. The same model scored 54.7 in the training framework: the gap is the
facts category, which depends on what Wikipedia returns on the day.

It's small and it shows: it sometimes searches when it shouldn't, poems are short, facts from memory are
hit or miss. The answers of the last run are in `bench/results.json`.

## Apps

Vertigo runs in the [ParadoxalAI app](https://paradoxalai.com/download) (Mac, Windows, Linux), the phone app and
the terminal, with a receipt on every answer.

## Licence

Apache 2.0, code and weights. contact@weareparis.eu
