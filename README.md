# Vertigo 3

A small French language model, trained by [ParadoxalAI](https://paradoxalai.com) on its own code and its own data,
with its own engine. It runs on any computer with Node.js: no GPU, no Python, no dependency to install.

What makes it different: when it does not know, it asks to read. It writes `[outil:cherche|…]`, the engine fetches
the opening of the matching French Wikipedia article, and the answer comes with its source. You can check it.

```
$ npm run ask "Qui a écrit Les Misérables ?"
Il s'agit d'un roman de Victor Hugo publié en 1862.

Source : Les Misérables · https://fr.wikipedia.org/wiki/Les_Misérables
```

## Run it

Node.js 22.6 or later. Clone, then:

```bash
npm test                                   # the engine gives the same numbers as the training framework
npm run ask "Pourquoi le ciel est bleu ?"  # ask anything, in French
npm run ask -- --no-wiki "Donne-moi trois idées de repas rapides"
npm run bench                              # the 150-question bench (about 40 minutes, greedy decoding)
```

About 60 ms per token on a MacBook. The model takes 100 MB of memory.

## What is in here

| Folder | What |
|---|---|
| `model/` | The weights (98 MB, 8-bit with one scale per row), the tokenizer, and three test vectors |
| `engine/` | The engine in TypeScript: `tokenizer.ts`, `model.ts` (the transformer, incremental decoding), `vertigo.ts` (generation and the Wikipedia tool), `ask.ts` (the command line), `test.ts` |
| `bench/` | 150 French questions in six categories and the rule-based scorer, `score.ts` |

The model: 97 million parameters, 12 layers, context of 1024 tokens, a 16k vocabulary. Pretrained on French
Wikipedia and French web text, then trained to converse and to call its tool. Everything was trained on one
gaming PC with an 8 GB GPU.

## The bench

150 questions nobody trained on, scored by rules, no judge model: a fact is right or wrong, a tool call is the
right one or not, a poem has at least the asked number of words and no refusal. The score is the mean of the
six categories.

| Model | Total | faits | outils | calcul | lecture | demandes | conversation |
|---|---|---|---|---|---|---|---|
| Vertigo 3 | **54.7** | 45.0 | 100.0 | 33.3 | 60.0 | 30.0 | 60.0 |

It is a small model. It makes mistakes, it sometimes asks for Wikipedia when it should not, and its poems are
short. The number is here so that the next version has something to beat, in public.

## In the apps

Vertigo answers in the [ParadoxalAI app](https://paradoxalai.com/download) on Mac, Windows and Linux, in the
phone app, and in the terminal. Every answer comes with a receipt: what was read, what was computed, how long it
took.

## Licence

Apache 2.0, for the code and the weights. If you build something with it, tell us: contact@weareparis.eu.
