# What are the records, and what did the first experiment measure?

The first experiment used lists of integer pairs, not conversations. An illustrative history begins:

```text
key 12 -> value 7
key 9  -> value 22
key 3  -> value 15
... 37 more pairs ...
```

The 40-record condition means each list has exactly 40 such pairs. Keys are distinct within a list and come from 128 possible integers; values are independently drawn from 32 possible integers. Different lists have different assignments. There is no persistent rule that key 12 means value 7.

The question is simply "What value was assigned to key 12?" The correct answer in that example is 7. An answer counts as correct only if the model selects that exact value. A blind uniform guess has a 1/32, or 3.125%, success rate.

The recent workspace contains the newest eight pairs. The older 32 pairs are blended into eight numerical memory slots, four pairs per slot. These slots do not hold readable notes: each is a 16-by-32 matrix of floating-point numbers. Learned key features determine how facts are written and read. The slot count and overwrite schedule are programmed rules, not learned policies.

For the reported old-query condition, the selected key belongs to those older 32 pairs. Across three training seeds and 512 fresh lists per seed, the trained student answered 92.3% correctly. Before training, its random features already scored 87.4%. Correct history mattered: removing or substituting another list's memory reduced accuracy to about 3%. Exact dictionary lookup got every answer right.

This demonstrated that the code can store information in a learned representation, use it later, and train that representation through an answer-prediction objective. It did not establish language understanding, useful text compression, superior retrieval, learned forgetting, or practical KV-cache savings. The toy matrices actually occupy more bytes than the original integer pairs.

Longer lists deliberately exceed the ring's capacity. After overwrite, some correct answers no longer exist in the state; declining accuracy is expected. Even when evidence survives, blended memories can interfere with one another. The first results also show old memory interfering with answers about recent records.

## Moving to a real language model

The next experiment replaces integer pairs with sentences such as:

```text
The color of the lantern is blue.
The color of the bicycle is red.
The color of the notebook is green.
The color of the umbrella is white.
According to the facts, the color of the lantern is
```

This is a real pretrained language model predicting the next text token. A full-context run sees all four facts. A restricted run sees only the last two facts plus the question. A small trained module compresses the removed first two facts into four learned vectors inserted ahead of the restricted context.

The model's main weights stay frozen. The compressor is trained to match the full-context model's next-token distribution. The compressor sees only the old facts, never the future question. This is an initial test of whether learned vectors can provide usable information to a frozen language model; it is not yet the complete recurrent-ring design.

Eight color words are used so the result is easy to score. Reports distinguish selecting the correct color among those eight from predicting its actual token as the highest-probability output across the entire vocabulary. Neither score measures unrestricted conversation quality.

## Why not start with Llama 8B?

An 8B model in BF16 has roughly 16 GB of weights alone. Activations, working memory and training overhead come on top. Quantized inference, and some carefully configured adapter training, can fit in 16 GB, so the limitation is not absolute. But quantization, longer runtimes and tighter memory budgets would complicate the first diagnostic experiment.

A 135M model leaves ample room for testing the memory interface and catching failures. Its limitation is ability: if it cannot solve a task even with all the text, it is a poor teacher for that task. We measure that before training and report the failures alongside any successful short-context result.
